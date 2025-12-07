from abc import abstractmethod, ABC
from hydra.utils import instantiate
from copy import deepcopy

import torch
from numpy import inf
from torch.nn.utils import clip_grad_norm_
from tqdm.auto import tqdm

from src.datasets.data_utils import inf_loop
from src.metrics.tracker import MetricTracker
from src.utils.io_utils import ROOT_PATH
from src.utils.mel_spec import MelSpectrogram, MelSpectrogramConfig

from src.datasets.data_utils import get_dataloaders
from src.loss.loss_builder import LossBuilder

class BaseTrainer(ABC):
    def __init__(
        self,
        config,
        logger,
        writer,
        epoch_len=None,
        skip_oom=True,
    ):
        """
        Args:
            config (DictConfig): experiment config containing training config.
            logger (Logger): logger that logs output.
            writer (WandBWriter | CometMLWriter): experiment tracker.
            epoch_len (int | None): number of steps in each epoch for
                iteration-based training. If None, use epoch-based
                training (len(dataloader)).
            skip_oom (bool): skip batches with the OutOfMemory error.
            batch_transforms (dict[Callable] | None): transforms that
                should be applied on the whole batch. Depend on the
                tensor name.
        """
        self.is_train = True

        self.config = config
        self.cfg_trainer = self.config.trainer

        if config.trainer.device == "auto":
            self.device = "cuda" if torch.cuda.is_available() else "cpu"
        else:
            self.device = config.trainer.device
        self.skip_oom = skip_oom

        self._setup_models()
        self._setup_losses()
        self._setup_optimizers()
        self._setup_lr_schedulers()
        self._load_checkpoints()

        dataloaders, self.batch_transforms = get_dataloaders(config, self.device)

        self.logger = logger
        self.log_step = config.trainer.get("log_step", 50)

        self.train_dataloader = dataloaders["train"]
        if epoch_len is None:
            self.epoch_len = len(self.train_dataloader)
        else:
            self.train_dataloader = inf_loop(self.train_dataloader)
            self.epoch_len = epoch_len

        self.evaluation_dataloaders = {
            k: v for k, v in dataloaders.items() if k != "train"
        }

        self._last_epoch = 0
        self.start_epoch = 1
        self.epochs = self.cfg_trainer.n_epochs
        self.sr = self.cfg_trainer.sr
        self.examples_to_log_on_val = self.cfg_trainer.examples_to_log_on_val

        self.save_period = (
            self.cfg_trainer.save_period
        )
        self.monitor = self.cfg_trainer.get(
            "monitor", "off"
        )

        if self.monitor == "off":
            self.mnt_mode = "off"
            self.mnt_best = 0
        else:
            self.mnt_mode, self.mnt_metric = self.monitor.split()
            assert self.mnt_mode in ["min", "max"]

            self.mnt_best = inf if self.mnt_mode == "min" else -inf
            self.early_stop = self.cfg_trainer.get("early_stop", inf)
            if self.early_stop <= 0:
                self.early_stop = inf

        self.writer = writer

        self.metrics = instantiate(config.metrics)
        self.train_metrics = MetricTracker(
            *self.config.writer.loss_names,
            "grad_norm",
            *[m.name for m in self.metrics["train"]],
            writer=self.writer,
        )
        self.evaluation_metrics = MetricTracker(
            *self.config.writer.loss_names,
            *[m.name for m in self.metrics["inference"]],
            writer=self.writer,
        )

        self.checkpoint_dir = (
            ROOT_PATH / config.trainer.save_dir / config.writer.run_name
        )

        mel_conf: MelSpectrogramConfig = instantiate(config.mel)
        self.mel = MelSpectrogram(mel_conf)
        mel_conf_for_loss = deepcopy(mel_conf)
        mel_conf_for_loss.f_max = None
        self.mel_for_loss = MelSpectrogram(mel_conf_for_loss)


    def _setup_models(self):
        self.models = {}
        for model_name, model_config in self.config.models.items():
            self.models[model_name] = instantiate(model_config.model)
        self.logger.info(f'Models successfully initialized: {list(self.models.keys())}')
    
    def _setup_losses(self):
        self.loss_builders = {}
        for model_name, model_config in self.config.models.items():
            self.loss_builders[model_name] = LossBuilder(instantiate(model_config.losses))
        self.logger.info('Losses successfuly initialized')
    
    def _setup_optimizers(self):
        self.optimizers = {}
        for model_name, model_config in self.config.models.items():
            self.optimizers[model_name] = instantiate(model_config.optimizer)
        self.logger.info('Optimizers and schedulers successfully initialized')
    
    def _setup_lr_schedulers(self):
        self.lr_schedulers = {}
        for model_name, model_config in self.config.models.items():
            self.lr_schedulers[model_name] = instantiate(model_config.lr_scheduler)
        self.logger.info('Lr schedulers and schedulers successfully initialized')

    def _step_lr_schedulers(self):
        for _, lr_scheduler in self.lr_schedulers:
            lr_scheduler.step()

    def train(self):
        try:
            self._train_process()
        except KeyboardInterrupt as e:
            self.logger.info("Saving model on keyboard interrupt")
            self._save_checkpoints(self._last_epoch, save_best=False)
            raise e

    def _train_process(self):
        not_improved_count = 0
        for epoch in range(self.start_epoch, self.epochs + 1):
            self._last_epoch = epoch
            result = self._train_epoch(epoch)

            logs = {"epoch": epoch}
            logs.update(result)

            for key, value in logs.items():
                self.logger.info(f"    {key:15s}: {value}")

            best, stop_process, not_improved_count = self._monitor_performance(
                logs, not_improved_count
            )

            if epoch % self.save_period == 0 or best:
                self._save_checkpoints(epoch, save_best=best, only_best=True)

            if stop_process:
                break

    def _train_epoch(self, epoch):
        self.is_train = True
        self.to_train()
        self.train_metrics.reset()
        self.writer.set_step((epoch - 1) * self.epoch_len)
        self.writer.add_scalar("epoch", epoch)
        for batch_idx, batch in enumerate(
            tqdm(self.train_dataloader, desc="train", total=self.epoch_len)
        ):
            try:
                batch = self.process_batch(
                    batch,
                    metrics=self.train_metrics,
                )
            except torch.cuda.OutOfMemoryError as e:
                if self.skip_oom:
                    self.logger.warning("OOM on batch. Skipping batch.")
                    torch.cuda.empty_cache()
                    continue
                else:
                    raise e

            for model_name in self.models.keys():
                self.train_metrics.update(
                    f"{model_name}_grad_norm",
                    self._get_grad_norm(model_name)
                )

            if batch_idx % self.log_step == 0:
                self.writer.set_step((epoch - 1) * self.epoch_len + batch_idx)
                for model_name in self.models.keys():
                    self.logger.debug(
                        "Train Epoch: {} {} {} total loss: {:.6f}".format(
                            epoch,
                            self._progress(batch_idx),
                            model_name,
                            batch[f"{model_name}_loss"]
                        )
                    )

                for model_name in self.models.keys():
                    self.writer.add_scalar(
                        f"{model_name}_learning rate",
                        self.lr_schedulers[model_name].get_last_lr()[0]
                    )

                self._log_scalars(self.train_metrics)
                self._log_batch(batch_idx, batch)

                last_train_metrics = self.train_metrics.result()
                self.train_metrics.reset()
            if batch_idx + 1 >= self.epoch_len:
                break

        logs = last_train_metrics

        for part, dataloader in self.evaluation_dataloaders.items():
            val_logs = self._evaluation_epoch(epoch, part, dataloader)
            logs.update(**{f"{part}_{name}": value for name, value in val_logs.items()})

        return logs
    
    def to_train(self):
        for model in self.models.values():
            model.train()

    def to_eval(self):
        for model in self.models.values():
            model.eval()

    def _evaluation_epoch(self, epoch, part, dataloader):
        self.is_train = False
        self.to_eval()

        self.evaluation_metrics.reset()
        with torch.no_grad():
            for batch_idx, batch in tqdm(
                enumerate(dataloader),
                desc=part,
                total=len(dataloader),
            ):
                batch = self.process_batch(
                    batch,
                    metrics=self.evaluation_metrics,
                )
            self.writer.set_step(epoch * self.epoch_len, part)
            self._log_scalars(self.evaluation_metrics)
            self._log_batch(
                batch_idx, batch, part
            )

        return self.evaluation_metrics.result()

    def _monitor_performance(self, logs, not_improved_count):
        best = False
        stop_process = False
        if self.mnt_mode != "off":
            try:
                if self.mnt_mode == "min":
                    improved = logs[self.mnt_metric] <= self.mnt_best
                elif self.mnt_mode == "max":
                    improved = logs[self.mnt_metric] >= self.mnt_best
                else:
                    improved = False
            except KeyError:
                self.logger.warning(
                    f"Warning: Metric '{self.mnt_metric}' is not found. "
                    "Model performance monitoring is disabled."
                )
                self.mnt_mode = "off"
                improved = False

            if improved:
                self.mnt_best = logs[self.mnt_metric]
                not_improved_count = 0
                best = True
            else:
                not_improved_count += 1

            if not_improved_count >= self.early_stop:
                self.logger.info(
                    "Validation performance didn't improve for {} epochs. "
                    "Training stops.".format(self.early_stop)
                )
                stop_process = True
        return best, stop_process, not_improved_count

    def move_batch_to_device(self, batch):
        for tensor_for_device in self.cfg_trainer.device_tensors:
            batch[tensor_for_device] = batch[tensor_for_device].to(self.device)
        return batch

    def transform_batch(self, batch):
        transform_type = "train" if self.is_train else "inference"
        transforms = self.batch_transforms.get(transform_type)
        if transforms is not None:
            for transform_name in transforms.keys():
                batch[transform_name] = transforms[transform_name](
                    batch[transform_name]
                )
        return batch

    def _clip_grad_norm(self, model_name):
        if self.config["trainer"].get("max_grad_norm", None) is not None:
            clip_grad_norm_(
                self.models[model_name].parameters(),
                self.config["trainer"]["max_grad_norm"]
            )

    @torch.no_grad()
    def _get_grad_norm(self, model_name, norm_type=2):
        parameters = self.models[model_name].parameters()
        if isinstance(parameters, torch.Tensor):
            parameters = [parameters]
        parameters = [p for p in parameters if p.grad is not None]
        total_norm = torch.norm(
            torch.stack([torch.norm(p.grad.detach(), norm_type) for p in parameters]),
            norm_type,
        )
        return total_norm.item()

    def _progress(self, batch_idx):
        base = "[{}/{} ({:.0f}%)]"
        if hasattr(self.train_dataloader, "n_samples"):
            current = batch_idx * self.train_dataloader.batch_size
            total = self.train_dataloader.n_samples
        else:
            current = batch_idx
            total = self.epoch_len
        return base.format(current, total, 100.0 * current / total)

    @abstractmethod
    def _log_batch(self, batch_idx, batch, mode="train"):
        raise NotImplementedError
    
    @abstractmethod
    def process_batch(self, batch):
        raise NotImplementedError

    def _log_scalars(self, metric_tracker: MetricTracker):
        if self.writer is None:
            return
        for metric_name in metric_tracker.keys():
            self.writer.add_scalar(f"{metric_name}", metric_tracker.avg(metric_name))

    def _save_checkpoint(self, model_name, epoch, save_best=False, only_best=False):
        arch = type(self.models[model_name]).__name__
        state = {
            "arch": arch,
            "epoch": epoch,
            "state_dict": self.models[model_name].state_dict(),
            "optimizer": self.optimizers[model_name].state_dict(),
            "lr_scheduler": self.lr_schedulers[model_name].state_dict(),
            "monitor_best": self.mnt_best,
            "config": self.config,
        }

        model_checkpoint_dir = self.checkpoint_dir / model_name
        filename = str(model_checkpoint_dir / f"checkpoint-epoch{epoch}.pth")

        model_checkpoint_dir.mkdir(parents=True, exist_ok=True)

        if only_best and not save_best:
            self.logger.warning(
                "only_best=True, но save_best=False — сохраняется только обычный checkpoint."
            )

        if not only_best:
            torch.save(state, filename)
            if self.config.writer.log_checkpoints:
                self.writer.add_checkpoint(filename, str(self.checkpoint_dir.parent))
            self.logger.info(f"Saving checkpoint: {filename} ...")

        if save_best:
            best_path = str(model_checkpoint_dir / "model_best.pth")
            torch.save(state, best_path)
            if self.config.writer.log_checkpoints:
                self.writer.add_checkpoint(best_path, str(self.checkpoint_dir.parent))
            self.logger.info("Saving current best: model_best.pth ...")


    def _load_checkpoint(self, model_name, resume_path):
        resume_path = str(resume_path)
        self.logger.info(f"Loading checkpoint for {model_name}: {resume_path} ...")

        checkpoint = torch.load(resume_path, map_location=self.device)

        if checkpoint["config"]["model"] != self.config["model"]:
            self.logger.warning(
                f"Warning: Architecture of {model_name} configuration given in the config file is different "
                "from that of the checkpoint. This may yield an exception when state_dict is loaded."
            )

        self.models[model_name].load_state_dict(checkpoint["state_dict"])

        self.start_epoch = checkpoint["epoch"] + 1
        self.mnt_best = checkpoint["monitor_best"]

        if (
            checkpoint["config"]["optimizer"] != self.config["optimizer"]
            or checkpoint["config"]["lr_scheduler"] != self.config["lr_scheduler"]
        ):
            self.logger.warning(
                f"Warning: For {model_name} ptimizer or lr_scheduler given in the config file is different "
                "from that of the checkpoint. Optimizer and scheduler parameters "
                "are not resumed."
            )
        else:
            self.optimizers[model_name].load_state_dict(checkpoint["optimizer"])
            self.lr_schedulers[model_name].load_state_dict(checkpoint["lr_scheduler"])

        self.logger.info(
            f"Checkpoint for {model_name} loaded. Resume training from epoch {self.start_epoch}"
        )

    def _load_checkpoints(self):
        for model_name, model_config in self.config.models.items():
            self._load_checkpoint(model_name, model_config.checkpoint_path)

    def _save_checkpoints(self, epoch, save_best=False, only_best=False):
        for model_name in self.models.keys():
            self._save_checkpoint(model_name, epoch, save_best, only_best)
