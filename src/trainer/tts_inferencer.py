import torch
import torchaudio
from tqdm.auto import tqdm

from pathlib import Path
from src.metrics.tracker import MetricTracker
from src.trainer.base_trainer import BaseTrainer
from hydra.utils import instantiate

from src.datasets.data_utils import get_dataloaders

class TTSInferencer(BaseTrainer):
    """
    Inferencer (Like Trainer but for Inference) class

    The class is used to process data without
    the need of optimizers, writers, etc.
    Required to evaluate the model on the dataset, save predictions, etc.
    """

    def __init__(
        self,
        config,
        save_path
    ):
        """
        Initialize the Inferencer.

        Args:
            config (DictConfig): run config containing inferencer config.
            save_path (str): path to save model predictions and other
                information.
            metrics (dict): dict with the definition of metrics for
                inference (metrics[inference]). Each metric is an instance
                of src.metrics.BaseMetric.
            batch_transforms (dict[nn.Module] | None): transforms that
                should be applied on the whole batch. Depend on the
                tensor name.
            skip_model_load (bool): if False, require the user to set
                pre-trained checkpoint path. Set this argument to True if
                the model desirable weights are defined outside of the
                Inferencer Class.
        """
        self.is_train = False

        self.config = config
        self.cfg_inferencer = self.config.inferencer

        if config.inferencer.device == "auto":
            self.device = "cuda" if torch.cuda.is_available() else "cpu"
        else:
            self.device = config.inferencer.device

        dataloaders, _ = get_dataloaders(config, self.device)

        self.inf_dataloader = dataloaders["inf"]
        self.save_dir = Path(save_path)

        self._setup_models()
        checkpoint = torch.load(config.models.generator.checkpoint_path, map_location=self.device)
        self.models['generator'].load_state_dict(checkpoint["state_dict"])

        self.metrics = instantiate(config.metrics)
        self.evaluation_metrics = MetricTracker(
            *[m.name for m in self.metrics["inference"]],
        )

        self.sample_rate = 22050
        bundle = torchaudio.pipelines.TACOTRON2_WAVERNN_PHONE_LJSPEECH
        self.processor = bundle.get_text_processor()
        self.tacotron2 = bundle.get_tacotron2().to(self.device)

    @torch.no_grad()
    def process_batch(self, batch_idx, batch, metrics):
        with torch.inference_mode():
            processed, lengths = self.processor(batch['text'])
            processed = processed.to(self.device)
            lengths = lengths.to(self.device)

            mel, _, _ = self.tacotron2.infer(processed, lengths)

        outputs = self.models['generator']({'mel': mel})
        gen_wav = outputs["gen_wav"]

        batch["mel"] = mel
        batch["gen_wav"] = gen_wav

        if metrics is not None and "inference" in self.metrics:
            for met in self.metrics["inference"]:
                metrics.update(met.name, met(**batch))

        if self.save_dir is not None:
            self.save_dir.mkdir(parents=True, exist_ok=True)

            batch_size = gen_wav.shape[0]
            utt_ids = batch.get("utt_id", None)

            for i in range(batch_size):
                wav = gen_wav[i].detach().cpu()

                if wav.dim() == 1:
                    wav = wav.unsqueeze(0)

                if utt_ids is not None:
                    fname = f"{utt_ids[i]}.wav"
                else:
                    global_id = batch_idx * batch_size + i
                    fname = f"sample_{global_id}.wav"

                torchaudio.save(
                    str(self.save_dir / fname),
                    wav,
                    self.sample_rate,
                )

        return batch


    def run_inference(self):
        """
        Run inference

        Returns:
            logs (dict): metrics, calculated on the partition.
        """

        self.is_train = False
        self.to_eval()
        self.evaluation_metrics.reset()

        if self.save_dir is not None:
            self.save_dir.mkdir(exist_ok=True, parents=True)

        with torch.no_grad():
            for batch_idx, batch in tqdm(
                enumerate(self.inf_dataloader),
                total=len(self.inf_dataloader),
            ):
                batch = self.process_batch(
                    batch_idx=batch_idx,
                    batch=batch,
                    metrics=self.evaluation_metrics,
                )

        return self.evaluation_metrics.result()
