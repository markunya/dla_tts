from src.metrics.tracker import MetricTracker
from src.trainer.base_trainer import BaseTrainer
from src.utils.model_utils import requires_grad

class GanTrainer(BaseTrainer):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        assert 'generator' in self.models
        assert 'discriminator' in self.models

    def process_batch(self, batch, metrics: MetricTracker):
        gen = self.models['generator']
        gen_optimizer = self.optimizers['generator']
        gen_loss_builder = self.loss_builders['generator']

        disc = self.models['discriminator']
        disc_optimizer = self.optimizers['discriminator']
        disc_loss_builder = self.loss_builders['discriminator']

        batch = self.move_batch_to_device(batch)
        batch = self.transform_batch(batch)

        metric_funcs = self.metrics["train"] if self.is_train else self.metrics["inference"]

        batch['mel'] = self.mel(batch['wav'])

        gen_out = gen(batch)
        batch['gen_wav'] = gen_out['gen_wav']
        batch['gen_mel'] = self.mel_for_loss(batch['gen_wav'])
        batch['real_mel'] = self.mel_for_loss(batch['wav'])

        if self.is_train:
            requires_grad(disc, True)
            disc_optimizer.zero_grad()

            batch_disc = {k: v for k, v in batch.items()}
            batch_disc['gen_wav'] = batch['gen_wav'].detach()
            batch_disc['gen_mel'] = self.mel_for_loss(batch_disc['gen_wav'])

            disc_out = disc(batch_disc)
            disc_loss, disc_losses_dict = disc_loss_builder.calculate_loss(**disc_out)
            disc_losses_dict = {f'discriminator_{key}': val for key, val in disc_losses_dict.items()}
            disc_loss.backward()
            self._clip_grad_norm('discriminator')
            disc_optimizer.step()

            requires_grad(disc, False)
            gen_optimizer.zero_grad()

            disc_out = disc(batch)
            gen_loss, gen_losses_dict = gen_loss_builder.calculate_loss(
                **(disc_out | batch)
            )
            gen_losses_dict = {f'generator_{key}': val for key, val in gen_losses_dict.items()}
            gen_loss.backward()
            self._clip_grad_norm('generator')
            gen_optimizer.step()

            batch.update({**gen_losses_dict, **disc_losses_dict})

            for loss_name in self.config.writer.loss_names:
                if loss_name in batch:
                    metrics.update(loss_name, batch[loss_name].item())

        for met in metric_funcs:
            metrics.update(met.name, met(**batch))

        return batch


    def _log_batch(self, batch):
        self.log_audios(batch['wav'], 'real_wav')
        self.log_audios(batch['gen_wav'], 'gen_wav')
        self.log_mel(batch['mel'], 'real_mel')
        self.log_mel(batch['gen_mel'], 'gen_mel')
        

    def log_audios(self, wav, audio_name):
        for i in range(min(self.examples_to_log_on_val, wav.shape[0])):
            audio_for_writer = wav[i].detach().cpu()
            self.writer.add_audio(
                f"{audio_name}_{i+1}",
                audio_for_writer,
                sample_rate=self.sr
            )
    
    def log_mel(self, mel, mel_name):
        for i in range(min(self.examples_to_log_on_val, mel.shape[0])):
            mel_for_writer = mel[i].detach().cpu()

            if mel_for_writer.dim() == 3:
                mel_for_writer = mel_for_writer.squeeze(0)

            mel_min = mel_for_writer.min()
            mel_max = mel_for_writer.max()
            mel_for_writer = (mel_for_writer - mel_min) / (mel_max - mel_min + 1e-9)

            self.writer.add_image(
                f"{mel_name}_{i+1}",
                mel_for_writer,
                step=self.step,
            )