import torch
import torchaudio
from src.metrics.base_metric import BaseMetric

class AudioMetricWrapper(BaseMetric):
    def __init__(
        self,
        metric,
        name,
        in_sr=22050,
        target_sr=16000,
        device="auto"
    ):
        super().__init__(name)
        if device == "auto":
            device = "cuda" if torch.cuda.is_available() else "cpu"
        self.metric = metric.to(device)
        self.resampler = torchaudio.transforms.Resample(
            orig_freq=in_sr,
            new_freq=target_sr,
            resampling_method="sinc_interpolation"
        ).to(device)

    def __call__(self, wav: torch.Tensor, gen_wav: torch.Tensor, **batch):
        wav = self.resampler(wav)
        gen_wav = self.resampler(gen_wav)
        return self.metric(gen_wav, wav)
