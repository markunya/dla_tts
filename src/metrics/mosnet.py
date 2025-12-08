import torch
import torchaudio
import numpy as np
from src.model.mosnet import Wav2Vec2MOS
from src.metrics.base_metric import BaseMetric

class MOSNet(BaseMetric):
    def __init__(
        self,
        name,
        in_sr=22050,
        target_sr=16000,
        device='auto'
    ):
        super().__init__(name)
        if device == "auto":
            device = "cuda" if torch.cuda.is_available() else "cpu"

        self.model = Wav2Vec2MOS(
            "src/metrics/weights/wave2vec2mos.pth",
            device
        ).to(device)
        self.resampler = torchaudio.transforms.Resample(
            orig_freq=in_sr,
            new_freq=target_sr,
            resampling_method="sinc_interpolation"
        ).to(device)

    def __call__(self, gen_wav, **kwargs) -> float:
        mos_scores = []

        for gen_wav_ in gen_wav:
            gen_wav_ = gen_wav_ / gen_wav_.abs().max()
            gen_wav_resampled = self.resampler(gen_wav_)

            input_values = self.model.processor(
                gen_wav_resampled.cpu().numpy(),
                return_tensors="pt",
                sampling_rate=self.model.sample_rate
            ).input_values.to(gen_wav.device)

            with torch.no_grad():
                mos_score = self.model(input_values).item()
            mos_scores.append(mos_score)

        return float(np.mean(mos_scores).item())