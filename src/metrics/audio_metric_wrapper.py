import torch
from src.metrics.base_metric import BaseMetric

class AudioMetricWrapper(BaseMetric):
    def __init__(self, metric, name, device="auto"):
        super().__init__(name)
        if device == "auto":
            device = "cuda" if torch.cuda.is_available() else "cpu"
        self.metric = metric.to(device)

    def __call__(self, wav: torch.Tensor, gen_wav: torch.Tensor, **batch):
        return self.metric(gen_wav, wav)
