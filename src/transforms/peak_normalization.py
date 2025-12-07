import torch.nn as nn


class PeakNormalization(nn.Module):
    def __init__(self, peak_target: float = 0.99, eps: float = 1e-8):
        super().__init__()
        self.peak_target = peak_target
        self.eps = eps

    def forward(self, item: dict):
        if "wav" not in item:
            raise ValueError("wav not in item")

        x = item["wav"]
        peak = x.abs().amax().clamp_min(self.eps)
        gain = self.peak_target / peak

        item["wav"] = item["wav"] * gain

        item[
            "normalization_gain"
        ] = gain
        return item
