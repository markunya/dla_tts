from torch import nn
from src.utils.model_utils import get_norm
from src.model.hifigan.common import LEACKY_RELU_SLOPE

import torch
import torch.nn.functional as F
from typing import Literal

class HifiScaleDiscriminator(nn.Module):
    def __init__(
        self,
        norm_type: Literal["weight", "spectral"] = "weight"
    ):
        super().__init__()

        norm = get_norm(norm_type)
        self.convs = nn.ModuleList([
            norm(nn.Conv1d(1, 128, 15, 1, padding=7)),
            norm(nn.Conv1d(128, 128, 41, 2, groups=4, padding=20)),
            norm(nn.Conv1d(128, 256, 41, 2, groups=16, padding=20)),
            norm(nn.Conv1d(256, 512, 41, 4, groups=16, padding=20)),
            norm(nn.Conv1d(512, 1024, 41, 4, groups=16, padding=20)),
            norm(nn.Conv1d(1024, 1024, 41, 1, groups=16, padding=20)),
            norm(nn.Conv1d(1024, 1024, 5, 1, padding=2)),
        ])
        self.conv_post = norm(nn.Conv1d(1024, 1, 3, 1, padding=1))

    def forward(self, x):
        fmaps = []
        for l in self.convs:
            x = l(x)
            x = F.leaky_relu(x, LEACKY_RELU_SLOPE)
            fmaps.append(x)
        x = self.conv_post(x)
        fmaps.append(x)
        x = torch.flatten(x, 1, -1)

        return x, fmaps

class HifiMultiScaleDiscriminator(torch.nn.Module):
    def __init__(
        self,
        same_scale=False
    ):
        super().__init__()
        self.discriminators = nn.ModuleList([
            HifiScaleDiscriminator(norm_type='spectral'),
            HifiScaleDiscriminator(),
            HifiScaleDiscriminator(),
        ])
        self.meanpools = nn.ModuleList([
            nn.AvgPool1d(4, 2, padding=2),
            nn.AvgPool1d(4, 2, padding=2)
        ]) if not same_scale else None

    def forward(self, real_wav, gen_wav):
        discs_real_out = []
        discs_gen_out = []
        fmaps_real = []
        fmaps_gen = []
        for i, d in enumerate(self.discriminators):
            if i != 0 and self.meanpools is not None:
                real_wav = self.meanpools[i-1](real_wav)
                gen_wav = self.meanpools[i-1](gen_wav)
            disc_real_out, fmap_real = d(real_wav)
            disc_gen_out, fmap_gen = d(gen_wav)
            discs_real_out.append(disc_real_out)
            fmaps_real.append(fmap_real)
            discs_gen_out.append(disc_gen_out)
            fmaps_gen.append(fmap_gen)

        return discs_real_out, discs_gen_out, fmaps_gen, fmaps_real
