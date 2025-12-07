from torch import nn
from src.utils.model_utils import get_norm, get_padding
from src.model.hifigan.common import LEACKY_RELU_SLOPE

import torch
import torch.nn.functional as F
from typing import Literal

class HifiPeriodDiscriminator(torch.nn.Module):
    def __init__(
        self,
        period,
        kernel_size=5,
        stride=3,
        norm_type: Literal["weight", "spectral"] = "weight"
    ):
        super().__init__()
        self.period = period
        norm = get_norm(norm_type)
        self.convs = nn.ModuleList([
            norm(nn.Conv2d(1, 32, (kernel_size, 1), (stride, 1), padding=(get_padding(5, 1), 0))),
            norm(nn.Conv2d(32, 128, (kernel_size, 1), (stride, 1), padding=(get_padding(5, 1), 0))),
            norm(nn.Conv2d(128, 512, (kernel_size, 1), (stride, 1), padding=(get_padding(5, 1), 0))),
            norm(nn.Conv2d(512, 1024, (kernel_size, 1), (stride, 1), padding=(get_padding(5, 1), 0))),
            norm(nn.Conv2d(1024, 1024, (kernel_size, 1), 1, padding=(2, 0))),
        ])
        self.conv_post = norm(nn.Conv2d(1024, 1, (3, 1), 1, padding=(1, 0)))

    def forward(self, x):
        fmaps = []

        b, c, t = x.shape
        if t % self.period != 0:
            n_pad = self.period - (t % self.period)
            x = F.pad(x, (0, n_pad), "reflect")
            t = t + n_pad
        x = x.view(b, c, t // self.period, self.period)

        for l in self.convs:
            x = l(x)
            x = F.leaky_relu(x, LEACKY_RELU_SLOPE)
            fmaps.append(x)
        x = self.conv_post(x)
        fmaps.append(x)
        x = torch.flatten(x, 1, -1)

        return x, fmaps

class HifiMultiPeriodDiscriminator(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.discriminators = nn.ModuleList([
            HifiPeriodDiscriminator(2),
            HifiPeriodDiscriminator(3),
            HifiPeriodDiscriminator(5),
            HifiPeriodDiscriminator(7),
            HifiPeriodDiscriminator(11),
        ])

    def forward(self, real_wav, gen_wav):
        discs_real_out = []
        discs_gen_out = []
        fmaps_real = []
        fmaps_gen = []
        for disc in self.discriminators:
            disc_real_out, fmap_real = disc(real_wav)
            disc_gen_out, fmap_gen = disc(gen_wav)
            discs_real_out.append(disc_real_out)
            fmaps_real.append(fmap_real)
            discs_gen_out.append(disc_gen_out)
            fmaps_gen.append(fmap_gen)

        return discs_real_out, discs_gen_out, fmaps_real, fmaps_gen
