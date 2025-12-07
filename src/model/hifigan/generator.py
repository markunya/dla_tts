import torch
import torch.nn.functional as F
import torch.nn as nn
from src.model.base_model import BaseModel
from typing import Literal, Sequence
from tqdm import tqdm
from src.utils.model_utils import get_padding, init_weights, get_norm
from src.model.hifigan.common import ResBlock, LEACKY_RELU_SLOPE
from torch.nn.utils import remove_weight_norm


class HifiGenerator(BaseModel):
    def __init__(
        self,
        resblock_kernel_sizes: Sequence[int],
        upsample_rates: Sequence[int],
        upsample_initial_channel: int,
        upsample_kernel_sizes: Sequence[int],
        resblock_dilation_sizes,
        norm_type: Literal["weight", "spectral"] = "weight",
    ):
        super().__init__()

        self.num_kernels = len(resblock_kernel_sizes)
        self.num_upsamples = len(upsample_rates)

        if len(upsample_kernel_sizes) != self.num_upsamples:
            raise ValueError(
                f"upsample_kernel_sizes must have length {self.num_upsamples}, "
                f"got {len(upsample_kernel_sizes)}"
            )

        self.norm_type = norm_type

        norm = get_norm(norm_type)

        self.conv_pre = norm(
            nn.Conv1d(
                in_channels=80,
                out_channels=upsample_initial_channel,
                kernel_size=7,
                stride=1,
                padding=3,
            )
        )

        def make_upsample(idx: int, rate: int, kernel_size: int) -> nn.Module:
            in_ch = upsample_initial_channel // (2 ** idx)
            out_ch = upsample_initial_channel // (2 ** (idx + 1))
            return norm(
                nn.ConvTranspose1d(
                    in_channels=in_ch,
                    out_channels=out_ch,
                    kernel_size=kernel_size,
                    stride=rate,
                    padding=(kernel_size - rate) // 2,
                )
            )

        self.ups = nn.ModuleList(
            [
                make_upsample(i, u, k)
                for i, (u, k) in enumerate(zip(upsample_rates, upsample_kernel_sizes))
            ]
        )

        self.resblocks = nn.ModuleList()
        for i in range(self.num_upsamples):
            ch = upsample_initial_channel // (2 ** (i + 1))
            for k, d in zip(resblock_kernel_sizes, resblock_dilation_sizes):
                self.resblocks.append(ResBlock(ch, k, d, norm_type))

        final_channels = upsample_initial_channel // (2 ** self.num_upsamples)
        self.conv_post = norm(
            nn.Conv1d(
                in_channels=final_channels,
                out_channels=1,
                kernel_size=7,
                stride=1,
                padding=3,
            )
        )

        self.ups.apply(init_weights)
        self.conv_post.apply(init_weights)

    def forward(self, batch: dict) -> torch.Tensor:
        x = batch['mel']
        if len(x.shape) == 4:
            x = x.squeeze(1)

        x = self.conv_pre(x)
        for i in range(self.num_upsamples):
            x = F.leaky_relu(x, LEACKY_RELU_SLOPE)
            x = self.ups[i](x)

            xs = None
            start = i * self.num_kernels
            end = start + self.num_kernels

            for resblock in self.resblocks[start:end]:
                out = resblock(x)
                xs = out if xs is None else xs + out

            x = xs / self.num_kernels

        x = F.leaky_relu(x, LEACKY_RELU_SLOPE)
        x = self.conv_post(x)
        x = torch.tanh(x)
        return {'gen_wav': x}

    def remove_weight_norm(self) -> None:
        tqdm.write("Removing weight norm...")
        for layer in self.ups:
            remove_weight_norm(layer)
        for block in self.resblocks:
            block.remove_weight_norm()
        remove_weight_norm(self.conv_pre)
        remove_weight_norm(self.conv_post)

