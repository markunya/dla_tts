import torch.nn.functional as F

from torch import nn
from src.utils.model_utils import get_norm, get_padding, init_weights
from torch.nn.utils import remove_weight_norm
from typing import Literal, Sequence

LEACKY_RELU_SLOPE = 0.2


class ResBlock(nn.Module):
    def __init__(
        self,
        channels: int,
        kernel_size: int = 3,
        dilation: Sequence[int] = (1, 3, 5),
        norm_type: Literal["weight", "spectral"] = "weight",
    ) -> None:
        super().__init__()

        if len(dilation) != 3:
            raise ValueError(f"dilation must have length 3, got {len(dilation)}")

        self.norm_type = norm_type

        def make_conv(d: int) -> nn.Module:
            return get_norm(norm_type)(
                nn.Conv1d(
                    in_channels=channels,
                    out_channels=channels,
                    kernel_size=kernel_size,
                    stride=1,
                    dilation=d,
                    padding=get_padding(kernel_size, d),
                )
            )

        self.convs1 = nn.ModuleList([make_conv(d) for d in dilation])
        self.convs1.apply(init_weights)

        self.convs2 = nn.ModuleList([make_conv(1) for _ in dilation])
        self.convs2.apply(init_weights)

    def forward(self, x):
        for c1, c2 in zip(self.convs1, self.convs2):
            xt = F.leaky_relu(x, LEACKY_RELU_SLOPE)
            xt = c1(xt)
            xt = F.leaky_relu(xt, LEACKY_RELU_SLOPE)
            xt = c2(xt)
            x = xt + x
        return x

    def remove_weight_norm(self) -> None:
        if self.norm_type != "weight":
            raise RuntimeWarning("No weight norm to remove")

        for convs in (self.convs1, self.convs2):
            for layer in convs:
                remove_weight_norm(layer)
