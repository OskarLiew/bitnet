from math import prod

import torch
from torch import Tensor, nn

from .ste import ste

EPS = 1e-9


class BitLinear(nn.Linear):
    def __init__(
        self,
        in_features: int,
        out_features: int,
        bias: bool = True,
        b: int = 8,
    ) -> None:
        super().__init__(in_features, out_features, bias)

        self.in_features = in_features
        self.out_features = out_features
        self.qb = 2 ** (b - 1)

        self.linear_norm = nn.LayerNorm(in_features)

    def binarize_weights(self) -> Tensor:
        return torch.sign(self.weight - self.weight.mean())

    def quantize_activations(self, x: Tensor) -> tuple[Tensor, Tensor]:
        gamma = x.abs().max()
        quantized = torch.clamp(x * self.qb / gamma, -self.qb + EPS, self.qb - EPS)
        return quantized, gamma

    def dequantize_activations(self, xq: Tensor, gamma: Tensor) -> Tensor:
        beta = self.weight.abs().sum() / prod(self.weight.shape)
        return xq * beta * gamma / self.qb

    def forward(self, x: Tensor) -> Tensor:
        # Normalize input
        out = self.linear_norm(x)

        # Binarize weights
        bin_weights = self.binarize_weights()

        # Quantize activations
        out, gamma = self.quantize_activations(out)

        # Apply linear transformation
        out = nn.functional.linear(out, ste(bin_weights, self.weight), bias=self.bias)

        # Dequantize
        return self.dequantize_activations(out, gamma)


def replace_with_bitlinear(model: nn.Module, b: int = 8) -> None:
    """
    Replaces all linear layers in a model with BitLinear layers.
    """
    for name, module in model.named_children():
        if isinstance(module, nn.Linear) and name != "lm_head":
            setattr(
                model,
                name,
                BitLinear(
                    in_features=module.in_features,
                    out_features=module.out_features,
                    bias=module.bias is not None,
                    b=b,
                ),
            )
        else:
            replace_with_bitlinear(module)
