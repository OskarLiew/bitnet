from bitlinear import BitLinear
from torch import Tensor, nn

EPS = 1e-9


class TernaryBitLinear(BitLinear):
    def binarize_weights(self) -> Tensor:
        w = self.weight / (self.weight.abs().mean() + EPS)
        return w.round().clamp(-1, 1)


def replace_with_bitlinear(model: nn.Module, b: int = 8) -> None:
    """
    Replaces all linear layers in a model with BitLinear layers.
    """
    for name, module in model.named_children():
        if isinstance(module, nn.Linear) and name != "lm_head":
            setattr(
                model,
                name,
                TernaryBitLinear(
                    in_features=module.in_features,
                    out_features=module.out_features,
                    bias=module.bias is not None,
                    b=b,
                ),
            )
        else:
            replace_with_bitlinear(module)
