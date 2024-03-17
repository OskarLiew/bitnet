from torch import Tensor


def ste(x_quantized: Tensor, x: Tensor) -> Tensor:
    """Straight-through estimator (STE). Uses the addition and subtraction trick with
    `x` and cleaver detaching of gradients so forward passes only see `x_quantized`
    and backward passes only see `x`, i.e. it gets passed straight-through."""
    return x + (x_quantized - x).detach()
