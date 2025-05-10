from typing import Callable, Optional

import torch
import torch.nn.functional as F
from torch import Tensor
from torch.optim.optimizer import ParamsT


# https://arxiv.org/abs/2412.13148
# if diag == True, beta is 0.4, ns_steps is 2
# if diag == False, beta is 0.8, ns_steps is 10

from torch import Tensor

# for arguments check

def must_be_positive(name: str, value: int|float|Tensor) -> None:
    if not 0 < value:
        raise ValueError(f'{name} must be positive value: {value}')

def must_be_non_negative(name: str, value: int|float|Tensor) -> None:
    if not 0 <= value:
        raise ValueError(f'{name} must be non-negative value: {value}')

def must_be_in_open_interval(name: str, value: float|Tensor, low: float, high: float) -> None:
    if not low < value < high:
        raise ValueError(f'{name} must be in ({low}, {high})')

def must_be_in_closed_interval(name: str, value: float|Tensor, low: float, high: float) -> None:
    if not low <= value <= high:
        raise ValueError(f'{name} must be in [{low}, {high}]')

def must_be_in_left_open_interval(name: str, value: float|Tensor, low: float, high: float) -> None:
    if not low < value <= high:
        raise ValueError(f'{name} must be in ({low}, {high}]')

def must_be_in_right_open_interval(name: str, value: float|Tensor, low: float, high: float) -> None:
    if not low <= value < high:
        raise ValueError(f'{name} must be in [{low}, {high})')

import math
from typing import Iterable, List, Optional, Tuple, Union

from torch import Size, Tensor
from torch.nn import Bilinear, Embedding, EmbeddingBag, Module, MultiheadAttention, Parameter
from torch.nn.modules.conv import _ConvNd

class ParameterInfo:
    matrix_shape: Size

    def __init__(self, matrix_shape: Size) -> None:
        super().__init__()
        if len(matrix_shape) != 3:
            raise RuntimeError(f"length of matrix_shape must be 3: {len(matrix_shape)}")
        self.matrix_shape = matrix_shape

def get_parameter_info(p: Tensor) -> Optional[ParameterInfo]:
    if not isinstance(p, Parameter):
        raise RuntimeError(f"except Parameter but {type(p)}")

    if hasattr(p, "parameter_info"):
        return p.parameter_info
    else:
        return None


def get_matrix_shape(p: Parameter) -> Size:
    ret = _get_matrix_shape(p)
    if ret is None:
        raise RuntimeError("parameter is not matrix")
    return ret


def _get_matrix_shape(p: Tensor) -> Optional[Size]:
    pi = get_parameter_info(p)
    if pi is not None:
        return pi.matrix_shape
    elif p.dim() == 2:
        return p.size()
    else:
        return None


def is_matrix(p: Tensor) -> bool:
    s = _get_matrix_shape(p)
    if s is None:
        return False
    return s[-2] > 1 and s[-1] > 1


def set_parameter_info(model: Module) -> None:
    for m in model.modules():
        match m:
            case _ConvNd():
                if get_parameter_info(m.weight) is not None:
                    continue  # donothing
                if m.transposed:
                    raise RuntimeError("ParameterInfo is not supoprt ConvTransposeNd")
                else:
                    in_features = m.in_channels * math.prod(m.kernel_size) // m.groups
                    out_features = m.out_channels // m.groups
                    setattr(m.weight, "parameter_info", ParameterInfo(Size([m.groups, out_features, in_features])))
            case MultiheadAttention():
                if m.in_proj_weight is not None:
                    if get_parameter_info(m.in_proj_weight) is not None:
                        continue  # donothing
                    out_features = m.in_proj_weight.size(0) // 3
                    in_features = m.in_proj_weight.size(1)
                    setattr(m.in_proj_weight, "parameter_info", ParameterInfo(Size([3, out_features, in_features])))
            case Bilinear():
                raise RuntimeError("ParameterInfo is not supoprt Bilinear")
            case Embedding():
                if get_parameter_info(m.weight) is not None:
                    continue  # donothing
                num_groups, out_features = m.weight.size()
                setattr(m.weight, "parameter_info", ParameterInfo(Size([num_groups, out_features, 1])))
            case EmbeddingBag():
                raise RuntimeError("ParameterInfo is not supoprt EmbddingBag")
            case _:
                pass



def _diag(x: Tensor) -> Tensor:
    return torch.diagonal(x, dim1=1, dim2=2)


def whitening(G: Tensor, beta: float, steps: int, diag: bool, eps: float=1e-8) -> Tensor:
    # [m, n] m <= n
    assert G.ndim >= 2

    if diag:
        X = G.bfloat16()
    else:
        X = G
    if G.size(-2) > G.size(-1):
        X = X.mT
    *_, m, n = X.size()
    X = X.view(-1, m, n)

    X = X / (X.norm(dim=(1, 2), keepdim=True) + eps)
    Y = X @ X.mT  # [B, m, m]
    ID = torch.eye(m, dtype=X.dtype, device=X.device).unsqueeze(0)
    Z = ID
    IDx3 = ID * 3

    for _ in range(steps):
        if diag:
            diagZ = _diag(Z)
            diagY = _diag(Y)
            Y_next = beta * Y * (3 - diagZ * diagY).unsqueeze(1)
            Z_next = beta * (IDx3 - diagZ.unsqueeze(2) * Y) * diagZ.unsqueeze(1)
        else:
            Y_next = beta * Y @ (IDx3 - Z @ Y)
            Z_next = beta * (IDx3 - Z @ Y) @ Z
        Y, Z = Y_next, Z_next

    ret = Z @ X
    if G.size(-2) > G.size(-1):
        ret = ret.mT

    return ret.to(G.dtype)


class SWAN(torch.optim.Optimizer):

    def __init__(
        self,
        params: ParamsT,
        lr: float = 0.02,
        beta: float = 0.4,
        weight_decay: float = 0.01,
        momentum: float = 0.0,
        diag: bool = True,
        ns_steps: int = 2,
        eps: float = 1e-8,
        do_compile: bool = False,
    ) -> None:

        must_be_non_negative("lr", lr)
        must_be_in_open_interval('beta', beta, 0.0, 1.0)
        must_be_non_negative('weight_decay', weight_decay)
        must_be_in_right_open_interval('momentum', momentum, 0.0, 1.0)
        must_be_positive('ns_steps', ns_steps)
        must_be_non_negative('eps', eps)

        if do_compile:
            self.whitening = torch.compile(whitening)
        else:
            self.whitening = whitening

        defaults = {
            'lr': lr,
            'beta': beta,
            'weight_decay': weight_decay,
            'momentum': momentum,
            'diag': diag,
            'ns_steps': ns_steps,
            'eps': eps,
            'step': 0,
        }
        super().__init__(params, defaults)

    @torch.no_grad()
    def step(self, closure: Optional[Callable[[], float]] = None) -> Optional[float]:  # type:ignore[override]
        loss = None
        if closure is not None:
            with torch.enable_grad():
                loss = closure()

        for group in self.param_groups:

            lr = group["lr"]
            beta = group["beta"]
            ns_steps = group["ns_steps"]
            diag = group["diag"]
            weight_decay = group["weight_decay"]
            momentum = group["momentum"]
            eps = group["eps"]

            for p in group["params"]:
                if p.grad is None:
                    continue

                matrix_shape = get_matrix_shape(p)

                g = p.grad
                if momentum > 0.0:
                    state = self.state[p]
                    if "momentum_buffer" not in state:
                        state["momentum_buffer"] = torch.zeros_like(g)
                    buf: Tensor = state["momentum_buffer"]
                    buf.lerp_(g, 1 - momentum)
                    g = buf

                g = g.view(matrix_shape)

                g = F.rms_norm(g.mT, [matrix_shape[-2]], eps=eps).mT
                g = self.whitening(g, beta, ns_steps, diag)
                g = F.rms_norm(g, matrix_shape[-2:], eps=eps)

                p.mul_(1 - lr * weight_decay)
                p.sub_(g.view_as(p), alpha=lr * 0.2)

        return loss