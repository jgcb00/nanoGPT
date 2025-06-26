import math
import warnings
from typing import Callable, Iterable, Tuple

import torch
import torch.optim as optim
from torch import nn
from torch.optim import Optimizer


class CosineDecay:
    """
    Applies cosine decay to a parameter (death_rate), using PyTorch's built-in
    `torch.optim.lr_scheduler.CosineAnnealingLR`.

    Args:
        death_rate (float): Initial value to be decayed.
        T_max (int): Maximum number of iterations for the decay.
        eta_min (float, optional): Minimum value of the parameter after decay.
            Defaults to 0.
        last_epoch (int, optional): The index of the last epoch. Defaults to -1.
    """

    def __init__(
        self, death_rate: float, T_max: int, eta_min: float = 0, last_epoch: int = -1
    ):
        self.sgd = optim.SGD(
            torch.nn.ParameterList([torch.nn.Parameter(torch.zeros(1))]),
            lr=death_rate,
        )
        self.cosine_stepper = torch.optim.lr_scheduler.CosineAnnealingLR(
            self.sgd, T_max + 1, eta_min, last_epoch
        )
        self.T_max = T_max
        self.eta_min = eta_min

    def step(self, current_step: int) -> None:
        """
        Performs one step of the cosine decay scheduler.

        Args:
            current_step (int): Current step index.
        """
        self.cosine_stepper.step(current_step)

    def get_dr(self, current_step: int) -> float:
        """
        Returns the updated rate (death_rate) at the given step.

        Args:
            current_step (int): Current step index.

        Returns:
            float: The decayed parameter.
        """
        if current_step >= self.T_max:
            return self.eta_min
        self.step(current_step)
        return self.sgd.param_groups[0]["lr"]


class SPAMAdamW(Optimizer):
    """
    Implements the Adam algorithm with the weight decay fix, as introduced in
    "Decoupled Weight Decay Regularization" (https://arxiv.org/abs/1711.05101).

    This implementation adds Stochastic Persistent Adam with Momentum (SPAM) features:
    - Periodic momentum resets
    - Threshold-based gradient masking
    - Warmup after momentum resets

    Args:
        params (Iterable[nn.parameter.Parameter]): Iterable of parameters to optimize or
            dictionaries defining parameter groups.
        lr (float, optional): Learning rate. Defaults to 1e-3.
        betas (Tuple[float, float], optional): Coefficients used for computing
            running averages of gradient and its square. Defaults to (0.9, 0.999).
        eps (float, optional): Term added to the denominator to improve numerical
            stability. Defaults to 1e-6.
        weight_decay (float, optional): Weight decay (L2 penalty). Defaults to 0.0.
        correct_bias (bool, optional): Whether or not to correct bias in Adam.
            Defaults to True.
        no_deprecation_warning (bool, optional): Disable deprecation warning.
            Defaults to False.
        warmup_epoch (int, optional): Number of epochs to warm up after momentum reset. Defaults to 50.
        threshold (int, optional): Threshold for gradient masking. Defaults to 5000.
        reset_interval (int, optional): Interval for resetting momentum. Defaults to 500.
        grad_accu_steps (int, optional): Gradient accumulation steps before
            threshold-based masking applies. Defaults to 20.
    """

    def __init__(
        self,
        params: Iterable[nn.parameter.Parameter],
        lr: float = 1e-3,
        betas: Tuple[float, float] = (0.9, 0.999),
        eps: float = 1e-6,
        weight_decay: float = 0.0,
        correct_bias: bool = True,
        no_deprecation_warning: bool = False,
        warmup_epoch: int = 150,
        threshold: int = 5000,
        reset_interval: int = 500,
        grad_accu_steps: int = 5,
    ):
        if not no_deprecation_warning:
            warnings.warn(
                "This implementation of AdamW is deprecated and will be removed in "
                "a future version. Use `torch.optim.AdamW` instead, or set "
                "`no_deprecation_warning=True` to disable this warning.",
                FutureWarning,
            )

        if lr < 0.0:
            raise ValueError(f"Invalid learning rate: {lr} - should be >= 0.0")
        if not 0.0 <= betas[0] < 1.0:
            raise ValueError(
                f"Invalid beta parameter: {betas[0]} - should be in [0.0, 1.0)"
            )
        if not 0.0 <= betas[1] < 1.0:
            raise ValueError(
                f"Invalid beta parameter: {betas[1]} - should be in [0.0, 1.0)"
            )
        if not 0.0 <= eps:
            raise ValueError(f"Invalid epsilon value: {eps} - should be >= 0.0")

        assert (
            grad_accu_steps > 0
        ), "grad_accu_steps should be greater than 0 otherwise the model won't train"

        defaults = {
            "lr": lr,
            "betas": betas,
            "eps": eps,
            "weight_decay": weight_decay,
            "correct_bias": correct_bias,
        }
        super().__init__(params, defaults)

        # Initialize state dictionary
        self.state = self.state or defaultdict(dict)

        # Add special state variables for the optimizer
        self.state["total_step"] = 0

        # Store configuration parameters
        self.warmup_epoch = warmup_epoch
        self.warmup = CosineDecay(0.99, warmup_epoch)  # Warmup after momentum reset
        self.thres = threshold
        self.reset_interval = reset_interval
        self.grad_accu_steps = (
            grad_accu_steps  # apply threshold limit after this many steps
        )

    @torch.no_grad()
    def step(self, closure: Callable = None) -> float:
        """
        Performs a single optimization step.

        Args:
            closure (Callable, optional): A closure that re-evaluates the model and
                returns the loss.

        Returns:
            float: The loss, if the closure was provided, otherwise None.
        """
        loss = None
        if closure is not None:
            loss = closure()

        if self.state["total_step"] % self.reset_interval == 0:
            self.warmup = CosineDecay(0.99, self.warmup_epoch)  # Reset warmup scheduler

        # Calculate scale factor based on the cosine decay
        scale_factor = 1 - self.warmup.get_dr(
            self.state["total_step"] % self.reset_interval
        )
        for group in self.param_groups:
            for p in group["params"]:
                if p.grad is None:
                    continue

                grad = p.grad
                if grad.is_sparse:
                    raise RuntimeError(
                        "Adam does not support sparse gradients. Use SparseAdam instead."
                    )

                state = self.state[p]

                # Initialize EMA states if necessary
                if "exp_avg" not in state:
                    state["exp_avg"] = torch.zeros_like(grad)
                    state["exp_avg_sq"] = torch.zeros_like(grad)

                # Reset momentum when total_step hits reset_interval
                if self.state["total_step"] % self.reset_interval == 0:
                    state["exp_avg"].zero_()
                    state["exp_avg_sq"].zero_()

                exp_avg, exp_avg_sq = state["exp_avg"], state["exp_avg_sq"]
                beta1, beta2 = group["betas"]

                # Threshold-based gradient masking
                current_step = self.state["total_step"] + 1

                if (
                    self.thres != 0
                    and current_step % self.reset_interval >= self.grad_accu_steps
                ):
                    # Only apply after accumulation steps
                    mask = (grad**2) > (self.thres * exp_avg_sq)
                    grad = grad.clone()  # Clone to avoid modifying the original grad
                    grad[mask] = grad[mask].sign() * torch.sqrt(
                        exp_avg_sq[mask] * self.thres
                    )

                # Update exponential moving averages
                exp_avg.mul_(beta1).add_(grad, alpha=1.0 - beta1)
                exp_avg_sq.mul_(beta2).addcmul_(grad, grad, value=1.0 - beta2)

                denom = exp_avg_sq.sqrt().add_(group["eps"])

                step_size = group["lr"]
                if group["correct_bias"]:
                    # Bias correction based on steps since last reset
                    steps_since_reset = current_step % self.reset_interval
                    if steps_since_reset == 0:
                        steps_since_reset = self.reset_interval

                    bias_correction1 = 1.0 - beta1**steps_since_reset
                    bias_correction2 = 1.0 - beta2**steps_since_reset
                    step_size *= math.sqrt(bias_correction2) / bias_correction1

                # Compute normalized gradient
                norm_grad = exp_avg / denom
                p.add_(norm_grad, alpha=-step_size * scale_factor)

                # Weight decay
                if group["weight_decay"] > 0:
                    p.add_(p, alpha=(-group["lr"] * group["weight_decay"]))

        # Bookkeeping
        self.state["total_step"] += 1

        return loss


# Add this import at the top of the file if needed
from collections import defaultdict
