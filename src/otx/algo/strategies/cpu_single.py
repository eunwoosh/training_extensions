"""Lightning strategy for single CPU device."""

# Copyright (C) 2023 Intel Corporation
# SPDX-License-Identifier: Apache-2.0
#

from __future__ import annotations

from logging import warning
from typing import TYPE_CHECKING
from typing import Callable, Any
from functools import partial
from contextlib import contextmanager

import torch
import intel_extension_for_pytorch as ipex
from intel_extension_for_pytorch.optim._optimizer_utils import IPEX_FUSED_OPTIMIZER_LIST_CPU as IPEX_FUSED_OPTIMIZER_LIST
from torch.nn import Module
from torch.optim import Optimizer, LBFGS
from lightning.pytorch.strategies import StrategyRegistry
from lightning.pytorch.strategies.single_device import SingleDeviceStrategy
from lightning.pytorch.plugins.precision import Precision

if TYPE_CHECKING:
    import lightning.pytorch as pl
    from lightning_fabric.plugins import CheckpointIO
    from lightning_fabric.utilities.types import _DEVICE


class SingleCPUStrategy(SingleDeviceStrategy):
    """Strategy for training on single CPU device."""

    strategy_name = "cpu_single"

    def __init__(
        self,
        device: _DEVICE = "cpu",
        accelerator: pl.accelerators.Accelerator | None = None,
        checkpoint_io: CheckpointIO | None = None,
        precision_plugin: Precision | None = None,
    ):
        # precision_plugin = IPEXBF16Precision()
        super().__init__(
            accelerator=accelerator,
            device=device,
            checkpoint_io=checkpoint_io,
            precision_plugin=precision_plugin,
        )

    @property
    def is_distributed(self) -> bool:
        """Returns true if the strategy supports distributed training."""
        return False

    def setup_optimizers(self, trainer: pl.Trainer) -> None:
        """Sets up optimizers."""
        super().setup_optimizers(trainer)
        if len(self.optimizers) > 1:  # type: ignore[has-type]
            msg = "CPU strategy doesn't support multiple optimizers"
            raise RuntimeError(msg)
        if trainer.task != "SEMANTIC_SEGMENTATION":
            if len(self.optimizers) == 1:  # type: ignore[has-type]
                ipex.optimize(self.model.model, optimizer=self.optimizers[0], inplace=True)
                # ipex.optimize(self.model.model, optimizer=self.optimizers[0], inplace=True, dtype=torch.bfloat16)
            else:  # for inference
                trainer.model.eval()
                self.model.model = ipex.optimize(trainer.model.model)


StrategyRegistry.register(
    SingleCPUStrategy.strategy_name,
    SingleCPUStrategy,
    description="Strategy that enables training on single CPU",
)

class IPEXBF16Precision(Precision):
    """Create Precision Plugin for IPEX BFloat16."""

    precision: str | int = 'bf16'

    @contextmanager
    def forward_context(self):
        """AMP for managing model forward/training_step/evaluation_step/predict_step."""
        with torch.cpu.amp.autocast():
            yield

    def optimizer_step(
        self,
        optimizer: Optimizer,
        model: "pl.LightningModule" | Module,
        closure: Callable[[], Any],
        **kwargs: Any
    ) -> Any:
        """Bf16 optimizer step."""
        """Hook to run the optimizer step."""
        if type(optimizer) in IPEX_FUSED_OPTIMIZER_LIST:
            return super().optimizer_step(optimizer, model=model, closure=closure, **kwargs)

        if isinstance(model, pl.LightningModule):
            closure = partial(self._wrap_closure, model, optimizer, closure)

        # Only `torch.optim.LBFGS`  need to reevaluate closure multiple times
        # in optimizer.step(...) now.
        if isinstance(optimizer, LBFGS):
            RuntimeError(
                "IPEX BFloat16 and the LBFGS optimizer are not compatible "
                f"(optimizer",
                "Hint: Set 'use_ipex' to False or not set 'precision' to 'bf16'"
                " if LBFGS optimizer is necessary"
            )

        # Detect custom optimzer
        if type(optimizer).__name__ not in dir(torch.optim):
            warning("Seems like you are using a custom optimizer,"
                    "please make sure that 'optimizer.step(closure)'"
                    " does not need to be called in training stage")

        # For optimizer not in IPEX_FUSED_OPTIMIZER_LIST,
        # `closure()` needs to be called to backward the loss to avoid `.grad` being None
        closure_result = closure()
        optimizer.step(**kwargs)

        return closure_result