"""Lightning strategy for single CPU device."""

# Copyright (C) 2023 Intel Corporation
# SPDX-License-Identifier: Apache-2.0
#

from __future__ import annotations

from typing import TYPE_CHECKING, Any

import torch
import intel_extension_for_pytorch as ipex
from lightning.pytorch.strategies import StrategyRegistry
from lightning.pytorch.strategies.single_device import SingleDeviceStrategy
from lightning.pytorch.plugins.precision import MixedPrecision

if TYPE_CHECKING:
    import lightning.pytorch as pl
    from lightning_fabric.plugins import CheckpointIO
    from lightning_fabric.utilities.types import _DEVICE
    from lightning.pytorch.plugins.precision import Precision


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
                # ipex.optimize(self.model.model, optimizer=self.optimizers[0], inplace=True)
                ipex.optimize(self.model, optimizer=self.optimizers[0], inplace=True, dtype=torch.bfloat16)
            else:  # for inference
                trainer.model.eval()
                self.model.model = ipex.optimize(trainer.model.model)

    def lightning_module_state_dict(self) -> dict[str, Any]:
        """Returns model state."""
        assert self.lightning_module is not None
        state_dict = self.lightning_module.state_dict()
        return state_dict


StrategyRegistry.register(
    SingleCPUStrategy.strategy_name,
    SingleCPUStrategy,
    description="Strategy that enables training on single CPU",
)

class IPEXBF16Precision(MixedPrecision):
    """Create Precision Plugin for IPEX BFloat16."""

    precision: str | int = 'bf16'

    def autocast_context_manager(self) -> torch.autocast:
        return torch.cpu.amp.autocast(True, dtype=torch.bfloat16)
