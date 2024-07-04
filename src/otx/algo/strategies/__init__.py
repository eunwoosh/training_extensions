# Copyright (C) 2024 Intel Corporation
# SPDX-License-Identifier: Apache-2.0
#
"""Lightning strategy for single XPU device."""

from .xpu_single import SingleXPUStrategy
from .cpu_single import SingleCPUStrategy

__all__ = ["SingleXPUStrategy", "SingleCPUStrategy"]
