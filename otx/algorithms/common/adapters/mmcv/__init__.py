"""Adapters for mmcv support."""

# Copyright (C) 2021-2022 Intel Corporation
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing,
# software distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions
# and limitations under the License.

from .hooks import (
    CancelTrainingHook,
    EarlyStoppingHook,
    EnsureCorrectBestCheckpointHook,
    OTXLoggerHook,
    OTXProgressHook,
    ReduceLROnPlateauLrUpdaterHook,
    StopLossNanTrainingHook,
)
from .runner import EpochRunnerWithCancel, IterBasedRunnerWithCancel
from .utils import remove_from_config

__all__ = [
    "EpochRunnerWithCancel",
    "IterBasedRunnerWithCancel",
    "CancelTrainingHook",
    "OTXLoggerHook",
    "OTXProgressHook",
    "EarlyStoppingHook",
    "ReduceLROnPlateauLrUpdaterHook",
    "EnsureCorrectBestCheckpointHook",
    "StopLossNanTrainingHook",
    "remove_from_config",
]
