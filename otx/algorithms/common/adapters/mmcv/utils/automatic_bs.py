"""Algorithm to find a proper batch size which is fit to current GPU device for tasks using mmcv."""

# Copyright (C) 2023 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

from copy import deepcopy
from typing import Callable, List

import numpy as np

from otx.algorithms.common.adapters.torch.utils import adapt_batch_size as adapt_torch_model_bs
from otx.algorithms.common.utils.logger import get_logger

logger = get_logger()


def adapt_batch_size(train_func: Callable, cfg, datasets: List, validate: bool = False):
    """Decrease batch size if default batch size isn't fit to current GPU device.

    This function just setup for single iteration training to reduce time for adapting.
    The core part of adapting batch size is done in adapt_batch_size in the torch.utils package.

    Args:
        train_func (Callable): The function to train a model.
            Only cfg, dataset and meta are passed to the function when invoking it.
        cfg: Configuration of a training.
        meta (Dict): A dict records some meta information of a training.
        datasets (List): List of datasets.
        validate (bool): Whether do vlidation or not.
    """

    def train_func_single_iter(batch_size):
        copied_cfg = deepcopy(cfg)
        _set_batch_size(copied_cfg, batch_size)

        # setup for training a single iter to reduce time
        copied_cfg.runner["max_epochs"] = 1
        if not validate:
            for hook in copied_cfg.custom_hooks:
                if hook["type"] == "AdaptiveTrainSchedulingHook":
                    hook["enable_eval_before_run"] = False

        new_datasets = [SubDataset(datasets[0], batch_size)]

        train_func(
            dataset=new_datasets,
            cfg=copied_cfg,
            validate=validate,
        )

    default_bs = _get_batch_size(cfg)

    available_bs = adapt_torch_model_bs(
        train_func=train_func_single_iter,
        current_bs=default_bs,
        trainset_size=len(datasets[0]),
    )

    if default_bs != available_bs:
        _set_batch_size(cfg, available_bs)
        origin_lr = cfg.optimizer.lr
        cfg.optimizer.lr *= available_bs / default_bs

        logger.info("Adapting batch size is done.")
        logger.info(f"Batch size is adapted : {default_bs} -> {available_bs}")
        logger.info(f"learning rate is adapted : {origin_lr} -> {cfg.optimizer.lr}")
    else:
        logger.info("Adapting batch size is done. Current batch size is availble.")


def _get_batch_size(cfg) -> int:
    if "action" in str(cfg.domain).lower():
        return cfg.data.videos_per_gpu
    return cfg.data.train_dataloader["samples_per_gpu"]


def _set_batch_size(cfg, batch_size: int):
    if "action" in str(cfg.domain).lower():
        cfg.data.videos_per_gpu = batch_size
    else:
        cfg.data.train_dataloader["samples_per_gpu"] = batch_size


class SubDataset:
    """Wrapper class to make dataset pretend to have specified number of images.

    Args:
        fullset: Original dataset.
        num_samples (int): Number of images to pretend to have. It should be positive.
    """

    def __init__(self, fullset, num_sampels: int):
        if num_sampels <= 0:
            raise ValueError(f"num_sampels should be positive. But, current value is {num_sampels}.")

        self.fullset = fullset
        self.num_sampels = num_sampels

    def __len__(self) -> int:
        """Get length of subset."""
        return self.num_sampels

    def __getitem__(self, indx) -> dict:
        """Get dataset at index."""
        return self.fullset[indx]

    def __getattr__(self, name):
        """When trying to get other attributes, not dataset, get values from fullset."""
        if name == "__setstate__":
            raise AttributeError(name)
        return getattr(self.fullset, name)

    @property
    def flag(self):
        """Getter of flag for detection task.

        Sampler of the detection task decides length of dataset checking sum of flag array.
        To consider that case, return flag array with length of num_samples.

        """
        return np.zeros(self.num_sampels, dtype=np.uint8)
