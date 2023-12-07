# Copyright (C) 2023 Intel Corporation
# SPDX-License-Identifier: Apache-2.0
#
"""Module for OTXDetectionDataset."""

from __future__ import annotations

from typing import Callable
import collections

import copy
import numpy as np
import torch
from datumaro import Bbox, DatasetSubset, Image
from torchvision import tv_tensors

from otx.core.data.entity.base import T_OTXDataEntity
from otx.core.data.entity.base import ImageInfo
from otx.core.data.entity.detection import DetBatchDataEntity, DetDataEntity

from .base import OTXDataset, Transforms


class OTXDetectionDataset(OTXDataset[DetDataEntity]):
    """OTXDataset class for detection task."""

    def __init__(self, dm_subset: DatasetSubset, transforms: Transforms) -> None:
        super().__init__(dm_subset, transforms)

    def _get_item_impl(self, index: int) -> DetDataEntity | None:
        item = self.dm_subset.get(id=self.ids[index], subset=self.dm_subset.name)
        img = item.media_as(Image)
        img_data = self._get_img_data(img)
        img_shape = img.size

        bbox_anns = [ann for ann in item.annotations if isinstance(ann, Bbox)]

        bboxes = (
            np.stack([ann.points for ann in bbox_anns], axis=0).astype(np.float32)
            if len(bbox_anns) > 0
            else np.zeros((0, 4), dtype=np.float32)
        )

        entity = DetDataEntity(
            image=img_data,
            img_info=ImageInfo(
                img_idx=index,
                img_shape=img_shape,
                ori_shape=img_shape,
                pad_shape=img_shape,
                scale_factor=(1.0, 1.0),
            ),
            bboxes=tv_tensors.BoundingBoxes(
                bboxes,
                format=tv_tensors.BoundingBoxFormat.XYXY,
                canvas_size=img_shape,
            ),
            labels=torch.as_tensor([ann.label for ann in bbox_anns]),
        )

        return self._apply_transforms(entity)

    @property
    def collate_fn(self) -> Callable:
        """Collection function to collect DetDataEntity into DetBatchDataEntity in data loader."""
        return DetBatchDataEntity.collate_fn


class YOLOXOTXDetectionDataset(OTXDataset[DetDataEntity]):
    """OTXDataset class for detection task."""

    def __init__(self, dm_subset: DatasetSubset, transforms: Transforms) -> None:
        super().__init__(dm_subset, transforms)

    def _get_normal_data(self, idx):
        for _ in range(self.max_refetch):
            item = self.dm_subset.get(id=self.ids[idx], subset=self.dm_subset.name)
            img = item.media_as(Image)
            img_data = self._get_img_data(img)
            img_shape = img.size

            bbox_anns = [ann for ann in item.annotations if isinstance(ann, Bbox)]

            bboxes = (
                np.stack([ann.points for ann in bbox_anns], axis=0).astype(np.float32)
                if len(bbox_anns) > 0
                else np.zeros((0, 4), dtype=np.float32)
            )

            entity = DetDataEntity(
                image=img_data,
                img_info=ImageInfo(
                    img_idx=idx,
                    img_shape=img_shape,
                    ori_shape=img_shape,
                    pad_shape=img_shape,
                    scale_factor=(1.0, 1.0),
                ),
                bboxes=tv_tensors.BoundingBoxes(
                    bboxes,
                    format=tv_tensors.BoundingBoxFormat.XYXY,
                    canvas_size=img_shape,
                ),
                labels=torch.as_tensor([ann.label for ann in bbox_anns]),
            )

            for transform in self.transforms[:2]:
                entity = transform(entity)
                # MMCV transform can produce None. Please see
                # https://github.com/open-mmlab/mmengine/blob/26f22ed283ae4ac3a24b756809e5961efe6f9da8/mmengine/dataset/base_dataset.py#L59-L66
            
            if entity is not None:
                return entity

        assert entity is not None

    def __getitem__(self, idx):
        results = copy.deepcopy(self._get_normal_data(idx))
        for transform in self.transforms[2:]:
            if hasattr(transform, 'get_indexes'):
                for i in range(15):
                    # Make sure the results passed the loading pipeline
                    # of the original dataset is not None.
                    indexes = transform.get_indexes(self)
                    if not isinstance(indexes, collections.abc.Sequence):
                        indexes = [indexes]
                    mix_results = [
                        copy.deepcopy(self._get_normal_data(index)) for index in indexes
                    ]
                    if None not in mix_results:
                        results['mix_results'] = mix_results
                        break
                else:
                    raise RuntimeError(
                        'The loading pipeline of the original dataset'
                        ' always return None. Please check the correctness '
                        'of the dataset and its pipeline.')

            for i in range(15):
                # To confirm the results passed the training pipeline
                # of the wrapper is not None.
                updated_results = transform(copy.deepcopy(results))
                if updated_results is not None:
                    results = updated_results
                    break
            else:
                raise RuntimeError(
                    'The training pipeline of the dataset wrapper'
                    ' always return None.Please check the correctness '
                    'of the dataset and its pipeline.')

            if isinstance(results, dict) and 'mix_results' in results:
                results.pop('mix_results')

        return results

    @property
    def collate_fn(self) -> Callable:
        """Collection function to collect DetDataEntity into DetBatchDataEntity in data loader."""
        return DetBatchDataEntity.collate_fn
