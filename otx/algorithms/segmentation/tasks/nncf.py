"""NNCF Task of OTX Segmentation."""

# Copyright (C) 2022 Intel Corporation
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

from functools import partial
from typing import List, Optional

import otx.algorithms.segmentation.adapters.mmseg.nncf.patches  # noqa: F401  # pylint: disable=unused-import
from otx.algorithms.common.tasks.nncf_base import NNCFBaseTask
from otx.algorithms.segmentation.adapters.mmseg.nncf import build_nncf_segmentor
from otx.api.entities.datasets import DatasetEntity
from otx.api.entities.metrics import (
    CurveMetric,
    InfoMetric,
    LineChartInfo,
    MetricsGroup,
    Performance,
    ScoreMetric,
    VisualizationInfo,
    VisualizationType,
)
from otx.api.entities.model import ModelEntity
from otx.api.entities.optimization_parameters import OptimizationParameters
from otx.mpa.utils.logger import get_logger

from .inference import SegmentationInferenceTask

logger = get_logger()


class SegmentationNNCFTask(NNCFBaseTask, SegmentationInferenceTask):  # pylint: disable=too-many-ancestors
    """SegmentationNNCFTask."""

    def _initialize_post_hook(self, options=None):
        super()._initialize_post_hook(options)

        export = options.get("export", False)
        options["model_builder"] = partial(
            self.model_builder,
            nncf_model_builder=build_nncf_segmentor,
            return_compression_ctrl=False,
            is_export=export,
        )

    def _optimize(
        self,
        dataset: DatasetEntity,
        optimization_parameters: Optional[OptimizationParameters] = None,
    ):
        results = self._run_task(
            "SegTrainer",
            mode="train",
            dataset=dataset,
            parameters=optimization_parameters,
        )
        return results

    def _optimize_post_hook(
        self,
        dataset: DatasetEntity,
        output_model: ModelEntity,
    ):
        # Get training metrics group from learning curves
        training_metrics, best_score = self._generate_training_metrics_group(self._learning_curves)
        performance = Performance(
            score=ScoreMetric(value=best_score, name=self.metric),
            dashboard_metrics=training_metrics,
        )

        logger.info(f"Final model performance: {str(performance)}")
        output_model.performance = performance

    def _generate_training_metrics_group(self, learning_curves):
        """Get Training metrics (epochs & scores).

        Parses the mmsegmentation logs to get metrics from the latest training run
        :return output List[MetricsGroup]
        """
        output: List[MetricsGroup] = []
        # Model architecture
        architecture = InfoMetric(name="Model architecture", value=self._model_name)
        visualization_info_architecture = VisualizationInfo(
            name="Model architecture", visualisation_type=VisualizationType.TEXT
        )
        output.append(
            MetricsGroup(
                metrics=[architecture],
                visualization_info=visualization_info_architecture,
            )
        )
        # Learning curves
        best_score = -1
        for key, curve in learning_curves.items():
            metric_curve = CurveMetric(xs=curve.x, ys=curve.y, name=key)
            if key == f"val/{self.metric}":
                best_score = max(curve.y)
            visualization_info = LineChartInfo(name=key, x_axis_label="Epoch", y_axis_label=key)
            output.append(MetricsGroup(metrics=[metric_curve], visualization_info=visualization_info))
        return output, best_score
