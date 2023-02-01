"""Sample Code of otx training for action detection."""

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

import argparse
import sys

from mmcv.utils import get_logger

from otx.algorithms.common.utils import get_task_class
from otx.api.configuration.helper import create
from otx.api.entities.inference_parameters import InferenceParameters
from otx.api.entities.model import ModelEntity
from otx.api.entities.model_template import parse_model_template
from otx.api.entities.optimization_parameters import OptimizationParameters
from otx.api.entities.resultset import ResultSetEntity
from otx.api.entities.subset import Subset
from otx.api.entities.task_environment import TaskEnvironment
from otx.api.usecases.tasks.interfaces.export_interface import ExportType
from otx.api.usecases.tasks.interfaces.optimization_interface import OptimizationType
from otx.cli.datasets import get_dataset_class
from otx.cli.utils.io import generate_label_schema

logger = get_logger(name="sample")


def parse_args():
    """Parse function for getting model template & check export."""
    parser = argparse.ArgumentParser(description="Sample showcasing the new API")
    parser.add_argument("template_file_path", help="path to template file")
    parser.add_argument("--export", action="store_true")
    return parser.parse_args()


TRAIN_ANN_FILES = "data/custom_action_recognition/custom_dataset/val.csv"
TRAIN_DATA_ROOTS = "data/custom_action_recognition/custom_dataset/rawframes"
VAL_ANN_FILES = "data/custom_action_recognition/custom_dataset/val.csv"
VAL_DATA_ROOTS = "data/custom_action_recognition/custom_dataset/rawframes"


def load_test_dataset(model_template):
    """Load Sample dataset for detection."""
    dataset_class = get_dataset_class(model_template.task_type)
    dataset = dataset_class(
        train_subset={"ann_file": TRAIN_ANN_FILES, "data_root": TRAIN_DATA_ROOTS},
        val_subset={"ann_file": VAL_ANN_FILES, "data_root": VAL_DATA_ROOTS},
    )
    labels_schema = generate_label_schema(dataset, model_template.task_type)
    return dataset, labels_schema


# pylint: disable=too-many-locals, too-many-statements
def main(args):
    """Main function of Detection Sample."""
    logger.info("Fine tuning sample dataset")
    logger.info("Sample dataset can be found at data/custom_action_recognition/custom_dataset")

    logger.info("Load model template")
    model_template = parse_model_template(args.template_file_path)

    logger.info("Get dataset")
    dataset, labels_schema = load_test_dataset(model_template)

    logger.info("Set hyperparameters")
    params = create(model_template.hyper_parameters.data)
    params.learning_parameters.num_iters = 5

    logger.info("Setup environment")
    environment = TaskEnvironment(
        model=None,
        hyper_parameters=params,
        label_schema=labels_schema,
        model_template=model_template,
    )

    logger.info("Create base Task")
    task_impl_path = model_template.entrypoints.base
    task_cls = get_task_class(task_impl_path)
    task = task_cls(task_environment=environment)

    logger.info("Train model")
    output_model = ModelEntity(
        dataset,
        environment.get_model_configuration(),
    )
    task.train(dataset, output_model)

    logger.info("Get predictions on the validation set")
    validation_dataset = dataset.get_subset(Subset.VALIDATION)
    predicted_validation_dataset = task.infer(
        validation_dataset.with_empty_annotations(),
        InferenceParameters(is_evaluation=False),
    )
    resultset = ResultSetEntity(
        model=output_model,
        ground_truth_dataset=validation_dataset,
        prediction_dataset=predicted_validation_dataset,
    )
    logger.info("Estimate quality on validation set")
    task.evaluate(resultset)
    logger.info(str(resultset.performance))

    if args.export:
        logger.info("Export model")
        exported_model = ModelEntity(
            dataset,
            environment.get_model_configuration(),
        )
        task.export(ExportType.OPENVINO, exported_model)

        logger.info("Create OpenVINO Task")
        environment.model = exported_model
        openvino_task_impl_path = model_template.entrypoints.openvino
        openvino_task_cls = get_task_class(openvino_task_impl_path)
        openvino_task = openvino_task_cls(environment)

        logger.info("Get predictions on the validation set")
        predicted_validation_dataset = openvino_task.infer(
            validation_dataset.with_empty_annotations(),
            InferenceParameters(is_evaluation=True),
        )

        resultset = ResultSetEntity(
            model=output_model,
            ground_truth_dataset=validation_dataset,
            prediction_dataset=predicted_validation_dataset,
        )
        logger.info("Estimate quality on validation set")
        openvino_task.evaluate(resultset)
        logger.info(str(resultset.performance))

        logger.info("Run POT optimization")
        optimized_model = ModelEntity(
            dataset,
            environment.get_model_configuration(),
        )
        openvino_task.optimize(
            OptimizationType.POT,
            dataset.get_subset(Subset.TRAINING),
            optimized_model,
            OptimizationParameters(),
        )

        logger.info("Get predictions on the validation set")
        predicted_validation_dataset = openvino_task.infer(
            validation_dataset.with_empty_annotations(),
            InferenceParameters(is_evaluation=True),
        )
        resultset = ResultSetEntity(
            model=optimized_model,
            ground_truth_dataset=validation_dataset,
            prediction_dataset=predicted_validation_dataset,
        )
        logger.info("Performance of optimized model:")
        openvino_task.evaluate(resultset)
        logger.info(str(resultset.performance))


if __name__ == "__main__":
    sys.exit(main(parse_args()) or 0)
