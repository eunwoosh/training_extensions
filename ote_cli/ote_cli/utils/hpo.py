"""
Utils for HPO with hpopt
"""

# Copyright (C) 2021-2022 Intel Corporation
# SPDX-License-Identifier: Apache-2.0
#

import glob
import os
import re
import shutil
import time
from enum import Enum
from functools import partial
from inspect import isclass
from math import floor
from os import path as osp
from typing import Any, Callable, Dict, Optional, Union

import torch
import yaml
from ote_sdk.configuration.helper import create
from ote_sdk.entities.model import ModelEntity
from ote_sdk.entities.model_template import TaskType
from ote_sdk.entities.subset import Subset
from ote_sdk.entities.task_environment import TaskEnvironment
from ote_sdk.entities.train_parameters import TrainParameters, UpdateProgressCallback

from ote_cli.datasets import get_dataset_class
from ote_cli.utils.importing import get_impl_class
from ote_cli.utils.io import generate_label_schema, read_model, save_model_data

try:
    import hpopt
    from hpopt.hpo_runner import run_hpo_loop
    from hpopt.hyperband import HyperBand
    from hpopt.hpo_base import TrialStatus
except ImportError:
    print("cannot import hpopt module")
    hpopt = None

import logging
hpopt_logger = logging.getLogger("hpopt")
hpopt_logger.setLevel(logging.DEBUG)

def _check_hpo_enabled_task(task_type):
    return task_type in [
        TaskType.CLASSIFICATION,
        TaskType.DETECTION,
        TaskType.SEGMENTATION,
        TaskType.INSTANCE_SEGMENTATION,
        TaskType.ROTATED_DETECTION,
        TaskType.ANOMALY_CLASSIFICATION,
        TaskType.ANOMALY_DETECTION,
        TaskType.ANOMALY_SEGMENTATION,
    ]

def check_hpopt_available():
    """Check whether hpopt is avaiable"""

    if hpopt is None:
        return False
    return True

class TaskManager:
    def __init__(self, task_type: TaskType):
        self._task_type = task_type

    @property
    def task_type(self):
        return self._task_type

    def is_mpa_framework_task(self):
        return (
            self.is_cls_framework_task()
            or self.is_det_framework_task()
            or self.is_seg_framework_task()
        )

    def is_cls_framework_task(self):
        return self._task_type == TaskType.CLASSIFICATION

    def is_det_framework_task(self):
        return self._task_type in [
            TaskType.DETECTION,
            TaskType.INSTANCE_SEGMENTATION,
            TaskType.ROTATED_DETECTION,
        ]

    def is_seg_framework_task(self):
        return self._task_type == TaskType.SEGMENTATION

    def is_anomaly_framework_task(self):
        return self._task_type in [
            TaskType.ANOMALY_CLASSIFICATION,
            TaskType.ANOMALY_DETECTION,
            TaskType.ANOMALY_SEGMENTATION,
        ]

    def get_batch_size_name(self):
        batch_size_name = None
        if self.is_mpa_framework_task():
            batch_size_name = "learning_parameters.batch_size"
        elif self.is_anomaly_framework_task():
            batch_size_name = "learning_parameters.train_batch_size"

        return batch_size_name
        
    def get_epoch_name(self):
        epoch_name = None
        if self.is_mpa_framework_task():
            epoch_name = "num_iters"
        elif self.is_anomaly_framework_task():
            epoch_name = "max_epochs"

        return epoch_name

    def copy_weight(self, src: str, det: str):
        if self.is_mpa_framework_task():
            for weight_candidate in glob.iglob(osp.join(src, "**/epoch_*.pth"), recursive=True):
                if not (osp.islink(weight_candidate) or osp.exists(osp.join(det, osp.basename(weight_candidate)))):
                    shutil.copy(weight_candidate, det)

    def get_latest_weight(self, workdir: str):
        latest_weight = None
        if self.is_mpa_framework_task():
            pattern = re.compile(r"(\d+)\.pth")
            current_latest_epoch = -1
            latest_weight = None

            for weight_name in glob.iglob(osp.join(workdir, "**/epoch_*.pth"), recursive=True):
                ret = pattern.search(weight_name)
                epoch = int(ret.group(1))
                if current_latest_epoch < epoch:
                    current_latest_epoch = epoch
                    latest_weight = weight_name

        return latest_weight

    def get_best_weights(self, workdir):
        best_weights = None
        if self.is_mpa_framework_task():
            best_weights = list(glob.iglob(osp.join(workdir, "**/best*.pth"), recursive=True))

        return best_weights

class TaskEnvironmentManager:
    def __init__(self, environment: TaskEnvironment):
        self._environment = environment
        self.task = TaskManager(environment.model_template.task_type)

    def get_task(self):
        return self._environment.model_template.task_type

    def get_model_template(self):
        return self._environment.model_template

    def get_model_template_path(self):
        return self._environment.model_template.model_template_path

    def set_hyper_parameter_from_flatten_format_dict(self, hyper_parameter: Dict):
        env_hp = self._environment.get_hyper_parameters()

        for param_key, param_val in hyper_parameter.items():
            param_key = param_key.split(".")

            target = env_hp
            for val in param_key[:-1]:
                target = getattr(target, val)
            setattr(target, param_key[-1], param_val)

    def get_dict_type_hyper_parameter(self):
        learning_parameters = self._environment.get_hyper_parameters().learning_parameters
        learning_parameters = self._convert_parameter_group_to_dict(learning_parameters)
        hyper_parameter = {f"learning_parameters.{key}" : val for key, val in learning_parameters.items()}
        return hyper_parameter

    @staticmethod
    def _convert_parameter_group_to_dict(parameter_group):
        groups = getattr(parameter_group, "groups", None)
        parameters = getattr(parameter_group, "parameters", None)

        total_arr = []
        for val in [groups, parameters]:
            if val is not None:
                total_arr.extend(val)
        if not total_arr:
            return parameter_group

        ret = {}
        for key in total_arr:
            val = TaskEnvironmentManager._convert_parameter_group_to_dict(getattr(parameter_group, key))
            if not (isclass(val) or isinstance(val, Enum)):
                ret[key] = val

        return ret

    def get_max_epoch(self):
        return getattr(self._environment.get_hyper_parameters().learning_parameters, self.task.get_epoch_name())

    def save_initial_weight(self, save_path: str):
        if self._environment.model is None:
            if self.task.is_anomaly_framework_task():
                # if task isn't anomaly, then save model weight during first trial
                task = self.get_train_task(self._environment)
                model = self.get_output_model()
                task.save_model(model)
                save_model_data(model, save_path)
                return True
        else:
            save_model_data(self._environment.model, self.save_path)
            return True
        return False

    def get_train_task(self):
        impl_class = get_impl_class(self._environment.model_template.entrypoints.base)
        return impl_class(task_environment=self._environment)

    def get_batch_size_name(self):
        return self.task.get_batch_size_name()

    def load_model_weight(self, model_weight_path: str):
        self._environment.model = read_model(self._environment.get_model_configuration(), model_weight_path, None)

    def resume_model_weight(self, model_weight_path: str):
        self.load_model_weight(model_weight_path)
        self._environment.model.model_adapters["resume"] = True

    def get_output_model(self, dataset = None):
        return ModelEntity(
            dataset,
            self._environment.get_model_configuration(),
        )

    def set_epoch(self, epoch: int):
        hp = {f"learning_parameters.{self.task.get_epoch_name()}" : epoch}
        self.set_hyper_parameter_from_flatten_format_dict(hp)

class HpoRunner:
    def __init__(
        self,
        environment: TaskEnvironment,
        dataset,
        hpo_workdir: str,
        hpo_time_ratio: int = 4,
    ):
        self._environment = TaskEnvironmentManager(environment)
        self._dataset = dataset
        self._hpo_workdir = hpo_workdir
        self._hpo_time_ratio = hpo_time_ratio
        self._hpo_config: Dict = self._set_hpo_config()
        self._train_dataset_size = None
        self._val_dataset_size = None
        self._fixed_hp = {}
        self._initial_weight_name = "initial_weight.pth"

        self._align_batch_size_search_space_to_dataset_size()

    def _set_hpo_config(self):
        hpo_config_path = osp.join(
            osp.dirname(self._environment.get_model_template_path()),
            "hpo_config.yaml",
        )
        with open(hpo_config_path, "r", encoding="utf-8") as f:
            hpopt_cfg = yaml.safe_load(f)

        return hpopt_cfg

    def _align_batch_size_search_space_to_dataset_size(self):
        batch_size_name = self._environment.get_batch_size_name()
        train_dataset_size = self._get_train_dataset_size()

        if batch_size_name in self._hpo_config["hp_space"]:
            if "range" in self._hpo_config["hp_space"][batch_size_name]:
                max_val = self._hpo_config["hp_space"][batch_size_name]["range"][1]
                min_val = self._hpo_config["hp_space"][batch_size_name]["range"][0]
                if max_val > train_dataset_size:
                    max_val = train_dataset_size
                    self._hpo_config["hp_space"][batch_size_name]["range"][1] = max_val

                # If trainset size is lower than min batch size range,
                # fix batch size to trainset size
                if min_val >= max_val:
                    print(
                        "Train set size is equal or lower than batch size range."
                        "Batch size is fixed to train set size."
                    )
                    del self._hpo_config["hp_space"][batch_size_name]
                    self._fixed_hp[batch_size_name] = train_dataset_size
                    self._environment.set_hyper_parameter_from_flatten_format_dict(self._fixed_hp)
            else:
                raise NotImplementedError

    def _get_train_dataset_size(self):
        if self._train_dataset_size is None:
            self._train_dataset_size = len(self._dataset.get_subset(Subset.TRAINING))
        return self._train_dataset_size

    def _get_val_dataset_size(self):
        if self._val_dataset_size is None:
            self._val_dataset_size = len(self._dataset.get_subset(Subset.TRAINING))
        return self._val_dataset_size

    def run_hpo(self, train_func: Callable, dataset_path: Dict[str, str]):
        self._environment.save_initial_weight(self._get_initial_model_weight_path())
        hpo_algo = self._get_hpo_algo()
        resource_type = "gpu" if torch.cuda.is_available() else "cpu"
        best_config = run_hpo_loop(
            hpo_algo,
            partial(
                train_func,
                model_template=self._environment.get_model_template(),
                dataset_paths=dataset_path,
                task_type=self._environment.get_task(),
                hpo_workdir=self._hpo_workdir,
                initial_weight_name=self._initial_weight_name,
                metric=self._hpo_config["metric"]
            ),
            resource_type,
        )
        self._restore_fixed_hp(best_config)
        hpo_algo.print_result()

        return best_config

    def _restore_fixed_hp(self, hyper_parameter: Dict[str, Any]):
        for key, val in self._fixed_hp.items():
            hyper_parameter[key] = val

    def _get_hpo_algo(self):
        hpo_algo_type = self._hpo_config.get("search_algorithm", "asha")

        if hpo_algo_type == "asha":
            hpo_algo = self._prepare_asha()
        elif hpo_algo_type == "smbo":
            hpo_algo = self._prepare_smbo()
        else:
            raise ValueError(f"Supported HPO algorithms are asha and smbo. your value is {hpo_algo_type}.")

        return hpo_algo

    def _prepare_asha(self):
        train_dataset_size = self._get_train_dataset_size()
        val_dataset_size = self._get_val_dataset_size()

        args = {
            "search_space" : self._hpo_config["hp_space"],
            "save_path" : self._hpo_workdir,
            "mode" : self._hpo_config.get("mode", "max"),
            "num_workers" : 1,
            "num_full_iterations" : self._environment.get_max_epoch(),
            "full_dataset_size" : train_dataset_size,
            "non_pure_train_ratio" : val_dataset_size / (train_dataset_size + val_dataset_size),
            "metric" : self._hpo_config.get("metric", "mAP"),
            "expected_time_ratio" : self._hpo_time_ratio,
            "prior_hyper_parameters" : self._get_default_hyper_parameters(),
            "asynchronous_bracket" : True,
            "asynchronous_sha" : False if torch.cuda.device_count() == 1 else True
        }

        print(f"[OTE_CLI] [DEBUG-HPO] ASHA args for create hpopt = {args}")

        return HyperBand(**args)

    def _prepare_smbo(self):
        raise NotImplementedError

    def _get_default_hyper_parameters(self):
        default_hyper_parameters = {}
        hp_from_env = self._environment.get_dict_type_hyper_parameter()

        for key, val in hp_from_env.items():
            if key in self._hpo_config["hp_space"]:
                default_hyper_parameters[key] = val

        if not default_hyper_parameters:
            return None
        return default_hyper_parameters

    def _get_initial_model_weight_path(self):
        return osp.join(self._hpo_workdir, self._initial_weight_name)

def run_hpo(args, environment, dataset, task_type):
    """Update the environment with better hyper-parameters found by HPO"""
    if not check_hpopt_available():
        print("hpopt isn't available. hpo is skipped.")
        return None

    if not _check_hpo_enabled_task(task_type):
        print(
            "Currently supported task types are classification, detection, segmentation and anomaly"
            f"{task_type} is not supported yet."
        )
        return None

    hpo_save_path = os.path.abspath(os.path.join(os.path.dirname(args.save_model_to), "hpo"))
    hpo_runner = HpoRunner(environment, dataset, hpo_save_path, args.hpo_time_ratio)
    dataset_paths = {
        "train_ann_file": args.train_ann_files,
        "train_data_root": args.train_data_roots,
        "val_ann_file": args.val_ann_files,
        "val_data_root": args.val_data_roots,
    }

    print(
        f"{time.strftime('%Y-%m-%d %H:%M:%S', time.localtime())} [HPO] started hyper-parameter optimization"
    )
    best_config = hpo_runner.run_hpo(run_trial, dataset_paths)
    print(
        f"{time.strftime('%Y-%m-%d %H:%M:%S', time.localtime())} [HPO] completed hyper-parameter optimization"
    )

    TaskEnvironmentManager(environment).set_hyper_parameter_from_flatten_format_dict(best_config)

class Trainer:
    def __init__(
        self,
        hp_config: Dict[str, Any],
        report_func: Callable,
        model_template,
        dataset_paths: Dict[str, str],
        task_type: TaskType,
        hpo_workdir: str,
        initial_weight_name: str,
        metric: str
    ):
        self._hp_config = hp_config
        self._report_func = report_func
        self._model_template = model_template
        self._dataset_paths = dataset_paths
        self._task = TaskManager(task_type)
        self._hpo_workdir = hpo_workdir
        self._initial_weight_name = initial_weight_name
        self._metric = metric
        self._epoch = floor(self._hp_config["configuration"]["iterations"])
        del self._hp_config["configuration"]["iterations"]

    def run(self):
        """Run each training of each trial with given hyper parameters"""
        hyper_parameters = self._prepare_hyper_parameter()
        dataset = self._prepare_dataset()

        environment = self._prepare_environment(hyper_parameters, dataset)
        self._set_hyper_parameter(environment)

        need_to_save_initial_weight = False
        resume_weight_path = self._get_resume_weight_path()
        if resume_weight_path is not None:
            environment.resume_model_weight(resume_weight_path)
        else:
            initial_weight = self._load_fixed_initial_weight()
            if initial_weight is not None:
                environment.load_model_weight(initial_weight)
            else:
                need_to_save_initial_weight = True

        task = environment.get_train_task()
        if need_to_save_initial_weight:
            self._add_initial_weight_saving_hook(task)

        output_model = environment.get_output_model(dataset)
        score_report_callback = self._prepare_score_report_callback(task)
        task.train(dataset=dataset, output_model=output_model, train_parameters=score_report_callback)
        self._finalize_trial(task)

    def _prepare_hyper_parameter(self):
        return create(self._model_template.hyper_parameters.data)

    def _prepare_dataset(self):
        dataset_class = get_dataset_class(self._task.task_type)
        dataset = dataset_class(
            train_subset={
                "ann_file": self._dataset_paths.get("train_ann_file", None),
                "data_root": self._dataset_paths.get("train_data_root", None),
            },
            val_subset={
                "ann_file": self._dataset_paths.get("val_ann_file", None),
                "data_root": self._dataset_paths.get("val_data_root", None),
            }
        )
        dataset = HpoDataset(dataset, self._hp_config)

        return dataset

    def _set_hyper_parameter(self, environment: TaskEnvironmentManager):
        environment.set_hyper_parameter_from_flatten_format_dict(self._hp_config["configuration"])
        environment.set_epoch(self._epoch)

    def _prepare_environment(self, hyper_parameters, dataset):
        enviroment = TaskEnvironment(
            model=None,
            hyper_parameters=hyper_parameters,
            label_schema=generate_label_schema(dataset, self._task.task_type),
            model_template=self._model_template
        )

        return TaskEnvironmentManager(enviroment)

    def _get_resume_weight_path(self):
        trial_work_dir = self._get_weight_dir_path()
        if not osp.exists(trial_work_dir):
            return None
        return self._task.get_latest_weight(trial_work_dir)

    def _load_fixed_initial_weight(self):
        initial_weight_path = self._get_initial_weight_path()
        if osp.exists(initial_weight_path):
            return initial_weight_path
        return None

    def _prepare_task(self, environment: TaskEnvironment):
        task_class = get_impl_class(environment.model_template.entrypoints.base)
        return task_class(task_environment=environment)

    def _add_initial_weight_saving_hook(self, task):
        initial_weight_path = self._get_initial_weight_path()
        task.update_override_configurations(
            {
                "custom_hooks": [
                    dict(
                        type="SaveInitialWeightHook",
                        save_path=osp.dirname(initial_weight_path),
                        file_name=osp.basename(initial_weight_path),
                        after_save_func=self._change_model_weight_to_otx_format
                    )
                ]
            }
        )

    def _change_model_weight_to_otx_format(self):
        initial_weight_path = self._get_initial_weight_path()
        initial_weight = torch.load(initial_weight_path, map_location="cpu")
        initial_weight["model"] = initial_weight["state_dict"]
        torch.save(initial_weight, initial_weight_path)

    def _prepare_score_report_callback(self, task):
        return TrainParameters(False, HpoCallback(self._report_func, self._metric, self._epoch, task))

    def _get_initial_weight_path(self):
        return osp.join(self._hpo_workdir, self._initial_weight_name)

    def _finalize_trial(self, task):
        weight_dir_path = self._get_weight_dir_path()
        os.makedirs(weight_dir_path, exist_ok=True)
        self._task.copy_weight(task.output_path, weight_dir_path)
        for best_weight in self._task.get_best_weights(task.output_path):
            shutil.copy(best_weight, weight_dir_path)
        self._report_func(0, 0, done=True)

    def _get_weight_dir_path(self):
        return osp.join(self._hpo_workdir, "weight", self._hp_config["id"])

def run_trial(
    hp_config:  Dict,
    report_func: Callable,
    model_template,
    dataset_paths: Dict[str, str],
    task_type: TaskType,
    hpo_workdir: str,
    initial_weight_name: str,
    metric: str
):
    trainer = Trainer(
        hp_config,
        report_func,
        model_template,
        dataset_paths,
        task_type,
        hpo_workdir,
        initial_weight_name,
        metric
    )
    trainer.run()

def mocking_run_trial(
    hp_config:  Dict,
    report_func: Callable,
    model_template,
    dataset_paths: Dict[str, str],
    task_type: TaskType,
    hpo_workdir: str,
    initial_weight_name: str,
    metric: str,
):
    print("mocking_run_trial start!")

    import logging
    hpopt_logger = logging.getLogger("hpopt")
    hpopt_logger.setLevel(logging.DEBUG)

    lr = hp_config["configuration"]["learning_parameters.learning_rate"]
    bs = hp_config["configuration"]["learning_parameters.batch_size"]
    obj_func = 100 - (0.001 - lr) ** 2 - ((16 - bs) ** 2) / 100
    iteration = floor(hp_config["configuration"]["iterations"])
    validation_interval = 2
    print(f"iteratoin : {iteration}")
    if iteration > 50:
        iteration = 50
        print("iteration is modified to 50.")

    for i in range(validation_interval, iteration+1, validation_interval):
        score = obj_func + i / 100
        progress = i
        stop_train = report_func(score, progress)
        if stop_train == TrialStatus.STOP:
            break

    report_func(0, 0, done=True)

class HpoCallback(UpdateProgressCallback):
    """Callback class to report score to hpopt"""

    def __init__(self, report_func: Callable, metric: str, max_epoch: int, task):
        super().__init__()
        self._report_func = report_func
        self.metric = metric
        self._max_epoch = max_epoch
        self._task = task

    def __call__(self, progress: Union[int, float], score: Optional[float] = None):
        if score is not None:
            epoch = round(self._max_epoch * progress / 100)
            print("*"*100, f"In hpo callback : {score} / {progress} / {epoch}")
            if self._report_func(score=score, progress=epoch) == TrialStatus.STOP:
                self._task.cancel_training()

class HpoDataset:
    """
    Wrapper class for DatasetEntity of dataset.
    It's used to make subset during HPO.
    """

    def __init__(self, fullset, config=None, indices=None):
        self.fullset = fullset
        self.indices = indices
        subset_ratio = config["train_environment"]["subset_ratio"]
        self.subset_ratio = 1 if subset_ratio is None else subset_ratio

    def __len__(self) -> int:
        return len(self.indices)

    def __getitem__(self, indx) -> dict:
        return self.fullset[self.indices[indx]]

    def __getattr__(self, name):
        if name == "__setstate__":
            raise AttributeError(name)
        return getattr(self.fullset, name)

    def get_subset(self, subset: Subset):
        """
        Get subset according to subset_ratio if trainin dataset is requeseted.
        """

        dataset = self.fullset.get_subset(subset)
        if subset != Subset.TRAINING or self.subset_ratio > 0.99:
            return dataset

        indices = torch.randperm(
            len(dataset), generator=torch.Generator().manual_seed(42)
        )
        indices = indices.tolist()  # type: ignore
        indices = indices[: int(len(dataset) * self.subset_ratio)]

        return HpoDataset(dataset, config=None, indices=indices)
