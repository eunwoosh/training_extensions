# Copyright (C) 2024 Intel Corporation
# SPDX-License-Identifier: Apache-2.0
#

from __future__ import annotations

import argparse
import logging
import json
import subprocess
from dataclasses import dataclass, asdict as dcls_todict
from tempfile import TemporaryDirectory
from typing import Literal
from pathlib import Path

from dacite import from_dict as dcls_fromdict
import pandas as pd

logger = logging.Logger(__name__)


@dataclass
class OvInferResult:
    task: str
    model: str
    device: str
    hint: Literal["latency", "throughput"]
    num_data: int
    total_time: float = -1.0
    fps: float = -1.0
    latency_median: float = -1.0
    latency_average: float = -1.0
    latency_min: float = -1.0
    latency_max: float = -1.0


def get_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Benchmark OpenVINO inference using OTX exportable code.")
    parser.add_argument(
        "--dir",
        required=True,
        help=(
            "Directory which includes OpenVINO IRs. "
            "Directory structure should be 'dir / `task_name` / `model_name` / ...'."
        )
    )
    parser.add_argument("--data-root", required=True, help="Performance test result directory.")
    parser.add_argument(
        "--output",
        default="./ov_infer_result",
        help="Place to store OpenVINO inference result."
    )
    parser.add_argument("--device", default="CPU", help="device.")
    parser.add_argument("--model-name", default="exported_model.xml", help="OpenVINO IR file name.")
    parser.add_argument("--repeat", type=int, default=5, help="repeat.")

    return parser.parse_args()


class OVInferBench:
    def __init__(
        self,
        ov_ir_dir: Path,
        model_name: str = "exported_model.xml"
    ) -> None:
        self._ov_ir_dir = ov_ir_dir
        self._model_name = model_name
        self._model_tried: list[set[str,str]] = []

    def run_benchmark(
        self,
        data_root: Path,
        output_dir: Path,
        device: str = "CPU",
        hint: Literal["latency", "throughput"] = "latency",
        repeat: int = 5,
    ) -> None:
        if repeat < 0:
            msg = "repeat should be positive"
            raise ValueError(msg)

        (output_dir / "output").mkdir(parents=True, exist_ok=True)
        num_data = len(list(data_root.iterdir()))

        self._model_tried = []
        for ir_file in self._ov_ir_dir.rglob(self._model_name):
            otx_work_dir = ir_file.parent.parent
            if not self._need_to_run(otx_work_dir):
                continue
            task, model = self._get_task_and_model(otx_work_dir)
            self._model_tried.append({task, model})

            for repeat_index in range(repeat):
                result = OvInferResult(task, model, device, hint, num_data)
                total_time, throughput, latency = self._run_ov_inference(
                    model_path=ir_file,
                    data_root=data_root,
                    device=device,
                    hint=hint
                )
                if total_time is not None:
                    result.total_time = float(total_time)
                if throughput is not None:
                    result.fps = float(throughput)
                for key, value in latency.items():
                    if value is not None:
                        setattr(result, f"latency_{key}", float(value))
                
                with (output_dir / "output" / f"{task}_{model}_repeat_{repeat_index}.json").open("w") as f:
                    json.dump(dcls_todict(result), f)

        self._gather_infer_result(output_dir)

    def _need_to_run(self, otx_work_dir: Path) -> bool:
        task, model = self._get_task_and_model(otx_work_dir)
        return {task, model} not in self._model_tried

    def _get_task_and_model(self, otx_work_dir: Path) -> tuple[str, str]:
        dir_arr = str(otx_work_dir.relative_to(self._ov_ir_dir)).split('/')
        if len(dir_arr) < 2:
            msg = f"Directory structure should be otx_work_dir / `task_name` / `model_name` / ..."
            raise RuntimeError(msg)
        return dir_arr[0], dir_arr[1]

    def _run_ov_inference(
        self,
        model_path: Path,
        data_root: Path,
        device: str = "CPU",
        hint: Literal["latency", "throughput"] = "latency",
    ) -> tuple[str | None, str | None, dict[str, str | None]]:
        if not model_path.exists():
            msg = f"{model_path} doesn't exist."
            raise RuntimeError(msg)

        num_data = len(list(data_root.iterdir()))
        if num_data == 0:
            msg = f"There is no dataset in {data_root}"
            raise RuntimeError(msg)

        with TemporaryDirectory() as temp_dir:
            command = (
                f"benchmark_app -m {model_path} -hint {hint} -d {device} --api async -niter {num_data} "
                f"-inference_only False -i {data_root} -data_shape [1,720,1280,3]  "
                f"-report_folder {temp_dir} -report_type no_counters  -json_stats"
            )
            subprocess.run(command, shell=True)
            benchmark_report = Path(temp_dir) / "benchmark_report.json"
            if not benchmark_report.exists():
                logger.warning(f"benchmark report doesn't exist. Error may be raised whlie running {model_path}.")
                return None, None, None
            return self._parse_benchmark_output(benchmark_report)

    def _parse_benchmark_output(self, benchmark_output_file: Path) -> tuple[str | None, str | None, dict[str, str | None]]:
        with benchmark_output_file.open() as f:
            benchmark_output = json.load(f)["execution_results"]

        total_time = benchmark_output.get("total execution time (ms)")
        throughput = benchmark_output.get("throughput")
        latency = {
            "median" : benchmark_output.get("total execution time (ms)"),
            "average" : benchmark_output.get("avg latency"),
            "min" : benchmark_output.get("min latency"),
            "max" : benchmark_output.get("max latency")
        }

        return total_time, throughput, latency

    def _gather_infer_result(self, output_dir: Path) -> None:
        infer_result_arr = []
        for json_file in (output_dir / "output").iterdir():
            with json_file.open() as f:
                infer_result_arr.append(dcls_fromdict(data_class=OvInferResult, data=json.load(f)))
        total_data = pd.DataFrame(infer_result_arr)
        total_data.to_csv(output_dir / "raw_result.csv")
        summary_data = total_data.groupby(['task', 'model', 'device', 'hint', 'num_data'], as_index=False).agg(
            {
                'total_time' : ['mean','std'],
                'fps' : ['mean','std'],
                'latency_median' : ['mean','std'],
                'latency_average' : ['mean','std'],
                'latency_min' : ['mean','std'],
                'latency_max' : ['mean','std']
            }
        )
        summary_data.to_csv(output_dir / "result.csv")


if __name__ == "__main__":
    args = get_args()
    infer_bench = OVInferBench(Path(args.dir), args.model_name)
    infer_bench.run_benchmark(Path(args.data_root), Path(args.output), args.device)
