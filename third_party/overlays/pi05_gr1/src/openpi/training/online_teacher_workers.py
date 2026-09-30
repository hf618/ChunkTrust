from __future__ import annotations

import dataclasses
import logging
import multiprocessing as mp
import os
import pathlib
import queue
import time
import traceback
from typing import Any

import numpy as np


@dataclasses.dataclass(frozen=True)
class WorkerSpec:
    worker_id: int
    device_id: str
    repo_ids: tuple[str, ...]


def _checkpoint_dir_for_spec(spec: dict[str, Any], checkpoint_base_dir: str) -> pathlib.Path:
    return (
        pathlib.Path(checkpoint_base_dir)
        / str(spec.get("checkpoint_root_tag") or spec["train_config_name"])
        / str(spec["model_name"])
        / str(spec["checkpoint_id"])
    ).resolve()


def _worker_main(
    *,
    worker_id: int,
    device_id: str,
    specs_by_repo_id: dict[str, dict[str, Any]],
    checkpoint_base_dir: str,
    request_queue: mp.Queue,
    response_queue: mp.Queue,
) -> None:
    # Import JAX/openpi only after constraining this process to a single GPU.
    os.environ["CUDA_VISIBLE_DEVICES"] = str(device_id)
    os.environ.setdefault("JAX_PLATFORMS", "cuda")
    try:
        import dataclasses as _dataclasses

        import jax
        import jax.numpy as jnp

        import openpi.models.model as _model
        import openpi.shared.nnx_utils as nnx_utils
        from openpi.training import checkpoints as _checkpoints
        from openpi.training import config as _config

        runtimes: dict[str, dict[str, Any]] = {}
        preload_start = time.perf_counter()
        for i, (repo_id, raw_spec) in enumerate(specs_by_repo_id.items()):
            load_start = time.perf_counter()
            checkpoint_dir = _checkpoint_dir_for_spec(raw_spec, checkpoint_base_dir)
            params = _model.restore_params(checkpoint_dir / "params", restore_type=np.ndarray, dtype=jnp.bfloat16)
            teacher_config = _dataclasses.replace(_config.get_config(str(raw_spec["train_config_name"])))
            model_config = _checkpoints.resolve_model_config_for_checkpoint(
                teacher_config.model,
                checkpoint_dir,
                restored_params=params,
                qha_candidate_mode_override="dense_full",
            )
            if hasattr(model_config, "use_qha"):
                model_config = _dataclasses.replace(model_config, use_qha=False, qha_candidate_mode="dense_full")
            model = model_config.load(params)
            model.eval()
            runtimes[repo_id] = {
                "predict_fn": nnx_utils.module_jit(model.predict_qha_supervision_outputs),
                "rng": jax.random.key(worker_id * 1000 + i),
                "checkpoint_dir": str(checkpoint_dir),
                "load_wall_s": time.perf_counter() - load_start,
            }

        response_queue.put(
            {
                "type": "ready",
                "worker_id": worker_id,
                "device_id": str(device_id),
                "repo_ids": tuple(specs_by_repo_id),
                "preload_wall_s": time.perf_counter() - preload_start,
            }
        )

        while True:
            request = request_queue.get()
            if request is None or request.get("type") == "shutdown":
                return
            request_id = request["request_id"]
            groups = request["groups"]
            wall_start = time.perf_counter()
            out_groups = []
            for group in groups:
                repo_id = group["repo_id"]
                runtime = runtimes[repo_id]
                runtime["rng"], call_rng = jax.random.split(runtime["rng"])
                outputs = runtime["predict_fn"](call_rng, group["observation"])
                out_groups.append(
                    {
                        "repo_id": repo_id,
                        "positions": group["positions"],
                        "outputs": {key: np.asarray(jax.device_get(value)) for key, value in outputs.items()},
                    }
                )
            response_queue.put(
                {
                    "type": "result",
                    "worker_id": worker_id,
                    "request_id": request_id,
                    "groups": out_groups,
                    "wall_s": time.perf_counter() - wall_start,
                }
            )
    except BaseException:
        response_queue.put(
            {
                "type": "error",
                "worker_id": worker_id,
                "device_id": str(device_id),
                "traceback": traceback.format_exc(),
            }
        )


class GpuTeacherWorkerPool:
    def __init__(
        self,
        *,
        worker_specs: list[WorkerSpec],
        specs_by_repo_id: dict[str, dict[str, Any]],
        checkpoint_base_dir: str,
        startup_timeout_s: int,
        request_timeout_s: int,
    ):
        # Use forkserver so child workers do not re-import scripts/train.py before
        # _worker_main can pin CUDA_VISIBLE_DEVICES and then import JAX/openpi.
        self._ctx = mp.get_context("forkserver")
        self._response_queue = self._ctx.Queue()
        self._request_queues: dict[int, mp.Queue] = {}
        self._processes: dict[int, mp.Process] = {}
        self._request_timeout_s = int(request_timeout_s)
        self._next_request_id = 0
        self._closed = False
        for worker in worker_specs:
            request_queue = self._ctx.Queue()
            assigned_specs = {repo_id: specs_by_repo_id[repo_id] for repo_id in worker.repo_ids}
            process = self._ctx.Process(
                target=_worker_main,
                kwargs={
                    "worker_id": worker.worker_id,
                    "device_id": worker.device_id,
                    "specs_by_repo_id": assigned_specs,
                    "checkpoint_base_dir": checkpoint_base_dir,
                    "request_queue": request_queue,
                    "response_queue": self._response_queue,
                },
                daemon=True,
            )
            process.start()
            self._request_queues[worker.worker_id] = request_queue
            self._processes[worker.worker_id] = process

        self._wait_until_ready(len(worker_specs), timeout_s=int(startup_timeout_s))

    def _wait_until_ready(self, expected: int, *, timeout_s: int) -> None:
        deadline = time.perf_counter() + timeout_s
        ready: set[int] = set()
        while len(ready) < expected:
            remaining = deadline - time.perf_counter()
            if remaining <= 0:
                self.close()
                raise TimeoutError(f"Timed out waiting for {expected} teacher workers to start.")
            try:
                message = self._response_queue.get(timeout=remaining)
            except queue.Empty as exc:
                self.close()
                raise TimeoutError(f"Timed out waiting for {expected} teacher workers to start.") from exc
            if message.get("type") == "ready":
                worker_id = int(message["worker_id"])
                ready.add(worker_id)
                logging.info(
                    "Teacher GPU worker %d ready on device=%s repo_ids=%s preload_wall_s=%.3f",
                    worker_id,
                    message.get("device_id"),
                    list(message.get("repo_ids", ())),
                    float(message.get("preload_wall_s", 0.0)),
                )
                continue
            if message.get("type") == "error":
                self.close()
                raise RuntimeError(
                    f"Teacher worker {message.get('worker_id')} failed during startup:\n{message.get('traceback')}"
                )

    def infer(self, worker_groups: dict[int, list[dict[str, Any]]]) -> tuple[list[dict[str, Any]], dict[str, float]]:
        if self._closed:
            raise RuntimeError("Teacher worker pool is closed.")
        request_id = self._next_request_id
        self._next_request_id += 1
        pending = {worker_id for worker_id, groups in worker_groups.items() if groups}
        for worker_id in pending:
            self._request_queues[worker_id].put(
                {
                    "type": "infer",
                    "request_id": request_id,
                    "groups": worker_groups[worker_id],
                }
            )

        deadline = time.perf_counter() + self._request_timeout_s
        results: list[dict[str, Any]] = []
        max_worker_wall_s = 0.0
        while pending:
            remaining = deadline - time.perf_counter()
            if remaining <= 0:
                self.close()
                raise TimeoutError(f"Timed out waiting for teacher worker request {request_id}.")
            try:
                message = self._response_queue.get(timeout=remaining)
            except queue.Empty as exc:
                self.close()
                raise TimeoutError(f"Timed out waiting for teacher worker request {request_id}.") from exc
            if message.get("type") == "error":
                self.close()
                raise RuntimeError(
                    f"Teacher worker {message.get('worker_id')} failed:\n{message.get('traceback')}"
                )
            if message.get("type") != "result" or int(message.get("request_id", -1)) != int(request_id):
                continue
            worker_id = int(message["worker_id"])
            if worker_id in pending:
                pending.remove(worker_id)
                max_worker_wall_s = max(max_worker_wall_s, float(message.get("wall_s", 0.0)))
                results.extend(list(message.get("groups", [])))
        return results, {"teacher_worker_max_request_wall_s": max_worker_wall_s}

    def close(self) -> None:
        if self._closed:
            return
        self._closed = True
        for request_queue in self._request_queues.values():
            try:
                request_queue.put({"type": "shutdown"})
            except Exception:
                pass
        for process in self._processes.values():
            process.join(timeout=5)
            if process.is_alive():
                process.terminate()
                process.join(timeout=5)
