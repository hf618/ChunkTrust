#!/usr/bin/env python3
"""Non-blocking sidecar video recorder for real-robot rollout evaluation."""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
import queue
import re
import shutil
import subprocess
import threading
import time
from typing import Any

import numpy as np


CAMERA_STREAMS = {
    "top": "image_top",
    "right": "image_right",
    "left": "image_left",
}


def add_recording_args(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--record-video", type=int, choices=[0, 1], default=0)
    parser.add_argument("--record-root", type=Path, default=Path("records/eval_videos"))
    parser.add_argument("--record-task-name", default="")
    parser.add_argument("--record-checkpoint-dir", type=Path, default=None)
    parser.add_argument("--record-episode-idx", type=int, default=-1)
    parser.add_argument("--record-overwrite", type=int, choices=[0, 1], default=0)
    parser.add_argument("--record-fps", type=float, default=30.0)
    parser.add_argument("--record-max-queue", type=int, default=90)
    parser.add_argument(
        "--record-overview-device",
        default="auto",
        help="Overview camera device path, or 'auto' to use the first OsmoAction6 /dev/video* device.",
    )
    parser.add_argument("--record-overview-width", type=int, default=1920)
    parser.add_argument("--record-overview-height", type=int, default=1080)
    parser.add_argument("--record-overview-fps", type=float, default=30.0)
    parser.add_argument("--record-overview-input-format", choices=["mjpeg", "h264"], default="mjpeg")
    parser.add_argument("--record-overview-codec", choices=["libx264", "copy"], default="libx264")
    parser.add_argument("--record-overview-preset", default="ultrafast")


def create_eval_video_recorder(args: argparse.Namespace, robot: Any, *, rollout_kind: str) -> EvalVideoRecorder | None:
    if not getattr(args, "record_video", 0):
        return None
    return EvalVideoRecorder(
        robot=robot,
        root=Path(args.record_root),
        task_name=_resolve_task_name(args),
        episode_idx=int(args.record_episode_idx),
        overwrite=bool(args.record_overwrite),
        fps=float(args.record_fps),
        max_queue=int(args.record_max_queue),
        overview_device=_resolve_overview_device(str(args.record_overview_device)),
        overview_width=int(args.record_overview_width),
        overview_height=int(args.record_overview_height),
        overview_fps=float(args.record_overview_fps),
        overview_input_format=str(args.record_overview_input_format),
        overview_codec=str(args.record_overview_codec),
        overview_preset=str(args.record_overview_preset),
        rollout_kind=rollout_kind,
        rollout_args=args,
    )


def _sanitize_name(value: str, *, fallback: str = "unknown_task") -> str:
    value = value.strip().lower()
    value = re.sub(r"[^a-z0-9._-]+", "_", value)
    value = re.sub(r"_+", "_", value).strip("._-")
    if not value:
        return fallback
    return value[:80].strip("._-") or fallback


def _resolve_task_name(args: argparse.Namespace) -> str:
    explicit = str(getattr(args, "record_task_name", "") or "").strip()
    if explicit:
        return _sanitize_name(explicit)

    checkpoint_dir = getattr(args, "record_checkpoint_dir", None)
    if checkpoint_dir:
        path = Path(checkpoint_dir)
        # Checkpoint paths usually end in a numeric step, e.g. task_name/40000.
        candidate = path.parent.name if path.name.isdigit() else path.name
        if candidate:
            return _sanitize_name(candidate)

    prompt = str(getattr(args, "prompt", "") or "").strip()
    if prompt:
        return _sanitize_name(prompt[:80])

    return "unknown_task"


def _resolve_overview_device(device: str) -> str:
    device = str(device or "").strip()
    if device and device.lower() != "auto":
        return device
    return _find_osmo_action_device()


def _find_osmo_action_device() -> str:
    try:
        result = subprocess.run(
            ["v4l2-ctl", "--list-devices"],
            check=False,
            capture_output=True,
            text=True,
            timeout=5.0,
        )
    except (OSError, subprocess.TimeoutExpired) as exc:
        raise RuntimeError("Failed to run v4l2-ctl while auto-detecting overview camera") from exc
    if result.returncode != 0:
        raise RuntimeError(f"v4l2-ctl --list-devices failed: {result.stderr.strip()}")

    lines = result.stdout.splitlines()
    for index, line in enumerate(lines):
        if "OsmoAction6" not in line and "Osmo Action" not in line and "OsmoAc" not in line:
            continue
        for next_line in lines[index + 1 :]:
            if next_line and not next_line[0].isspace():
                break
            candidate = next_line.strip()
            if candidate.startswith("/dev/video"):
                return candidate
    raise RuntimeError(
        "Could not auto-detect DJI OsmoAction6 overview camera. "
        "Check camera USB webcam mode or pass --record-overview-device /dev/videoX."
    )


class EvalVideoRecorder:
    def __init__(
        self,
        *,
        robot: Any,
        root: Path,
        task_name: str,
        episode_idx: int,
        overwrite: bool,
        fps: float,
        max_queue: int,
        overview_device: str,
        overview_width: int,
        overview_height: int,
        overview_fps: float,
        overview_input_format: str,
        overview_codec: str,
        overview_preset: str,
        rollout_kind: str,
        rollout_args: argparse.Namespace,
    ) -> None:
        import cv2

        self._cv2 = cv2
        self.robot = robot
        self.root = root
        self.task_name = _sanitize_name(task_name)
        self.task_dir = self.root / self.task_name
        self.episode_idx_requested = episode_idx
        self.episode_idx: int | None = None
        self.episode_dir: Path | None = None
        self.overwrite = overwrite
        self.fps = max(float(fps), 1e-6)
        self.max_queue = max(1, int(max_queue))
        self.overview_device = overview_device
        self.overview_width = overview_width
        self.overview_height = overview_height
        self.overview_fps = overview_fps
        self.overview_input_format = overview_input_format
        self.overview_codec = overview_codec
        self.overview_preset = overview_preset
        self.rollout_kind = rollout_kind
        self.rollout_args = rollout_args

        self._stop_event = threading.Event()
        self._sample_thread: threading.Thread | None = None
        self._writer_threads: list[threading.Thread] = []
        self._queues = {name: queue.Queue(maxsize=self.max_queue) for name in CAMERA_STREAMS}
        self._overview_proc: subprocess.Popen | None = None
        self._overview_cmd: list[str] | None = None
        self._overview_returncode: int | None = None
        self._started = False
        self._start_time_ns: int | None = None
        self._start_monotonic_ns: int | None = None
        self._end_time_ns: int | None = None
        self._end_monotonic_ns: int | None = None
        self._stats_lock = threading.Lock()
        self._stats: dict[str, Any] = {
            "sampled_frames": {name: 0 for name in CAMERA_STREAMS},
            "written_frames": {name: 0 for name in CAMERA_STREAMS},
            "queue_full_drops": {name: 0 for name in CAMERA_STREAMS},
            "missing_frame_skips": {name: 0 for name in CAMERA_STREAMS},
            "lock_busy_skips": 0,
        }

    def start(self) -> None:
        if self._started:
            return
        self._prepare_episode_dir()
        self._start_time_ns = time.time_ns()
        self._start_monotonic_ns = time.monotonic_ns()
        self._write_overview_start_json()
        self._write_initial_metadata()
        self._start_overview_ffmpeg()

        for name in CAMERA_STREAMS:
            thread = threading.Thread(target=self._writer_loop, args=(name,), daemon=True)
            thread.start()
            self._writer_threads.append(thread)

        self._sample_thread = threading.Thread(target=self._sample_loop, daemon=True)
        self._sample_thread.start()
        self._started = True
        print(f"recording videos: {self.episode_dir}")

    def stop(self, *, status: str = "stopped") -> None:
        if not self._started:
            return
        self._stop_event.set()
        if self._sample_thread is not None:
            self._sample_thread.join(timeout=2.0)
        for q in self._queues.values():
            self._signal_writer_stop(q)
        for thread in self._writer_threads:
            thread.join(timeout=5.0)
        self._stop_overview_ffmpeg()
        self._end_time_ns = time.time_ns()
        self._end_monotonic_ns = time.monotonic_ns()
        self._write_final_metadata(status=status)
        self._started = False
        print(f"recording stopped: {self.episode_dir}")

    @staticmethod
    def _signal_writer_stop(q: queue.Queue) -> None:
        while True:
            try:
                q.put_nowait(None)
                return
            except queue.Full:
                try:
                    q.get_nowait()
                    q.task_done()
                except queue.Empty:
                    continue

    def _prepare_episode_dir(self) -> None:
        self.task_dir.mkdir(parents=True, exist_ok=True)
        auto_episode_idx = self.episode_idx_requested < 0
        episode_idx = self._next_episode_idx() if auto_episode_idx else self.episode_idx_requested
        if episode_idx < 0:
            raise ValueError("--record-episode-idx must be >= 0, or omit it for automatic numbering")

        while True:
            episode_dir = self.task_dir / f"episode_{episode_idx:06d}"
            if not episode_dir.exists():
                episode_dir.mkdir(parents=True, exist_ok=False)
                break
            if auto_episode_idx:
                episode_idx += 1
                continue
            if not self.overwrite:
                raise FileExistsError(
                    f"Recording episode directory already exists: {episode_dir}. "
                    "Use --record-overwrite 1 to replace it, or omit --record-episode-idx for automatic numbering."
                )
            shutil.rmtree(episode_dir)
            episode_dir.mkdir(parents=True, exist_ok=False)
            break

        self.episode_idx = episode_idx
        self.episode_dir = episode_dir

    def _next_episode_idx(self) -> int:
        max_idx = -1
        for path in self.task_dir.glob("episode_*"):
            if not path.is_dir():
                continue
            match = re.fullmatch(r"episode_(\d+)", path.name)
            if match:
                max_idx = max(max_idx, int(match.group(1)))
        return max_idx + 1

    def _write_overview_start_json(self) -> None:
        assert self.episode_dir is not None
        payload = {
            "device": self.overview_device,
            "width": self.overview_width,
            "height": self.overview_height,
            "fps": self.overview_fps,
            "input_format": self.overview_input_format,
            "codec": self.overview_codec,
            "preset": self.overview_preset,
            "start_time_ns": self._start_time_ns,
            "start_monotonic_ns": self._start_monotonic_ns,
            "video_path": str(self.episode_dir / "overview.mp4"),
        }
        (self.episode_dir / "overview_start.json").write_text(json.dumps(payload, indent=2) + "\n")

    def _write_initial_metadata(self) -> None:
        self._write_metadata(status="running")

    def _write_final_metadata(self, *, status: str) -> None:
        self._write_metadata(status=status)

    def _write_metadata(self, *, status: str) -> None:
        assert self.episode_dir is not None
        with self._stats_lock:
            stats = json.loads(json.dumps(self._stats))
        payload = {
            "status": status,
            "rollout_kind": self.rollout_kind,
            "task_name": self.task_name,
            "task_dir": str(self.task_dir),
            "episode_idx_requested": self.episode_idx_requested,
            "episode_idx": self.episode_idx,
            "episode_dir": str(self.episode_dir),
            "start_time_ns": self._start_time_ns,
            "start_monotonic_ns": self._start_monotonic_ns,
            "end_time_ns": self._end_time_ns,
            "end_monotonic_ns": self._end_monotonic_ns,
            "record_fps": self.fps,
            "record_max_queue": self.max_queue,
            "overview": {
                "device": self.overview_device,
                "width": self.overview_width,
                "height": self.overview_height,
                "fps": self.overview_fps,
                "input_format": self.overview_input_format,
                "codec": self.overview_codec,
                "preset": self.overview_preset,
                "video": str(self.episode_dir / "overview.mp4"),
                "ffmpeg_cmd": self._overview_cmd,
                "ffmpeg_returncode": self._overview_returncode,
            },
            "videos": {
                "top": str(self.episode_dir / "top.mp4"),
                "right": str(self.episode_dir / "right.mp4"),
                "left": str(self.episode_dir / "left.mp4"),
            },
            "stats": stats,
            "rollout_args": vars(self.rollout_args),
        }
        (self.episode_dir / "episode_metadata.json").write_text(json.dumps(payload, indent=2, default=str) + "\n")

    def _start_overview_ffmpeg(self) -> None:
        assert self.episode_dir is not None
        cmd = [
            "ffmpeg",
            "-hide_banner",
            "-loglevel",
            "warning",
            "-y",
            "-f",
            "v4l2",
            "-input_format",
            self.overview_input_format,
            "-framerate",
            str(self.overview_fps),
            "-video_size",
            f"{self.overview_width}x{self.overview_height}",
            "-i",
            self.overview_device,
            "-an",
        ]
        if self.overview_codec == "copy":
            cmd.extend(["-c:v", "copy"])
        else:
            cmd.extend(
                [
                    "-c:v",
                    self.overview_codec,
                    "-preset",
                    self.overview_preset,
                    "-tune",
                    "zerolatency",
                    "-pix_fmt",
                    "yuv420p",
                ]
            )
        cmd.extend(
            [
                str(self.episode_dir / "overview.mp4"),
            ]
        )
        self._overview_cmd = cmd
        log_file = open(self.episode_dir / "overview_ffmpeg.log", "w")
        self._overview_log_file = log_file
        self._overview_proc = subprocess.Popen(cmd, stdout=log_file, stderr=log_file)

    def _stop_overview_ffmpeg(self) -> None:
        if self._overview_proc is None:
            return
        if self._overview_proc.poll() is None:
            self._overview_proc.terminate()
            try:
                self._overview_proc.wait(timeout=5.0)
            except subprocess.TimeoutExpired:
                self._overview_proc.kill()
                self._overview_proc.wait(timeout=5.0)
        self._overview_returncode = self._overview_proc.returncode
        if hasattr(self, "_overview_log_file"):
            self._overview_log_file.close()
        self._overview_proc = None

    def _sample_loop(self) -> None:
        period = 1.0 / self.fps
        next_tick = time.monotonic()
        while not self._stop_event.is_set():
            now = time.monotonic()
            if now < next_tick:
                time.sleep(min(next_tick - now, 0.005))
                continue
            next_tick += period
            self._sample_once()
            if next_tick < time.monotonic() - period:
                next_tick = time.monotonic() + period

    def _sample_once(self) -> None:
        node = getattr(self.robot, "robot_dora_node", None)
        if node is None:
            return
        lock = getattr(node, "lock", None)
        if lock is not None and not lock.acquire(blocking=False):
            with self._stats_lock:
                self._stats["lock_busy_skips"] += 1
            return
        try:
            images = getattr(node, "recv_images", {})
            snapshots: dict[str, np.ndarray] = {}
            for name, source_key in CAMERA_STREAMS.items():
                frame = None
                for event_id, value in images.items():
                    if source_key in event_id:
                        frame = value
                        break
                if frame is None:
                    with self._stats_lock:
                        self._stats["missing_frame_skips"][name] += 1
                    continue
                snapshots[name] = np.asarray(frame).copy()
        finally:
            if lock is not None:
                lock.release()

        time_ns = time.time_ns()
        monotonic_ns = time.monotonic_ns()
        for name, frame in snapshots.items():
            item = {
                "frame": frame,
                "time_ns": time_ns,
                "monotonic_ns": monotonic_ns,
            }
            try:
                self._queues[name].put_nowait(item)
                with self._stats_lock:
                    self._stats["sampled_frames"][name] += 1
            except queue.Full:
                with self._stats_lock:
                    self._stats["queue_full_drops"][name] += 1

    def _writer_loop(self, name: str) -> None:
        cv2 = self._cv2
        video_path = self.episode_dir / f"{name}.mp4"
        timestamp_path = self.episode_dir / f"{name}_timestamps.csv"
        writer = None
        frame_index = 0
        with timestamp_path.open("w", newline="") as file:
            csv_writer = csv.writer(file)
            csv_writer.writerow(["frame_index", "time_ns", "monotonic_ns", "source_key", "width", "height"])
            while True:
                item = self._queues[name].get()
                if item is None:
                    self._queues[name].task_done()
                    break
                frame = self._normalize_frame(item["frame"])
                height, width = frame.shape[:2]
                if writer is None:
                    writer = cv2.VideoWriter(
                        str(video_path),
                        cv2.VideoWriter_fourcc(*"mp4v"),
                        self.fps,
                        (width, height),
                    )
                    if not writer.isOpened():
                        raise RuntimeError(f"Failed to open video writer: {video_path}")
                writer.write(cv2.cvtColor(frame, cv2.COLOR_RGB2BGR))
                csv_writer.writerow(
                    [
                        frame_index,
                        item["time_ns"],
                        item["monotonic_ns"],
                        CAMERA_STREAMS[name],
                        width,
                        height,
                    ]
                )
                frame_index += 1
                with self._stats_lock:
                    self._stats["written_frames"][name] += 1
                self._queues[name].task_done()
        if writer is not None:
            writer.release()

    @staticmethod
    def _normalize_frame(frame: np.ndarray) -> np.ndarray:
        arr = np.asarray(frame)
        if arr.ndim != 3:
            raise RuntimeError(f"Expected 3D image frame, got {arr.shape}")
        if arr.shape[0] == 3 and arr.shape[-1] != 3:
            arr = np.transpose(arr, (1, 2, 0))
        if arr.dtype != np.uint8:
            if np.issubdtype(arr.dtype, np.floating):
                arr = np.clip(arr * 255.0, 0, 255).astype(np.uint8)
            else:
                arr = np.clip(arr, 0, 255).astype(np.uint8)
        if arr.shape[-1] != 3:
            raise RuntimeError(f"Expected RGB image frame, got {arr.shape}")
        return np.ascontiguousarray(arr)
