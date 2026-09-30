from chunktrust.paths import resolve_legacy_path as _ct_path
from pathlib import Path


ISAAC_ROOT = Path(_ct_path('CHUNKTRUST_WORKSPACE_ROOT', 'Isaac-GR00T'))
STARVLA_ROOT = Path(_ct_path('CHUNKTRUST_WORKSPACE_ROOT', 'starVLA'))


def replace_once(text: str, old: str, new: str, path: Path) -> str:
    if old not in text:
        raise RuntimeError(f"Pattern not found in {path}: {old[:120]!r}")
    return text.replace(old, new, 1)


def is_horizon_ready(text: str) -> bool:
    return "horizon_candidates" in text or "HORIZON_CANDIDATES" in text


def patch_stride_selector() -> None:
    src = ISAAC_ROOT / "gr00t/eval/stride_selector.py"
    dst = STARVLA_ROOT / "examples/Robocasa_tabletop/eval_files/stride_selector.py"
    dst.write_text(src.read_text(encoding="utf-8"), encoding="utf-8")


def patch_model_interface() -> None:
    path = STARVLA_ROOT / "examples/Robocasa_tabletop/eval_files/model2robocasa_interface.py"
    text = path.read_text(encoding="utf-8")
    if is_horizon_ready(text):
        return

    text = replace_once(
        text,
        """        stride_exec_mode: str = "expected_round",
        stride_expected_temp: float = 0.75,
        stride_ts_alpha: float = 80.0,
        stride_z_mode: str = "window_rms",
        stride_tau_window: int = 3,
        stride_signal_mode: str = "intra",
        stride_use_speed_uniformity: bool = True,
        stride_speed_window: int = 60,
        stride_speed_weight: float = 0.3,
        stride_speed_beta: float = 4.0,
        stride_speed_fallback: str = "ehh_only",
        replan_log_path: Optional[str] = None,
""",
        """        stride_exec_mode: str = "expected_round",
        stride_expected_temp: float = 0.75,
        stride_ts_alpha: float = 80.0,
        stride_ts_epsilon: float = 0.05,
        stride_ts_seed: int = 0,
        stride_ts_update_mode: str = "legacy",
        stride_ts_kernel_bandwidth: float = 10.0,
        stride_ts_forget_rho: float = 1.0,
        stride_ts_update_eta: float = 1.0,
        stride_z_mode: str = "window_rms",
        stride_tau_window: int = 3,
        stride_signal_mode: str = "intra",
        stride_use_speed_uniformity: bool = True,
        stride_speed_window: int = 60,
        stride_speed_weight: float = 0.3,
        stride_speed_beta: float = 4.0,
        stride_speed_fallback: str = "ehh_only",
        replan_log_path: Optional[str] = None,
""",
        path,
    )

    text = replace_once(
        text,
        """        self.stride_exec_mode = str(stride_exec_mode).strip().lower()
        self.stride_expected_temp = float(stride_expected_temp)
        self.stride_ts_alpha = float(stride_ts_alpha)
        self.stride_z_mode = str(stride_z_mode).strip().lower()
        self.stride_tau_window = int(stride_tau_window)
""",
        """        self.stride_exec_mode = str(stride_exec_mode).strip().lower()
        self.stride_expected_temp = float(stride_expected_temp)
        self.stride_ts_alpha = float(stride_ts_alpha)
        self.stride_ts_epsilon = float(stride_ts_epsilon)
        self.stride_ts_seed = int(stride_ts_seed)
        self.stride_ts_update_mode = str(stride_ts_update_mode).strip().lower()
        self.stride_ts_kernel_bandwidth = float(stride_ts_kernel_bandwidth)
        self.stride_ts_forget_rho = float(stride_ts_forget_rho)
        self.stride_ts_update_eta = float(stride_ts_update_eta)
        self.stride_z_mode = str(stride_z_mode).strip().lower()
        self.stride_tau_window = int(stride_tau_window)
""",
        path,
    )

    text = replace_once(
        text,
        """        self._episode_npz_paths: dict[int, Path] = {}
        self._denoise_overhead_seconds: dict[int, float] = {}
        self._env_episode_indices: dict[int, int] = {}
""",
        """        self._episode_npz_paths: dict[int, Path] = {}
        self._denoise_overhead_seconds: dict[int, float] = {}
        self._env_episode_indices: dict[int, int] = {}
        self._stride_ts_states: dict[int, dict[int, list[float]]] = {}
        self._stride_ts_rngs: dict[int, np.random.Generator] = {}
""",
        path,
    )

    text = replace_once(
        text,
        """        self._episode_npz_paths.pop(env_idx, None)
        self._denoise_overhead_seconds[env_idx] = 0.0
        self._get_denoise_logger(env_idx).reset_episode()
""",
        """        self._episode_npz_paths.pop(env_idx, None)
        self._denoise_overhead_seconds[env_idx] = 0.0
        self._stride_ts_states[env_idx] = {}
        self._stride_ts_rngs[env_idx] = np.random.default_rng(self.stride_ts_seed + env_idx)
        if env_idx < len(self.executed_action_history):
            history = self.executed_action_history[env_idx]
            action_dim = int(history.shape[1]) if getattr(history, "ndim", 0) == 2 else 0
            self.executed_action_history[env_idx] = np.zeros((0, action_dim), dtype=np.float32)
        self._get_denoise_logger(env_idx).reset_episode()
""",
        path,
    )

    text = replace_once(
        text,
        """            "stride_q_proxy": np.asarray(stride_info.get("q_proxy", []), dtype=np.float32),
            "stride_q_mix": np.asarray(stride_info.get("q_mix", []), dtype=np.float32),
            "stride_speed_uniformity_cv": self._float_list_or_nan(
                stride_info.get("speed_uniformity_cv", [])
            ),
""",
        """            "stride_q_proxy": np.asarray(stride_info.get("q_proxy", []), dtype=np.float32),
            "stride_q_mix": np.asarray(stride_info.get("q_mix", []), dtype=np.float32),
            "stride_score": np.asarray(stride_info.get("score", []), dtype=np.float32),
            "stride_speed_uniformity_cv": self._float_list_or_nan(
                stride_info.get("speed_uniformity_cv", [])
            ),
""",
        path,
    )

    text = replace_once(
        text,
        """            "stride_expected_prob": np.asarray(
                stride_info.get("expected_prob", []),
                dtype=np.float32,
            ),
            "stride_expected_stride": np.asarray(
                float(stride_info.get("expected_stride", np.nan)),
                dtype=np.float32,
            ),
            "stride_chosen_stride": np.asarray(
                int(stride_info.get("chosen_stride", 0)),
                dtype=np.int32,
            ),
            "stride_ehh_curve_by_stride": np.asarray(
                stride_info.get("ehh_curve_by_stride", []),
                dtype=np.float32,
            ),
            "stride_info_json": np.asarray(
                json.dumps(self._json_safe(stride_info), ensure_ascii=False)
            ),
""",
        """            "stride_expected_prob": np.asarray(
                stride_info.get("expected_prob", []),
                dtype=np.float32,
            ),
            "stride_theta_sample": np.asarray(
                stride_info.get("theta_sample", []),
                dtype=np.float32,
            ),
            "stride_posterior_alpha_before": np.asarray(
                stride_info.get("posterior_alpha_before", []),
                dtype=np.float32,
            ),
            "stride_posterior_beta_before": np.asarray(
                stride_info.get("posterior_beta_before", []),
                dtype=np.float32,
            ),
            "stride_posterior_alpha_after": np.asarray(
                stride_info.get("posterior_alpha_after", []),
                dtype=np.float32,
            ),
            "stride_posterior_beta_after": np.asarray(
                stride_info.get("posterior_beta_after", []),
                dtype=np.float32,
            ),
            "stride_expected_stride": np.asarray(
                float(stride_info.get("expected_stride", np.nan)),
                dtype=np.float32,
            ),
            "stride_chosen_stride": np.asarray(
                int(stride_info.get("chosen_stride", 0)),
                dtype=np.int32,
            ),
            "stride_update_center": np.asarray(
                float(stride_info.get("update_center", np.nan)),
                dtype=np.float32,
            ),
            "stride_update_feedback": np.asarray(
                float(stride_info.get("update_feedback", np.nan)),
                dtype=np.float32,
            ),
            "stride_ehh_curve_by_stride": np.asarray(
                stride_info.get("ehh_curve_by_stride", []),
                dtype=np.float32,
            ),
            "stride_selector_mode": np.asarray(
                str(stride_info.get("selector_mode", "")),
            ),
            "stride_choose_mode": np.asarray(
                str(stride_info.get("choose_mode", "")),
            ),
            "stride_info_json": np.asarray(
                json.dumps(self._json_safe(stride_info), ensure_ascii=False)
            ),
""",
        path,
    )

    text = replace_once(
        text,
        """        if len(self.executed_action_history) != raw_actions.shape[0]:
            self.executed_action_history = [
                np.zeros((0, raw_actions.shape[-1]), dtype=np.float32)
                for _ in range(raw_actions.shape[0])
            ]
""",
        """        if len(self.executed_action_history) != raw_actions.shape[0]:
            self.executed_action_history = [
                np.zeros((0, raw_actions.shape[-1]), dtype=np.float32)
                for _ in range(raw_actions.shape[0])
            ]
        for batch_idx in range(raw_actions.shape[0]):
            self._stride_ts_states.setdefault(batch_idx, {})
            self._stride_ts_rngs.setdefault(
                batch_idx,
                np.random.default_rng(self.stride_ts_seed + batch_idx),
            )
""",
        path,
    )

    text = replace_once(
        text,
        """                    expected_temp=self.stride_expected_temp,
                    stride_alpha=self.stride_ts_alpha,
                    z_mode=self.stride_z_mode,
                    tau_window=self.stride_tau_window,
                    xt_step=np.asarray(raw_actions[batch_idx, :used_h, :], dtype=np.float32),
                    history_actions=self.executed_action_history[batch_idx],
                    signal_mode=self.stride_signal_mode,
                    use_speed_uniformity=self.stride_use_speed_uniformity,
                    speed_window=self.stride_speed_window,
                    speed_weight=self.stride_speed_weight,
                    speed_beta=self.stride_speed_beta,
                    speed_fallback=self.stride_speed_fallback,
                )
""",
        """                    expected_temp=self.stride_expected_temp,
                    stride_alpha=self.stride_ts_alpha,
                    ts_epsilon=self.stride_ts_epsilon,
                    ts_update_mode=self.stride_ts_update_mode,
                    ts_kernel_bandwidth=self.stride_ts_kernel_bandwidth,
                    ts_forget_rho=self.stride_ts_forget_rho,
                    ts_update_eta=self.stride_ts_update_eta,
                    z_mode=self.stride_z_mode,
                    tau_window=self.stride_tau_window,
                    xt_step=np.asarray(raw_actions[batch_idx, :used_h, :], dtype=np.float32),
                    history_actions=self.executed_action_history[batch_idx],
                    signal_mode=self.stride_signal_mode,
                    use_speed_uniformity=self.stride_use_speed_uniformity,
                    speed_window=self.stride_speed_window,
                    speed_weight=self.stride_speed_weight,
                    speed_beta=self.stride_speed_beta,
                    speed_fallback=self.stride_speed_fallback,
                    ts_state=self._stride_ts_states[batch_idx],
                    ts_rng=self._stride_ts_rngs[batch_idx],
                )
""",
        path,
    )

    path.write_text(text, encoding="utf-8")


def patch_simulation_env() -> None:
    path = STARVLA_ROOT / "examples/Robocasa_tabletop/eval_files/simulation_env.py"
    text = path.read_text(encoding="utf-8")
    if is_horizon_ready(text):
        return

    text = replace_once(
        text,
        """    stride_exec_mode: str = "expected_round"
    stride_expected_temp: float = 0.75
    stride_ts_alpha: float = 80.0
    stride_z_mode: str = "window_rms"
    stride_tau_window: int = 3
""",
        """    stride_exec_mode: str = "expected_round"
    stride_expected_temp: float = 0.75
    stride_ts_alpha: float = 80.0
    stride_ts_epsilon: float = 0.05
    stride_ts_seed: int = 0
    stride_ts_update_mode: str = "legacy"
    stride_ts_kernel_bandwidth: float = 10.0
    stride_ts_forget_rho: float = 1.0
    stride_ts_update_eta: float = 1.0
    stride_z_mode: str = "window_rms"
    stride_tau_window: int = 3
""",
        path,
    )

    text = replace_once(
        text,
        """        stride_exec_mode=args.stride_exec_mode,
        stride_expected_temp=args.stride_expected_temp,
        stride_ts_alpha=args.stride_ts_alpha,
        stride_z_mode=args.stride_z_mode,
        stride_tau_window=args.stride_tau_window,
""",
        """        stride_exec_mode=args.stride_exec_mode,
        stride_expected_temp=args.stride_expected_temp,
        stride_ts_alpha=args.stride_ts_alpha,
        stride_ts_epsilon=args.stride_ts_epsilon,
        stride_ts_seed=args.stride_ts_seed,
        stride_ts_update_mode=args.stride_ts_update_mode,
        stride_ts_kernel_bandwidth=args.stride_ts_kernel_bandwidth,
        stride_ts_forget_rho=args.stride_ts_forget_rho,
        stride_ts_update_eta=args.stride_ts_update_eta,
        stride_z_mode=args.stride_z_mode,
        stride_tau_window=args.stride_tau_window,
""",
        path,
    )

    path.write_text(text, encoding="utf-8")


def patch_batch_script() -> None:
    path = STARVLA_ROOT / "examples/Robocasa_tabletop/eval_files/batch_eval_single_gpu.sh"
    text = path.read_text(encoding="utf-8")
    if is_horizon_ready(text):
        return

    text = replace_once(
        text,
        """RESULT_ROOT="${ROOT_DIR}/examples/Robocasa_tabletop/eval_result"
SUMMARY_DIR="${RESULT_ROOT}/_summary/${MODEL_NAME}/${EVAL_TAG}/${CKPT_TAG}"
""",
        """RESULT_ROOT="${RESULT_ROOT:-/home/hfd24/Fanding/United/robocasa-gr1-tabletop-tasks/eval_results}"
SUMMARY_DIR="${RESULT_ROOT}/_summary/${MODEL_NAME}/${CKPT_TAG}/${EVAL_TAG}"
""",
        path,
    )

    text = replace_once(
        text,
        """  TASK_DIR="${RESULT_ROOT}/${TASK_NAME}/${MODEL_NAME}/${EVAL_TAG}/${CKPT_TAG}"
""",
        """  TASK_DIR="${RESULT_ROOT}/${TASK_NAME}/${MODEL_NAME}/${CKPT_TAG}/${EVAL_TAG}"
""",
        path,
    )

    text = replace_once(
        text,
        """STRIDE_EXEC_MODE="${STRIDE_EXEC_MODE:-expected_round}"
STRIDE_EXPECTED_TEMP="${STRIDE_EXPECTED_TEMP:-0.75}"
STRIDE_TS_ALPHA="${STRIDE_TS_ALPHA:-80.0}"
STRIDE_Z_MODE="${STRIDE_Z_MODE:-window_rms}"
""",
        """STRIDE_EXEC_MODE="${STRIDE_EXEC_MODE:-expected_round}"
STRIDE_EXPECTED_TEMP="${STRIDE_EXPECTED_TEMP:-0.75}"
STRIDE_TS_ALPHA="${STRIDE_TS_ALPHA:-80.0}"
STRIDE_TS_EPSILON="${STRIDE_TS_EPSILON:-0.05}"
STRIDE_TS_SEED="${STRIDE_TS_SEED:-0}"
STRIDE_TS_UPDATE_MODE="${STRIDE_TS_UPDATE_MODE:-legacy}"
STRIDE_TS_KERNEL_BANDWIDTH="${STRIDE_TS_KERNEL_BANDWIDTH:-10.0}"
STRIDE_TS_FORGET_RHO="${STRIDE_TS_FORGET_RHO:-1.0}"
STRIDE_TS_UPDATE_ETA="${STRIDE_TS_UPDATE_ETA:-1.0}"
STRIDE_Z_MODE="${STRIDE_Z_MODE:-window_rms}"
""",
        path,
    )

    text = replace_once(
        text,
        """    --args.stride_exec_mode "${STRIDE_EXEC_MODE}"
    --args.stride_expected_temp "${STRIDE_EXPECTED_TEMP}"
    --args.stride_ts_alpha "${STRIDE_TS_ALPHA}"
    --args.stride_z_mode "${STRIDE_Z_MODE}"
""",
        """    --args.stride_exec_mode "${STRIDE_EXEC_MODE}"
    --args.stride_expected_temp "${STRIDE_EXPECTED_TEMP}"
    --args.stride_ts_alpha "${STRIDE_TS_ALPHA}"
    --args.stride_ts_epsilon "${STRIDE_TS_EPSILON}"
    --args.stride_ts_seed "${STRIDE_TS_SEED}"
    --args.stride_ts_update_mode "${STRIDE_TS_UPDATE_MODE}"
    --args.stride_ts_kernel_bandwidth "${STRIDE_TS_KERNEL_BANDWIDTH}"
    --args.stride_ts_forget_rho "${STRIDE_TS_FORGET_RHO}"
    --args.stride_ts_update_eta "${STRIDE_TS_UPDATE_ETA}"
    --args.stride_z_mode "${STRIDE_Z_MODE}"
""",
        path,
    )

    text = replace_once(
        text,
        """  "stride_exec_mode": "${STRIDE_EXEC_MODE}",
  "stride_expected_temp": ${STRIDE_EXPECTED_TEMP},
  "stride_ts_alpha": ${STRIDE_TS_ALPHA},
  "stride_z_mode": "${STRIDE_Z_MODE}",
""",
        """  "stride_exec_mode": "${STRIDE_EXEC_MODE}",
  "stride_expected_temp": ${STRIDE_EXPECTED_TEMP},
  "stride_ts_alpha": ${STRIDE_TS_ALPHA},
  "stride_ts_epsilon": ${STRIDE_TS_EPSILON},
  "stride_ts_seed": ${STRIDE_TS_SEED},
  "stride_ts_update_mode": "${STRIDE_TS_UPDATE_MODE}",
  "stride_ts_kernel_bandwidth": ${STRIDE_TS_KERNEL_BANDWIDTH},
  "stride_ts_forget_rho": ${STRIDE_TS_FORGET_RHO},
  "stride_ts_update_eta": ${STRIDE_TS_UPDATE_ETA},
  "stride_z_mode": "${STRIDE_Z_MODE}",
""",
        path,
    )

    path.write_text(text, encoding="utf-8")


def main() -> None:
    patch_stride_selector()
    patch_model_interface()
    patch_simulation_env()
    patch_batch_script()
    print("patched starVLA stride Thompson/kernel_forget support")


if __name__ == "__main__":
    main()
