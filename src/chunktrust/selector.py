"""Small, model-independent interface to the frozen PI-family AHS implementation.

Inputs must use the same action coordinates and dimensions as the selected
backend. GR00T/StarVLA paper integrations retain their own frozen selectors.
"""
from dataclasses import asdict, dataclass
import copy
import numpy as np
from ._pi_ahs import Selector


@dataclass(frozen=True)
class AHSConfig:
    horizon: int = 50
    candidates: tuple[int, ...] = (10, 20, 30, 40, 50)
    action_dim: int = 14
    temperature: float = 0.8
    epsilon: float = 0.05
    kernel_bandwidth: float = 10.0
    forget_rho: float = 0.99
    update_eta: float = 1.0
    history_window: int = 60
    mix_mode: str = "both"

    def __post_init__(self):
        object.__setattr__(self, "candidates", tuple(self.candidates))
        if not self.candidates or list(self.candidates) != sorted(set(self.candidates)):
            raise ValueError("candidates must be nonempty, unique and increasing")
        if self.candidates[0] < 1 or self.candidates[-1] > self.horizon:
            raise ValueError("candidate horizons must lie within the prediction horizon")
        if self.action_dim < 1 or self.history_window < 4 or self.temperature <= 0:
            raise ValueError("invalid dimensions, history window, or temperature")
        if not 0 <= self.epsilon <= 1 or not 0 < self.forget_rho <= 1:
            raise ValueError("invalid exploration or forgetting parameter")
        if self.kernel_bandwidth <= 0 or self.update_eta < 0:
            raise ValueError("invalid posterior update parameters")
        if self.mix_mode not in {"both", "intra", "inter"}:
            raise ValueError("mix_mode must be both, intra, or inter")


class AHS(Selector):
    """One instance per episode. Append only actually executed targets to history."""
    def __init__(self, config: AHSConfig | None = None, seed: int = 0):
        self.config = config or AHSConfig()
        self.reset(seed)

    def reset(self, seed: int = 0):
        super().__init__(self.config.candidates, seed)
        self.fft_horizon = self.config.horizon
        self.horizon_expected_temp = self.config.temperature
        self.horizon_ts_epsilon = self.config.epsilon
        self.horizon_ts_kernel_bandwidth = self.config.kernel_bandwidth
        self.horizon_ts_forget_rho = self.config.forget_rho
        self.horizon_ts_update_eta = self.config.update_eta
        self.horizon_inter_window = self.config.history_window
        self.horizon_mix_mode = self.config.mix_mode
        self._executed_action_history = np.zeros((0, self.config.action_dim), np.float32)

    def evidence(self, velocity, actions, history=None):
        """Score candidate prefixes without drawing randomness or updating memory.

        velocity: [denoising_steps, available_actions, action_dim]. Fourier
        transforms run along available_actions, zero-padded to the full H.
        actions/history: predicted / actually executed targets in the same space.
        """
        velocity, actions = np.asarray(velocity), np.asarray(actions)
        if velocity.ndim != 3 or actions.ndim != 2:
            raise ValueError("expected velocity[T,H,D] and actions[H,D]")
        if velocity.shape[0] < 1 or velocity.shape[1:] != actions.shape:
            raise ValueError("velocity and action shapes do not align")
        if actions.shape[1] != self.config.action_dim or not max(self.config.candidates) <= len(actions) <= self.fft_horizon:
            raise ValueError("available actions cannot support the configured candidates/dimensions")
        if history is None:
            history = np.zeros((0, self.config.action_dim), np.float32)
        history = np.asarray(history, dtype=np.float32)
        if history.ndim != 2 or history.shape[1] != self.config.action_dim:
            raise ValueError("history must have shape [executed_actions, action_dim]")
        if any(not np.isfinite(x).all() for x in (velocity, actions, history)):
            raise ValueError("inputs must be finite")
        self._executed_action_history = history[-self.config.history_window:]
        info = self._compute_horizon_info(velocity, self.fft_horizon, xt_step=actions)
        info.update(fft_horizon=self.fft_horizon, available_horizon=len(actions))
        return info

    def select(self, velocity, actions, history=None):
        return self._select_exec_k_horizon_from_info(self.evidence(velocity, actions, history), len(actions))

    def state_dict(self):
        return {"version": 1, "config": asdict(self.config),
                "posterior": copy.deepcopy(self._horizon_ts_state),
                "rng": copy.deepcopy(self._horizon_ts_rng.bit_generator.state)}

    def load_state_dict(self, state):
        if state.get("version") != 1 or AHSConfig(**state["config"]) != self.config:
            raise ValueError("state configuration differs from this selector")
        posterior = {int(k): list(v) for k, v in state["posterior"].items()}
        if set(posterior) != set(self.config.candidates) or any(len(v) != 2 or not np.isfinite(v).all() or min(v) <= 0 for v in posterior.values()):
            raise ValueError("invalid posterior")
        rng = np.random.default_rng()
        rng.bit_generator.state = copy.deepcopy(state["rng"])
        self._horizon_ts_state, self._horizon_ts_rng = posterior, rng
