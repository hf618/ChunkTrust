"""Inspect a stored evidence trace without importing a policy or simulator."""
import argparse
import json
from pathlib import Path
import numpy as np
from .selector import AHS, AHSConfig


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("trace", type=Path, help="NPZ with velocity[T,H,D], actions[H,D], optional history[N,D]")
    parser.add_argument("--candidates", default="10,20,30,40,50")
    parser.add_argument("--seed", type=int, default=0)
    args = parser.parse_args()
    with np.load(args.trace, allow_pickle=False) as data:
        velocity, actions = data["velocity"], data["actions"]
        history = data["history"] if "history" in data else None
        config = AHSConfig(horizon=actions.shape[0], action_dim=actions.shape[1], candidates=tuple(map(int, args.candidates.split(','))))
        k, info = AHS(config, seed=args.seed).select(velocity, actions, history)
    print(json.dumps({"selected_horizon": k, "q_intra": info["q_intra"].tolist(),
                      "p_inter": [float(v) if np.isfinite(v) else None for v in info["p_inter"]],
                      "q_mix": info["q_mix"].tolist()}, indent=2, allow_nan=False))
