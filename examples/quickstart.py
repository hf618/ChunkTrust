"""Synthetic interface example. This is not an experimental result."""
import numpy as np
from chunktrust import AHS, AHSConfig, HybridQHARuntimeSelector

rng = np.random.default_rng(7)
velocity = rng.normal(size=(10, 50, 14)).astype(np.float32)
actions = rng.normal(size=(50, 14)).astype(np.float32)
history = rng.normal(size=(60, 14)).astype(np.float32)
selector = AHS(seed=42)
k, evidence = selector.select(velocity, actions, history)
print(f"AHS selected {k} actions from a 50-action prediction")
print("Intra-chunk:", evidence["q_intra"])
print("Inter-chunk:", evidence["p_inter"])

# In deployment this distribution comes from the trained QHA head, using
# exactly the candidate basis expected by that checkpoint and configuration.
prior = np.full(5, 0.2)
hybrid = HybridQHARuntimeSelector(candidate_horizons=selector.config.candidates, ts_seed=42)
result = hybrid.select(prior, evidence["q_mix"], max_exec_length=50)
print(f"Illustrative uniform-prior fusion selected {result['exec_length']} actions")
