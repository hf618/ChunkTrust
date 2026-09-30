from __future__ import annotations

from openpi.training.data_loader import TaskBalancedCoverageBatchSampler


def test_task_balanced_coverage_visits_every_sample_before_wrapping() -> None:
    lengths = [5, 9, 13]
    sampler = TaskBalancedCoverageBatchSampler(task_lengths=lengths, batch_size=6, seed=7)
    batches = list(sampler)
    assert len(batches) == 7

    offsets = [0, 5, 14]
    for task_index, (offset, length) in enumerate(zip(offsets, lengths, strict=True)):
        observed = {idx - offset for batch in batches for idx in batch if offset <= idx < offset + length}
        assert observed == set(range(length)), task_index


def test_task_balanced_coverage_changes_order_each_epoch() -> None:
    sampler = TaskBalancedCoverageBatchSampler(task_lengths=[10, 10], batch_size=4, seed=11)
    assert list(sampler) != list(sampler)
