"""Read this experiment's NumPy traces without losing BF16 values.

NumPy's .npy format inside .npz stores ml_dtypes.bfloat16 as opaque V2.
Only the three known model trace fields may use that representation. The
recording host and reading host must have the same endianness for legacy V2.
This decoder is offline only; it does not change inference or saved traces.
"""
from pathlib import Path

import ml_dtypes
import numpy as np

BF16_FIELDS = frozenset(('v', 'guided_update', 'correction_rms'))


def load_trace(path: str | Path) -> dict[str, np.ndarray]:
    arrays = {}
    with np.load(path, allow_pickle=False) as archive:
        for name in archive.files:
            value = archive[name]
            if value.dtype.kind == 'V':
                if name not in BF16_FIELDS or value.dtype != np.dtype('V2'):
                    raise ValueError(f'unsupported opaque trace dtype: {name} {value.dtype}')
                value = value.view(ml_dtypes.bfloat16).astype(np.float32)
            arrays[name] = value
    return arrays
