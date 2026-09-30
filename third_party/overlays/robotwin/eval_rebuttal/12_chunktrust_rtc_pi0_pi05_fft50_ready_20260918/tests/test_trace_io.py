import sys
import tempfile
import unittest
from pathlib import Path

import ml_dtypes
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'runtime'))
from trace_io import load_trace


class TraceSerialization(unittest.TestCase):
    def test_actual_bfloat16_archive_roundtrip(self):
        original = np.asarray([0., -1., 1.00390625, .001, 1024.], dtype=ml_dtypes.bfloat16)
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / 'trace.npz'
            np.savez_compressed(path, v=original, actions=np.ones((2, 14), np.float32), expired=2)
            with np.load(path) as archive:
                self.assertEqual(archive['v'].dtype, np.dtype('V2'))
            decoded = load_trace(path)
        np.testing.assert_array_equal(decoded['v'], original.astype(np.float32))
        self.assertEqual(decoded['actions'].dtype, np.float32)
        self.assertEqual(decoded['expired'], 2)

    def test_unknown_opaque_field_is_rejected(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / 'trace.npz'
            np.savez_compressed(path, actions=np.zeros((2,), dtype='V2'))
            with self.assertRaises(ValueError):
                load_trace(path)


if __name__ == '__main__':
    unittest.main()
