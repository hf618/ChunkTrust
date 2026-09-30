"""Persist a test result tied to the actual execution sources."""
import time
import unittest
from common import EXP, PROTOCOL_ID, write_json
from run_block import execution_fingerprint

if __name__ == '__main__':
    suite=unittest.defaultTestLoader.discover(str(EXP/'tests'))
    result=unittest.TextTestRunner(verbosity=2).run(suite)
    write_json(EXP/'artifacts/unit_test_report.json',dict(
        protocol_id=PROTOCOL_ID,execution_hash=execution_fingerprint(),checked_at=time.time(),
        status='PASS' if result.wasSuccessful() else 'FAIL',tests=result.testsRun,
        errors=len(result.errors),failures=len(result.failures)))
    if not result.wasSuccessful():raise SystemExit(1)
