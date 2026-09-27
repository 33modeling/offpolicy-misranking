from pathlib import Path
import tempfile
import unittest

from srgc_rebuttal.cost_ledger import PhaseLedger


class CostLedgerTests(unittest.TestCase):
    def test_completed_retries_count_once_and_open_timer_is_not_zero(self):
        with tempfile.TemporaryDirectory() as folder:
            ledger = PhaseLedger(Path(folder))
            first = {"id": "first", "phase": "selection", "checkpoint": 25, "gpu_count": 4, "state": "started"}
            ledger.record(first)
            self.assertFalse(ledger.totals()["complete"])
            ledger.record({**first, "state": "finished", "gpu_seconds": 20.0})
            # A restart reruns this checkpoint: both actual charges remain.
            second = {**first, "id": "retry"}
            ledger.record(second)
            ledger.record({**second, "state": "finished", "gpu_seconds": 23.0})
            self.assertEqual(ledger.totals()["known_gpu_seconds"]["selection_gpu_seconds"], 43.0)
            self.assertEqual(len(list(Path(folder).glob("*.json"))), 2)
            ledger.record({**first, "id": "interrupted", "phase": "training"})
            report = ledger.totals()
            self.assertFalse(report["complete"])
            self.assertEqual(report["unfinished_phases"][0]["phase"], "training")

    def test_negative_time_and_double_start_rejected(self):
        with tempfile.TemporaryDirectory() as folder:
            ledger = PhaseLedger(Path(folder))
            event = {"id": "x", "phase": "training", "checkpoint": 26, "gpu_count": 4, "state": "started"}
            ledger.record(event)
            with self.assertRaises(ValueError):
                ledger.record(event)
            with self.assertRaises(ValueError):
                ledger.record({**event, "state": "finished", "gpu_seconds": -1.0})
