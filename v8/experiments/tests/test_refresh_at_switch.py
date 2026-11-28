"""V9 X16: bank_srrefresh_t* rebuilds the SR cache once at the switch; earlier method hashes are unchanged."""

import unittest

from helpers import HAS_CODE  # noqa: F401


class RefreshAtSwitch(unittest.TestCase):
    def test_entry_once(self):
        from switching.cache import refresh_due
        due = [t for t in range(40, 300) if refresh_due(t, entry_update=50, interval=None, origin="entry_once")]
        self.assertEqual(due, [50])
        self.assertTrue(refresh_due(75, entry_update=25, interval=50))   # existing origin unchanged
        self.assertFalse(refresh_due(50, entry_update=25, interval=50))

    def test_method_and_hashes(self):
        from switching.methods import method, method_hash
        m = method("bank_srrefresh_t050")
        self.assertEqual((m["initial_selector"], m["family"], m["branch_update"]), ("sr", "fixed_srrefresh", 50))
        self.assertEqual(m["cache"]["refresh_origin"], "entry_once")
        self.assertNotEqual(method_hash(m), method_hash(method("bank_sr_t050")))
        from switching.common import sha256_json
        from switching.methods import EXECUTION_FIELDS
        for mid in ("bank_sr_t050", "on_continued", "sr_early_refreshed_k025", "sr_hold025", "dynsample_early"):
            mm = method(mid)
            self.assertNotEqual(mm["cache"].get("refresh_origin"), "entry_once")
        base = method("bank_sr_t050")
        self.assertEqual(method_hash(base), sha256_json({k: base.get(k) for k in EXECUTION_FIELDS}))


if __name__ == "__main__":
    unittest.main()
