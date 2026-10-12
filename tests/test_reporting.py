"""Narrow regression tests for idempotent report edits and document checks."""
import sys
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from fill_thesis_from_report import CAPTIONS, replace_tables
from eval_ista import differences


class Reporting(unittest.TestCase):
    def test_table_updates_preserve_discussion_and_are_idempotent(self):
        text = "\n\n".join(caption + "\n\n| old | value |\n|---|---|\n| a | 0 |\n\n"
                             + "Reviewed discussion that must remain."
                             for caption in CAPTIONS.values())
        values = {key: "| a | b |\n|---|---|\n| c | 1 |" for key in CAPTIONS}
        updated = replace_tables(text, values)
        self.assertEqual(updated.count("Reviewed discussion that must remain."), 5)
        self.assertEqual(replace_tables(updated, values), updated)
        self.assertEqual(updated.count("| c | 1 |"), 5)

    def test_metric_comparison_includes_nested_patients(self):
        a = {"psnr": 2.0, "calibers": {"per_case": {"a": 1.0}}, "n": 3}
        b = {"psnr": 2.0, "calibers": {"per_case": {"a": 1.5}}, "n": 3}
        self.assertEqual(dict(differences(a, b))["calibers.per_case.a"], 0.5)


if __name__ == "__main__":
    unittest.main()
