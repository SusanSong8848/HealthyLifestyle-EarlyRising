import os
import sys
import tempfile
import unittest

import pandas as pd

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(ROOT, "src"))

from clean_utils import load_manifest_split_indices


class SharedSplitUsageTests(unittest.TestCase):
    def _write_manifest(self, rows):
        handle = tempfile.NamedTemporaryFile(
            mode="w",
            suffix=".csv",
            delete=False,
            newline="",
            encoding="utf-8",
        )
        handle.close()
        pd.DataFrame(rows).to_csv(handle.name, index=False)
        self.addCleanup(
            lambda: os.path.exists(handle.name) and os.unlink(handle.name)
        )
        return handle.name

    def test_aligns_manifest_by_person_id_not_row_order(self):
        data = pd.DataFrame({"Person_ID": ["P3", "P1", "P4", "P2"]})
        manifest_path = self._write_manifest(
            [
                {"Person_ID": "P1", "split": "train"},
                {"Person_ID": "P2", "split": "val"},
                {"Person_ID": "P3", "split": "train"},
                {"Person_ID": "P4", "split": "val"},
            ]
        )

        train_idx, val_idx = load_manifest_split_indices(data, manifest_path)

        self.assertEqual(train_idx.tolist(), [0, 1])
        self.assertEqual(val_idx.tolist(), [2, 3])

    def test_rejects_different_person_id_sets(self):
        data = pd.DataFrame({"Person_ID": ["P1", "P2"]})
        manifest_path = self._write_manifest(
            [
                {"Person_ID": "P1", "split": "train"},
                {"Person_ID": "P3", "split": "val"},
            ]
        )

        with self.assertRaisesRegex(ValueError, "different Person_ID sets"):
            load_manifest_split_indices(data, manifest_path)

    def test_rejects_duplicate_manifest_ids(self):
        data = pd.DataFrame({"Person_ID": ["P1", "P2"]})
        manifest_path = self._write_manifest(
            [
                {"Person_ID": "P1", "split": "train"},
                {"Person_ID": "P1", "split": "val"},
            ]
        )

        with self.assertRaisesRegex(ValueError, "Duplicate Person_ID"):
            load_manifest_split_indices(data, manifest_path)

    def test_rejects_stale_manifest_labels(self):
        data = pd.DataFrame(
            {
                "Person_ID": ["P1", "P2"],
                "Early_Waker": ["Yes", "No"],
            }
        )
        manifest_path = self._write_manifest(
            [
                {
                    "Person_ID": "P1",
                    "split": "train",
                    "task1_label": "No",
                },
                {
                    "Person_ID": "P2",
                    "split": "val",
                    "task1_label": "No",
                },
            ]
        )

        with self.assertRaisesRegex(ValueError, "does not match"):
            load_manifest_split_indices(
                data,
                manifest_path,
                label_checks={"task1_label": data["Early_Waker"]},
            )


if __name__ == "__main__":
    unittest.main()
