"""
数据审计脚本 (D27-02)
对原始数据和三版清洗结果做自动化审计。
"""
import os, sys, json, hashlib
import pandas as pd
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from config import DATA_PATH, ROOT_DIR

def sha256(path: str) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(8192), b""):
            h.update(chunk)
    return h.hexdigest()

def audit_file(filepath: str, label: str) -> dict:
    df = pd.read_csv(filepath)
    return {
        "label": label,
        "path": filepath,
        "sha256": sha256(filepath),
        "rows": len(df),
        "cols": len(df.columns),
        "missing_cells": int(df.isnull().sum().sum()),
        "duplicate_rows": int(df.duplicated().sum()),
        "person_id_missing": int(df["Person_ID"].isna().sum()) if "Person_ID" in df.columns else -1,
        "person_id_duplicates": int(df["Person_ID"].duplicated().sum()) if "Person_ID" in df.columns else -1,
        "columns": list(df.columns),
    }

if __name__ == "__main__":
    print("=" * 50)
    print("Data Audit Report")
    print("=" * 50)

    results = []

    # 1. Original data
    if os.path.exists(DATA_PATH):
        r = audit_file(DATA_PATH, "原始数据 (A题数据集.csv)")
        results.append(r)
        print(f"\n  [原始数据] {r['rows']}×{r['cols']}, 缺失={r['missing_cells']}")

    # 2. Check for legacy cleaning outputs
    legacy_dir = os.path.join(ROOT_DIR, "data", "legacy")
    if os.path.exists(legacy_dir):
        for f in os.listdir(legacy_dir):
            if f.endswith(".csv"):
                r = audit_file(os.path.join(legacy_dir, f), f"遗留: {f}")
                results.append(r)
                print(f"  [遗留] {f}: {r['rows']}×{r['cols']}")

    # 3. Current base_semantic_clean.csv
    clean_path = os.path.join(ROOT_DIR, "data", "processed", "base_semantic_clean.csv")
    if os.path.exists(clean_path):
        r = audit_file(clean_path, "当前公共清洗版")
        results.append(r)
        print(f"\n  [公共清洗] {r['rows']}×{r['cols']}, 缺失={r['missing_cells']}")
        # Validate
        assert r["rows"] == 10000, f"Expected 10000 rows, got {r['rows']}"
        assert r["missing_cells"] == 0, f"Expected 0 missing, got {r['missing_cells']}"
        assert r["person_id_duplicates"] == 0
        print("  Validation: PASSED ✓")

    # Save
    output_path = os.path.join(ROOT_DIR, "docs", "data_inventory.csv")
    pd.DataFrame(results).to_csv(output_path, index=False)
    print(f"\n  Saved: {output_path}")