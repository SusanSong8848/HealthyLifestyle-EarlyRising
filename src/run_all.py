"""
一键复现脚本 (D27 + D29)
按正确顺序运行所有任务，从原始数据到最终预测。
"""
import os, sys, subprocess, time

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
PYTHON = r"D:\python\python.exe"

SCRIPTS = [
    ("preprocess.py", "数据预处理"),
    ("task1_final.py", "任务1 — Early Waker 二分类"),
    ("task2_health_score.py", "任务2 — Health Score 四分类"),
    ("task3_wellness_category.py", "任务3 — Wellness Category 四分类"),
]

def run_script(script, description):
    path = os.path.join(ROOT, "src", script)
    print(f"\n{'='*60}")
    print(f"  Running: {script} — {description}")
    print(f"{'='*60}")
    t0 = time.time()
    result = subprocess.run([PYTHON, path], cwd=ROOT,
                           capture_output=False, text=True)
    elapsed = time.time() - t0
    if result.returncode != 0:
        print(f"  ERROR: {script} failed with exit code {result.returncode}")
        return False
    print(f"  COMPLETED in {elapsed:.1f}s")
    return True

if __name__ == "__main__":
    t_start = time.time()
    print("=" * 60)
    print("  RUN ALL — 一键复现")
    print(f"  Python: {PYTHON}")
    print("=" * 60)

    for script, desc in SCRIPTS:
        if not run_script(script, desc):
            print(f"\n  Aborted at {script}")
            sys.exit(1)

    print(f"\n{'='*60}")
    print(f"  ALL SCRIPTS COMPLETED in {time.time()-t_start:.0f}s")
    print(f"  outputs/ → task1/, task2/, task3/")
    print(f"{'='*60}")