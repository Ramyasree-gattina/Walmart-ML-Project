import subprocess
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
PIPELINES = (
    "run_lab3_baseline.py",
    "run_lab4_tracking.py",
    "run_lab5_pipeline.py",
    "run_lab6_registry.py",
)


def main():
    for pipeline in PIPELINES:
        print(f"\n[INFO] Running pipelines/{pipeline}...")
        subprocess.run([sys.executable, str(ROOT / "pipelines" / pipeline)], cwd=ROOT, check=True)
    print("\n[SUCCESS] All project stages completed!")


if __name__ == "__main__":
    main()
