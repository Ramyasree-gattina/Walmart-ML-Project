import subprocess
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]


def main():
    print("[INFO] Starting Lab 4: MLflow Experiment Tracking")
    for script in ("preprocess.py", "train_mlflow.py", "validate_reproducibility.py"):
        print(f"\n[INFO] ---> Executing src/{script}...")
        subprocess.run([sys.executable, str(ROOT / "src" / script)], cwd=ROOT, check=True)
    print("\n[SUCCESS] Lab 4 pipeline fully executed!")


if __name__ == "__main__":
    main()
