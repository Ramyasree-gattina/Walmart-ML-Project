import subprocess
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]


def main():
    print("[INFO] Starting Lab 3: Baseline ML Pipeline")
    for script in ("preprocess.py", "train.py", "evaluate.py"):
        print(f"\n[INFO] ---> Executing src/{script}...")
        subprocess.run([sys.executable, str(ROOT / "src" / script)], cwd=ROOT, check=True)
    print("\n[SUCCESS] Lab 3 pipeline fully executed!")


if __name__ == "__main__":
    main()
