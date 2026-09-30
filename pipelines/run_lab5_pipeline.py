import subprocess
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]


def main():
    print("[INFO] Starting Lab 5: Production Data Pipeline")
    for script in ("validate_data.py", "preprocess_pipeline.py", "validate_outputs.py"):
        print(f"\n[INFO] ---> Executing src/{script}...")
        subprocess.run([sys.executable, str(ROOT / "src" / script)], cwd=ROOT, check=True)
    print("\n[SUCCESS] Lab 5 production pipeline fully executed!")


if __name__ == "__main__":
    main()
