import subprocess
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]


def main():
    print("[INFO] Starting Lab 6: Model Registry and Lifecycle")
    subprocess.run(
        [sys.executable, str(ROOT / "src" / "register_model.py")], cwd=ROOT, check=True
    )
    print("[SUCCESS] Lab 6 model registered and quality-gated!")


if __name__ == "__main__":
    main()
