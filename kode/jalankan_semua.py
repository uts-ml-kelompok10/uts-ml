"""Run all three official-data experiments with the current Python interpreter."""
from pathlib import Path
import subprocess
import sys


if __name__ == "__main__":
    folder = Path(__file__).resolve().parent
    for task in ("regresi.py", "klasifikasi.py", "clustering.py"):
        subprocess.run([sys.executable, str(folder / task), *sys.argv[1:]], check=True)
