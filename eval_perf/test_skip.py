import sys
from pathlib import Path

SCRIPT_DIR = Path(__file__).resolve().parent
SCRIPT_DIR_STR = str(SCRIPT_DIR)
if SCRIPT_DIR_STR not in sys.path:
    sys.path.insert(0, SCRIPT_DIR_STR)

from common import run


if __name__ == "__main__":
    run("skip")
