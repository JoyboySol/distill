import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from distill.commands.rewrite_completed_index_paths import *  # noqa: F401,F403
from distill.commands._compat import run_compat_main
from distill.commands.rewrite_completed_index_paths import main


if __name__ == "__main__":
    run_compat_main("rewrite-completed-index-paths", main)
