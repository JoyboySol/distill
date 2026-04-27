import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from distill.commands import manifest_token_stats as _impl
from distill.commands.manifest_token_stats import *  # noqa: F401,F403
from distill.commands._compat import run_compat_main
from distill.commands.manifest_token_stats import main

_build_task_row = _impl._build_task_row


def summarize_manifest_token_stats(*args, **kwargs):
    original = _impl._build_task_row
    _impl._build_task_row = _build_task_row
    try:
        return _impl.summarize_manifest_token_stats(*args, **kwargs)
    finally:
        _impl._build_task_row = original


if __name__ == "__main__":
    run_compat_main("manifest-token-stats", main)
