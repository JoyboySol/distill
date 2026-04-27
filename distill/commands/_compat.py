import sys
from typing import Callable, Sequence


def run_compat_main(command_name: str,
                    target_main: Callable[[Sequence[str] | None], object]):
    print(
        ("[deprecated] Direct script entrypoints will be removed in a future "
         f"cleanup. Use `python -m distill {command_name} ...` instead."),
        file=sys.stderr,
    )
    return target_main(sys.argv[1:])
