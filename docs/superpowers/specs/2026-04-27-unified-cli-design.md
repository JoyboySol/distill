# Unified CLI Design

**Context**

The project already has a primary pipeline entrypoint in `distill/cli.py` and
`python -m distill`, but many common maintenance tools still live as standalone
`scripts/*.py` entrypoints. This splits discoverability, duplicates CLI
patterns, and makes it harder to evolve command UX consistently.

## Goals

- Provide one unified CLI surface for the pipeline and common Python tools.
- Make `python -m distill <subcommand> ...` the primary interface.
- Preserve backward compatibility for existing direct script calls during the
  migration window.
- Register an installable `distill` console command for editable/package-based
  workflows.

## Decisions

- Introduce a root CLI dispatcher in `distill/cli.py`.
- Keep the pipeline run behavior as a dedicated `run` subcommand.
- Preserve legacy pipeline invocation compatibility so existing
  `python -m distill --config ...` calls still execute the pipeline.
- Move command implementations into package modules under `distill/commands/`.
- Convert old `scripts/*.py` files into thin compatibility shims that forward
  to the packaged command entrypoints and emit a deprecation warning.
- Add a minimal `pyproject.toml` with a console-script entrypoint:
  `distill = distill.cli:main`.

## Command Layout

- `distill run`
- `distill list-configs`
- `distill stats`
- `distill avg-correct-tokens`
- `distill manifest-token-stats`
- `distill merge-correct-segments`
- `distill rebuild-correct-segments`
- `distill rejudge-saved-outputs`
- `distill eval-existing-outputs`
- `distill prepare-opencode-reasoning`
- `distill hydrate-opencode-reasoning-split1`
- `distill rewrite-completed-index-paths`

## Constraints

- Existing `distill.cli` imports used by tests should remain compatible.
- The pipeline parser logic and config resolution behavior should not regress.
- The migration should avoid changing command semantics beyond normalizing the
  entrypoint surface.
