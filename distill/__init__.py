from .runtime.settings import PipelineConfig, resolve_base_urls

__all__ = [
    "DistillPipeline",
    "PipelineConfig",
    "RoundRobinPipeline",
    "resolve_base_urls",
]


def __getattr__(name):
    if name in {"DistillPipeline", "RoundRobinPipeline"}:
        from .core.pipeline import DistillPipeline, RoundRobinPipeline

        exports = {
            "DistillPipeline": DistillPipeline,
            "RoundRobinPipeline": RoundRobinPipeline,
        }
        return exports[name]
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
