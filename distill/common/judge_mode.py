from typing import Any, List, Tuple


DEFAULT_JUDGE_FAMILY_ORDER: Tuple[str, ...] = (
    "code",
    "instruction_following",
    "mcq",
    "math",
)


def _split_mode_items(raw: Any) -> List[str]:
    if raw is None:
        return []
    if isinstance(raw, (list, tuple, set)):
        items: List[str] = []
        for item in raw:
            items.extend(_split_mode_items(item))
        return items
    text = str(raw)
    return [
        item.strip().lower() for item in text.replace(",", "\n").splitlines()
        if item.strip()
    ]


def normalize_judge_mode(raw: Any) -> str:
    parts = _split_mode_items(raw)
    if not parts:
        return "auto"

    if len(parts) == 1 and parts[0] in {"auto", "none"}:
        return parts[0]

    if "auto" in parts:
        raise ValueError(
            "judge_mode 'auto' cannot be combined with explicit judge families."
        )
    if "none" in parts:
        raise ValueError(
            "judge_mode 'none' cannot be combined with explicit judge families."
        )

    normalized: List[str] = []
    seen = set()
    for part in parts:
        if part not in DEFAULT_JUDGE_FAMILY_ORDER:
            raise ValueError(
                f"Unsupported judge_mode entry '{part}'. "
                f"Expected one of auto, none, {', '.join(DEFAULT_JUDGE_FAMILY_ORDER)}."
            )
        if part in seen:
            continue
        normalized.append(part)
        seen.add(part)
    return ",".join(normalized)


def resolve_judge_family_order(raw: Any) -> Tuple[str, ...]:
    normalized = normalize_judge_mode(raw)
    if normalized == "none":
        return ()
    if normalized == "auto":
        return DEFAULT_JUDGE_FAMILY_ORDER
    return tuple(item for item in normalized.split(",") if item)


def judge_mode_disables_all(raw: Any) -> bool:
    return normalize_judge_mode(raw) == "none"
