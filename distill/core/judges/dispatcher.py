from typing import Any, Dict, List, Optional

from ...common.judge_mode import resolve_judge_family_order
from .code import code_judge_type_hint, judge_code
from .instruction_following import instruction_following_hint, \
    judge_instruction_following
from .math import judge_math, try_math_reference
from .mcq import judge_mcq, try_mcq_boxed_reference
from .shared import assistant_text


def judge_output_impl(row_data: Dict[str, Any],
                      messages: List[Dict[str, Any]],
                      label_field: Optional[str] = None,
                      judge_mode: Optional[str] = None) -> Dict[str, Any]:
    family_order = resolve_judge_family_order(judge_mode)
    content = assistant_text(messages)

    for family in family_order:
        if family == "code":
            code_result = judge_code(row_data, messages)
            if code_result is not None:
                return code_result
            continue

        if family == "instruction_following":
            instruction_result = judge_instruction_following(row_data, messages)
            if instruction_result is not None:
                return instruction_result
            continue

        if family == "mcq":
            mcq_result = judge_mcq(row_data, content)
            if mcq_result is not None:
                return mcq_result
            continue

        if family == "math":
            math_result = judge_math(row_data,
                                     content,
                                     label_field=label_field)
            if math_result is not None:
                return math_result
            continue

    return {
        "judge_type": None,
        "judge_backend": None,
        "is_correct": None,
        "judge_status": "not_applicable",
        "judge_detail": None,
    }


def judge_type_hint(row_data: Dict[str, Any],
                    messages: List[Dict[str, Any]],
                    label_field: Optional[str] = None,
                    judge_mode: Optional[str] = None) -> Optional[str]:
    family_order = resolve_judge_family_order(judge_mode)
    for family in family_order:
        if family == "code":
            hint = code_judge_type_hint(row_data, messages)
            if hint is not None:
                return hint
            continue

        if family == "instruction_following":
            hint = instruction_following_hint(row_data)
            if hint is not None:
                return hint
            continue

        if family == "mcq":
            if try_mcq_boxed_reference(row_data) is not None:
                return "mcq_boxed"
            continue

        if family == "math":
            if try_math_reference(row_data,
                                  preferred_field=label_field) is not None:
                return "math"
            continue

    return None
