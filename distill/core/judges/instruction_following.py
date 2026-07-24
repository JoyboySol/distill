import ast
import os
import json
import re
from typing import Any, Dict, List, Optional, Tuple

from .shared import assistant_text


SUPPORTED_BACKEND = "instruction_following_v1"
YULAN_IFEVAL_BACKEND = "lm_eval_ifeval"
YULAN_IFEVAL_SUITES = {"ifeval", "ifeval_extracted"}
NUMBER_WORDS = {
    "one": 1,
    "two": 2,
    "three": 3,
    "four": 4,
    "five": 5,
    "six": 6,
    "seven": 7,
    "eight": 8,
    "nine": 9,
    "ten": 10,
}
FREQUENCY_WORDS = {
    "once": 1,
    "twice": 2,
    "thrice": 3,
}
ORDINAL_WORDS = {
    "first": 1,
    "second": 2,
    "third": 3,
    "fourth": 4,
    "fifth": 5,
    "sixth": 6,
    "seventh": 7,
    "eighth": 8,
    "ninth": 9,
    "tenth": 10,
    "final": -1,
    "last": -1,
}


def _split_words(text: str) -> List[str]:
    return re.findall(r"[^\W_]+", text, flags=re.UNICODE)


def _normalized_last_word(text: str) -> Optional[str]:
    words = _split_words(text)
    if not words:
        return None
    return words[-1].lower()


def _normalized_first_word(text: str) -> Optional[str]:
    words = _split_words(text)
    if not words:
        return None
    return words[0].lower()


def _paragraphs(text: str) -> List[str]:
    return [part.strip() for part in re.split(r"\n\s*\n", text) if part.strip()]


def _count_placeholders(text: str) -> int:
    return len(re.findall(r"\[[^\[\]\n]+\]", text))


def _count_uppercase_words(text: str) -> int:
    return sum(1 for word in _split_words(text) if len(word) > 1 and word.isupper())


def _extract_quoted_phrases(text: str) -> List[str]:
    matches = re.findall(r'"([^"\n]+)"|(?<!\w)\'([^\'\n]+)\'(?!\w)', text)
    return [match[0] or match[1] for match in matches]


def _extract_excluded_words(prompt: str) -> List[str]:
    match = re.search(
        r'excluding (?:the )?words? (.+?)(?:[.](?:\s|$)|$)',
        prompt,
        re.IGNORECASE,
    )
    if not match:
        return []
    return _extract_quoted_phrases(match.group(1))


def _normalize_prompt_keywords(values: List[str]) -> List[str]:
    normalized = []
    for value in values:
        cleaned = value.strip().strip(".,!?;:")
        if cleaned:
            normalized.append(cleaned)
    return normalized


def _extract_included_keywords(prompt: str) -> List[str]:
    match = re.search(
        r'keywords?\s*(?::|such as|like)?\s*(.+?)(?:[.](?:\s|$)|$)',
        prompt,
        re.IGNORECASE,
    )
    if not match:
        return []
    return _extract_quoted_phrases(match.group(1))


def _extract_specific_ending(prompt: str) -> Optional[str]:
    patterns = [
        r'end (?:your )?response with "([^"]+)"',
        r'end (?:your )?response with the phrase [\'"]([^\'"]+)[\'"]',
        r'end with the phrase [\'"]([^\'"]+)[\'"]',
        r'end(?: [^"\n]*)? with the exact (?:sentence|line):\s*"([^"]+)"',
        r'conclude(?: [^"\n]*)? with the exact (?:sentence|line):\s*"([^"]+)"',
        r'the last sentence [^"\n]* must be:\s*"([^"]+)"',
        r'ensure [^"\n]* ends with the sentence:\s*"([^"]+)"',
        r'final paragraph concludes with the word [\'"]([^\'"]+)[\'"]',
        r'end your (?:article|paragraph|response) with the word [\'"]([^\'"]+)[\'"]',
    ]
    for pattern in patterns:
        match = re.search(pattern, prompt, re.IGNORECASE)
        if match:
            return match.group(1)
    return None


def _extract_number_token(prompt: str, pattern: str) -> Optional[int]:
    match = re.search(pattern, prompt, re.IGNORECASE)
    if not match:
        return None
    token = match.group(1).lower()
    if token.isdigit():
        return int(token)
    return NUMBER_WORDS.get(token)


def _parse_count_token(raw_token: str) -> Optional[int]:
    token = raw_token.lower()
    if token.isdigit():
        return int(token)
    return NUMBER_WORDS.get(token, FREQUENCY_WORDS.get(token))


def _extract_numbered_requirement(prompt: str, unit: str) -> Optional[int]:
    patterns = [
        rf'exactly (\d+) {unit}s?',
        rf'(\d+)[-\s]{unit}s? long',
        rf'(\d+)[-\s]{unit}',
        rf'in (\d+) {unit}s?',
    ]
    for pattern in patterns:
        match = re.search(pattern, prompt, re.IGNORECASE)
        if match:
            return int(match.group(1))
    word_patterns = [
        rf'exactly ({("|".join(NUMBER_WORDS))}) {unit}s?',
        rf'({("|".join(NUMBER_WORDS))})[-\s]{unit}s? long',
        rf'({("|".join(NUMBER_WORDS))})[-\s]{unit}',
        rf'in ({("|".join(NUMBER_WORDS))}) {unit}s?',
    ]
    for pattern in word_patterns:
        match = re.search(pattern, prompt, re.IGNORECASE)
        if match:
            return NUMBER_WORDS[match.group(1).lower()]
    return None


def _extract_word_requirement(prompt: str) -> Optional[Dict[str, Any]]:
    patterns = [
        (r'exactly (\d+) words?', "exactly"),
        (rf'exactly ({"|".join(NUMBER_WORDS)}) words?', "exactly"),
        (r'between (\d+) and (\d+) words?', "between"),
        (rf'between ({"|".join(NUMBER_WORDS)}) and ({"|".join(NUMBER_WORDS)}) words?', "between"),
        (r'no more than (\d+) words?', "at_most"),
        (rf'no more than ({"|".join(NUMBER_WORDS)}) words?', "at_most"),
        (r'not exceeding (\d+) words?', "at_most"),
        (rf'not exceeding ({"|".join(NUMBER_WORDS)}) words?', "at_most"),
        (r'not to exceed (\d+) words?', "at_most"),
        (rf'not to exceed ({"|".join(NUMBER_WORDS)}) words?', "at_most"),
        (r'not exceed (\d+) words?', "at_most"),
        (rf'not exceed ({"|".join(NUMBER_WORDS)}) words?', "at_most"),
        (r'not more than (\d+) words?', "at_most"),
        (rf'not more than ({"|".join(NUMBER_WORDS)}) words?', "at_most"),
        (r'no longer than (\d+) words?', "at_most"),
        (rf'no longer than ({"|".join(NUMBER_WORDS)}) words?', "at_most"),
        (r'maximum of (\d+) words?', "at_most"),
        (rf'maximum of ({"|".join(NUMBER_WORDS)}) words?', "at_most"),
        (r'under (\d+) words?', "less_than"),
        (rf'under ({"|".join(NUMBER_WORDS)}) words?', "less_than"),
        (r'less than (\d+) words?', "less_than"),
        (rf'less than ({"|".join(NUMBER_WORDS)}) words?', "less_than"),
        (r'(\d+)-word\b', "exactly"),
        (rf'({"|".join(NUMBER_WORDS)})-word\b', "exactly"),
    ]
    for pattern, relation in patterns:
        match = re.search(pattern, prompt, re.IGNORECASE)
        if not match:
            continue
        if relation == "between":
            lower = _parse_count_token(match.group(1))
            upper = _parse_count_token(match.group(2))
            if lower is not None and upper is not None:
                return {
                    "value": lower,
                    "upper_value": upper,
                    "relation": relation,
                }
            continue
        value = _parse_count_token(match.group(1))
        if value is not None:
            return {"value": value, "relation": relation}
    return None


def _extract_placeholder_requirement(prompt: str) -> Optional[int]:
    patterns = [
        r'at least (\d+) placeholders?',
        rf'at least ({"|".join(NUMBER_WORDS)}) placeholders?',
        r'provide (\d+) placeholders?',
        rf'provide ({"|".join(NUMBER_WORDS)}) placeholders?',
        r'include (\d+) placeholders?',
        rf'include ({"|".join(NUMBER_WORDS)}) placeholders?',
        r'contains? exactly (\d+) placeholders?',
        rf'contains? exactly ({"|".join(NUMBER_WORDS)}) placeholders?',
        r'contains? (\d+) placeholders?',
        rf'contains? ({"|".join(NUMBER_WORDS)}) placeholders?',
    ]
    for pattern in patterns:
        value = _extract_number_token(prompt, pattern)
        if value is not None:
            return value
    return None


def _extract_bullet_requirement(prompt: str) -> Optional[Dict[str, Any]]:
    patterns = [
        (r'(\d+) bullet lists?', "list_count"),
        (rf'({"|".join(NUMBER_WORDS)}) bullet lists?', "list_count"),
        (r'(\d+) bullet points?', "item_count"),
        (rf'({"|".join(NUMBER_WORDS)}) bullet points?', "item_count"),
        (r'bullet list with (\d+) items?', "item_count"),
        (rf'bullet list with ({"|".join(NUMBER_WORDS)}) items?', "item_count"),
    ]
    for pattern, mode in patterns:
        value = _extract_number_token(prompt, pattern)
        if value is not None:
            return {"value": value, "mode": mode}
    return None


def _extract_choose_one_options(prompt: str) -> List[str]:
    patterns = [
        r'choose from the following options:\s*\((.+?)\)',
        r'choose from the following:\s*\((.+?)\)',
        r'one of the following exact phrases:\s*(.+?)(?:[.](?:\s|$)|$)',
        r'choos(?:e|ing) one of the following options(?: as [^:]+)?:\s*(.+)$',
        r'choose one from the following options:\s*(.+)$',
        r'choose one from options:\s*(.+)$',
        r'choose one of the following themes(?: [^:]+)?:\s*(.+)$',
        r'choose one of the following settings:\s*(.+)$',
        r'answer with one of the following options:\s*(.+)$',
        r'classify your answers? into\s*(.+?)(?:[.](?:\s|$)|$)',
    ]
    for pattern in patterns:
        match = re.search(pattern, prompt, re.IGNORECASE)
        if not match:
            continue
        values = _normalize_prompt_keywords(_extract_quoted_phrases(match.group(1)))
        if values:
            return values
        raw = match.group(1).strip().strip("()")
        if ", " in raw:
            values = _normalize_prompt_keywords([part.strip() for part in raw.split(",")])
            if len(values) >= 2:
                return values
        if " or " in raw.lower():
            values = _normalize_prompt_keywords(
                re.split(r'\s+or\s+', raw, maxsplit=10, flags=re.IGNORECASE))
            if len(values) >= 2:
                return values
    either_match = re.search(
        r'choose either ([^.:\n]+?) or ([^.:\n]+?)(?:[.](?:\s|$)|$)',
        prompt,
        re.IGNORECASE,
    )
    if either_match:
        return _normalize_prompt_keywords([either_match.group(1), either_match.group(2)])
    include_match = re.search(
        r'choose whether to include ([^.:\n]+?) or ([^.:\n]+?), but not both',
        prompt,
        re.IGNORECASE,
    )
    if include_match:
        return _normalize_prompt_keywords([include_match.group(1), include_match.group(2)])
    return []


def _extract_ordinal_token(prompt: str, pattern: str) -> Optional[int]:
    match = re.search(pattern, prompt, re.IGNORECASE)
    if not match:
        return None
    token = match.group(1).lower()
    if token.isdigit():
        return int(token)
    return ORDINAL_WORDS.get(token)


def _extract_capital_word_requirement(prompt: str) -> Optional[Dict[str, Any]]:
    patterns = [
        (r'exactly (\d+) words? in all capital letters', "exactly", "all"),
        (rf'exactly ({"|".join(NUMBER_WORDS)}) words? in all capital letters', "exactly", "all"),
        (r'words? in all capital letters is exactly (\d+)', "exactly", "all"),
        (rf'words? in all capital letters is exactly ({"|".join(NUMBER_WORDS)})', "exactly", "all"),
        (r'at least (\d+) (?:words?|instances?) of words? in all capital letters', "at least", "all"),
        (rf'at least ({"|".join(NUMBER_WORDS)}) (?:words?|instances?) of words? in all capital letters', "at least", "all"),
        (r'at least (\d+) (?:words?|instances?) in all capital letters', "at least", "all"),
        (rf'at least ({"|".join(NUMBER_WORDS)}) (?:words?|instances?) in all capital letters', "at least", "all"),
        (r'no more than (\d+) words? in all capital letters', "at most", "all"),
        (rf'no more than ({"|".join(NUMBER_WORDS)}) words? in all capital letters', "at most", "all"),
        (r'no more than (\d+) capitalized words? in each paragraph', "at most", "each_paragraph"),
        (rf'no more than ({"|".join(NUMBER_WORDS)}) capitalized words? in each paragraph', "at most", "each_paragraph"),
    ]
    for pattern, relation, scope in patterns:
        value = _extract_number_token(prompt, pattern)
        if value is not None:
            return {"value": value, "relation": relation, "scope": scope}
    return None


def _extract_first_word_nth_paragraph_requirement(
        prompt: str) -> Optional[Dict[str, Any]]:
    patterns = [
        r'first word of the ([a-z]+|\d+)(?:st|nd|rd|th)? paragraph (?:should be|is) "([^"]+)"',
        r'first word of the ([a-z]+|\d+)(?:st|nd|rd|th)? paragraph (?:should be|is) \'([^\']+)\'',
        r'begin the ([a-z]+|\d+)(?:st|nd|rd|th)? paragraph with the word "([^"]+)"',
        r'begin the ([a-z]+|\d+)(?:st|nd|rd|th)? paragraph with the word \'([^\']+)\'',
        r'the ([a-z]+|\d+)(?:st|nd|rd|th)? paragraph starts with the word "([^"]+)"',
        r'the ([a-z]+|\d+)(?:st|nd|rd|th)? paragraph starts with the word \'([^\']+)\'',
    ]
    for pattern in patterns:
        match = re.search(pattern, prompt, re.IGNORECASE)
        if not match:
            continue
        ordinal = match.group(1).lower()
        expected = match.group(2)
        index = int(ordinal) if ordinal.isdigit() else ORDINAL_WORDS.get(ordinal)
        if index is not None:
            return {"index": index, "value": expected}
    return None


def _extract_frequency_requirements(prompt: str) -> List[Dict[str, Any]]:
    requirements: List[Dict[str, Any]] = []
    patterns = [
        r'(?:word|phrase)\s+"([^"]+)"\s+(at least|exactly)\s+(\d+)\s+times',
        rf'(?:word|phrase)\s+"([^"]+)"\s+(at least|exactly)\s+({"|".join(NUMBER_WORDS)})\s+times',
        r"(?:word|phrase)\s+'([^']+)'\s+(at least|exactly)\s+(\d+)\s+times",
        rf"(?:word|phrase)\s+'([^']+)'\s+(at least|exactly)\s+({'|'.join(NUMBER_WORDS)})\s+times",
    ]
    for pattern in patterns:
        for term, relation, raw_count in re.findall(pattern, prompt, re.IGNORECASE):
            count = int(raw_count) if raw_count.isdigit() else NUMBER_WORDS[
                raw_count.lower()]
            requirements.append({
                "term": term,
                "count": count,
                "relation": relation.lower(),
            })
    return requirements


def _extract_sentence_term_frequency_requirement(
        prompt: str) -> Optional[Dict[str, Any]]:
    count_tokens = "|".join([*NUMBER_WORDS.keys(), *FREQUENCY_WORDS.keys()])
    patterns = [
        r'each sentence must include the word [\'"]([^\'"]+)[\'"] (at least|exactly) (\d+) times',
        rf'each sentence must include the word [\'"]([^\'"]+)[\'"] (at least|exactly) ({count_tokens})(?: times)?',
        r'each sentence must contain the word [\'"]([^\'"]+)[\'"] (at least|exactly) (\d+) times',
        rf'each sentence must contain the word [\'"]([^\'"]+)[\'"] (at least|exactly) ({count_tokens})(?: times)?',
        r'uses the word [\'"]([^\'"]+)[\'"] (at least|exactly) (\d+) times',
        rf'uses the word [\'"]([^\'"]+)[\'"] (at least|exactly) ({count_tokens})(?: times)?',
    ]
    for idx, pattern in enumerate(patterns):
        match = re.search(pattern, prompt, re.IGNORECASE)
        if not match:
            continue
        term = match.group(1)
        relation = match.group(2).lower()
        raw_count = match.group(3)
        count = _parse_count_token(raw_count)
        if count is None:
            continue
        return {
            "term": term,
            "count": count,
            "relation": relation,
            "scope": "each_sentence" if idx < 4 else "all",
        }
    return None


def _extract_highlighted_section_requirement(prompt: str) -> Optional[Dict[str,
                                                                             Any]]:
    patterns = [
        (r'include exactly (\d+) highlighted sections?', "exactly"),
        (rf'include exactly ({"|".join(NUMBER_WORDS)}) highlighted sections?', "exactly"),
        (r'include (\d+) highlighted sections?', "exactly"),
        (rf'include ({"|".join(NUMBER_WORDS)}) highlighted sections?', "exactly"),
        (r'contain(?:s)? (\d+) highlighted sections?', "exactly"),
        (rf'contain(?:s)? ({"|".join(NUMBER_WORDS)}) highlighted sections?', "exactly"),
        (r'with (\d+) highlighted sections?(?!\s+each)', "exactly"),
        (rf'with ({"|".join(NUMBER_WORDS)}) highlighted sections?(?!\s+each)', "exactly"),
        (r'at least (\d+) highlighted sections?', "at least"),
        (rf'at least ({"|".join(NUMBER_WORDS)}) highlighted sections?', "at least"),
        (r'at least (\d+) bold text sections?', "at least"),
        (rf'at least ({"|".join(NUMBER_WORDS)}) bold text sections?', "at least"),
        (r'at least [*_]*(\d+)[*_]* bold text sections?', "at least"),
        (rf'at least [*_]*({"|".join(NUMBER_WORDS)})[*_]* bold text sections?', "at least"),
        (r'into (\d+) highlighted sections?', "exactly"),
        (rf'into ({"|".join(NUMBER_WORDS)}) highlighted sections?', "exactly"),
        (r'divided into (\d+) highlighted sections?', "exactly"),
        (rf'divided into ({"|".join(NUMBER_WORDS)}) highlighted sections?', "exactly"),
        (r'(\d+) distinct sections, each highlighted', "exactly"),
        (rf'({"|".join(NUMBER_WORDS)}) distinct sections, each highlighted', "exactly"),
        (r'highlight (\d+) sections?', "exactly"),
        (rf'highlight ({"|".join(NUMBER_WORDS)}) sections?', "exactly"),
        (r'highlight (\d+) [a-z]+', "exactly"),
        (rf'highlight ({"|".join(NUMBER_WORDS)}) [a-z]+', "exactly"),
        (r'more than (\d+) [a-z]+', "at least_plus_one"),
        (rf'more than ({"|".join(NUMBER_WORDS)}) [a-z]+', "at least_plus_one"),
    ]
    for pattern, relation in patterns:
        value = _extract_number_token(prompt, pattern)
        if value is not None:
            if relation == "at least_plus_one":
                return {"value": value + 1, "relation": "at least"}
            return {"value": value, "relation": relation}
    return None


def _extract_letter_frequency_requirement(prompt: str) -> Optional[Dict[str,
                                                                         Any]]:
    patterns = [
        (r'each sentence must contain the letter [\'"]([^\'"]+)[\'"] at least (\d+) times', "each_sentence", "at least"),
        (rf'each sentence must contain the letter [\'"]([^\'"]+)[\'"] at least ({"|".join(NUMBER_WORDS)}) times', "each_sentence", "at least"),
        (r'at least one sentence where the letter [\'"]([^\'"]+)[\'"] appears at least (\d+) times', "any_sentence", "at least"),
        (rf'at least one sentence where the letter [\'"]([^\'"]+)[\'"] appears at least ({"|".join(NUMBER_WORDS)}) times', "any_sentence", "at least"),
        (r'the letter [\'"]([^\'"]+)[\'"] (?:must )?appears? at least (\d+) times', "all", "at least"),
        (rf'the letter [\'"]([^\'"]+)[\'"] (?:must )?appears? at least ({"|".join(NUMBER_WORDS)}) times', "all", "at least"),
        (r'the letter [\'"]([^\'"]+)[\'"] (?:must )?appears? exactly (\d+) times', "all", "exactly"),
        (rf'the letter [\'"]([^\'"]+)[\'"] (?:must )?appears? exactly ({"|".join(NUMBER_WORDS)}) times', "all", "exactly"),
        (r'contains? the letter [\'"]([^\'"]+)[\'"] at least (\d+) times', "all", "at least"),
        (rf'contains? the letter [\'"]([^\'"]+)[\'"] at least ({"|".join(NUMBER_WORDS)}) times', "all", "at least"),
    ]
    for pattern, scope, relation in patterns:
        match = re.search(pattern, prompt, re.IGNORECASE)
        if not match:
            continue
        letter = match.group(1)
        raw_count = match.group(2)
        count = _parse_count_token(raw_count)
        if count is None:
            continue
        return {
            "letter": letter.lower(),
            "count": count,
            "scope": scope,
            "relation": relation,
        }
    return None


def _count_term_occurrences(text: str, term: str) -> int:
    pattern = re.compile(rf'(?<!\w){re.escape(term)}(?!\w)', re.IGNORECASE)
    return len(pattern.findall(text))


def _sentence_blocks(text: str) -> List[str]:
    parts = re.split(r"(?<=[.!?])\s+|\n+", text.strip())
    return [part.strip() for part in parts if part.strip()]


def _all_letter_chars(text: str) -> List[str]:
    return [ch for ch in text if ch.isalpha()]


def _ascii_letter_chars(text: str) -> List[str]:
    return [ch for ch in text if ch.isascii() and ch.isalpha()]


def _is_english_text(text: str) -> bool:
    letters = _all_letter_chars(text)
    ascii_letters = _ascii_letter_chars(text)
    return bool(ascii_letters) and len(ascii_letters) == len(letters)


def _detect_language(text: str) -> Optional[str]:
    try:
        from langdetect import DetectorFactory, LangDetectException, detect
    except Exception:
        return None
    DetectorFactory.seed = 0
    try:
        return detect(text)
    except LangDetectException:
        return None


def _language_matches(text: str, expected: str) -> Tuple[bool, Dict[str, Any]]:
    expected = expected.lower()
    if expected in {"en", "eng", "english"}:
        ok = _is_english_text(text)
        return ok, {"expected_language": expected, "backend": "ascii_english"}
    detected = _detect_language(text)
    return detected == expected, {
        "expected_language": expected,
        "detected_language": detected,
        "backend": "langdetect",
    }


def _count_bullet_lists(text: str) -> int:
    count = 0
    in_list = False
    for line in text.splitlines():
        is_bullet = bool(re.match(r"^\s*(?:[-*]|\d+\.)\s+", line))
        if is_bullet and not in_list:
            count += 1
            in_list = True
        elif not is_bullet:
            in_list = False
    return count


def _count_bullet_items(text: str) -> int:
    return sum(
        1 for line in text.splitlines()
        if re.match(r"^\s*(?:[-*]|\d+\.)\s+", line))


def _count_sections(text: str) -> int:
    count = 0
    for line in text.splitlines():
        stripped = line.strip()
        if not stripped:
            continue
        if re.match(r"^(?:#+\s*)?section\s+\d+\b", stripped, re.IGNORECASE):
            count += 1
            continue
        if re.match(r"^\*\*[^*]+\*\*$", stripped):
            count += 1
            continue
        if stripped.endswith(":") and len(_split_words(stripped)) <= 12:
            count += 1
    return count


def _count_marked_sections(text: str, marker: str) -> int:
    marker = re.escape(marker.strip())
    if not marker:
        return 0
    return len(
        re.findall(rf"(?im)^\s*{marker}\s*\d+\b", text)
    )


def _count_highlighted_sections(text: str) -> int:
    matches = re.findall(
        r"\*\*[^*\n]+\*\*|__[^_\n]+__|(?<!\*)\*[^*\n]+\*(?!\*)|(?<!_)_[^_\n]+_(?!_)",
        text,
    )
    return len(matches)


def _has_title_like_line(text: str) -> bool:
    for idx, line in enumerate(text.splitlines()):
        stripped = line.strip()
        if not stripped:
            continue
        if stripped.startswith("#"):
            return True
        if stripped.startswith("[") and stripped.endswith("]"):
            return True
        if len(_split_words(stripped)) <= 12 and not stripped.endswith((".", "?", "!")):
            next_lines = text.splitlines()[idx + 1:]
            if next_lines and not next_lines[0].strip():
                return True
        return False
    return False


def _contains_prompt_text(answer: str, prompt_to_repeat: str, count: int = 1) -> bool:
    return answer.count(prompt_to_repeat) >= count


def _all_words_square_bracketed(text: str) -> bool:
    tokens = re.findall(r"\S+", text)
    if not tokens:
        return False
    return all(re.fullmatch(r"\[[^\[\]\s]+\][.,!?;:]?", token) for token in tokens)


def _bigrams_wrapped(text: str) -> bool:
    stripped = text.strip()
    if not stripped:
        return False
    spans = re.findall(r"<<([^<>\n]+)>>", stripped)
    if not spans:
        return False
    outside = re.sub(r"<<[^<>\n]+>>", " ", stripped).strip()
    if outside:
        return False
    return all(len(_split_words(span)) == 2 for span in spans)


def _same_start_end_word(text: str) -> Tuple[bool, Dict[str, Any]]:
    words = _split_words(text)
    if not words:
        return False, {"first": None, "last": None}
    first = words[0].lower()
    last = words[-1].lower()
    return first == last, {"first": first, "last": last}


def _has_palindrome(text: str) -> Tuple[bool, Dict[str, Any]]:
    palindromes = [
        word for word in _split_words(text)
        if len(word) > 2 and word.lower() == word.lower()[::-1]
    ]
    return bool(palindromes), {"palindromes": palindromes[:10]}


def _lowercase_word_max_frequency(text: str, max_count: int) -> Tuple[bool, Dict[str, Any]]:
    counts: Dict[str, int] = {}
    for word in _split_words(text):
        if word.islower():
            counts[word] = counts.get(word, 0) + 1
    violations = {word: count for word, count in counts.items() if count > max_count}
    return not violations, {"max_allowed": max_count, "violations": violations}


def _all_words_unique(text: str) -> Tuple[bool, Dict[str, Any]]:
    words = [word.lower() for word in _split_words(text)]
    seen = set()
    duplicates = []
    for word in words:
        if word in seen:
            duplicates.append(word)
        seen.add(word)
    return not duplicates, {"duplicates": duplicates[:20]}


def _no_adjacent_consecutive_initials(text: str) -> Tuple[bool, Dict[str, Any]]:
    words = _split_words(text)
    violations = []
    initials = [word[0].lower() for word in words if word and word[0].isalpha()]
    for left, right in zip(initials, initials[1:]):
        if left.isascii() and right.isascii() and abs(ord(left) - ord(right)) == 1:
            violations.append(f"{left}{right}")
    return not violations, {"violations": violations[:20]}


def _composition_matches(text: str, sentence_count: int, words_per_sentence: int) -> Tuple[bool, Dict[str, Any]]:
    paragraphs = _paragraphs(text)
    if not paragraphs:
        paragraphs = [text]
    details = []
    all_ok = True
    for paragraph in paragraphs:
        sentences = _sentence_blocks(paragraph)
        word_counts = [len(_split_words(sentence)) for sentence in sentences]
        ok = len(sentences) == sentence_count and all(
            count == words_per_sentence for count in word_counts)
        details.append({
            "sentence_count": len(sentences),
            "word_counts": word_counts,
            "passed": ok,
        })
        all_ok = all_ok and ok
    return all_ok, {
        "expected_sentences_per_paragraph": sentence_count,
        "expected_words_per_sentence": words_per_sentence,
        "paragraphs": details,
    }


def _compare_count(actual: int, expected: int, relation: str) -> bool:
    if relation == "at least":
        return actual >= expected
    if relation == "at most":
        return actual <= expected
    if relation == "less than":
        return actual < expected
    return actual == expected


def _check_constraint(spec: Dict[str, Any], answer: str) -> Tuple[bool, Dict[str, Any]]:
    kind = spec["kind"]
    if kind == "last_word":
        actual = _normalized_last_word(answer)
        expected = str(spec["value"]).lower()
        return actual == expected, {"expected": expected, "actual": actual}

    if kind == "last_word_each_sentence":
        sentences = _sentence_blocks(answer)
        expected = str(spec["value"]).lower()
        actual = [_normalized_last_word(sentence) for sentence in sentences]
        return bool(actual) and all(word == expected for word in actual), {
            "expected": expected,
            "actual": actual,
        }

    if kind == "first_word":
        actual = _normalized_first_word(answer)
        expected = str(spec["value"]).lower()
        return actual == expected, {"expected": expected, "actual": actual}

    if kind == "exclude_keywords":
        lowered = answer.lower()
        values = [str(word) for word in spec["values"] if word]
        missing = [word for word in values if word.lower() in lowered]
        return not missing, {"forbidden_present": missing}

    if kind == "include_keywords":
        lowered = answer.lower()
        values = [str(word) for word in spec["values"] if word]
        missing = [word for word in values if word.lower() not in lowered]
        return not missing, {"missing_keywords": missing}

    if kind == "punctuation_forbidden":
        symbol = str(spec["value"])
        present = symbol in answer
        return not present, {"symbol": symbol, "present": present}

    if kind == "sentence_hyphens":
        compact = answer.strip()
        ok = bool(compact) and " " not in compact and "-" in compact
        return ok, {"value": compact[:120]}

    if kind == "bigram_wrapping":
        ok = _bigrams_wrapped(answer)
        return ok, {"value": answer[:160]}

    if kind == "square_brackets":
        ok = _all_words_square_bracketed(answer)
        return ok, {"value": answer[:160]}

    if kind == "start_end_same_word":
        return _same_start_end_word(answer)

    if kind == "palindrome":
        return _has_palindrome(answer)

    if kind == "unique_words":
        return _all_words_unique(answer)

    if kind == "no_adjacent_consecutive_initials":
        return _no_adjacent_consecutive_initials(answer)

    if kind == "num_paragraphs":
        actual = len(_paragraphs(answer))
        expected = int(spec["value"])
        return actual == expected, {"expected": expected, "actual": actual}

    if kind == "specific_ending":
        expected = str(spec["value"]).strip()
        actual = answer.rstrip()
        return actual.endswith(expected), {"expected": expected}

    if kind == "json_format":
        try:
            parsed = json.loads(answer.strip())
        except Exception as exc:
            return False, {"error": type(exc).__name__}
        return isinstance(parsed, (dict, list)), {"parsed_type": type(parsed).__name__}

    if kind == "num_sentences":
        actual = len(_sentence_blocks(answer))
        expected = int(spec["value"])
        relation = str(spec.get("relation") or "exactly")
        return _compare_count(actual, expected, relation), {
            "expected": expected,
            "relation": relation,
            "actual": actual,
        }

    if kind == "num_sections":
        actual = _count_sections(answer)
        expected = int(spec["value"])
        return actual == expected, {"expected": expected, "actual": actual}

    if kind == "marked_sections":
        actual = _count_marked_sections(answer, str(spec["marker"]))
        expected = int(spec["value"])
        return actual == expected, {
            "marker": spec["marker"],
            "expected": expected,
            "actual": actual,
        }

    if kind == "num_words":
        actual = len(_split_words(answer))
        expected = int(spec["value"])
        relation = str(spec.get("relation") or "exactly")
        if relation == "between":
            upper = int(spec["upper_value"])
            return expected <= actual <= upper, {
                "expected_lower": expected,
                "expected_upper": upper,
                "relation": relation,
                "actual": actual,
            }
        return _compare_count(actual, expected, relation), {
            "expected": expected,
            "relation": relation,
            "actual": actual,
        }

    if kind == "nth_paragraph_first_word":
        paragraphs = _paragraphs(answer)
        index = int(spec["index"])
        target_idx = len(paragraphs) - 1 if index == -1 else index - 1
        actual = (_normalized_first_word(paragraphs[target_idx])
                  if 0 <= target_idx < len(paragraphs) else None)
        expected = str(spec["value"]).lower()
        return actual == expected, {
            "expected": expected,
            "actual": actual,
            "paragraph_count": len(paragraphs),
        }

    if kind == "repeat_prompt":
        expected = str(spec["value"])
        return _contains_prompt_text(answer, expected), {"expected": expected[:120]}

    if kind == "repeat_phrase_count":
        phrase = str(spec["value"])
        count = int(spec["count"])
        actual = answer.count(phrase)
        return actual == count, {"phrase": phrase, "expected": count, "actual": actual}

    if kind == "copy_span":
        span = str(spec["value"])
        return span in answer, {"expected": span}

    if kind == "placeholder_count":
        actual = _count_placeholders(answer)
        expected = int(spec["value"])
        return actual >= expected, {"expected_at_least": expected, "actual": actual}

    if kind == "postscript":
        ok = bool(
            re.search(r'(^|\n)\s*(?:P\.?\s*S\.?|Postscript)\s*[:.]',
                      answer,
                      re.IGNORECASE))
        return ok, {"found": ok}

    if kind == "title":
        ok = _has_title_like_line(answer)
        return ok, {"first_nonempty_line": next((line.strip() for line in answer.splitlines() if line.strip()), None)}

    if kind == "bullet_list_count":
        actual = _count_bullet_lists(answer)
        expected = int(spec["value"])
        return actual == expected, {"expected": expected, "actual": actual}

    if kind == "bullet_item_count":
        actual = _count_bullet_items(answer)
        expected = int(spec["value"])
        return actual == expected, {"expected": expected, "actual": actual}

    if kind == "highlighted_section_count":
        actual = _count_highlighted_sections(answer)
        expected = int(spec["value"])
        relation = str(spec.get("relation") or "exactly")
        return _compare_count(actual, expected, relation), {
            "expected": expected,
            "relation": relation,
            "actual": actual,
        }

    if kind == "quotation":
        stripped = answer.strip()
        ok = (stripped.startswith('"') and stripped.endswith('"')) or (
            stripped.startswith("'") and stripped.endswith("'"))
        return ok, {"value": stripped[:120]}

    if kind == "english_lowercase":
        letters = _all_letter_chars(answer)
        ascii_letters = _ascii_letter_chars(answer)
        ok = bool(ascii_letters) and len(ascii_letters) == len(letters) and all(
            ch == ch.lower() for ch in ascii_letters)
        return ok, {"checked_letters": len(letters)}

    if kind == "english_uppercase":
        letters = _all_letter_chars(answer)
        ascii_letters = _ascii_letter_chars(answer)
        ok = bool(ascii_letters) and len(ascii_letters) == len(letters) and all(
            ch == ch.upper() for ch in ascii_letters)
        return ok, {"checked_letters": len(letters)}

    if kind == "response_language":
        expected = str(spec.get("value") or "english").lower()
        return _language_matches(answer, expected)

    if kind == "capital_word_frequency":
        expected = int(spec["value"])
        relation = str(spec.get("relation") or "exactly")
        scope = str(spec.get("scope") or "all")
        if scope == "each_paragraph":
            counts = [_count_uppercase_words(paragraph) for paragraph in _paragraphs(answer)]
            ok = bool(counts) and all(
                _compare_count(actual, expected, relation) for actual in counts)
            return ok, {
                "expected": expected,
                "relation": relation,
                "scope": scope,
                "actual_counts": counts,
            }
        actual = _count_uppercase_words(answer)
        return _compare_count(actual, expected, relation), {
            "expected": expected,
            "relation": relation,
            "scope": scope,
            "actual": actual,
        }

    if kind == "two_responses":
        normalized = answer.lower()
        ok = (
            "summary 1" in normalized and "summary 2" in normalized
        ) or (
            "response 1" in normalized and "response 2" in normalized
        ) or (
            len(re.findall(r"\n\s*\*{4,}\s*\n", answer)) >= 1
        )
        return ok, {"value": answer[:160]}

    if kind == "choose_one_option":
        normalized = answer.lower()
        options = [str(option) for option in spec["values"]]
        stripped = normalized.strip().strip(".,!?;:()[]{}\"'")
        if all(len(option.strip()) == 1 for option in options):
            matches = [
                option for option in options if option.lower().strip() == stripped
            ]
        else:
            matches = [
                option for option in options if option.lower() in normalized
            ]
        return len(matches) == 1, {
            "options": options,
            "matched": matches,
        }

    if kind == "term_frequency":
        actual = _count_term_occurrences(answer, str(spec["term"]))
        expected = int(spec["count"])
        relation = str(spec.get("relation") or "exactly")
        return _compare_count(actual, expected, relation), {
            "term": spec["term"],
            "expected": expected,
            "relation": relation,
            "actual": actual,
        }

    if kind == "multi_term_frequency":
        results = []
        all_pass = True
        for term, expected in spec["terms"]:
            actual = _count_term_occurrences(answer, str(term))
            passed = actual == int(expected)
            all_pass = all_pass and passed
            results.append({
                "term": term,
                "expected": int(expected),
                "actual": actual,
                "passed": passed,
            })
        return all_pass, {"results": results}

    if kind == "letter_frequency":
        letter = str(spec["letter"]).lower()
        actual = answer.lower().count(letter)
        expected = int(spec["count"])
        relation = str(spec.get("relation") or "exactly")
        return _compare_count(actual, expected, relation), {
            "letter": letter,
            "expected": expected,
            "relation": relation,
            "actual": actual,
        }

    if kind == "lowercase_word_max_frequency":
        return _lowercase_word_max_frequency(answer, int(spec["value"]))

    if kind == "composition":
        return _composition_matches(
            answer,
            int(spec["sentence_count"]),
            int(spec["words_per_sentence"]),
        )

    if kind == "sentence_letter_frequency":
        sentences = _sentence_blocks(answer)
        letter = str(spec["letter"]).lower()
        expected = int(spec["count"])
        relation = str(spec.get("relation") or "at least")
        counts = [sentence.lower().count(letter) for sentence in sentences]
        scope = str(spec.get("scope") or "each_sentence")
        if scope == "all":
            actual = answer.lower().count(letter)
            ok = _compare_count(actual, expected, relation)
            return ok, {
                "letter": letter,
                "expected": expected,
                "relation": relation,
                "scope": scope,
                "actual": actual,
            }
        if scope == "each_sentence":
            ok = bool(counts) and all(
                _compare_count(count, expected, relation) for count in counts)
        else:
            ok = bool(counts) and any(
                _compare_count(count, expected, relation) for count in counts)
        return ok, {
            "letter": letter,
            "expected": expected,
            "relation": relation,
            "scope": scope,
            "counts": counts,
        }

    if kind == "sentence_term_frequency":
        sentences = _sentence_blocks(answer)
        term = str(spec["term"])
        expected = int(spec["count"])
        relation = str(spec.get("relation") or "at least")
        scope = str(spec.get("scope") or "all")
        if scope == "each_sentence":
            counts = [_count_term_occurrences(sentence, term) for sentence in sentences]
            ok = bool(counts) and all(
                _compare_count(count, expected, relation) for count in counts)
            return ok, {
                "term": term,
                "expected": expected,
                "relation": relation,
                "scope": scope,
                "counts": counts,
            }
        actual = _count_term_occurrences(answer, term)
        return _compare_count(actual, expected, relation), {
            "term": term,
            "expected": expected,
            "relation": relation,
            "scope": scope,
            "actual": actual,
        }

    if kind == "keyword_specific_position":
        sentences = _sentence_blocks(answer)
        sentence_index = int(spec["sentence_index"]) - 1
        word_index = int(spec["word_index"]) - 1
        words = (
            _split_words(sentences[sentence_index])
            if 0 <= sentence_index < len(sentences) else []
        )
        actual = words[word_index].lower() if 0 <= word_index < len(words) else None
        expected = str(spec["keyword"]).lower()
        return actual == expected, {
            "expected": expected,
            "actual": actual,
            "sentence_count": len(sentences),
            "word_count_in_sentence": len(words),
            "sentence_index": sentence_index + 1,
            "word_index": word_index + 1,
        }

    raise KeyError(kind)


def _structured_constraint_specs(row_data: Dict[str, Any]) -> Optional[List[Dict[str, Any]]]:
    raw = row_data.get("ground_truth")
    if not isinstance(raw, str) or not raw.strip():
        return None
    try:
        payload = ast.literal_eval(raw)
    except Exception:
        return None
    if not isinstance(payload, list) or not payload:
        return None

    specs: List[Dict[str, Any]] = []
    for entry in payload:
        if not isinstance(entry, dict):
            return None
        instruction_ids = entry.get("instruction_id") or []
        kwargs_list = entry.get("kwargs") or []
        for idx, instruction_id in enumerate(instruction_ids):
            kwargs = kwargs_list[idx] if idx < len(kwargs_list) else None
            spec = _spec_from_instruction_id(str(instruction_id), kwargs)
            if spec is None:
                specs.append({
                    "kind": "__unsupported__",
                    "source": str(instruction_id),
                })
                continue
            specs.append(spec)
    return specs


def _spec_from_instruction_id(instruction_id: str,
                              kwargs: Optional[Dict[str, Any]]) -> Optional[Dict[str, Any]]:
    kwargs = kwargs or {}
    if instruction_id == "detectable_format:sentence_hyphens":
        return {"kind": "sentence_hyphens", "source": instruction_id}
    if instruction_id == "detectable_format:bigram_wrapping":
        return {"kind": "bigram_wrapping", "source": instruction_id}
    if instruction_id == "detectable_format:square_brackets":
        return {"kind": "square_brackets", "source": instruction_id}
    if instruction_id == "last_word:last_word_answer":
        return {"kind": "last_word", "value": kwargs.get("last_word"), "source": instruction_id}
    if instruction_id == "last_word:last_word_sent":
        return {"kind": "last_word_each_sentence", "value": kwargs.get("last_word"), "source": instruction_id}
    if instruction_id in {"first_word:first_word_answer", "first_word:first_word_sent"}:
        return {"kind": "first_word", "value": kwargs.get("first_word"), "source": instruction_id}
    if instruction_id in {"keywords:word_once", "keywords:existence"}:
        values = kwargs.get("keywords")
        if values is None:
            values = [kwargs.get("keyword")]
        return {"kind": "include_keywords", "values": values, "source": instruction_id}
    if instruction_id in {"keywords:exclude_word_harder", "keywords:forbidden_words"}:
        values = kwargs.get("forbidden_words")
        if values is None:
            values = [kwargs.get("keyword")]
        return {"kind": "exclude_keywords", "values": values, "source": instruction_id}
    if instruction_id in {"keywords:frequency", "keywords:word_count_different_numbers"}:
        return {
            "kind": "term_frequency",
            "term": kwargs.get("keyword"),
            "count": kwargs.get("frequency"),
            "relation": kwargs.get("relation") or "exactly",
            "source": instruction_id,
        }
    if instruction_id in {"keywords:letter_frequency", "letters:letter_counting2"}:
        return {
            "kind": "letter_frequency",
            "letter": kwargs.get("letter"),
            "count": kwargs.get("let_frequency"),
            "relation": kwargs.get("let_relation") or "exactly",
            "source": instruction_id,
        }
    if instruction_id == "letters:letter_counting":
        return {
            "kind": "letter_frequency",
            "letter": kwargs.get("letter"),
            "count": kwargs.get("N"),
            "relation": kwargs.get("relation") or "exactly",
            "source": instruction_id,
        }
    if instruction_id == "keywords:start_end":
        return {"kind": "start_end_same_word", "source": instruction_id}
    if instruction_id == "keywords:palindrome":
        return {"kind": "palindrome", "source": instruction_id}
    if instruction_id == "keywords:no_adjacent_consecutive":
        return {"kind": "no_adjacent_consecutive_initials", "source": instruction_id}
    if instruction_id == "length_constraints:number_paragraphs":
        return {"kind": "num_paragraphs", "value": kwargs.get("num_paragraphs"), "source": instruction_id}
    if instruction_id == "length_constraints:number_sentences":
        return {
            "kind": "num_sentences",
            "value": kwargs.get("num_sentences"),
            "relation": kwargs.get("relation") or "exactly",
            "source": instruction_id,
        }
    if instruction_id == "length_constraints:number_words":
        return {
            "kind": "num_words",
            "value": kwargs.get("num_words"),
            "relation": kwargs.get("relation") or "exactly",
            "source": instruction_id,
        }
    if instruction_id == "length_constraints:nth_paragraph_first_word":
        return {
            "kind": "nth_paragraph_first_word",
            "index": kwargs.get("nth_paragraph"),
            "value": kwargs.get("first_word"),
            "source": instruction_id,
        }
    if instruction_id == "detectable_content:number_placeholders":
        return {"kind": "placeholder_count", "value": kwargs.get("num_placeholders"), "source": instruction_id}
    if instruction_id == "detectable_content:postscript":
        return {"kind": "postscript", "source": instruction_id}
    if instruction_id == "detectable_format:number_highlighted_sections":
        return {
            "kind": "highlighted_section_count",
            "value": kwargs.get("num_highlights"),
            "relation": kwargs.get("relation") or "at least",
            "source": instruction_id,
        }
    if instruction_id == "detectable_format:number_bullet_lists":
        return {
            "kind": "bullet_item_count",
            "value": kwargs.get("num_bullets"),
            "source": instruction_id,
        }
    if instruction_id == "detectable_format:multiple_sections":
        return {
            "kind": "marked_sections",
            "marker": kwargs.get("section_spliter"),
            "value": kwargs.get("num_sections"),
            "source": instruction_id,
        }
    if instruction_id == "detectable_format:json_format":
        return {"kind": "json_format", "source": instruction_id}
    if instruction_id == "startend:end_checker":
        return {"kind": "specific_ending", "value": kwargs.get("end_phrase"), "source": instruction_id}
    if instruction_id == "startend:quotation":
        return {"kind": "quotation", "source": instruction_id}
    if instruction_id == "change_case:english_lowercase":
        return {"kind": "english_lowercase", "source": instruction_id}
    if instruction_id == "change_case:english_capital":
        return {"kind": "english_uppercase", "source": instruction_id}
    if instruction_id == "change_case:capital_word_frequency":
        return {
            "kind": "capital_word_frequency",
            "value": kwargs.get("capital_frequency"),
            "relation": kwargs.get("capital_relation") or "exactly",
            "source": instruction_id,
        }
    if instruction_id == "language:response_language":
        return {
            "kind": "response_language",
            "value": kwargs.get("language"),
            "source": instruction_id,
        }
    if instruction_id == "copy:copying_simple":
        return {"kind": "repeat_prompt", "value": kwargs.get("prompt_to_repeat"), "source": instruction_id}
    if instruction_id == "copy:copy":
        return {"kind": "repeat_prompt", "value": kwargs.get("prompt_to_repeat"), "source": instruction_id}
    if instruction_id == "copy:copying_multiple":
        return {
            "kind": "repeat_phrase_count",
            "value": kwargs.get("prompt_to_repeat"),
            "count": kwargs.get("N"),
            "source": instruction_id,
        }
    if instruction_id == "copy:repeat_phrase":
        return {
            "kind": "repeat_phrase_count",
            "value": kwargs.get("phrase"),
            "count": kwargs.get("small_n"),
            "source": instruction_id,
        }
    if instruction_id == "new:copy_span_idx":
        prompt = str(kwargs.get("prompt_to_repeat") or "")
        start = kwargs.get("n_start")
        end = kwargs.get("n_end")
        if isinstance(start, int) and isinstance(end, int) and prompt:
            return {
                "kind": "copy_span",
                "value": prompt[start:end + 1],
                "source": instruction_id,
            }
    if instruction_id == "punctuation:punctuation_dot":
        return {"kind": "punctuation_forbidden", "value": ".", "source": instruction_id}
    if instruction_id == "punctuation:punctuation_exclamation":
        return {"kind": "punctuation_forbidden", "value": "!", "source": instruction_id}
    if instruction_id == "punctuation:no_comma":
        return {"kind": "punctuation_forbidden", "value": ",", "source": instruction_id}
    if instruction_id == "detectable_format:title":
        return {"kind": "title", "source": instruction_id}
    if instruction_id == "count:count_unique":
        return {"kind": "unique_words", "source": instruction_id}
    if instruction_id == "count:lowercase_counting":
        return {
            "kind": "lowercase_word_max_frequency",
            "value": kwargs.get("N"),
            "source": instruction_id,
        }
    if instruction_id == "count:count_increment_word":
        return {
            "kind": "multi_term_frequency",
            "terms": [(kwargs.get("keyword1"), 1), (kwargs.get("keyword2"), 2)],
            "source": instruction_id,
        }
    if instruction_id == "count:counting_composition":
        return {
            "kind": "composition",
            "sentence_count": kwargs.get("n_sent"),
            "words_per_sentence": kwargs.get("n_words"),
            "source": instruction_id,
        }
    if instruction_id == "combination:repeat_prompt":
        return {"kind": "repeat_prompt", "value": kwargs.get("prompt_to_repeat"), "source": instruction_id}
    if instruction_id == "combination:two_responses":
        return {"kind": "two_responses", "source": instruction_id}
    return None


def _prompt_constraint_specs(row_data: Dict[str, Any]) -> Optional[List[Dict[str, Any]]]:
    prompt = row_data.get("prompt")
    constraints = row_data.get("constraints")
    if not isinstance(prompt, str) or not prompt.strip() or not isinstance(constraints, list):
        return None

    specs: List[Dict[str, Any]] = []
    for raw_constraint in constraints:
        constraint = str(raw_constraint).strip().lower()
        if constraint == "keywords:exclude words":
            values = _normalize_prompt_keywords(_extract_excluded_words(prompt))
            if values:
                specs.append({"kind": "exclude_keywords", "values": values, "source": raw_constraint})
                continue
        if constraint == "specific ending":
            ending = _extract_specific_ending(prompt)
            if ending:
                specs.append({"kind": "specific_ending", "value": ending, "source": raw_constraint})
                continue
        if constraint == "format:use json format":
            if re.search(r'\bjson\b', prompt, re.IGNORECASE):
                specs.append({"kind": "json_format", "source": raw_constraint})
                continue
        if constraint == "include keywords":
            values = _normalize_prompt_keywords(_extract_included_keywords(prompt))
            if values:
                specs.append({"kind": "include_keywords", "values": values, "source": raw_constraint})
                continue
        if constraint == "response language":
            if re.search(r'\bin english\b', prompt, re.IGNORECASE):
                specs.append({
                    "kind": "response_language",
                    "value": "english",
                    "source": raw_constraint,
                })
                continue
        if constraint == "in english and capital":
            if re.search(r'\bin english\b', prompt, re.IGNORECASE) and re.search(
                    r'(?:all )?capital letters?', prompt, re.IGNORECASE):
                specs.append({
                    "kind": "english_uppercase",
                    "source": raw_constraint,
                })
                continue
        if constraint == "case: frequency of capital words":
            requirement = _extract_capital_word_requirement(prompt)
            if requirement is not None:
                specs.append({
                    "kind": "capital_word_frequency",
                    "value": requirement["value"],
                    "relation": requirement["relation"],
                    "scope": requirement.get("scope") or "all",
                    "source": raw_constraint,
                })
                continue
        if constraint == "case:in english and lowercase":
            if re.search(r'\bin english\b', prompt, re.IGNORECASE) and re.search(
                    r'\blowercase\b', prompt, re.IGNORECASE):
                specs.append({
                    "kind": "english_lowercase",
                    "source": raw_constraint,
                })
                continue
        if constraint == "length constraints:number of paragraphs":
            count = _extract_numbered_requirement(prompt, "paragraph")
            if count is not None:
                specs.append({
                    "kind": "num_paragraphs",
                    "value": count,
                    "source": raw_constraint,
                })
                continue
        if constraint == "length constraints:number of sentences":
            count = _extract_numbered_requirement(prompt, "sentence")
            if count is not None:
                specs.append({
                    "kind": "num_sentences",
                    "value": count,
                    "source": raw_constraint,
                })
                continue
        if constraint == "length constraints:first word of the nth paragraph":
            requirement = _extract_first_word_nth_paragraph_requirement(prompt)
            if requirement is not None:
                specs.append({
                    "kind": "nth_paragraph_first_word",
                    "index": requirement["index"],
                    "value": requirement["value"],
                    "source": raw_constraint,
                })
                continue
        if constraint == "length constraints:number of words":
            requirement = _extract_word_requirement(prompt)
            if requirement is not None:
                spec = {
                    "kind": "num_words",
                    "value": requirement["value"],
                    "relation": requirement["relation"].replace("_", " "),
                    "source": raw_constraint,
                }
                if "upper_value" in requirement:
                    spec["upper_value"] = requirement["upper_value"]
                specs.append(spec)
                continue
        if constraint == "format:number of sections":
            count = _extract_numbered_requirement(prompt, "section")
            if count is not None:
                specs.append({
                    "kind": "num_sections",
                    "value": count,
                    "source": raw_constraint,
                })
                continue
        if constraint == "format:number of bullet lists":
            requirement = _extract_bullet_requirement(prompt)
            if requirement is not None:
                specs.append({
                    "kind": ("bullet_list_count" if requirement["mode"] == "list_count"
                             else "bullet_item_count"),
                    "value": requirement["value"],
                    "source": raw_constraint,
                })
                continue
        if constraint == "content:number of placeholders":
            count = _extract_placeholder_requirement(prompt)
            if count is not None:
                specs.append({
                    "kind": "placeholder_count",
                    "value": count,
                    "source": raw_constraint,
                })
                continue
        if constraint == "content:include a postscript":
            if re.search(r'\bpostscript\b|\bP\.?\s*S\.?\b', prompt, re.IGNORECASE):
                specs.append({
                    "kind": "postscript",
                    "source": raw_constraint,
                })
                continue
        if constraint == "format:choose one from options":
            values = _extract_choose_one_options(prompt)
            if values:
                specs.append({
                    "kind": "choose_one_option",
                    "values": values,
                    "source": raw_constraint,
                })
                continue
        if constraint == "keywords:frequency":
            requirements = _extract_frequency_requirements(prompt)
            if requirements:
                for requirement in requirements:
                    specs.append({
                        "kind": "term_frequency",
                        "term": requirement["term"],
                        "count": requirement["count"],
                        "relation": requirement["relation"],
                        "source": raw_constraint,
                    })
                continue
        if constraint == "format:number of highlighted sections":
            requirement = _extract_highlighted_section_requirement(prompt)
            if requirement is not None:
                specs.append({
                    "kind": "highlighted_section_count",
                    "value": requirement["value"],
                    "relation": requirement["relation"],
                    "source": raw_constraint,
                })
                continue
        if constraint == "keywords:letter frequency":
            requirement = _extract_letter_frequency_requirement(prompt)
            if requirement is not None:
                specs.append({
                    "kind": "sentence_letter_frequency",
                    "letter": requirement["letter"],
                    "count": requirement["count"],
                    "scope": requirement["scope"],
                    "relation": requirement.get("relation") or "at least",
                    "source": raw_constraint,
                })
                continue
            term_requirement = _extract_sentence_term_frequency_requirement(prompt)
            if term_requirement is not None:
                specs.append({
                    "kind": "sentence_term_frequency",
                    "term": term_requirement["term"],
                    "count": term_requirement["count"],
                    "scope": term_requirement["scope"],
                    "relation": term_requirement.get("relation") or "at least",
                    "source": raw_constraint,
                })
                continue
        if constraint == "punctuation:use no comma":
            specs.append({"kind": "punctuation_forbidden", "value": ",", "source": raw_constraint})
            continue
        if constraint == "repeat the prompt":
            specs.append({"kind": "repeat_prompt", "value": prompt, "source": raw_constraint})
            continue
        if constraint == "use quotation":
            specs.append({"kind": "quotation", "source": raw_constraint})
            continue
        if constraint == "format:title":
            specs.append({"kind": "title", "source": raw_constraint})
            continue
        if constraint == "give two responses":
            specs.append({"kind": "two_responses", "source": raw_constraint})
            continue
        if constraint == "keywords:exclude words":
            continue
        specs.append({
            "kind": "__unsupported__",
            "source": raw_constraint,
        })
    return specs


def _evaluate_specs(specs: List[Dict[str, Any]], answer: str) -> Dict[str, Any]:
    unsupported = [spec["source"] for spec in specs if spec.get("kind") == "__unsupported__"]
    supported_specs = [spec for spec in specs if spec.get("kind") != "__unsupported__"]

    if not supported_specs:
        return {
            "judge_type": "instruction_following",
            "judge_backend": SUPPORTED_BACKEND,
            "is_correct": True,
            "judge_status": "pass",
            "judge_detail": {
                "unsupported_constraints": unsupported or ["no_supported_constraints"],
                "evaluated_constraint_count": 0,
                "assumed_correct": True,
            },
        }

    results = []
    all_pass = True
    for spec in supported_specs:
        passed, detail = _check_constraint(spec, answer)
        results.append({
            "source": spec.get("source"),
            "kind": spec["kind"],
            "passed": passed,
            "detail": detail,
        })
        all_pass = all_pass and passed

    return {
        "judge_type": "instruction_following",
        "judge_backend": SUPPORTED_BACKEND,
        "is_correct": all_pass,
        "judge_status": "pass" if all_pass else "wrong_answer",
        "judge_detail": {
            "constraint_results": results,
            "unsupported_constraints": unsupported,
            "evaluated_constraint_count": len(supported_specs),
        },
    }


def _json_safe(value: Any) -> Any:
    try:
        json.dumps(value)
        return value
    except TypeError:
        if isinstance(value, dict):
            return {str(key): _json_safe(sub_value) for key, sub_value in value.items()}
        if isinstance(value, (list, tuple)):
            return [_json_safe(item) for item in value]
        if hasattr(value, "item"):
            try:
                return value.item()
            except Exception:
                pass
        return str(value)


def _parse_yulan_judge_spec(row_data: Dict[str, Any]) -> Optional[Dict[str, Any]]:
    raw_spec = row_data.get("judge_spec")
    if raw_spec is None:
        return None
    if isinstance(raw_spec, str):
        if not raw_spec.strip():
            return None
        try:
            raw_spec = json.loads(raw_spec)
        except json.JSONDecodeError:
            return {
                "__invalid__": True,
                "error": "invalid_json",
            }
    if not isinstance(raw_spec, dict):
        return {
            "__invalid__": True,
            "error": f"invalid_type:{type(raw_spec).__name__}",
        }
    return raw_spec


def _run_lm_eval_ifeval(prompt: str, response: str, key: str,
                        spec: Dict[str, Any]) -> Dict[str, Any]:
    os.environ.setdefault("LOCAL_RANK", "1")
    from lm_eval.tasks.ifeval.utils import process_results

    kwargs = spec.get("kwargs") or []
    normalized_kwargs = [
        item if isinstance(item, dict) else {} for item in kwargs
    ]
    doc = {
        "key": key,
        "prompt": prompt,
        "instruction_id_list": spec.get("instruction_id_list") or [],
        "kwargs": normalized_kwargs,
    }
    return process_results(doc, [response])


def _spec_from_yulan_instruction_id(instruction_id: str,
                                    kwargs: Optional[Dict[str, Any]],
                                    prompt: str) -> Optional[Dict[str, Any]]:
    spec = _spec_from_instruction_id(instruction_id, kwargs)
    if spec is not None:
        return spec

    kwargs = kwargs or {}
    if instruction_id in {"paragraphs:paragraphs", "paragraphs:paragraphs2"}:
        count = _extract_numbered_requirement(prompt, "paragraph")
        if count is not None:
            return {
                "kind": "num_paragraphs",
                "value": count,
                "source": instruction_id,
            }

    if instruction_id == "detectable_format:constrained_response":
        options = _extract_choose_one_options(prompt)
        if options:
            return {
                "kind": "choose_one_option",
                "values": options,
                "source": instruction_id,
            }

    if instruction_id == "keywords:keyword_specific_position":
        return {
            "kind": "keyword_specific_position",
            "source": instruction_id,
            "keyword": kwargs.get("keyword"),
            "sentence_index": kwargs.get("n"),
            "word_index": kwargs.get("m"),
        }

    return None


def _yulan_ifbench_specs(row_data: Dict[str, Any],
                         spec: Dict[str, Any]) -> List[Dict[str, Any]]:
    prompt = row_data.get("prompt")
    prompt = prompt if isinstance(prompt, str) else ""
    instruction_ids = spec.get("instruction_id_list") or []
    kwargs_list = spec.get("kwargs") or []
    specs: List[Dict[str, Any]] = []
    for idx, instruction_id in enumerate(instruction_ids):
        kwargs = kwargs_list[idx] if idx < len(kwargs_list) else None
        parsed = _spec_from_yulan_instruction_id(str(instruction_id), kwargs, prompt)
        if parsed is None:
            specs.append({
                "kind": "__unsupported__",
                "source": str(instruction_id),
            })
        else:
            specs.append(parsed)
    return specs


def _evaluate_specs_strict(specs: List[Dict[str, Any]],
                           answer: str) -> Dict[str, Any]:
    unsupported = [spec for spec in specs if spec.get("kind") == "__unsupported__"]
    supported_specs = [spec for spec in specs if spec.get("kind") != "__unsupported__"]
    results = []
    all_pass = not unsupported and bool(supported_specs)

    for spec in supported_specs:
        try:
            passed, detail = _check_constraint(spec, answer)
        except Exception as exc:
            passed = False
            detail = {
                "reason": "checker_error",
                "error_type": type(exc).__name__,
                "error": str(exc),
            }
        results.append({
            "source": spec.get("source"),
            "kind": spec["kind"],
            "passed": passed,
            "detail": detail,
        })
        all_pass = all_pass and passed

    return {
        "judge_type": "instruction_following",
        "judge_backend": "ifbench_rule_v1",
        "is_correct": all_pass,
        "judge_status": (
            "unsupported" if unsupported else
            "pass" if all_pass else "wrong_answer"
        ),
        "judge_detail": {
            "constraint_results": results,
            "unsupported_constraints": [
                {
                    "source": spec.get("source"),
                    "reason": spec.get("reason"),
                } for spec in unsupported
            ],
            "evaluated_constraint_count": len(supported_specs),
        },
    }


def _judge_yulan_ifeval_spec(row_data: Dict[str, Any],
                             answer: str) -> Optional[Dict[str, Any]]:
    spec = _parse_yulan_judge_spec(row_data)
    if spec is None:
        return None

    if spec.get("__invalid__"):
        return {
            "judge_type": "instruction_following",
            "judge_backend": YULAN_IFEVAL_BACKEND,
            "is_correct": False,
            "judge_status": "failed",
            "judge_detail": {
                "reason": "invalid_judge_spec",
                "error": spec.get("error"),
            },
        }

    suite = str(spec.get("suite") or "")
    if suite == "ifbench":
        result = _evaluate_specs_strict(
            _yulan_ifbench_specs(row_data, spec),
            answer,
        )
        result["judge_detail"]["suite"] = suite
        return result

    if suite not in YULAN_IFEVAL_SUITES:
        return {
            "judge_type": "instruction_following",
            "judge_backend": YULAN_IFEVAL_BACKEND,
            "is_correct": False,
            "judge_status": "unsupported",
            "judge_detail": {
                "reason": "unsupported_judge_suite",
                "suite": suite,
                "supported_suites": sorted(YULAN_IFEVAL_SUITES),
                "instruction_id_list": spec.get("instruction_id_list") or [],
            },
        }

    prompt = row_data.get("prompt")
    if not isinstance(prompt, str) or not prompt.strip():
        return {
            "judge_type": "instruction_following",
            "judge_backend": YULAN_IFEVAL_BACKEND,
            "is_correct": False,
            "judge_status": "failed",
            "judge_detail": {
                "reason": "missing_prompt",
                "suite": suite,
            },
        }

    try:
        key = str(row_data.get("id") or row_data.get("sample_id")
                  or row_data.get("source_row") or prompt)
        raw_result = _run_lm_eval_ifeval(prompt, answer, key, spec)
    except ImportError as exc:
        return {
            "judge_type": "instruction_following",
            "judge_backend": YULAN_IFEVAL_BACKEND,
            "is_correct": None,
            "judge_status": "missing_dependency",
            "judge_detail": {
                "reason": "missing_lm_eval",
                "error": str(exc),
                "install": "pip install lm-eval==0.4.12",
                "suite": suite,
            },
        }
    except Exception as exc:
        return {
            "judge_type": "instruction_following",
            "judge_backend": YULAN_IFEVAL_BACKEND,
            "is_correct": False,
            "judge_status": "failed",
            "judge_detail": {
                "reason": "lm_eval_ifeval_error",
                "error_type": type(exc).__name__,
                "error": str(exc),
                "suite": suite,
            },
        }

    strict = bool(spec.get("strict", True))
    pass_key = "prompt_level_strict_acc" if strict else "prompt_level_loose_acc"
    is_correct = bool(raw_result.get(pass_key))
    return {
        "judge_type": "instruction_following",
        "judge_backend": YULAN_IFEVAL_BACKEND,
        "is_correct": is_correct,
        "judge_status": "pass" if is_correct else "wrong_answer",
        "judge_detail": {
            "suite": suite,
            "strict": strict,
            "pass_key": pass_key,
            "instruction_id_list": spec.get("instruction_id_list") or [],
            "raw": _json_safe(raw_result),
        },
    }


def instruction_following_hint(row_data: Dict[str, Any]) -> Optional[str]:
    if _parse_yulan_judge_spec(row_data) is not None:
        return "instruction_following"
    if _structured_constraint_specs(row_data) is not None:
        return "instruction_following"
    if _prompt_constraint_specs(row_data) is not None:
        return "instruction_following"
    return None


def judge_instruction_following(row_data: Dict[str, Any],
                                messages: List[Dict[str, Any]]) -> Optional[Dict[str, Any]]:
    yulan_result = _judge_yulan_ifeval_spec(row_data, assistant_text(messages))
    if yulan_result is not None:
        return yulan_result

    specs = _structured_constraint_specs(row_data)
    if specs is None:
        specs = _prompt_constraint_specs(row_data)
    if specs is None:
        return None
    return _evaluate_specs(specs, assistant_text(messages))
