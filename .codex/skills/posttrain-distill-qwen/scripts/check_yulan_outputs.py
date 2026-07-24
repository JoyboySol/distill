#!/usr/bin/env python3
import collections
import glob
import hashlib
import json
import sys
import textwrap
from pathlib import Path


def main() -> int:
    if len(sys.argv) != 2:
        print("usage: check_yulan_outputs.py OUTPUT_ROOT", file=sys.stderr)
        return 2

    root = Path(sys.argv[1])
    files = sorted(glob.glob(str(root / "all" / "segments" / "segment_*.jsonl")))
    counts = collections.Counter()
    by_query = collections.defaultdict(list)
    total = system = reasoning = boxed = 0
    examples = []

    for file_path in files:
        with open(file_path, encoding="utf-8") as handle:
            for line in handle:
                if not line.strip():
                    continue
                record = json.loads(line)
                total += 1
                counts[str(record.get("is_correct"))] += 1
                messages = record.get("messages") or []
                if messages and messages[0].get("role") == "system":
                    system += 1
                assistants = [m for m in messages if m.get("role") == "assistant"]
                if assistants:
                    assistant = assistants[-1]
                    content = assistant.get("content") or ""
                    reason = assistant.get("reasoning_content") or ""
                    if reason:
                        reasoning += 1
                    if "\\boxed" in content:
                        boxed += 1
                    users = [m.get("content") for m in messages if m.get("role") == "user"]
                    if users:
                        by_query[users[-1]].append(hashlib.sha1(content.encode()).hexdigest())
                    if len(examples) < 3:
                        examples.append((record, reason, content))

    print(f"root={root}")
    print(f"segment_files={len(files)} total={total} counts={dict(counts)}")
    print(f"system={system} reasoning_content={reasoning} boxed={boxed}")
    if by_query:
        diversity = [len(set(values)) for values in by_query.values()]
        print(f"unique_outputs_by_input={diversity[:20]}")

    for idx, (record, reason, content) in enumerate(examples, 1):
        print(f"\nEXAMPLE {idx}")
        print(f"correct={record.get('is_correct')} detail={record.get('judge_detail')}")
        print(f"reasoning_len={len(reason)} content_len={len(content)}")
        print("reasoning_head=" + textwrap.shorten(reason.replace("\\n", " "), width=240))
        print("content=" + textwrap.shorten(content.replace("\\n", " "), width=360))

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
