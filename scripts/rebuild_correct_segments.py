import argparse
import json
import logging
import shutil
import re
from pathlib import Path


logger = logging.getLogger("rebuild_correct_segments")


def configure_logging(log_path: Path | None) -> None:
    handlers = [logging.StreamHandler()]
    if log_path is not None:
        log_path.parent.mkdir(parents=True, exist_ok=True)
        handlers.append(logging.FileHandler(log_path, encoding="utf-8"))
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s | %(levelname)s | %(message)s",
        handlers=handlers,
        force=True,
    )


def iter_segment_paths(segment_dir: Path):
    return sorted(segment_dir.glob("segment_*.jsonl"))


def segment_index(path: Path) -> int:
    match = re.search(r"segment_(\d+)\.jsonl$", path.name)
    if not match:
        return -1
    return int(match.group(1))


def sync_resume_state(output_root: Path,
                      scanned_records: int,
                      written_records: int,
                      overlong_records: int,
                      all_segment_dir: Path,
                      correct_segment_dir: Path,
                      completed_lines: list[str]) -> dict[str, int]:
    resume_dir = output_root / ".resume"
    resume_dir.mkdir(parents=True, exist_ok=True)

    completed_index_path = resume_dir / "completed_index.jsonl"
    completed_tmp_path = resume_dir / "completed_index.jsonl.tmp"
    with completed_tmp_path.open("w", encoding="utf-8") as f:
        for line in completed_lines:
            f.write(line)
    completed_tmp_path.replace(completed_index_path)

    all_next_segment_idx = max(
        (segment_index(path) for path in iter_segment_paths(all_segment_dir)),
        default=-1,
    ) + 1
    correct_next_segment_idx = max(
        (segment_index(path) for path in iter_segment_paths(correct_segment_dir)),
        default=-1,
    ) + 1

    resume_state = {
        "progress": {
            "written": scanned_records,
            "correct": written_records,
            "overlong": overlong_records,
        },
        "streams": {
            "all": {
                "next_segment_idx": all_next_segment_idx,
                "next_shard_idx": 0,
            },
            "correct": {
                "next_segment_idx": correct_next_segment_idx,
                "next_shard_idx": 0,
            },
        },
        "version": 2,
    }
    resume_state_path = resume_dir / "resume_state.json"
    resume_tmp_path = resume_dir / "resume_state.json.tmp"
    resume_tmp_path.write_text(
        json.dumps(resume_state, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    resume_tmp_path.replace(resume_state_path)
    logger.info("Synchronized resume state at %s", resume_state_path)

    return {
        "resume_written": scanned_records,
        "resume_correct": written_records,
        "resume_overlong": overlong_records,
        "resume_all_next_segment_idx": all_next_segment_idx,
        "resume_correct_next_segment_idx": correct_next_segment_idx,
    }


def rebuild_correct(all_segment_dir: Path,
                    correct_segment_dir: Path,
                    backup_dir: Path | None = None,
                    sync_resume: bool = False) -> dict[str, int]:
    if not all_segment_dir.exists():
        raise FileNotFoundError(f"all segment dir not found: {all_segment_dir}")

    if backup_dir is not None and correct_segment_dir.exists():
        if backup_dir.exists():
            shutil.rmtree(backup_dir)
        backup_dir.parent.mkdir(parents=True, exist_ok=True)
        shutil.move(str(correct_segment_dir), str(backup_dir))
        logger.info("Backed up existing correct segments to %s", backup_dir)
    elif correct_segment_dir.exists():
        shutil.rmtree(correct_segment_dir)
        logger.info("Removed existing correct segments at %s", correct_segment_dir)

    correct_segment_dir.mkdir(parents=True, exist_ok=True)

    written_segments = 0
    written_records = 0
    scanned_segments = 0
    scanned_records = 0
    overlong_records = 0
    completed_lines: list[str] = []

    for segment_path in iter_segment_paths(all_segment_dir):
        scanned_segments += 1
        correct_records = []
        with segment_path.open("r", encoding="utf-8") as src:
            for line in src:
                if not line.strip():
                    continue
                scanned_records += 1
                record = json.loads(line)
                completed_lines.append(
                    f"{record['source_file']}\t{int(record['source_row'])}\t{int(record.get('rollout_index', 0) or 0)}\n")
                if record.get("generation_finish_reason") == "length":
                    overlong_records += 1
                if record.get("is_correct") is True:
                    correct_records.append(record)

        if not correct_records:
            continue

        target_path = correct_segment_dir / segment_path.name
        with target_path.open("w", encoding="utf-8") as dst:
            for record in correct_records:
                dst.write(json.dumps(record, ensure_ascii=False) + "\n")
        written_segments += 1
        written_records += len(correct_records)

        logger.info(
            "Wrote %s with %s correct records",
            target_path.name,
            len(correct_records),
        )

    summary = {
        "scanned_segments": scanned_segments,
        "scanned_records": scanned_records,
        "written_segments": written_segments,
        "written_records": written_records,
        "overlong_records": overlong_records,
    }
    if sync_resume:
        output_root = all_segment_dir.parent.parent
        summary.update(
            sync_resume_state(
                output_root=output_root,
                scanned_records=scanned_records,
                written_records=written_records,
                overlong_records=overlong_records,
                all_segment_dir=all_segment_dir,
                correct_segment_dir=correct_segment_dir,
                completed_lines=completed_lines,
            ))
    logger.info(
        "Finished rebuild: scanned_segments=%s scanned_records=%s written_segments=%s written_records=%s overlong_records=%s",
        scanned_segments,
        scanned_records,
        written_segments,
        written_records,
        overlong_records,
    )
    return summary


def main():
    parser = argparse.ArgumentParser(
        description="Rebuild correct/segments from all/segments using is_correct=true.")
    parser.add_argument("--all-segment-dir", required=True, type=Path)
    parser.add_argument("--correct-segment-dir", required=True, type=Path)
    parser.add_argument("--backup-dir", type=Path)
    parser.add_argument("--log-path", type=Path)
    parser.add_argument("--summary-path", type=Path)
    parser.add_argument("--sync-resume", action="store_true")
    args = parser.parse_args()

    configure_logging(args.log_path)
    summary = rebuild_correct(
        all_segment_dir=args.all_segment_dir,
        correct_segment_dir=args.correct_segment_dir,
        backup_dir=args.backup_dir,
        sync_resume=args.sync_resume,
    )
    if args.summary_path is not None:
        args.summary_path.parent.mkdir(parents=True, exist_ok=True)
        args.summary_path.write_text(
            json.dumps(summary, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
            encoding="utf-8",
        )
        logger.info("Wrote summary to %s", args.summary_path)
    print(json.dumps(summary, ensure_ascii=False, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
