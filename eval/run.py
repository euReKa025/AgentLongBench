"""Unified run entry for standardized AgentLong datasets."""

from __future__ import annotations

import argparse
import json
import sys
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from typing import Any, Dict, List, Optional, Set, Tuple

from dotenv import load_dotenv
from tqdm import tqdm

from models import ModelManager
from models.runner_settings import RUNNER_SETTINGS

from eval.common.io_utils import load_jsonl
from eval.common.mapping import infer_context_from_path, require_single_question_type
from eval.common.question_logic import build_prompt, parse_response


def _load_completed_ids(output_path: Path) -> Set[str]:
    if not output_path.exists():
        return set()
    completed = set()
    with output_path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                obj = json.loads(line)
            except json.JSONDecodeError:
                continue
            raw = obj.get("raw_response", "")
            sid = obj.get("id")
            if sid is not None and str(raw).strip():
                completed.add(str(sid))
    return completed


def _load_existing_records(output_path: Path) -> Dict[str, Dict[str, Any]]:
    if not output_path.exists():
        return {}
    records: Dict[str, Dict[str, Any]] = {}
    with output_path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                obj = json.loads(line)
            except json.JSONDecodeError:
                continue
            sid = obj.get("id")
            if sid is not None:
                records[str(sid)] = obj
    return records


def _predict(
    manager: ModelManager,
    messages: List[Dict[str, str]],
    stream: bool,
) -> str:
    resp = manager.chat_completion(messages, stream=stream, **(RUNNER_SETTINGS.extra_params or {}))
    if stream:
        content = ""
        try:
            for chunk in resp:
                choices = chunk.get("choices", [])
                if choices:
                    delta = choices[0].get("delta", {})
                    content += (delta.get("content") or "")
        except Exception as exc:
            print(f"Stream error: {exc}")
            raise
        return content
    if resp.get("choices"):
        return resp["choices"][0].get("message", {}).get("content", "")
    return ""


def _process_sample(
    sample: Dict[str, Any],
    question_type: str,
    history_label: str,
    knowledge_label: str,
    service_name: str,
    stream: bool,
) -> Dict[str, Any]:
    manager = ModelManager(default_service=service_name)
    messages = build_prompt(question_type, sample, history_label, knowledge_label)
    raw_content = _predict(manager, messages, stream)
    pred_value, parse_kind = parse_response(question_type, history_label, raw_content)
    record: Dict[str, Any] = {
        "id": sample.get("id"),
        "sample_id": sample.get("sample_id"),
        "question_type": question_type,
        "pred_answer": pred_value,
        "parse_kind": parse_kind,
        "raw_response": raw_content,
    }
    for key in ("round", "i_round", "j_round"):
        if key in sample:
            record[key] = sample[key]
    return record


def run(
    dataset_path: Path,
    output_path: Path,
    *,
    offset: int = 0,
    limit: Optional[int] = None,
    service_name: Optional[str] = None,
    workers: int = 1,
    resume: bool = True,
    stream: bool = True,
) -> int:
    rows = load_jsonl(dataset_path)
    if limit is None:
        rows = rows[offset:]
    else:
        rows = rows[offset : offset + limit]

    question_type = require_single_question_type(rows)
    _, knowledge_label, _, history_label = infer_context_from_path(dataset_path)

    service = service_name or RUNNER_SETTINGS.service_name
    output_path.parent.mkdir(parents=True, exist_ok=True)

    existing_records = _load_existing_records(output_path) if resume else {}
    completed_ids: Set[str] = set()
    if resume:
        completed_ids = _load_completed_ids(output_path)
        print(f"Resume mode: found {len(completed_ids)} completed IDs")

    pending_rows = []
    for sample in rows:
        sid = sample.get("id")
        if sid is not None and str(sid) in completed_ids:
            continue
        pending_rows.append(sample)

    if not pending_rows:
        print("All samples already completed, nothing to do.")
        return 0

    print(f"Processing {len(pending_rows)} samples with {workers} worker(s)...")

    output_records = existing_records.copy()
    written = 0

    if workers == 1:
        for sample in tqdm(pending_rows, desc="Predicting answers"):
            record = _process_sample(sample, question_type, history_label, knowledge_label, service, stream)
            sid = sample.get("id")
            if sid is not None:
                output_records[str(sid)] = record
            written += 1
    else:
        with ThreadPoolExecutor(max_workers=workers) as executor:
            future_to_sample = {
                executor.submit(
                    _process_sample, sample, question_type, history_label, knowledge_label, service, stream
                ): sample
                for sample in pending_rows
            }
            with tqdm(total=len(pending_rows), desc="Predicting answers") as pbar:
                for future in as_completed(future_to_sample):
                    try:
                        record = future.result()
                        sample = future_to_sample[future]
                        sid = sample.get("id")
                        if sid is not None:
                            output_records[str(sid)] = record
                        written += 1
                        pbar.update(1)
                    except Exception as exc:
                        sample = future_to_sample[future]
                        print(f"\nError processing sample {sample.get('id')}: {exc}", file=sys.stderr)
                        pbar.update(1)

    with output_path.open("w", encoding="utf-8") as f:
        for record in output_records.values():
            f.write(json.dumps(record, ensure_ascii=False) + "\n")

    return written


def main() -> None:
    load_dotenv()
    parser = argparse.ArgumentParser(description="Unified run for standardized AgentLong datasets.")
    parser.add_argument("--dataset", type=Path, required=True, help="Dataset JSONL file.")
    parser.add_argument("--output", type=Path, required=True, help="Output prediction JSONL file.")
    parser.add_argument("--offset", type=int, default=0, help="Start from this sample index.")
    parser.add_argument("--limit", type=int, default=None, help="Optional cap on number of samples.")
    parser.add_argument("--service", type=str, default=None, help="Override service_name for ModelManager.")
    parser.add_argument("--workers", type=int, default=1, help="Number of parallel workers.")
    parser.add_argument("--no-resume", action="store_true", help="Disable resume mode.")
    parser.add_argument("--no-stream", action="store_false", dest="stream", default=True, help="Disable streaming.")
    args = parser.parse_args()

    count = run(
        args.dataset,
        args.output,
        offset=args.offset,
        limit=args.limit,
        service_name=args.service,
        workers=args.workers,
        resume=not args.no_resume,
        stream=args.stream,
    )
    print(f"Wrote {count} new predictions -> {args.output}")


if __name__ == "__main__":
    main()
