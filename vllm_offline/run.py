"""Unified offline vLLM runner for standardized AgentLong datasets."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any, Dict, Optional

from tqdm import tqdm

from eval.common.mapping import infer_context_from_path, require_single_question_type
from eval.common.question_logic import build_prompt, parse_response
from vllm_offline.common import (
    filter_pending,
    load_completed_ids,
    load_jsonl,
    open_output_file,
    slice_rows,
)
from vllm_offline.offline_llm import OfflineVLLMChat, VLLMConfig


def run(
    dataset_path: Path,
    output_path: Path,
    *,
    offset: int = 0,
    limit: Optional[int] = None,
    resume: bool = True,
    model: str,
    tokenizer: Optional[str],
    max_new_tokens: int,
    temperature: float,
    top_p: float,
    tensor_parallel_size: int,
    pipeline_parallel_size: int,
    dtype: str,
    gpu_memory_utilization: float,
    trust_remote_code: bool,
) -> int:
    rows = load_jsonl(dataset_path)
    rows = slice_rows(rows, offset=offset, limit=limit)

    question_type = require_single_question_type(rows)
    _, knowledge_label, _, history_label = infer_context_from_path(dataset_path)

    completed = load_completed_ids(output_path) if resume else set()
    pending = filter_pending(rows, completed)
    if not pending:
        print("All samples already completed, nothing to do.")
        return 0

    cfg = VLLMConfig(
        model=model,
        tokenizer=tokenizer,
        max_new_tokens=max_new_tokens,
        temperature=temperature,
        top_p=top_p,
        tensor_parallel_size=tensor_parallel_size,
        pipeline_parallel_size=pipeline_parallel_size,
        dtype=dtype,
        gpu_memory_utilization=gpu_memory_utilization,
        trust_remote_code=trust_remote_code,
    )
    llm = OfflineVLLMChat(cfg)

    written = 0
    with open_output_file(output_path, resume) as f:
        pbar = tqdm(total=len(pending), desc="Predicting (vLLM batch)")
        batch_messages = [
            build_prompt(question_type, sample, history_label, knowledge_label) for sample in pending
        ]
        batch_responses = llm.chat_batch(batch_messages)

        for sample, raw_response in zip(pending, batch_responses):
            pred_value, parse_kind = parse_response(question_type, history_label, raw_response)
            record: Dict[str, Any] = {
                "id": sample.get("id"),
                "sample_id": sample.get("sample_id"),
                "question_type": question_type,
                "pred_answer": pred_value,
                "parse_kind": parse_kind,
                "raw_response": raw_response,
            }
            for key in ("round", "i_round", "j_round"):
                if key in sample:
                    record[key] = sample[key]
            f.write(json.dumps(record, ensure_ascii=False) + "\n")
            written += 1
            pbar.update(1)
        f.flush()
        pbar.close()

    return written


def main() -> None:
    parser = argparse.ArgumentParser(description="Unified offline vLLM runner")
    parser.add_argument("--dataset", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--offset", type=int, default=0)
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument("--no-resume", action="store_true")

    parser.add_argument("--model", required=True, help="Local model path or name for vLLM.")
    parser.add_argument("--tokenizer", default=None, help="Optional tokenizer name/path.")
    parser.add_argument("--max-new-tokens", type=int, default=256)
    parser.add_argument("--temperature", type=float, default=0.2)
    parser.add_argument("--top-p", type=float, default=0.9)
    parser.add_argument("--tensor-parallel-size", type=int, default=1)
    parser.add_argument("--pipeline-parallel-size", "--pp", type=int, default=1)
    parser.add_argument("--dtype", type=str, default="auto")
    parser.add_argument("--gpu-memory-utilization", type=float, default=0.9)
    parser.add_argument("--trust-remote-code", action="store_true")

    args = parser.parse_args()
    written = run(
        args.dataset,
        args.output,
        offset=args.offset,
        limit=args.limit,
        resume=not args.no_resume,
        model=args.model,
        tokenizer=args.tokenizer,
        max_new_tokens=args.max_new_tokens,
        temperature=args.temperature,
        top_p=args.top_p,
        tensor_parallel_size=args.tensor_parallel_size,
        pipeline_parallel_size=args.pipeline_parallel_size,
        dtype=args.dtype,
        gpu_memory_utilization=args.gpu_memory_utilization,
        trust_remote_code=args.trust_remote_code,
    )
    print(f"Wrote {written} predictions -> {args.output}")


if __name__ == "__main__":
    main()
