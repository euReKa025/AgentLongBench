"""Unified evaluation entry for standardized AgentLong benchmark."""

from __future__ import annotations

import argparse
import re
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

from eval.common.io_utils import load_jsonl
from eval.common.mapping import (
    COUNT_FREQUENCY_TOOL,
    FIND_DUPLICATES_TOOL,
    FIND_TARGET_OFFSETS_TOOL,
    COUNT_CORRECTNESS_ENV,
    COUNT_FREQUENCY_ENV,
    FIND_ROUND_LARGEST_VALUE_ENV,
    WEIGHTED_SUMMATION_ENV,
    INTERSECTION,
    infer_context_from_path,
    require_single_question_type,
)


def _to_number(val: Any) -> Optional[int]:
    if val is None:
        return None
    if isinstance(val, int):
        return val
    if isinstance(val, float):
        return int(val)
    text = str(val)
    match = re.search(r"-?\d+(?:\.\d+)?", text)
    if not match:
        return None
    try:
        return int(float(match.group(0)))
    except ValueError:
        return None


def _normalize_name(name: str, *, knowledge_label: str) -> str:
    if not name:
        return ""
    normalized = str(name).strip().lower()
    if knowledge_label == "knowledge_free":
        return (
            normalized.replace(" ", "")
            .replace("-", "")
            .replace("_", "")
            .replace("'", "")
            .replace('"', "")
            .replace(".", "")
        )
    return (
        normalized.replace(" ", "")
        .replace("-", "")
        .replace("'", "")
        .replace('"', "")
        .replace(".", "")
    )


def _normalize_boolean(val: Any) -> Optional[bool]:
    if val is None:
        return None
    if isinstance(val, bool):
        return val
    if isinstance(val, int):
        return val > 0
    text = str(val).lower().strip()
    if any(word in text for word in ["yes", "true", "1"]):
        return True
    if any(word in text for word in ["no", "false", "0"]):
        return False
    return None


def _normalize_pair_list(val: Any, *, knowledge_label: str) -> Optional[List[str]]:
    if val is None:
        return None
    if isinstance(val, list):
        return [_normalize_name(name, knowledge_label=knowledge_label) for name in val if isinstance(name, str)]
    if isinstance(val, str):
        text = val.strip("[](){}")
        parts = re.split(r"[,;]|\s+and\s+", text, flags=re.IGNORECASE)
        if len(parts) >= 2:
            return [_normalize_name(p.strip(), knowledge_label=knowledge_label) for p in parts]
    return None


def _compare_pair_lists(pred: List[str], gt: List[str]) -> float:
    if not pred or not gt:
        return 0.0
    if len(pred) == 2 and len(gt) == 2:
        if pred[0] == gt[0] and pred[1] == gt[1]:
            return 1.0
    if len(pred) == 1 and len(gt) == 2:
        if pred[0] == gt[0]:
            return 0.5
    return 0.0


def _to_set(val: Any) -> List[str]:
    if val is None:
        return []
    if isinstance(val, list):
        return [str(v).strip() for v in val if str(v).strip()]
    if isinstance(val, str):
        parts = [v.strip() for chunk in val.replace("\n", ",").split(",") for v in chunk.split() if v.strip()]
        return parts
    return []


def _pred_key(pred: Dict[str, Any]) -> Optional[str]:
    sid = pred.get("id")
    if sid is not None:
        return str(sid)
    sample_id = pred.get("sample_id")
    round_id = pred.get("round")
    if sample_id is not None and round_id is not None:
        return f"{sample_id}_r{round_id}"
    return None


def evaluate(dataset_path: Path, pred_path: Path, verbose: bool = False) -> Dict[str, Any]:
    rows = load_jsonl(dataset_path)
    preds = load_jsonl(pred_path)

    question_type = require_single_question_type(rows)
    knowledge_key, knowledge_label, _, history_label = infer_context_from_path(dataset_path)

    data = {row["id"]: row for row in rows}

    matched = 0
    total = 0
    correct = 0.0

    if question_type == INTERSECTION and history_label == "Verbose-Response":
        f1s: List[float] = []
        for pred in preds:
            key = _pred_key(pred)
            if not key or key not in data:
                continue
            matched += 1
            gt_set = set(_to_set(data[key].get("answer")))
            pred_val = pred.get("pred_answer")
            if pred_val is None:
                pred_val = pred.get("pred_intersection")
            pred_set = set(_to_set(pred_val))
            if not gt_set and not pred_set:
                f1 = 1.0
            elif not gt_set or not pred_set:
                f1 = 0.0
            else:
                inter = len(gt_set & pred_set)
                prec = inter / len(pred_set) if pred_set else 0.0
                rec = inter / len(gt_set) if gt_set else 0.0
                f1 = 2 * prec * rec / (prec + rec) if (prec + rec) else 0.0
            if verbose:
                print(f"Sample {key}: GT={gt_set}, Pred={pred_set}, F1={f1:.4f}")
            f1s.append(f1)
        avg_f1 = sum(f1s) / len(f1s) if f1s else 0.0
        return {
            "question_type": question_type,
            "knowledge_type": knowledge_label,
            "history_type": history_label,
            "metric": "f1",
            "score": avg_f1,
            "matched": matched,
            "skipped": len(preds) - matched,
        }

    for pred in preds:
        key = _pred_key(pred)
        if not key or key not in data:
            continue
        matched += 1
        gt_data = data[key]
        gt = gt_data.get("answer")
        pred_val = pred.get("pred_answer")

        if question_type in {
            COUNT_FREQUENCY_TOOL,
            COUNT_CORRECTNESS_ENV,
            COUNT_FREQUENCY_ENV,
            FIND_ROUND_LARGEST_VALUE_ENV,
            WEIGHTED_SUMMATION_ENV,
        }:
            gt_num = _to_number(gt)
            pred_num = _to_number(pred_val)
            if gt_num is None:
                continue
            total += 1
            if pred_num is not None and pred_num == gt_num:
                correct += 1
            elif verbose:
                print(f"Sample {key}: GT={gt}, Pred={pred_val}")
            continue

        if question_type == FIND_DUPLICATES_TOOL:
            gt_bool = _normalize_boolean(gt)
            pred_bool = _normalize_boolean(pred_val)
            if gt_bool is None:
                continue
            total += 1
            if pred_bool is not None and pred_bool == gt_bool:
                correct += 1
            elif verbose:
                print(f"Sample {key}: GT={gt}, Pred={pred_val}")
            continue

        if question_type == FIND_TARGET_OFFSETS_TOOL:
            gt_list = _normalize_pair_list(gt, knowledge_label=knowledge_label)
            pred_list = _normalize_pair_list(pred_val, knowledge_label=knowledge_label)
            if gt_list is None:
                continue
            total += 1
            score = 0.0
            if pred_list is not None:
                score = _compare_pair_lists(pred_list, gt_list)
            correct += score
            if verbose:
                print(f"Sample {key}: GT={gt}, Pred={pred_val}, score={score}")
            continue

        if question_type == INTERSECTION and history_label == "Concise-Response":
            gt_norm = _normalize_name(str(gt or ""), knowledge_label=knowledge_label)
            pred_norm = _normalize_name(str(pred_val or ""), knowledge_label=knowledge_label)
            if not gt_norm:
                continue
            total += 1
            if pred_norm == gt_norm:
                correct += 1
            elif verbose:
                print(f"Sample {key}: GT={gt}, Pred={pred_val}")
            continue

    score = correct / total if total > 0 else 0.0
    return {
        "question_type": question_type,
        "knowledge_type": knowledge_label,
        "history_type": history_label,
        "metric": "accuracy",
        "score": score,
        "correct": correct,
        "total": total,
        "matched": matched,
        "skipped": len(preds) - matched,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Unified evaluation for standardized AgentLong datasets.")
    parser.add_argument("--dataset", type=Path, required=True, help="Dataset JSONL file.")
    parser.add_argument("--pred", type=Path, required=True, help="Prediction JSONL file.")
    parser.add_argument("--verbose", action="store_true", help="Print per-sample info.")
    args = parser.parse_args()

    metrics = evaluate(args.dataset, args.pred, verbose=args.verbose)
    print(f"Question Type: {metrics['question_type']}")
    print(f"Knowledge Type: {metrics['knowledge_type']}")
    print(f"History Type: {metrics['history_type']}")
    print(f"Metric: {metrics['metric']}")
    print(f"Score: {metrics['score']:.4f}")
    if metrics.get("total") is not None:
        print(f"Matched: {metrics['matched']}")
        if metrics.get("skipped"):
            print(f"Skipped: {metrics['skipped']}")
        if metrics.get("total") is not None:
            print(f"Correct: {metrics.get('correct')}/{metrics.get('total')}")


if __name__ == "__main__":
    main()
