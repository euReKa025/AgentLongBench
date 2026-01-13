# AgentLongBench

Unified benchmark and evaluation codebase for AgentLong, standardizing labels and data layout.
This folder is standalone.

## What this contains

- `benchmark/`: standardized datasets organized by knowledge type + history type + length + category.
- `eval/`: single entry run/evaluate logic using standardized labels.
- `vllm_offline/`: unified offline runner for vLLM.
- `scripts/`: conversion and helper scripts.
- `models/`: copied model registry/configs used by the runner.

## Standard labels (paper format)

Knowledge types:
- `ki` -> knowledge_intensive
- `kf` -> knowledge_free

History types:
- `c` -> Concise-Response
- `v` -> Verbose-Response

Question types:
- Count Frequency(Tool)
- Find Duplicates(Tool)
- Find Target Offsets(Tool)
- Count Correctness(Env)
- Count Frequency(Env)
- Find Round with Largest Value(Env)
- Weighted Summation(Env)
- Intersection

Categories:
- Tool Response: Count Frequency(Tool), Find Duplicates(Tool), Find Target Offsets(Tool)
- Env Response: Count Correctness(Env), Count Frequency(Env), Find Round with Largest Value(Env), Weighted Summation(Env)
- Final Guess: Intersection

## Data layout

```
benchmark/
  ki-c/ or ki-v/ or kf-c/ or kf-v/
    <length>/
      tool_response/
        <question_type_slug>.jsonl
      env_response/
        <question_type_slug>.jsonl
      final_guess/
        <question_type_slug>.jsonl
```

Each JSONL file contains a single standardized `question_type` only.

## Benchmark download

The `benchmark/` directory is large and not included in the repository. Use the provided download link https://huggingface.co/datasets/ign1s/AgentLongBench
to fetch the dataset and place it under `agentlong_bench/benchmark/`.

## Key scripts

- `scripts/convert_benchmark.py` converts raw benchmark data into the standardized layout under
  `agentlong_bench/benchmark`.
- `eval/run.py` runs inference for a single standardized dataset file (online API runner).
- `eval/evaluate.py` evaluates a single dataset file and its predictions.
- `vllm_offline/run.py` runs offline inference with vLLM for a single dataset file.

## Prediction format (output JSONL)

Each line includes:
- `id`, `sample_id`, `question_type`
- `pred_answer` (parsed prediction)
- `parse_kind` (parser tag)
- `raw_response` (model text)
- optional `round`, `i_round`, `j_round` if present in the dataset

## Notes

- Prompting logic is aligned with the original task definitions, selected by knowledge type.
- Evaluation expects one standardized `question_type` per dataset file.
