"""Thin wrapper around vLLM for local chat-style inference."""

from __future__ import annotations

from dataclasses import dataclass
from typing import List, Optional, Sequence

from transformers import AutoTokenizer
from vllm import LLM, SamplingParams


@dataclass
class VLLMConfig:
    model: str
    tokenizer: Optional[str] = None
    max_new_tokens: int = 512
    temperature: float = 0.2
    top_p: float = 0.9
    tensor_parallel_size: int = 1
    pipeline_parallel_size: int = 1
    dtype: str = "auto"
    gpu_memory_utilization: float = 0.9
    trust_remote_code: bool = False
    stop: Optional[Sequence[str]] = None
    max_model_len: int = 1300000
    enable_chunked_prefill: bool = True
    enforce_eager: bool = True


class OfflineVLLMChat:
    """Minimal chat-completion style interface built on vLLM."""

    def __init__(self, cfg: VLLMConfig):
        self.cfg = cfg
        tokenizer_name = cfg.tokenizer or cfg.model
        self.tokenizer = AutoTokenizer.from_pretrained(
            tokenizer_name,
            trust_remote_code=cfg.trust_remote_code,
        )
        self.llm = LLM(
            model=cfg.model,
            tokenizer=tokenizer_name,
            tensor_parallel_size=cfg.tensor_parallel_size,
            pipeline_parallel_size=cfg.pipeline_parallel_size,
            dtype=cfg.dtype,
            gpu_memory_utilization=cfg.gpu_memory_utilization,
            trust_remote_code=cfg.trust_remote_code,
            enable_chunked_prefill=cfg.enable_chunked_prefill,
            enforce_eager=cfg.enforce_eager,
            max_model_len=cfg.max_model_len,
            max_num_batched_tokens=131072,
        )

    def chat(self, messages: List[dict], *, max_new_tokens: Optional[int] = None) -> str:
        prompt = self.tokenizer.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=True,
        )
        sampling_params = SamplingParams(
            temperature=self.cfg.temperature,
            top_p=self.cfg.top_p,
            max_tokens=max_new_tokens or self.cfg.max_new_tokens,
            stop=self.cfg.stop,
        )
        outputs = self.llm.generate([prompt], sampling_params=sampling_params)
        if not outputs or not outputs[0].outputs:
            return ""
        return outputs[0].outputs[0].text

    def chat_batch(self, messages_list: List[List[dict]], *, max_new_tokens: Optional[int] = None) -> List[str]:
        all_processed_histories = []

        for messages in messages_list:
            processed_history = []
            last_function_name = None

            for msg in messages:
                role = msg.get("role")
                if role == "assistant":
                    new_msg = {
                        "role": "assistant",
                        "content": msg.get("content", ""),
                    }
                    if "tool_calls" in msg and msg["tool_calls"]:
                        last_function_name = msg["tool_calls"][0]["function"]["name"]
                        new_msg["function_call"] = {
                            "name": last_function_name,
                            "arguments": msg["tool_calls"][0]["function"]["arguments"],
                        }
                    processed_history.append(new_msg)
                elif role == "tool":
                    processed_history.append(
                        {
                            "role": "function",
                            "name": last_function_name,
                            "content": msg.get("content"),
                        }
                    )
                else:
                    processed_history.append(msg)

            all_processed_histories.append(processed_history)

        prompts = [
            self.tokenizer.apply_chat_template(
                msgs,
                tokenize=False,
                add_generation_prompt=True,
            )
            for msgs in all_processed_histories
        ]

        sampling_params = SamplingParams(
            temperature=self.cfg.temperature,
            top_p=self.cfg.top_p,
            max_tokens=max_new_tokens or self.cfg.max_new_tokens,
            stop=self.cfg.stop,
        )

        outputs = self.llm.generate(prompts, sampling_params=sampling_params)

        results = []
        for output in outputs:
            if output.outputs:
                results.append(output.outputs[0].text)
            else:
                results.append("")
        return results
