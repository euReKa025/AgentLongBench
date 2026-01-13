from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, Optional


@dataclass
class RunnerSettings:
    service_name: Optional[str] = None
    temperature: float = 0.7
    max_tokens: Optional[int] = None
    extra_params: Dict[str, Any] = field(default_factory=dict)


def load_runner_settings(config_path: Optional[Path] = None) -> RunnerSettings:
    default = RunnerSettings()
    path = config_path or Path(__file__).with_name("runner_config.json")
    if not path.exists():
        return default

    try:
        data = json.loads(path.read_text(encoding="utf-8"))
    except json.JSONDecodeError:
        return default

    return RunnerSettings(
        service_name=data.get("service_name", default.service_name),
        temperature=data.get("temperature", default.temperature),
        max_tokens=data.get("max_tokens", default.max_tokens),
        extra_params=data.get("extra_params", default.extra_params) or {},
    )


RUNNER_SETTINGS = load_runner_settings()
