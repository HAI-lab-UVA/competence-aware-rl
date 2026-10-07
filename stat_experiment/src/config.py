from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List


ROOT = Path(__file__).resolve().parents[1]
CONFIG_DIR = ROOT / "configs"
RESULTS_DIR = ROOT.parent / "results" / "stat"
ARTIFACTS_DIR = ROOT / "artifacts"
SCENARIO_CHOICES = ["2A_4T_5x3", "3A_6T_5x3", "3A_12T_10x6", "3A_25T_25x15"]


@dataclass(frozen=True)
class Scenario:
    name: str
    num_agents: int
    num_tasks: int
    width: int
    height: int
    num_bins: int
    adaptive_index: int
    partner_indices: List[int]
    max_steps: int

    @property
    def num_sub_actions(self) -> int:
        return 3 + self.num_tasks

    @property
    def obs_dim(self) -> int:
        return (
            self.num_agents * self.num_tasks
            + self.num_agents
            + 2 * self.num_tasks
        )


def _read_json(path: Path) -> Dict:
    with path.open() as f:
        return json.load(f)


def load_scenario(name: str) -> Scenario:
    raw = _read_json(CONFIG_DIR / "scenarios.json")
    if name not in raw:
        known = ", ".join(sorted(raw))
        raise ValueError(f"Unknown scenario '{name}'. Known scenarios: {known}")
    return Scenario(name=name, **raw[name])


def load_reliability_levels() -> Dict:
    return _read_json(CONFIG_DIR / "reliability_levels.json")


def ensure_results_dir(*parts: str) -> Path:
    path = RESULTS_DIR.joinpath(*parts)
    path.mkdir(parents=True, exist_ok=True)
    return path
