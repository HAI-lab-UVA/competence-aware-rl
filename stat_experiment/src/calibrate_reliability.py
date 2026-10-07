from __future__ import annotations

import argparse
import csv
from dataclasses import asdict

import numpy as np

from .config import SCENARIO_CHOICES, ensure_results_dir, load_reliability_levels, load_scenario
from .policies import NearestTaskPartner, load_fdqn_partner_policy
from .stat_runner import make_env, run_episode


class NearestFocal:
    def __init__(self, env):
        self.partner = NearestTaskPartner(env)

    def act(self, state_dict: dict, agent_idx: int, rng: np.random.Generator) -> int:
        return self.partner.act(state_dict, agent_idx, rng)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--scenario", required=True, choices=SCENARIO_CHOICES)
    ap.add_argument("--episodes", type=int, default=50)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--out", type=str, default=None)
    ap.add_argument("--partner-checkpoint", type=str, default=None)
    args = ap.parse_args()

    scenario = load_scenario(args.scenario)
    epsilons = load_reliability_levels()["candidate_epsilons"]
    out_dir = ensure_results_dir("calibration")
    out_path = args.out or str(out_dir / f"{scenario.name}_calibration.csv")

    rows = []
    for epsilon in epsilons:
        for ep in range(args.episodes):
            seed = args.seed * 100000 + ep
            env = make_env(scenario, seed)
            adaptive = NearestFocal(env)
            if args.partner_checkpoint:
                import torch
                device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
                partner = load_fdqn_partner_policy(args.partner_checkpoint, device)
            else:
                partner = NearestTaskPartner(env)
            result = run_episode(
                scenario=scenario,
                seed=seed,
                adaptive_policy=adaptive,
                partner_policy=partner,
                true_epsilon=float(epsilon),
                rng=np.random.default_rng(seed + 17),
            )
            row = asdict(result)
            row.update({"scenario": scenario.name, "epsilon": float(epsilon), "episode_seed": seed})
            rows.append(row)

    with open(out_path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)
    print(f"wrote {out_path}")


if __name__ == "__main__":
    main()
