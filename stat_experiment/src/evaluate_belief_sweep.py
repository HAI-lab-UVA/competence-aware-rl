from __future__ import annotations

import argparse
import csv
from dataclasses import asdict
from pathlib import Path

import numpy as np
import torch

from .config import ARTIFACTS_DIR, SCENARIO_CHOICES, ensure_results_dir, load_reliability_levels, load_scenario
from .models import FactorizedQNetwork
from .policies import FDQNPolicy, NearestTaskPartner, load_fdqn_partner_policy
from .stat_runner import make_env, run_episode


def model_path(scenario: str, architecture: str, seed: int, specialist: str | None, models_dir: Path | None = None) -> Path:
    base = (models_dir or ARTIFACTS_DIR / "models") / scenario / architecture / f"seed{seed}"
    if specialist:
        base = base / specialist
    return base / "latest.pt"


def load_policy(path: Path, belief: float | None, device: torch.device) -> FDQNPolicy:
    payload = torch.load(path, map_location=device)
    net = FactorizedQNetwork(
        payload["input_dim"],
        payload["num_agents"],
        payload["num_actions"],
        hidden_dim=payload["hidden_dim"],
    ).to(device)
    net.load_state_dict(payload["state_dict"])
    net.eval()
    return FDQNPolicy(net, device, belief_input=(belief if payload["use_belief"] else None))


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--scenario", required=True, choices=SCENARIO_CHOICES)
    ap.add_argument("--architecture", required=True, choices=["Baseline", "CF", "CCM", "CCM+CF"])
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--episodes", type=int, default=50)
    ap.add_argument("--models-dir", type=Path, default=None, help="Model root; defaults to released artifacts/models")
    ap.add_argument("--out", type=str, default=None)
    ap.add_argument("--partner-checkpoint", type=str, default=None)
    args = ap.parse_args()

    scenario = load_scenario(args.scenario)
    levels = load_reliability_levels()["default_levels"]
    labels = ["low", "medium", "high"]
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    out_dir = ensure_results_dir("evaluation", scenario.name, args.architecture)
    out_path = args.out or str(out_dir / f"seed{args.seed}_belief_sweep.csv")
    rows = []

    for true_label in labels:
        true_epsilon = float(levels[true_label]["epsilon"])
        belief_labels = ["none"] if args.architecture == "Baseline" else labels
        for belief_label in belief_labels:
            belief = None if belief_label == "none" else float(levels[belief_label]["belief"])
            specialist = belief_label if args.architecture in {"CCM", "CCM+CF"} else None
            adaptive_policy = load_policy(model_path(args.scenario, args.architecture, args.seed, specialist, args.models_dir), belief, device)

            for ep in range(args.episodes):
                ep_seed = args.seed * 1000000 + ep
                env = make_env(scenario, ep_seed)
                if args.partner_checkpoint:
                    partner = load_fdqn_partner_policy(args.partner_checkpoint, device)
                else:
                    partner = NearestTaskPartner(env)
                result = run_episode(
                    scenario=scenario,
                    seed=ep_seed,
                    adaptive_policy=adaptive_policy,
                    partner_policy=partner,
                    true_epsilon=true_epsilon,
                    rng=np.random.default_rng(ep_seed + 99),
                )
                row = asdict(result)
                row.update({
                    "scenario": scenario.name,
                    "architecture": args.architecture,
                    "train_seed": args.seed,
                    "true_competence": true_label,
                    "true_epsilon": true_epsilon,
                    "believed_competence": belief_label,
                    "belief": "" if belief is None else belief,
                    "episode_seed": ep_seed,
                })
                rows.append(row)

    Path(out_path).parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)
    print(f"wrote {out_path}")


if __name__ == "__main__":
    main()
