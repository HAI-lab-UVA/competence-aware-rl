from __future__ import annotations

import argparse
import csv
import random
from pathlib import Path
from typing import Optional

import numpy as np
import torch
import torch.nn.functional as F

from .config import SCENARIO_CHOICES, ensure_results_dir, load_reliability_levels, load_scenario
from .models import FactorizedQNetwork
from .policies import FDQNPolicy, NearestTaskPartner, load_fdqn_partner_policy, valid_actions
from .replay import ReplayBuffer
from .stat_runner import compose_partner_actions, make_env


CONFIGURATION_CHOICES = ["Baseline", "CF", "CCM", "CCM+CF"]


def obs_with_belief(obs: np.ndarray, belief: float, use_belief: bool) -> np.ndarray:
    values = np.asarray(obs, dtype=np.float32)
    if not use_belief:
        return values
    return np.concatenate([values, np.array([belief], dtype=np.float32)])


def train_one(
    scenario_name: str,
    configuration: str,
    seed: int,
    total_steps: int,
    specialist: Optional[str],
    hidden_dim: int,
    lr: float,
    gamma: float,
    batch_size: int,
    buffer_capacity: int,
    target_update: int,
    epsilon_train: float,
    partner_checkpoint: Optional[str],
) -> Path:
    scenario = load_scenario(scenario_name)
    levels = load_reliability_levels()["default_levels"]
    use_belief = configuration in {"CF", "CCM+CF"}
    input_dim = scenario.obs_dim + (1 if use_belief else 0)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    q_net = FactorizedQNetwork(input_dim, scenario.num_agents, scenario.num_sub_actions, hidden_dim).to(device)
    target_net = FactorizedQNetwork(input_dim, scenario.num_agents, scenario.num_sub_actions, hidden_dim).to(device)
    target_net.load_state_dict(q_net.state_dict())
    opt = torch.optim.Adam(q_net.parameters(), lr=lr)
    replay = ReplayBuffer(buffer_capacity)

    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    rng = np.random.default_rng(seed)

    out_dir = ensure_results_dir("models", scenario.name, configuration, f"seed{seed}")
    if specialist:
        out_dir = out_dir / specialist
        out_dir.mkdir(parents=True, exist_ok=True)
    log_path = out_dir / "train_log.csv"
    model_path = out_dir / "latest.pt"

    env = make_env(scenario, seed)
    if partner_checkpoint:
        partner = load_fdqn_partner_policy(partner_checkpoint, device)
    else:
        partner = NearestTaskPartner(env)
    state, _ = env.reset(seed=seed)
    episode_reward = 0.0
    episode_forced_idle = 0
    episode_conflicts = 0
    episode_corruption_opportunities = 0
    episode_corruptions_applied = 0
    episode = 0
    rows = []

    def choose_condition():
        if configuration in {"CCM", "CCM+CF"}:
            label = specialist or "high"
            return label, float(levels[label]["epsilon"]), float(levels[label]["belief"])
        label = str(rng.choice(["low", "medium", "high"]))
        return label, float(levels[label]["epsilon"]), float(levels[label]["belief"])

    cond_label, true_epsilon, belief = choose_condition()

    for step in range(1, total_steps + 1):
        adaptive_mask = np.asarray(state["agent_masks"][scenario.adaptive_index], dtype=np.float32)
        train_eps = max(0.05, epsilon_train * (1.0 - step / max(total_steps, 1)))
        if rng.random() < train_eps:
            va = valid_actions(adaptive_mask)
            adaptive_action = int(rng.choice(va)) if len(va) else 0
        else:
            policy = FDQNPolicy(q_net, device, belief_input=(belief if use_belief else None))
            adaptive_action = policy.act(state, scenario.adaptive_index, rng)

        joint_action = [0] * scenario.num_agents
        joint_action[scenario.adaptive_index] = adaptive_action
        joint_action, partner_stats = compose_partner_actions(state, scenario, partner, true_epsilon, rng, joint_action)
        next_state, reward, done, _, _info = env.step(joint_action)
        episode_forced_idle += int(_info.get("forced_idle", 0))
        episode_conflicts += int(_info.get("num_conflicts", 0))
        episode_corruption_opportunities += partner_stats.opportunity_count
        episode_corruptions_applied += partner_stats.applied_count

        next_mask = np.asarray(next_state["agent_masks"][scenario.adaptive_index], dtype=np.float32)
        replay.append((
            np.asarray(state["obs"], dtype=np.float32),
            adaptive_mask,
            adaptive_action,
            float(reward),
            np.asarray(next_state["obs"], dtype=np.float32),
            next_mask,
            bool(done),
            float(belief),
        ))
        episode_reward += float(reward)
        state = next_state

        if len(replay) >= batch_size:
            obs, _mask, action, rew, next_obs, next_mask_b, done_arr, belief_arr = replay.sample(batch_size)
            obs_t = torch.tensor(
                np.asarray([obs_with_belief(o, b, use_belief) for o, b in zip(obs, belief_arr)]),
                dtype=torch.float32,
                device=device,
            )
            next_obs_t = torch.tensor(
                np.asarray([obs_with_belief(o, b, use_belief) for o, b in zip(next_obs, belief_arr)]),
                dtype=torch.float32,
                device=device,
            )
            action_t = torch.tensor(action, dtype=torch.long, device=device).unsqueeze(1)
            rew_t = torch.tensor(rew, dtype=torch.float32, device=device).unsqueeze(1)
            done_t = torch.tensor(done_arr, dtype=torch.float32, device=device).unsqueeze(1)
            next_mask_t = torch.tensor(next_mask_b, dtype=torch.float32, device=device)

            q_all = q_net(obs_t)[:, scenario.adaptive_index, :]
            q_sa = q_all.gather(1, action_t)
            with torch.no_grad():
                q_next = target_net(next_obs_t)[:, scenario.adaptive_index, :]
                q_next = q_next.masked_fill(next_mask_t == 0, -1e8)
                target = rew_t + gamma * q_next.max(dim=1, keepdim=True).values * (1.0 - done_t)
            loss = F.mse_loss(q_sa, target)
            opt.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(q_net.parameters(), 1.0)
            opt.step()

        if step % target_update == 0:
            target_net.load_state_dict(q_net.state_dict())

        if done or step % scenario.max_steps == 0:
            rows.append({
                "step": step,
                "episode": episode,
                "episode_reward": episode_reward,
                "condition": cond_label,
                "true_epsilon": true_epsilon,
                "belief": belief,
                "forced_idle": episode_forced_idle,
                "conflicts": episode_conflicts,
                "partner_corruption_opportunities": episode_corruption_opportunities,
                "partner_corruptions_applied": episode_corruptions_applied,
                "partner_corruption_rate_observed": (episode_corruptions_applied / episode_corruption_opportunities) if episode_corruption_opportunities else 0.0,
            })
            episode += 1
            cond_label, true_epsilon, belief = choose_condition()
            state, _ = env.reset(seed=seed * 100000 + episode)
            if hasattr(partner, "env"):
                partner.env = env
            episode_reward = 0.0
            episode_forced_idle = 0
            episode_conflicts = 0
            episode_corruption_opportunities = 0
            episode_corruptions_applied = 0

    torch.save({
        "configuration": configuration,
        "scenario": scenario.name,
        "specialist": specialist,
        "seed": seed,
        "use_belief": use_belief,
        "input_dim": input_dim,
        "num_agents": scenario.num_agents,
        "num_actions": scenario.num_sub_actions,
        "hidden_dim": hidden_dim,
        "state_dict": q_net.state_dict(),
    }, model_path)

    if rows:
        with log_path.open("w", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
            writer.writeheader()
            writer.writerows(rows)
    return model_path


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--scenario", required=True, choices=SCENARIO_CHOICES)
    ap.add_argument("--configuration", required=True, choices=CONFIGURATION_CHOICES)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--specialist", choices=["low", "medium", "high"], default=None)
    ap.add_argument("--total-steps", type=int, default=50000)
    ap.add_argument("--hidden-dim", type=int, default=64)
    ap.add_argument("--lr", type=float, default=1e-3)
    ap.add_argument("--gamma", type=float, default=0.99)
    ap.add_argument("--batch-size", type=int, default=128)
    ap.add_argument("--buffer-capacity", type=int, default=100000)
    ap.add_argument("--target-update", type=int, default=1000)
    ap.add_argument("--epsilon-train", type=float, default=1.0)
    ap.add_argument("--partner-checkpoint", type=str, default=None)
    args = ap.parse_args()

    if args.configuration in {"CCM", "CCM+CF"} and args.specialist is None:
        raise SystemExit("--specialist is required for CCM and CCM+CF")
    if args.configuration in {"Baseline", "CF"} and args.specialist is not None:
        raise SystemExit("--specialist is only used for CCM and CCM+CF")

    path = train_one(
        scenario_name=args.scenario,
        configuration=args.configuration,
        seed=args.seed,
        total_steps=args.total_steps,
        specialist=args.specialist,
        hidden_dim=args.hidden_dim,
        lr=args.lr,
        gamma=args.gamma,
        batch_size=args.batch_size,
        buffer_capacity=args.buffer_capacity,
        target_update=args.target_update,
        epsilon_train=args.epsilon_train,
        partner_checkpoint=args.partner_checkpoint,
    )
    print(f"saved {path}")


if __name__ == "__main__":
    main()
