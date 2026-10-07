from __future__ import annotations

import argparse
import csv
import random
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F

from .config import SCENARIO_CHOICES, ensure_results_dir, load_scenario
from .models import FactorizedQNetwork
from .policies import valid_actions
from .replay import ReplayBuffer
from .stat_runner import make_env


def train_partner(
    scenario_name: str,
    seed: int,
    total_steps: int,
    hidden_dim: int,
    lr: float,
    gamma: float,
    batch_size: int,
    buffer_capacity: int,
    target_update: int,
    epsilon_train: float,
) -> Path:
    scenario = load_scenario(scenario_name)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    q_net = FactorizedQNetwork(
        scenario.obs_dim, scenario.num_agents, scenario.num_sub_actions, hidden_dim
    ).to(device)
    target_net = FactorizedQNetwork(
        scenario.obs_dim, scenario.num_agents, scenario.num_sub_actions, hidden_dim
    ).to(device)
    target_net.load_state_dict(q_net.state_dict())
    opt = torch.optim.Adam(q_net.parameters(), lr=lr)
    replay = ReplayBuffer(buffer_capacity)

    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    rng = np.random.default_rng(seed)

    out_dir = ensure_results_dir("partners", scenario.name, f"seed{seed}")
    model_path = out_dir / "latest.pt"
    log_path = out_dir / "train_log.csv"

    env = make_env(scenario, seed)
    state, _ = env.reset(seed=seed)
    episode_reward = 0.0
    episode_forced_idle = 0
    episode_conflicts = 0
    episode = 0
    rows = []

    for step in range(1, total_steps + 1):
        masks = np.asarray(state["agent_masks"], dtype=np.float32)
        train_eps = max(0.05, epsilon_train * (1.0 - step / max(total_steps, 1)))
        joint_action = []
        if rng.random() < train_eps:
            for i in range(scenario.num_agents):
                va = valid_actions(masks[i])
                joint_action.append(int(rng.choice(va)) if len(va) else 0)
        else:
            with torch.no_grad():
                obs_t = torch.tensor(
                    np.asarray(state["obs"], dtype=np.float32),
                    dtype=torch.float32,
                    device=device,
                ).unsqueeze(0)
                q = q_net(obs_t).squeeze(0).detach().cpu().numpy()
            for i in range(scenario.num_agents):
                qi = q[i].copy()
                qi[masks[i] == 0] = -1e8
                joint_action.append(int(np.argmax(qi)))

        next_state, reward, done, _, info = env.step(joint_action)
        next_masks = np.asarray(next_state["agent_masks"], dtype=np.float32)
        # Store joint action in the action slot; replay returns it as a vector.
        replay.append((
            np.asarray(state["obs"], dtype=np.float32),
            masks,
            np.asarray(joint_action, dtype=np.int64),
            float(reward),
            np.asarray(next_state["obs"], dtype=np.float32),
            next_masks,
            bool(done),
            0.0,
        ))
        episode_reward += float(reward)
        episode_forced_idle += int(info.get("forced_idle", 0))
        episode_conflicts += int(info.get("num_conflicts", 0))
        state = next_state

        if len(replay) >= batch_size:
            obs, _masks, actions, rew, next_obs, next_masks_b, done_arr, _belief = replay.sample(batch_size)
            obs_t = torch.tensor(obs, dtype=torch.float32, device=device)
            next_obs_t = torch.tensor(next_obs, dtype=torch.float32, device=device)
            actions_t = torch.tensor(actions, dtype=torch.long, device=device)
            rew_t = torch.tensor(rew, dtype=torch.float32, device=device).unsqueeze(1)
            done_t = torch.tensor(done_arr, dtype=torch.float32, device=device).unsqueeze(1)
            next_masks_t = torch.tensor(next_masks_b, dtype=torch.float32, device=device)

            q_all = q_net(obs_t)
            q_taken = []
            for i in range(scenario.num_agents):
                q_taken.append(q_all[:, i, :].gather(1, actions_t[:, i].unsqueeze(1)))
            q_joint = torch.sum(torch.cat(q_taken, dim=1), dim=1, keepdim=True)

            with torch.no_grad():
                q_next_all = target_net(next_obs_t)
                next_max = []
                for i in range(scenario.num_agents):
                    q_i = q_next_all[:, i, :].masked_fill(next_masks_t[:, i, :] == 0, -1e8)
                    next_max.append(q_i.max(dim=1, keepdim=True).values)
                q_next_joint = torch.sum(torch.cat(next_max, dim=1), dim=1, keepdim=True)
                target = rew_t + gamma * q_next_joint * (1.0 - done_t)

            loss = F.mse_loss(q_joint, target)
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
                "forced_idle": episode_forced_idle,
                "conflicts": episode_conflicts,
            })
            episode += 1
            state, _ = env.reset(seed=seed * 100000 + episode)
            episode_reward = 0.0
            episode_forced_idle = 0
            episode_conflicts = 0

    torch.save({
        "kind": "partner_fdqn",
        "scenario": scenario.name,
        "seed": seed,
        "input_dim": scenario.obs_dim,
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
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--total-steps", type=int, default=50000)
    ap.add_argument("--hidden-dim", type=int, default=64)
    ap.add_argument("--lr", type=float, default=1e-3)
    ap.add_argument("--gamma", type=float, default=0.99)
    ap.add_argument("--batch-size", type=int, default=128)
    ap.add_argument("--buffer-capacity", type=int, default=100000)
    ap.add_argument("--target-update", type=int, default=1000)
    ap.add_argument("--epsilon-train", type=float, default=1.0)
    args = ap.parse_args()
    path = train_partner(
        scenario_name=args.scenario,
        seed=args.seed,
        total_steps=args.total_steps,
        hidden_dim=args.hidden_dim,
        lr=args.lr,
        gamma=args.gamma,
        batch_size=args.batch_size,
        buffer_capacity=args.buffer_capacity,
        target_update=args.target_update,
        epsilon_train=args.epsilon_train,
    )
    print(f"saved {path}")


if __name__ == "__main__":
    main()
