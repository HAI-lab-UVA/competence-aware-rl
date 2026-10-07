from __future__ import annotations

import json
from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple

import numpy as np

from .config import Scenario
from .policies import PartnerPolicy, corrupt_assignment_action_with_info

from stat_env import STATGymEnv  # public STAT release (github.com/mariacardei/coordination_aware_MARL)


@dataclass
class PartnerActionStats:
    opportunity_count: int = 0
    applied_count: int = 0
    changed_count: int = 0
    policy_assignment_count: int = 0
    final_assignment_count: int = 0
    policy_assignment_cost_sum: float = 0.0
    final_assignment_cost_sum: float = 0.0
    delta_assignment_cost_sum: float = 0.0
    best_feasible_assignment_cost_sum: float = 0.0
    policy_assignment_regret_sum: float = 0.0
    final_assignment_regret_sum: float = 0.0
    worse_than_policy_count: int = 0
    corrupted_policy_cost_sum: float = 0.0
    corrupted_final_cost_sum: float = 0.0
    corrupted_delta_cost_sum: float = 0.0
    corrupted_best_feasible_assignment_cost_sum: float = 0.0
    corrupted_policy_assignment_regret_sum: float = 0.0
    corrupted_final_assignment_regret_sum: float = 0.0
    corrupted_worse_than_policy_count: int = 0


@dataclass
class EpisodeResult:
    reward: float
    normalized_return: float
    steps: int
    completion_steps: Optional[int]
    done: bool
    forced_idle: int
    conflicts: int
    conflicts_per_100_steps: float
    forced_idle_per_100_steps: float
    conflicts_per_assignment_decision: float
    forced_idle_per_assignment_decision: float
    unique_assigned_mean: float
    unique_assigned_sum: int
    assignment_diversity_mean: float
    adaptive_completed_tasks: int
    partner_completed_tasks: int
    partner_completed_tasks_total: int
    partner_completed_tasks_mean: float
    partner_completed_tasks_by_agent: str
    adaptive_workload_share: float
    partner_workload_share: float
    adaptive_workload_excess: float
    adaptive_assignments: int
    partner_assignments: int
    partner_assignments_total: int
    partner_assignments_mean: float
    partner_i_assignments: str
    adaptive_assignment_share: float
    partner_assignment_share: float
    adaptive_assignment_excess: float
    partner_policy_assignments: int
    partner_final_assignments: int
    partner_corruption_opportunities: int
    partner_corruptions_applied: int
    partner_assignment_changed: int
    partner_assignment_changed_rate: float
    partner_corruption_rate_observed: float
    partner_policy_assignment_cost_mean: float
    partner_final_assignment_cost_mean: float
    partner_delta_assignment_cost_mean: float
    partner_best_feasible_assignment_cost_mean: float
    partner_policy_assignment_regret_mean: float
    partner_final_assignment_regret_mean: float
    partner_total_travel_inefficiency: float
    partner_total_policy_regret: float
    partner_total_final_regret: float
    partner_worse_than_policy_count: int
    partner_worse_than_policy_rate: float
    partner_corrupted_policy_assignment_cost_mean: float
    partner_corrupted_final_assignment_cost_mean: float
    partner_corrupted_delta_assignment_cost_mean: float
    partner_corrupted_best_feasible_assignment_cost_mean: float
    partner_corrupted_policy_assignment_regret_mean: float
    partner_corrupted_final_assignment_regret_mean: float
    partner_corrupted_travel_inefficiency: float
    partner_corrupted_policy_regret: float
    partner_corrupted_final_regret: float
    partner_corrupted_worse_than_policy_count: int
    partner_corrupted_worse_than_policy_rate: float
    adaptive_action_counts: str
    partner_action_counts: str
    adaptive_task_assignment_counts: str
    partner_task_assignment_counts: str
    partner_i_action_counts: str
    partner_i_task_assignment_counts: str


def make_env(scenario: Scenario, seed: int) -> STATGymEnv:
    return STATGymEnv(
        seed=seed,
        agents=scenario.num_agents,
        tasks=scenario.num_tasks,
        width=scenario.width,
        height=scenario.height,
        policy=6,
        num_bins=scenario.num_bins,
    )


def _blank_action_counts(scenario: Scenario) -> Dict[str, int]:
    return {str(a): 0 for a in range(scenario.num_sub_actions)}


def _blank_task_counts(scenario: Scenario) -> Dict[str, int]:
    return {str(v): 0 for v in range(scenario.num_tasks)}


def _inc_action(counts: Dict[str, int], action: int) -> None:
    counts[str(int(action))] = counts.get(str(int(action)), 0) + 1


def _inc_task(counts: Dict[str, int], action: int) -> None:
    if int(action) >= 3:
        task = int(action) - 3
        counts[str(task)] = counts.get(str(task), 0) + 1


def _share(part: int, total: int) -> float:
    return float(part / total) if total > 0 else 0.0


def _mean(total: float, count: int) -> float:
    return float(total / count) if count > 0 else 0.0


def _assignment_cost(env: STATGymEnv, agent_idx: int, action: int) -> Optional[float]:
    if int(action) < 3:
        return None
    task_idx = int(action) - 3
    if task_idx < 0 or task_idx >= len(env.model.sorted_tasks):
        return None
    agent = env.model.sorted_agents[agent_idx]
    task = env.model.sorted_tasks[task_idx]
    dx = float(agent.pos[0]) - float(task.pos[0])
    dy = float(agent.pos[1]) - float(task.pos[1])
    return float(np.sqrt(dx * dx + dy * dy))


def _best_feasible_assignment_cost(env: STATGymEnv, agent_idx: int, mask: np.ndarray) -> Optional[float]:
    costs = [
        _assignment_cost(env, agent_idx, int(action))
        for action in np.flatnonzero(np.asarray(mask) > 0.5)
        if int(action) >= 3
    ]
    valid_costs = [cost for cost in costs if cost is not None]
    return min(valid_costs) if valid_costs else None


def compose_partner_actions(
    state: dict,
    scenario: Scenario,
    partner_policy: PartnerPolicy,
    epsilon: float,
    rng: np.random.Generator,
    base_joint_action: Optional[List[int]] = None,
    partner_action_counts: Optional[Dict[str, int]] = None,
    partner_task_counts: Optional[Dict[str, int]] = None,
    env: Optional[STATGymEnv] = None,
) -> Tuple[List[int], PartnerActionStats]:
    joint_action = list(base_joint_action) if base_joint_action is not None else [0] * scenario.num_agents
    stats = PartnerActionStats()
    for idx in scenario.partner_indices:
        policy_action = partner_policy.act(state, idx, rng)
        if policy_action >= 3:
            stats.policy_assignment_count += 1

        mask = np.asarray(state["agent_masks"][idx])
        final_action, opportunity, applied = corrupt_assignment_action_with_info(
            policy_action, mask, epsilon, rng
        )
        stats.opportunity_count += int(opportunity)
        stats.applied_count += int(applied)
        stats.changed_count += int(final_action != policy_action)
        stats.final_assignment_count += int(final_action >= 3)

        if env is not None and policy_action >= 3 and final_action >= 3:
            policy_cost = _assignment_cost(env, idx, policy_action)
            final_cost = _assignment_cost(env, idx, final_action)
            if policy_cost is not None and final_cost is not None:
                delta_cost = final_cost - policy_cost
                best_cost = _best_feasible_assignment_cost(env, idx, mask)
                policy_regret = policy_cost - best_cost if best_cost is not None else 0.0
                final_regret = final_cost - best_cost if best_cost is not None else 0.0
                is_worse = delta_cost > 1e-9
                stats.policy_assignment_cost_sum += policy_cost
                stats.final_assignment_cost_sum += final_cost
                stats.delta_assignment_cost_sum += delta_cost
                stats.best_feasible_assignment_cost_sum += best_cost if best_cost is not None else 0.0
                stats.policy_assignment_regret_sum += policy_regret
                stats.final_assignment_regret_sum += final_regret
                stats.worse_than_policy_count += int(is_worse)
                if applied:
                    stats.corrupted_policy_cost_sum += policy_cost
                    stats.corrupted_final_cost_sum += final_cost
                    stats.corrupted_delta_cost_sum += delta_cost
                    stats.corrupted_best_feasible_assignment_cost_sum += best_cost if best_cost is not None else 0.0
                    stats.corrupted_policy_assignment_regret_sum += policy_regret
                    stats.corrupted_final_assignment_regret_sum += final_regret
                    stats.corrupted_worse_than_policy_count += int(is_worse)

        joint_action[idx] = int(final_action)

        if partner_action_counts is not None:
            _inc_action(partner_action_counts, final_action)
        if partner_task_counts is not None:
            _inc_task(partner_task_counts, final_action)
    return joint_action, stats


def run_episode(
    scenario: Scenario,
    seed: int,
    adaptive_policy,
    partner_policy: PartnerPolicy,
    true_epsilon: float,
    max_steps: Optional[int] = None,
    rng: Optional[np.random.Generator] = None,
) -> EpisodeResult:
    rng = rng or np.random.default_rng(seed)
    env = make_env(scenario, seed)
    if hasattr(partner_policy, "env"):
        partner_policy.env = env

    state, _ = env.reset(seed=seed)
    total_reward = 0.0
    forced_idle = 0
    conflicts = 0
    unique = []
    adaptive_assignments = 0
    partner_assignments = 0
    partner_policy_assignments = 0
    partner_final_assignments = 0
    corruption_opportunities = 0
    corruptions_applied = 0
    assignment_changed = 0
    policy_assignment_cost_sum = 0.0
    final_assignment_cost_sum = 0.0
    delta_assignment_cost_sum = 0.0
    best_feasible_assignment_cost_sum = 0.0
    policy_assignment_regret_sum = 0.0
    final_assignment_regret_sum = 0.0
    worse_than_policy_count = 0
    corrupted_policy_cost_sum = 0.0
    corrupted_final_cost_sum = 0.0
    corrupted_delta_cost_sum = 0.0
    corrupted_best_feasible_assignment_cost_sum = 0.0
    corrupted_policy_assignment_regret_sum = 0.0
    corrupted_final_assignment_regret_sum = 0.0
    corrupted_worse_than_policy_count = 0
    partner_i_assignments = {str(i): 0 for i in scenario.partner_indices}
    adaptive_action_counts = _blank_action_counts(scenario)
    partner_action_counts = _blank_action_counts(scenario)
    adaptive_task_counts = _blank_task_counts(scenario)
    partner_task_counts = _blank_task_counts(scenario)
    partner_i_action_counts = {str(i): _blank_action_counts(scenario) for i in scenario.partner_indices}
    partner_i_task_counts = {str(i): _blank_task_counts(scenario) for i in scenario.partner_indices}
    done = False
    limit = max_steps or scenario.max_steps

    for step in range(limit):
        adaptive_action = adaptive_policy.act(state, scenario.adaptive_index, rng)
        _inc_action(adaptive_action_counts, adaptive_action)
        _inc_task(adaptive_task_counts, adaptive_action)

        joint_action = [0] * scenario.num_agents
        joint_action[scenario.adaptive_index] = int(adaptive_action)
        joint_action, partner_stats = compose_partner_actions(
            state,
            scenario,
            partner_policy,
            true_epsilon,
            rng,
            joint_action,
            partner_action_counts=partner_action_counts,
            partner_task_counts=partner_task_counts,
            env=env,
        )
        for idx in scenario.partner_indices:
            action = joint_action[idx]
            if action >= 3:
                partner_i_assignments[str(idx)] += 1
            _inc_action(partner_i_action_counts[str(idx)], action)
            _inc_task(partner_i_task_counts[str(idx)], action)

        adaptive_assignments += int(adaptive_action >= 3)
        partner_assignments += partner_stats.final_assignment_count
        partner_policy_assignments += partner_stats.policy_assignment_count
        partner_final_assignments += partner_stats.final_assignment_count
        corruption_opportunities += partner_stats.opportunity_count
        corruptions_applied += partner_stats.applied_count
        assignment_changed += partner_stats.changed_count
        policy_assignment_cost_sum += partner_stats.policy_assignment_cost_sum
        final_assignment_cost_sum += partner_stats.final_assignment_cost_sum
        delta_assignment_cost_sum += partner_stats.delta_assignment_cost_sum
        best_feasible_assignment_cost_sum += partner_stats.best_feasible_assignment_cost_sum
        policy_assignment_regret_sum += partner_stats.policy_assignment_regret_sum
        final_assignment_regret_sum += partner_stats.final_assignment_regret_sum
        worse_than_policy_count += partner_stats.worse_than_policy_count
        corrupted_policy_cost_sum += partner_stats.corrupted_policy_cost_sum
        corrupted_final_cost_sum += partner_stats.corrupted_final_cost_sum
        corrupted_delta_cost_sum += partner_stats.corrupted_delta_cost_sum
        corrupted_best_feasible_assignment_cost_sum += partner_stats.corrupted_best_feasible_assignment_cost_sum
        corrupted_policy_assignment_regret_sum += partner_stats.corrupted_policy_assignment_regret_sum
        corrupted_final_assignment_regret_sum += partner_stats.corrupted_final_assignment_regret_sum
        corrupted_worse_than_policy_count += partner_stats.corrupted_worse_than_policy_count

        next_state, reward, done, _, info = env.step(joint_action)
        total_reward += float(reward)
        forced_idle += int(info.get("forced_idle", 0))
        conflicts += int(info.get("num_conflicts", 0))
        unique.append(float(info.get("unique_tasks_assigned", 0)))
        state = next_state
        if done:
            break

    adaptive_completed_tasks = int(env.model.sorted_agents[scenario.adaptive_index].num_completed)
    partner_completed_tasks_by_agent = {
        str(i): int(env.model.sorted_agents[i].num_completed)
        for i in scenario.partner_indices
    }
    partner_completed_tasks = int(sum(partner_completed_tasks_by_agent.values()))
    total_completed_tasks = adaptive_completed_tasks + partner_completed_tasks
    total_assignments = adaptive_assignments + partner_assignments
    partner_count = len(scenario.partner_indices)
    equal_share = 1.0 / scenario.num_agents
    assignment_decisions = adaptive_assignments + partner_final_assignments
    observed_steps = step + 1
    return EpisodeResult(
        reward=float(total_reward),
        normalized_return=float(total_reward / max(scenario.num_tasks, 1)),
        steps=observed_steps,
        completion_steps=observed_steps if done else None,
        done=bool(done),
        forced_idle=int(forced_idle),
        conflicts=int(conflicts),
        conflicts_per_100_steps=100.0 * _share(conflicts, observed_steps),
        forced_idle_per_100_steps=100.0 * _share(forced_idle, observed_steps),
        conflicts_per_assignment_decision=_share(conflicts, assignment_decisions),
        forced_idle_per_assignment_decision=_share(forced_idle, assignment_decisions),
        unique_assigned_mean=float(np.mean(unique)) if unique else 0.0,
        unique_assigned_sum=int(np.sum(unique)) if unique else 0,
        assignment_diversity_mean=_share(float(np.mean(unique)) if unique else 0.0, scenario.num_agents),
        adaptive_completed_tasks=adaptive_completed_tasks,
        partner_completed_tasks=partner_completed_tasks,
        partner_completed_tasks_total=partner_completed_tasks,
        partner_completed_tasks_mean=_mean(partner_completed_tasks, partner_count),
        partner_completed_tasks_by_agent=json.dumps(partner_completed_tasks_by_agent, sort_keys=True),
        adaptive_workload_share=_share(adaptive_completed_tasks, total_completed_tasks),
        partner_workload_share=_share(partner_completed_tasks, total_completed_tasks),
        adaptive_workload_excess=_share(adaptive_completed_tasks, total_completed_tasks) - equal_share,
        adaptive_assignments=int(adaptive_assignments),
        partner_assignments=int(partner_assignments),
        partner_assignments_total=int(partner_assignments),
        partner_assignments_mean=_mean(partner_assignments, partner_count),
        partner_i_assignments=json.dumps(partner_i_assignments, sort_keys=True),
        adaptive_assignment_share=_share(adaptive_assignments, total_assignments),
        partner_assignment_share=_share(partner_assignments, total_assignments),
        adaptive_assignment_excess=_share(adaptive_assignments, total_assignments) - equal_share,
        partner_policy_assignments=int(partner_policy_assignments),
        partner_final_assignments=int(partner_final_assignments),
        partner_corruption_opportunities=int(corruption_opportunities),
        partner_corruptions_applied=int(corruptions_applied),
        partner_assignment_changed=int(assignment_changed),
        partner_assignment_changed_rate=_share(assignment_changed, corruption_opportunities),
        partner_corruption_rate_observed=_share(corruptions_applied, corruption_opportunities),
        partner_policy_assignment_cost_mean=_mean(policy_assignment_cost_sum, partner_policy_assignments),
        partner_final_assignment_cost_mean=_mean(final_assignment_cost_sum, partner_final_assignments),
        partner_delta_assignment_cost_mean=_mean(delta_assignment_cost_sum, partner_policy_assignments),
        partner_best_feasible_assignment_cost_mean=_mean(best_feasible_assignment_cost_sum, partner_policy_assignments),
        partner_policy_assignment_regret_mean=_mean(policy_assignment_regret_sum, partner_policy_assignments),
        partner_final_assignment_regret_mean=_mean(final_assignment_regret_sum, partner_policy_assignments),
        partner_total_travel_inefficiency=float(delta_assignment_cost_sum),
        partner_total_policy_regret=float(policy_assignment_regret_sum),
        partner_total_final_regret=float(final_assignment_regret_sum),
        partner_worse_than_policy_count=int(worse_than_policy_count),
        partner_worse_than_policy_rate=_share(worse_than_policy_count, partner_policy_assignments),
        partner_corrupted_policy_assignment_cost_mean=_mean(corrupted_policy_cost_sum, corruptions_applied),
        partner_corrupted_final_assignment_cost_mean=_mean(corrupted_final_cost_sum, corruptions_applied),
        partner_corrupted_delta_assignment_cost_mean=_mean(corrupted_delta_cost_sum, corruptions_applied),
        partner_corrupted_best_feasible_assignment_cost_mean=_mean(corrupted_best_feasible_assignment_cost_sum, corruptions_applied),
        partner_corrupted_policy_assignment_regret_mean=_mean(corrupted_policy_assignment_regret_sum, corruptions_applied),
        partner_corrupted_final_assignment_regret_mean=_mean(corrupted_final_assignment_regret_sum, corruptions_applied),
        partner_corrupted_travel_inefficiency=float(corrupted_delta_cost_sum),
        partner_corrupted_policy_regret=float(corrupted_policy_assignment_regret_sum),
        partner_corrupted_final_regret=float(corrupted_final_assignment_regret_sum),
        partner_corrupted_worse_than_policy_count=int(corrupted_worse_than_policy_count),
        partner_corrupted_worse_than_policy_rate=_share(corrupted_worse_than_policy_count, corruptions_applied),
        adaptive_action_counts=json.dumps(adaptive_action_counts, sort_keys=True),
        partner_action_counts=json.dumps(partner_action_counts, sort_keys=True),
        adaptive_task_assignment_counts=json.dumps(adaptive_task_counts, sort_keys=True),
        partner_task_assignment_counts=json.dumps(partner_task_counts, sort_keys=True),
        partner_i_action_counts=json.dumps(partner_i_action_counts, sort_keys=True),
        partner_i_task_assignment_counts=json.dumps(partner_i_task_counts, sort_keys=True),
    )
