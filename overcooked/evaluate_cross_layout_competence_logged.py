#!/usr/bin/env python3
"""Competence-only cross-layout evaluation with compressed timestep logs."""
from __future__ import annotations

import argparse
import csv
import dataclasses
import gzip
import hashlib
import json
import math
import multiprocessing as mp
import os
from pathlib import Path
import random
import statistics
import sys
from concurrent.futures import ProcessPoolExecutor, as_completed
from functools import lru_cache

LAYOUTS = {
    'cramped_room': 'Cramped_Room', 'coord_ring': 'Coord_Ring',
    'asymm_advantages': 'Asymm_Advantages', 'counter_circuit': 'Counter_Circuit',
    'forced_coord': 'Forced_Coordination',
}
ALIASES = {v.lower(): k for k, v in LAYOUTS.items()}
ALIASES.update({k: k for k in LAYOUTS})
ALIASES.update(coordination_ring='coord_ring', asymmetric_advantages='asymm_advantages',
               forced_coordination='forced_coord')


def layout(value):
    try:
        return ALIASES[value.lower()]
    except KeyError:
        raise argparse.ArgumentTypeError(f'Unknown layout: {value}')


def stable_seed(*values):
    return int.from_bytes(hashlib.sha256(json.dumps(values).encode()).digest()[:4], 'big') % (2**31)


def read_json(path):
    return json.loads(Path(path).read_text())


def atomic_json(path, data):
    path = Path(path)
    temp = path.with_suffix('.tmp')
    temp.write_text(json.dumps(data, indent=2, allow_nan=False))
    temp.replace(path)


def checkpoint(directory, prefix, suffix):
    return str(directory / f'{prefix}_{suffix}' / 'population' / 'final.npz')


def build_jobs(args):
    root = args.repo.resolve()
    splits = read_json(root / 'configs' / 'splits.json')
    scores = read_json(root / 'artifacts' / 'partners' / 'competence_labels.json')
    dirs = [Path(p).resolve() for p in args.models_dirs] if args.models_dirs else [root / 'artifacts' / 'agents']
    for directory in dirs:
        if not directory.is_dir():
            raise FileNotFoundError(directory)
    jobs, warnings = [], []
    partners = {}
    for target in args.target_layouts:
        candidates = sorted(set(splits[LAYOUTS[target]]['Test']))
        random.Random(stable_seed(args.seed, target, 'partners')).shuffle(candidates)
        selected = candidates[:args.max_clones] if args.max_clones else candidates
        if not selected:
            raise ValueError(f'No held-out partners for {target}')
        for clone in selected:
            if not (root / 'artifacts' / 'partners' / clone).is_file():
                raise FileNotFoundError(root / 'artifacts' / 'partners' / clone)
        partners[target] = selected
    for source in args.source_layouts:
        prefix = LAYOUTS[source]
        specs = []
        for directory in dirs:
            cfg_path = directory / f'{prefix}_CF' / 'agent_config.json'
            if not cfg_path.exists():
                warnings.append(f'No configuration: {cfg_path}')
                continue
            cfg = read_json(cfg_path)
            features = cfg.get('extra_features', [])
            if len(features) != 1:
                raise ValueError(f'Expected one score feature: {cfg_path}')
            metric = features[0]
            if not (metric == 'competence' or metric.startswith('competence_')):
                raise ValueError(f'Competence-only evaluator cannot use {metric}: {cfg_path}')
            if layout(cfg['layout']) != source:
                raise ValueError(f'Configuration layout mismatch: {cfg_path}')
            for family, suffixes in [('CF', ['CF']), ('CCM', ['CCM_low', 'CCM_middle', 'CCM_high']),
                                     ('CCM+CF', ['CCM+CF_low', 'CCM+CF_middle', 'CCM+CF_high'])]:
                paths = [checkpoint(directory, prefix, suffix) for suffix in suffixes]
                absent = [p for p in paths if not Path(p).is_file()]
                if absent:
                    warnings.append(f'Skipping {directory.name}/{source}/{family}: missing {absent}')
                    continue
                specs.append(dict(model_set=directory.name, metric=metric, family=family, paths=paths))
        baseline = root / 'artifacts' / 'baselines' / f'{source}.npz'
        if baseline.is_file():
            specs.append(dict(model_set='baselines', metric='', family='Baseline', paths=[str(baseline)]))
        else:
            warnings.append(f'Missing true baseline: {baseline}')
        for target, clones in partners.items():
            if args.off_diagonal_only and source == target:
                continue
            for spec in specs:
                metric = spec['metric']
                if metric:
                    for clone in clones:
                        for role in ('clone_0', 'clone_1'):
                            value = scores.get(Path(clone).stem, {}).get(role, {}).get(metric)
                            if value is None or not math.isfinite(float(value)):
                                raise ValueError(f'Missing/nonfinite {metric} score: {clone}/{role}')
                job = dict(**spec, source_layout=source, target_layout=target, clones=clones,
                           n_trials=args.n_trials, max_steps=args.max_steps, cook_time=args.cook_time,
                           seed=args.seed, repo=str(root), step_logs=args.step_logs)
                job['id'] = hashlib.sha256(json.dumps(job, sort_keys=True).encode()).hexdigest()[:20]
                jobs.append(job)
    if not jobs:
        raise ValueError('No evaluable checkpoint families found')
    return jobs, warnings


def initialize_worker(repo):
    # Spawn avoids forking an initialized JAX runtime. CPU prevents multiple
    # workers from competing for / preallocating one accelerator's memory.
    os.environ['JAX_PLATFORMS'] = 'cpu'
    os.environ['XLA_PYTHON_CLIENT_PREALLOCATE'] = 'false'
    os.environ['OMP_NUM_THREADS'] = '1'
    os.environ['OPENBLAS_NUM_THREADS'] = '1'
    os.environ['MKL_NUM_THREADS'] = '1'
    os.environ.setdefault('XLA_FLAGS', '--xla_cpu_multi_thread_eigen=false intra_op_parallelism_threads=1')
    sys.path.insert(0, repo)
    import jax
    import evaluate_agents_v3 as evaluator
    # Reuse identical observations and metrics, with fused policy inference.
    evaluator.forward = jax.jit(evaluator.forward)


@lru_cache(maxsize=64)
def load_model(path):
    import numpy as np
    import evaluate_agents_v3 as ev
    with np.load(path) as data:
        widths = []
        while f'layers/{len(widths)}/W' in data:
            widths.append(int(data[f'layers/{len(widths)}/W'].shape[0]))
        obs_dim = int(data['layers/0/W'].shape[1])
    return ev.load_agent_params(path, obs_dim, tuple(widths)), obs_dim


@lru_cache(maxsize=5)
def environment(target, seed):
    import jax
    import numpy as np
    import evaluate_agents_v3 as ev
    env = ev.make('overcooked', layout=ev.overcooked_layouts[target])
    _, state = env.reset(jax.random.PRNGKey(seed))
    return EpisodeEnvironment(env), ev._get_layout_stations(env), np.array(state.wall_map)


def json_value(value):
    """Convert state dataclasses and device arrays to portable JSON values."""
    if dataclasses.is_dataclass(value):
        return {f.name: json_value(getattr(value, f.name)) for f in dataclasses.fields(value)}
    if isinstance(value, dict):
        return {str(k): json_value(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [json_value(v) for v in value]
    if hasattr(value, 'tolist'):
        return value.tolist()
    return value


class EpisodeEnvironment:
    """Single-episode transitions without the library's terminal auto-reset."""
    def __init__(self, env):
        import jax
        self.env = env
        self.step_once = jax.jit(env.step_env)

    def __getattr__(self, name):
        return getattr(self.env, name)

    def step(self, key, state, actions):
        import jax
        # Match the transition key used by MultiAgentEnv.step exactly.
        step_key, _ = jax.random.split(key)
        return self.step_once(step_key, state, actions)


class LoggedEnvironment:
    def __init__(self, env, stream):
        self.env, self.stream, self.timestep = env, stream, 0

    def __getattr__(self, name):
        return getattr(self.env, name)

    def write(self, record):
        self.stream.write(json.dumps(json_value(record), separators=(',', ':'), allow_nan=False) + '\n')

    def reset(self, key):
        obs, state = self.env.reset(key)
        self.write(dict(type='initial_state', state=state))
        return obs, state

    def step(self, key, state, actions):
        obs, next_state, rewards, dones, info = self.env.step(key, state, actions)
        self.write(dict(type='step', timestep=self.timestep, actions=actions,
                        rewards=rewards, dones=dones, info=info, state_after=next_state))
        self.timestep += 1
        return obs, next_state, rewards, dones, info


def trace_path(job, clone, role, trial):
    episode_id = hashlib.sha256(json.dumps([clone, role, trial]).encode()).hexdigest()[:20]
    return str(Path('steps') / job['id'] / f'{episode_id}.jsonl.gz')


def valid_completed_row(row, job, out_dir):
    return not job['step_logs'] or (bool(row.get('step_log')) and
                                   (Path(out_dir) / row['step_log']).is_file())


def evaluate_job(job, out_dir):
    import evaluate_agents_v3 as ev
    root = Path(job['repo'])
    output = Path(out_dir) / 'jobs' / f"{job['id']}.json"
    rows = read_json(output) if output.exists() else []
    rows = [r for r in rows if valid_completed_row(r, job, out_dir)]
    completed = {(r['clone_name'], r['clone_role'], r['trial']) for r in rows}
    scores = read_json(root / 'artifacts' / 'partners' / 'competence_labels.json')
    metric = job['metric']
    thresholds = ev.compute_thresholds(read_json(root / 'configs' / 'splits.json'),
                                      LAYOUTS[job['source_layout']], metric, scores) if metric else None
    env, stations, wall = environment(job['target_layout'], job['seed'])
    for clone in job['clones']:
        clone_params, clone_dim = load_model(str(root / 'artifacts' / 'partners' / clone))
        if clone_dim != ev.FEATURE_DIM:
            raise ValueError(f'Unexpected clone observation dimension: {clone}')
        for clone_seat in (0, 1):
            role = f'clone_{clone_seat}'
            score = float(scores[Path(clone).stem][role][metric]) if metric else None
            group = ev.score_to_group(score, *thresholds) if thresholds else ''
            idx = ['Low', 'Middle', 'High'].index(group) if len(job['paths']) == 3 else 0
            model_path = job['paths'][idx]
            params, obs_dim = load_model(model_path)
            extra = []
            if job['family'] in ('CF', 'CCM+CF'):
                model_root = Path(model_path).parents[2]
                output_name = Path(model_path).parents[1].name
                stats = ev.load_extra_feature_stats(str(model_root), output_name)
                extra = [ev._zscore(score, metric, stats)]
            if obs_dim != ev.FEATURE_DIM + len(extra):
                raise ValueError(f'Unexpected observation dimension: {model_path}')
            for trial in range(job['n_trials']):
                if (clone, role, trial) in completed:
                    continue
                # Common random seeds across source layouts and model families.
                seed = stable_seed(job['seed'], job['target_layout'], clone, role, trial)
                row = {k: job[k] for k in ('model_set', 'metric', 'family', 'source_layout', 'target_layout')}
                row.update(clone_name=clone, clone_role=role, trial=trial, episode_seed=seed,
                           agent_seat=1-clone_seat, clone_score=score, selected_group=group,
                           checkpoint=model_path, step_log='')
                def run(episode_env):
                    result = ev.run_episode(params, clone_params, extra, [], 1-clone_seat,
                                            episode_env, stations, wall, job['cook_time'], job['max_steps'], seed)
                    result.pop('_transitions', None)
                    return result
                if job['step_logs']:
                    row['step_log'] = trace_path(job, clone, role, trial)
                    trace = Path(out_dir) / row['step_log']
                    trace.parent.mkdir(parents=True, exist_ok=True)
                    temp = trace.with_suffix('.tmp')
                    try:
                        with gzip.open(temp, 'wt', encoding='utf-8', compresslevel=1) as stream:
                            logged_env = LoggedEnvironment(env, stream)
                            from jaxmarl.environments.overcooked.overcooked import Actions
                            logged_env.write(dict(type='metadata', schema_version=1, **row,
                                                  agent_extra=extra, thresholds=thresholds,
                                                  max_steps=job['max_steps'], cook_time=job['cook_time'],
                                                  action_names={int(a): a.name for a in Actions}))
                            row.update(run(logged_env))
                            logged_env.write(dict(type='episode_end', steps=logged_env.timestep,
                                                  horizon_reached=logged_env.timestep == job['max_steps'],
                                                  result=row))
                        # Commit log before marking the episode complete.
                        temp.replace(trace)
                    finally:
                        temp.unlink(missing_ok=True)
                else:
                    row.update(run(env))
                rows.append(row)
                atomic_json(output, rows)
    return len(rows)


def write_csv(path, rows):
    if not rows:
        return
    temp = path.with_suffix('.tmp')
    with temp.open('w', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    temp.replace(path)


def collect(out, jobs):
    rows = []
    for job in jobs:
        path = out / 'jobs' / f"{job['id']}.json"
        if path.exists():
            rows.extend(read_json(path))
    write_csv(out / 'results.csv', rows)
    groups = {}
    keys = ('model_set', 'metric', 'family', 'source_layout', 'target_layout')
    for row in rows:
        groups.setdefault(tuple(row[k] for k in keys), []).append(row)
    summaries = []
    for key, group in sorted(groups.items()):
        summary = dict(zip(keys, key), episodes=len(group))
        for metric in ('deliveries', 'total_reward', 'total_steps', 'collisions'):
            values = [r[metric] for r in group]
            summary[f'{metric}_mean'] = statistics.mean(values)
            summary[f'{metric}_std'] = statistics.stdev(values) if len(values) > 1 else 0.0
        summaries.append(summary)
    write_csv(out / 'summary.csv', summaries)


def parse_args():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--repo', type=Path, default=Path(__file__).resolve().parent)
    ap.add_argument('--models-dirs', nargs='+', help='Competence model directories; default: artifacts/agents under repo')
    ap.add_argument('--source-layouts', nargs='+', type=layout, default=list(LAYOUTS))
    ap.add_argument('--target-layouts', nargs='+', type=layout, default=list(LAYOUTS))
    ap.add_argument('--off-diagonal-only', action='store_true')
    ap.add_argument('--n-trials', type=int, default=2)
    ap.add_argument('--max-clones', type=int, default=4, help='Held-out partners per destination; 0 = all')
    ap.add_argument('--max-steps', type=int, default=400)
    ap.add_argument('--cook-time', type=int, default=20, help='Feature normalization, matching existing evaluator')
    ap.add_argument('--workers', type=int, default=min(4, os.cpu_count() or 1))
    ap.add_argument('--seed', type=int, default=0)
    ap.add_argument('--out-dir', type=Path, default=Path('evaluation/cross_layout_competence_logged'))
    ap.add_argument('--dry-run', action='store_true', help='Print work count without loading JAX or running episodes')
    ap.add_argument('--no-step-logs', dest='step_logs', action='store_false',
                    help='Disable timestep logs (enabled by default)')
    args = ap.parse_args()
    for name in ('n_trials', 'max_steps', 'cook_time', 'workers'):
        if getattr(args, name) < 1:
            ap.error(f'--{name.replace("_", "-")} must be positive')
    if args.max_clones < 0:
        ap.error('--max-clones must be nonnegative')
    args.source_layouts = list(dict.fromkeys(args.source_layouts))
    args.target_layouts = list(dict.fromkeys(args.target_layouts))
    if args.models_dirs:
        args.models_dirs = list(dict.fromkeys(str(Path(p).resolve()) for p in args.models_dirs))
    return args


def main():
    args = parse_args()
    jobs, warnings = build_jobs(args)
    for warning in warnings:
        print(f'WARNING: {warning}', file=sys.stderr)
    episodes = sum(len(j['clones'])*2*j['n_trials'] for j in jobs)
    print(f'{len(jobs)} jobs; {episodes} episodes; up to {episodes*args.max_steps:,} steps; {args.workers} CPU workers', flush=True)
    if args.dry_run:
        return
    out = args.out_dir.resolve()
    out.mkdir(parents=True, exist_ok=True)
    # Exclusive lock protects shards from simultaneous invocations.
    lock = out / '.running'
    fd = os.open(lock, os.O_CREAT | os.O_EXCL | os.O_WRONLY)
    os.close(fd)
    try:
        inputs = {str(args.repo.resolve() / p) for p in ('artifacts/partners/competence_labels.json', 'configs/splits.json',
                  'evaluate_agents_v3.py', 'bc_features.py', 'network.py', 'checkpoint.py')}
        inputs.add(str(Path(__file__).resolve()))
        for j in jobs:
            inputs.update(j['paths'])
            inputs.update(str(args.repo.resolve() / 'artifacts' / 'partners' / c) for c in j['clones'])
            for p in j['paths']:
                stats = Path(p).with_name('agent_population.json')
                if stats.exists():
                    inputs.add(str(stats))
        signature = {p: [Path(p).stat().st_size, Path(p).stat().st_mtime_ns] for p in sorted(inputs)}
        manifest = dict(version=2, jobs=jobs, inputs=signature)
        manifest_path = out / 'manifest.json'
        if manifest_path.exists() and read_json(manifest_path) != manifest:
            raise ValueError('Run settings or inputs changed. Use a new --out-dir to avoid mixing results.')
        atomic_json(manifest_path, manifest)
        (out / 'jobs').mkdir(exist_ok=True)
        failures = []
        with ProcessPoolExecutor(max_workers=args.workers, mp_context=mp.get_context('spawn'),
                                 initializer=initialize_worker, initargs=(str(args.repo.resolve()),)) as pool:
            pending = []
            for j in jobs:
                shard = out / 'jobs' / f"{j['id']}.json"
                saved = read_json(shard) if shard.exists() else []
                completed = {(r['clone_name'], r['clone_role'], r['trial']) for r in saved
                             if valid_completed_row(r, j, out)}
                expected = {(c, r, t) for c in j['clones'] for r in ('clone_0', 'clone_1')
                            for t in range(j['n_trials'])}
                if completed != expected:
                    pending.append(j)
            print(f'{len(pending)} unfinished jobs ({len(jobs)-len(pending)} already complete)', flush=True)
            futures = {pool.submit(evaluate_job, j, str(out)): j for j in pending}
            for count, future in enumerate(as_completed(futures), 1):
                job = futures[future]
                try:
                    n = future.result()
                    print(f'[{count}/{len(jobs)}] {job["model_set"]} {job["family"]} '
                          f'{job["source_layout"]} -> {job["target_layout"]}: {n} episodes', flush=True)
                except Exception as exc:
                    failures.append(dict(job=job['id'], error=repr(exc)))
                    print(f'FAILED {job["id"]}: {exc}', file=sys.stderr, flush=True)
        collect(out, jobs)
        atomic_json(out / 'failures.json', failures)
        if failures:
            raise RuntimeError(f'{len(failures)} jobs failed; partial results saved. Fix cause and rerun to resume.')
        print(f'Results: {out / "results.csv"}\nSummary: {out / "summary.csv"}')
    finally:
        lock.unlink()


if __name__ == '__main__':
    main()
