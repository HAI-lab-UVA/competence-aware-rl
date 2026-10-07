# Competence-aware agents and evaluation framework

Implementation accompanying *Separating Belief from Ability: Evaluating Competence-Aware Adaptation in Multi-Agent Teams*.

This repository provides CF, CCM, CCM+CF, and competence-agnostic baselines in Overcooked-AI and STAT, together with trained policies, partner inputs, and controlled evaluation. Vary believed partner competence while keeping the partner policies fixed, then measure behavioral adaptation separately from team performance.

This is an implementation and evaluation release. It supports new training and evaluation runs; it does not provide the complete historical dataset, human-data collection workflow, or notebooks for regenerating every paper figure and table. Retraining and reevaluation can vary with hardware, dependencies, and randomness.

## Repository layout

```text
competence_conditioned_rl/
├── overcooked/
│   ├── configs/
│   ├── artifacts/
│   │   ├── agents/
│   │   ├── baselines/
│   │   └── partners/
│   ├── metrics/
│   └── environment.yml
├── stat_experiment/
│   ├── configs/
│   ├── artifacts/
│   │   ├── models/
│   │   └── partners/
│   ├── src/
│   └── requirements.txt
├── examples/
├── docs/
├── tests/
├── results/
├── CITATION.cff
└── LICENSE
```

| Directory | Contents |
|---|---|
| [overcooked/](overcooked/README.md) | PPO/PBT training, belief-sweep and cross-layout evaluation, partner splits, and framework metrics |
| [stat_experiment/](stat_experiment/README.md) | FDQN training, reliability calibration, scenario definitions, and controlled evaluation |
| examples/ | Short evaluation runs using the released models |
| docs/ | Architecture descriptions and evaluation framework definitions |
| tests/ | Metric and architecture-label boundary checks |
| results/ | Generated experiment outputs; ignored by Git |

Overcooked includes partner checkpoints and their competence labels, five self-play baselines, and 35 competence-aware policies. STAT includes the paper scenario's frozen partner and 40 adaptive-agent policies across five training seeds. Released checkpoints are stored in each domain's `artifacts/` directory, separately from new experiment outputs.

Overcooked training and evaluation entry points are Python files directly under `overcooked/`; STAT entry points are modules under `stat_experiment/src/`.

## Overcooked quick start

From the repository root:

```bash
conda env create -f overcooked/environment.yml
conda activate competence-rl
python examples/overcooked_quickstart.py
python overcooked/metrics/evaluate_metrics.py \
  --evaluation-root results/overcooked \
  --out-dir results/overcooked/metrics
```

The example uses Cramped Room, one trial per starting role and belief, and a 20-step horizon. It demonstrates execution, not paper performance. Outputs include evaluation CSVs, action logs, reconstructed features, and `trial_constructs.csv` under a fingerprinted metrics directory.

## STAT quick start

Use a separate environment for the PyTorch implementation:

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r stat_experiment/requirements.txt
python examples/stat_quickstart.py
```

The example uses `3A_12T_10x6`, seed 0, and one episode per reliability/belief condition. It evaluates all four architectures and writes CSVs under `results/stat/evaluation/`. CSVs include reward, workload and assignment shares, and coordination diagnostics. The environment dependency is pinned to the public STAT commit used by the original implementation.

## Training and full evaluation

See [Overcooked instructions](overcooked/README.md), [STAT instructions](stat_experiment/README.md), [architecture mapping](docs/architectures.md), and [evaluation framework](docs/evaluation_framework.md).

Use fresh output locations for new training runs. The original partner splits and labels are provided as fixed experimental inputs; raw human trajectories, survey exports, and label-generation tooling are outside this release.

## Training hardware

All models were trained on either a MacBook Pro with Apple Silicon M4 and 64 GB of RAM, or a university computing cluster running Linux with a CPU and at least 32 GB of RAM.

## Checks

In the Overcooked environment:

```bash
python -m unittest discover -s tests -v
```

`docs/artifact_checksums.json` records SHA-256 hashes of the released checkpoints. Tests and short examples check execution and metric boundaries; they do not certify exact historical reproduction or scientific validity.

## Citation

If you use these agents or the evaluation framework, please cite the accompanying paper:

> Anonymous Author(s). *Separating Belief from Ability: Evaluating Competence-Aware Adaptation in Multi-Agent Teams*. Anonymized manuscript.

```bibtex
@misc{competence_aware_adaptation,
  title  = {Separating Belief from Ability: Evaluating Competence-Aware Adaptation in Multi-Agent Teams},
  author = {Anonymous Author(s)},
  note   = {Anonymized manuscript}
}
```

## License and dependencies

Repository code retains the original MIT license. Overcooked uses JaxMARL (Apache-2.0); STAT is installed separately from its upstream repository. See [third-party notices](THIRD_PARTY_NOTICES.md). No upstream environment source is vendored here.
