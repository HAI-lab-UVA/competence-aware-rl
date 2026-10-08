# STAT competence-aware agents

Run commands from the repository root in the separate PyTorch environment described in the main README. The original module layout is retained. Model loading defaults to frozen `artifacts/models/`; training, calibration, and default evaluation outputs go to `results/stat/`.

## Evaluate released agents

```bash
python -m stat_experiment.src.evaluate_belief_sweep \
  --scenario 3A_12T_10x6 --configuration CF --seed 0 --episodes 100 \
  --partner-checkpoint stat_experiment/artifacts/partners/3A_12T_10x6/seed0/latest.pt
```

Repeat for Baseline, CF, CCM, and CCM+CF and seeds 0–4. Baseline has no belief input, so it is evaluated once per true reliability. Adaptive configurations sweep all three beliefs for each true reliability.

## Train a new partner and adaptive-agent policies

```bash
python -m stat_experiment.src.train_partner_fdqn \
  --scenario 3A_12T_10x6 --seed 0 --total-steps 100000

python -m stat_experiment.src.train_adaptive \
  --scenario 3A_12T_10x6 --configuration CF --seed 0 --total-steps 100000 \
  --partner-checkpoint results/stat/partners/3A_12T_10x6/seed0/latest.pt
```

Use `--configuration Baseline` for the competence-agnostic adaptive-agent policy. For CCM and CCM+CF, run each configuration with `--specialist low`, `medium`, and `high`. Repeat adaptive-agent training for seeds 0–4 for the paper-sized design. The released paper models share the frozen seed-0 partner; partner seed and adaptive-agent training seed are separate inputs.

Training writes `latest.pt` and its log under `results/stat/models/<scenario>/<configuration>/seed<seed>/[specialist]/`. Use a fresh results directory (or move prior generated runs aside) before starting another experiment with the same identifiers. Released artifacts are separate and are not training destinations.

Evaluate newly trained policies by adding:

```bash
--models-dir results/stat/models
```

Pass the appropriate new partner checkpoint as well. Omitting `--partner-checkpoint` selects a heuristic fallback intended for debugging, not the paper protocol.

## Calibration

```bash
python -m stat_experiment.src.calibrate_reliability \
  --scenario 3A_12T_10x6 --episodes 100 \
  --partner-checkpoint stat_experiment/artifacts/partners/3A_12T_10x6/seed0/latest.pt
```

The other scenario definitions remain available for new experiments, but their trained checkpoints are not bundled. CSV fields refer to task assignments, completed tasks, adaptive workload, and adaptive assignment allocation. See the framework document for primary metric definitions. Exact historical episode replay is not guaranteed across runtimes.
