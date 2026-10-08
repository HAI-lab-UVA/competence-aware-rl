"""Run a short Cramped Room belief sweep using released models."""
import argparse
from pathlib import Path
import subprocess
import sys
ROOT = Path(__file__).resolve().parents[1]

def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--steps', type=int, default=20)
    parser.add_argument('--trials', type=int, default=1)
    args = parser.parse_args()
    if args.steps < 1 or args.trials < 1:
        parser.error('Steps and trials must be positive')
    oc = ROOT / 'overcooked'
    subprocess.run([sys.executable, str(oc / 'evaluate_agents.py'),
        '--scripts', str(oc / 'configs/cramped_room.json'),
        '--models-dir', str(oc / 'artifacts/agents'),
        '--clones-dir', str(oc / 'artifacts/partners'),
        '--clones-scores', str(oc / 'artifacts/partners/competence_labels.json'),
        '--train-sets', str(oc / 'configs/splits.json'),
        '--baselines-dir', str(oc / 'artifacts/baselines'),
        '--out-dir', str(ROOT / 'results/overcooked/evaluation/agents_competence_with_logs'),
        '--n-trials', str(args.trials), '--max-steps', str(args.steps),
        '--robustness-sweep', '--save-trial-logs', '--quiet'], check=True, cwd=ROOT)

if __name__ == '__main__':
    main()
