"""Run a short STAT reliability/belief sweep using released models."""
import argparse
from pathlib import Path
import subprocess
import sys
ROOT = Path(__file__).resolve().parents[1]

def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--episodes', type=int, default=1)
    parser.add_argument('--architectures', nargs='+', default=['Baseline','CF','CCM','CCM+CF'],
                        choices=['Baseline','CF','CCM','CCM+CF'])
    args = parser.parse_args()
    if args.episodes < 1:
        parser.error('Episodes must be positive')
    partner = ROOT / 'stat_experiment/artifacts/partners/3A_12T_10x6/seed0/latest.pt'
    for architecture in args.architectures:
        subprocess.run([sys.executable, '-m', 'stat_experiment.src.evaluate_belief_sweep',
            '--scenario','3A_12T_10x6','--architecture',architecture,'--seed','0',
            '--episodes',str(args.episodes),'--partner-checkpoint',str(partner)],
            check=True, cwd=ROOT)

if __name__ == '__main__':
    main()
