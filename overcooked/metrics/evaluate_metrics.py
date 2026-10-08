"""Replay newly generated logs and export the paper's trial-level constructs."""
import argparse
from pathlib import Path
import pandas as pd
from resumable_analysis import run, inventory
from constructs import derive_constructs

def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--evaluation-root', type=Path, required=True,
                        help='Contains evaluation/agents_competence_with_logs/*_trial_logs.csv and matching results')
    parser.add_argument('--out-dir', type=Path, required=True)
    parser.add_argument('--limit-per-agent-layout', type=int, default=None,
                        help='Optional small-example limit; omit for full coverage')
    args = parser.parse_args()
    if not inventory(args.evaluation_root)[0]:
        parser.error('No trial logs found; evaluate with --save-trial-logs first')
    if args.limit_per_agent_layout is not None and args.limit_per_agent_layout < 1:
        parser.error('The limit must be positive')
    folder = run(args.evaluation_root, args.out_dir,
                 limit_per_agent_layout=args.limit_per_agent_layout)
    features = pd.read_csv(folder / 'trial_features.csv', low_memory=False)
    constructs = derive_constructs(features)
    columns = ['trial_uid', 'agent_name', 'layout', 'clone_name', 'clone_role',
               'agent_seat', 'agent_score_input', 'clone_score', 'agent_group', 'clone_group',
               'observed_steps', 'workload_allocation', 'task_distinctiveness',
               'coordination_interference', 'team_performance',
               'belief_error', 'signed_belief_error']
    constructs[columns].rename(columns={
        'agent_name': 'configuration', 'clone_name': 'partner_name', 'clone_role': 'partner_role',
        'agent_score_input': 'believed_competence', 'clone_score': 'true_competence',
        'agent_group': 'believed_competence_group', 'clone_group': 'true_competence_group',
    }).to_csv(folder / 'trial_constructs.csv', index=False)
    print(f'Wrote {folder / "trial_constructs.csv"}')

if __name__ == '__main__':
    main()
