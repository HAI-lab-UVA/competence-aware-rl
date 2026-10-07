from pathlib import Path
import sys
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'overcooked' / 'metrics'))
import unittest
import numpy as np
import pandas as pd
from constructs import derive_constructs

class ConstructTests(unittest.TestCase):
    def test_equal_distinct_and_undefined_work(self):
        rows=[]
        for a,c in [([1,0,0],[0,1,0]),([1,1,0],[1,1,0]),([0,0,0],[1,0,0]),([0,0,0],[0,0,0])]:
            row={'agent_score_input':7,'clone_score':1,'observed_steps':100,
                 'agent_bumps_clone_steps':1,'clone_bumps_agent_steps':2,
                 'same_tile_collision_steps':3,'attempted_swap_steps':4,'conflict_steps':10,
                 'agent_deliveries':2,'clone_deliveries':3,'team_deliveries':5}
            for actor,counts in [('agent',a),('clone',c)]:
                row.update({f'{actor}_{k}_task_count':v for k,v in zip(['onion','plate','soup'],counts)})
                row[f'{actor}_total_task_count']=sum(counts)
            rows.append(row)
        d=derive_constructs(pd.DataFrame(rows))
        self.assertEqual(d.loc[0,'task_distinctiveness'],1)
        self.assertEqual(d.loc[1,'task_distinctiveness'],0)
        self.assertTrue(np.isnan(d.loc[2,'task_distinctiveness']))
        self.assertTrue(np.isnan(d.loc[3,'workload_allocation']))
        self.assertEqual(d.loc[0,'workload_allocation'],.5)
        self.assertEqual(d.loc[0,'coordination_interference'],6)
        self.assertEqual(d.loc[0,'team_performance'],5)
        self.assertEqual(d.loc[0,'belief_error'],1)
