from pathlib import Path
import sys
import unittest
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'overcooked/metrics'))
from metric_core import movement_conflict, task_group, standardize_agent_name

class FrameworkTests(unittest.TestCase):
    def test_architecture_labels(self):
        for name in ['CF','CCM','CCM+CF','Baseline']:
            self.assertEqual(standardize_agent_name('Cramped_Room_'+name),name)
        self.assertEqual(standardize_agent_name('Cramped_Room_CCM_low'),'Cramped_Room_CCM_low')

    def test_directional_conflicts_and_swaps(self):
        self.assertTrue(movement_conflict([(0,0),(1,0)],[(1,0),(1,0)])['agent_bumps_clone'])
        self.assertTrue(movement_conflict([(0,0),(1,0)],[(0,0),(0,0)])['clone_bumps_agent'])
        self.assertTrue(movement_conflict([(0,0),(2,0)],[(1,0),(1,0)])['same_tile_collision'])
        swap=movement_conflict([(0,0),(1,0)],[(1,0),(0,0)])
        self.assertTrue(swap['attempted_swap'])
        self.assertFalse(swap['same_tile_collision'])

    def test_productive_task_categories(self):
        self.assertEqual(task_group('deliver_soup','cramped_room'),'soup')
        self.assertEqual(task_group('fill_plate','cramped_room'),'plate')
        self.assertIsNone(task_group('place_onion_counter','cramped_room'))
        self.assertEqual(task_group('place_onion_counter','counter_circuit'),'onion')
        self.assertEqual(task_group('place_plate_counter','forced_coord'),'plate')
        self.assertIsNone(task_group('place_soup_counter','forced_coord'))
