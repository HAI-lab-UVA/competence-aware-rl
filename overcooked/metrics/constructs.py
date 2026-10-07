"""Paper constructs from trial features; no model fitting or plotting."""
import numpy as np

def safe_ratio(n, d):
    return n / d.where(d > 0)

def derive_constructs(frame):
    d = frame.copy()
    totals = {}
    composition = {}
    for actor in ['agent','clone']:
        counts = d[[f'{actor}_{k}_task_count' for k in ['onion','plate','soup']]].astype(float)
        counts.columns = ['onion','plate','soup']
        assert counts.ge(0).all().all() and counts.notna().all().all()
        totals[actor] = counts.sum(axis=1, min_count=3)
        np.testing.assert_allclose(totals[actor], d[f'{actor}_total_task_count'])
        composition[actor] = counts.div(totals[actor].where(totals[actor] > 0), axis=0)
    d['workload_allocation'] = safe_ratio(totals['agent'], totals['agent'] + totals['clone'])
    d['task_distinctiveness'] = .5 * (composition['agent']-composition['clone']).abs().sum(axis=1, min_count=3)
    d['task_distinctiveness_undefined'] = (totals['agent'] == 0) | (totals['clone'] == 0)
    three = d[['agent_bumps_clone_steps','clone_bumps_agent_steps','same_tile_collision_steps']].sum(axis=1, min_count=3)
    np.testing.assert_allclose(three + d.attempted_swap_steps, d.conflict_steps)
    d['coordination_interference'] = 100 * safe_ratio(three, d.observed_steps)
    d['team_performance'] = d.agent_deliveries + d.clone_deliveries
    np.testing.assert_allclose(d.team_performance, d.team_deliveries)
    d['belief_error'] = (d.agent_score_input-d.clone_score).abs()/6
    d['signed_belief_error'] = (d.agent_score_input-d.clone_score)/6
    for c in ['workload_allocation','task_distinctiveness','belief_error']:
        assert d[c].dropna().between(-1e-10,1+1e-10).all(), c
    assert d.loc[d.task_distinctiveness_undefined,'task_distinctiveness'].isna().all()
    return d
