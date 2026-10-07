from pathlib import Path
import csv, hashlib, json, re
from collections import Counter
from functools import lru_cache
LAYOUT_NAME_MAP = {}


LAYOUT_OVERRIDES = {}

SCHEMA_VERSION = 'events-v1'

INTERACT_ACTION = 5

def digest(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as f:
        for block in iter(lambda: f.read(1024 * 1024), b''):
            h.update(block)
    return h.hexdigest()

def preview(rows, n=5):
    for row in rows[:n]:
        print(json.dumps(row, ensure_ascii=False, default=str))

def movement_conflict(pre_positions, candidate_positions):
    """Classify terrain-adjusted destinations BEFORE player conflict resolution.

    Inputs are in semantic actor order: adaptive agent, clone.
    Types are mutually exclusive; a swap is not also a directional bump.
    """
    p0, p1 = map(tuple, pre_positions)
    q0, q1 = map(tuple, candidate_positions)
    m0, m1 = (q0 != p0, q1 != p1)
    same = q0 == q1
    flags = {'agent_bumps_clone': same and m0 and (not m1), 'clone_bumps_agent': same and m1 and (not m0), 'same_tile_collision': same and m0 and m1, 'attempted_swap': m0 and m1 and (q0 == p1) and (q1 == p0)}
    assert sum(flags.values()) <= 1
    return {**flags, 'conflict': any(flags.values())}

def terrain_adjusted_destinations(env, state, actions):
    import jax.numpy as jnp
    from jaxmarl.environments.overcooked.overcooked import DIR_TO_VEC, Actions
    action_ids = jnp.array([actions['agent_0'], actions['agent_1']])
    action = env.action_set.take(indices=action_ids)
    moving = ((action != Actions.stay) & (action != Actions.interact))[:, None]
    target = jnp.minimum(jnp.maximum(state.agent_pos + moving * DIR_TO_VEC[jnp.minimum(action, 3)] + ~moving * state.agent_dir, 0), jnp.array((env.width - 1, env.height - 1), dtype=jnp.uint32))
    walls = state.wall_map[target[:, 1], target[:, 0]]
    goals = jnp.any(jnp.all(target[:, None, :] == state.goal_pos[None, :, :], axis=2), axis=1)
    bounced = (walls | goals)[:, None] | ~moving
    candidates = (bounced * state.agent_pos + ~bounced * target).astype(jnp.uint32)
    return candidates.tolist()

class TerminalSafeReplayEnv:
    """Preserve the actual final transition instead of the automatic reset state.

    Inspected JaxMARL MultiAgentEnv.step splits key into transition/reset keys.
    For nonterminal steps, use the original implementation unchanged.
    """

    def __init__(self, env):
        self.env = env
        self.terminal_corrections = 0
        self.movement_evidence = []

    def reset(self, key):
        self.movement_evidence = []
        return self.env.reset(key)

    def step(self, key, state, actions):
        import jax
        candidates = terrain_adjusted_destinations(self.env, state, actions)
        original = self.env.step(key, state, actions)
        if bool(original[3].get('__all__', False)):
            transition_key, _ = jax.random.split(key)
            actual = self.env.step_env(transition_key, state, actions)
            assert bool(actual[3]['__all__'])
            for actor, reward in original[2].items():
                assert float(reward) == float(actual[2][actor])
            self.terminal_corrections += 1
            result = actual
        else:
            result = original
        pre = state.agent_pos.tolist()
        flags = movement_conflict(pre, candidates)
        expected = pre if flags['conflict'] else candidates
        post = result[1].agent_pos.tolist()
        if post != expected:
            raise ValueError('Movement reconstruction disagrees with simulator positions')
        self.movement_evidence.append({'pre_positions': pre, 'candidate_positions': candidates, 'post_positions': post})
        return result

@lru_cache(maxsize=None)
def environment_bundle(layout):
    env = make('overcooked', layout=overcooked_layouts[layout])
    return (TerminalSafeReplayEnv(env), _get_layout_stations(env))

def standardize_agent_name(name):
    """Return paper architecture labels; preserve specialist identifiers."""
    for architecture in ('CCM+CF', 'CCM', 'CF', 'Baseline'):
        if name == architecture or name.endswith('_' + architecture):
            return architecture
    return name

def token(value):
    return re.sub('[^a-z0-9]+', '_', str(value).lower()).strip('_')

def resolve_layout(row):
    if row.get('layout', '').strip():
        return LAYOUT_NAME_MAP.get(row['layout'], row['layout'])
    names = [row.get('agent_name', ''), row.get('clone_name', '')]
    explicit = {LAYOUT_OVERRIDES[n] for n in names if n in LAYOUT_OVERRIDES}
    if explicit:
        if len(explicit) != 1:
            raise ValueError(f'Conflicting layout overrides: {names}')
        return explicit.pop()
    aliases = set(LAYOUT_NAME_MAP) | set(LAYOUT_NAME_MAP.values())
    matches = {LAYOUT_NAME_MAP.get(a, a) for a in aliases if any((f'_{token(a)}_' in f'_{token(n)}_' for n in names))}
    if len(matches) != 1:
        raise ValueError(f'Unresolved/ambiguous layout {matches}: {names}; set LAYOUT_OVERRIDES')
    return matches.pop()

REQUIRED_META = ['agent_name', 'clone_name', 'clone_role', 'agent_score_input', 'trial', 'script', 'agent_seat', 'aligned', 'episode_seed', 'total_steps']

def metadata(record):
    row = record['raw']
    missing = [k for k in REQUIRED_META if not row.get(k, '').strip()]
    if missing:
        raise ValueError(f"Missing metadata {missing}: {record['source_file']}:{record['source_row']}")
    layout = resolve_layout(row)
    key = {k: row[k] for k in REQUIRED_META}
    key['layout'] = layout
    uid_source = [record['source_file'], record['source_row']]
    return {'trial_uid': hashlib.sha256(json.dumps(uid_source).encode()).hexdigest(), 'trial_key': json.dumps(key, sort_keys=True), 'source_file': record['source_file'], 'source_row': record['source_row'], 'agent_name_raw': row['agent_name'], 'agent_name': standardize_agent_name(row['agent_name']), 'layout': layout, **{k: row[k] for k in REQUIRED_META if k != 'agent_name'}, **{k: row.get(k, '') for k in ['agent_group', 'clone_score', 'clone_group']}}

EVENT_SPECS = {'pick_onion': ('empty', 'onion'), 'pick_plate': ('empty', 'plate'), 'pick_soup': ('empty', 'soup'), 'place_onion_pot': ('onion', 'empty'), 'place_onion_counter': ('onion', 'empty'), 'place_plate_counter': ('plate', 'empty'), 'place_soup_counter': ('soup', 'empty'), 'fill_plate': ('plate', 'soup'), 'deliver_soup': ('soup', 'empty')}

LAYOUT_RULES = {}

def json_safe(value):
    if isinstance(value, dict):
        return {str(k): json_safe(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [json_safe(v) for v in value]
    if hasattr(value, 'tolist'):
        return json_safe(value.tolist())
    if value is None or isinstance(value, (str, int, float, bool)):
        return value
    raise TypeError(f'Unsupported decoder value: {type(value)}')

def reconstruct(decoded, meta):
    observations, events, issues = ([], [], [])

    def issue(step, actor, reason):
        issues.append({'trial_uid': meta['trial_uid'], 'step': step, 'actor': actor, 'reason': reason})
    if not isinstance(decoded.get('steps'), (list, tuple)):
        raise ValueError('Decoder must return a steps sequence')
    previous = None
    for step_record in decoded['steps']:
        step = json_safe(step_record['step'])
        if isinstance(step, bool) or not isinstance(step, int):
            raise ValueError(f'Noninteger step: {step!r}')
        if previous is not None:
            if step <= previous:
                raise ValueError('Duplicate/reversed timesteps: inspect trial/reset boundaries')
            if step != previous + 1:
                issue(step, 'team', 'step_gap')
        previous = step
        for actor in ('agent', 'clone'):
            side = json_safe(step_record[actor])
            required = {'action', 'events', 'held_pre', 'held_post', 'pos', 'dir'}
            if required - side.keys():
                raise ValueError(f'Missing actor fields: {required - side.keys()}')
            if not isinstance(side['events'], list) or not all((isinstance(e, str) for e in side['events'])):
                raise ValueError('events must be a sequence of event names')
            event_labels = list(side['events'])
            if side['action'] == INTERACT_ACTION and side['held_pre'] == 'empty' and (side['held_post'] == 'soup') and ('pick_soup' not in event_labels):
                event_labels.append('pick_soup')
            if side['held_pre'] != side['held_post'] and (not event_labels):
                issue(step, actor, 'unclassified_inventory_change')
            target = [side['pos'][i] + side['dir'][i] for i in (0, 1)]
            base = {**meta, 'step': step, 'actor': actor}
            observations.append({**base, 'action': side['action'], 'held_pre': side['held_pre'], 'held_post': side['held_post'], 'position_pre': side['pos'], 'direction_pre': side['dir'], 'movement_outcome': side.get('movement_outcome'), 'reward_reported': side.get('reward'), 'collision_reported': json_safe(step_record.get('collision')), 'events_reported': side['events'], 'decoder_actor_record': side})
            if len(side['events']) != len(set(side['events'])):
                issue(step, actor, 'duplicate_event_label')
            for ordinal, event in enumerate(event_labels):
                expected = EVENT_SPECS.get(event)
                status = 'unknown_event' if expected is None else 'held_transition_consistent' if tuple([side['held_pre'], side['held_post']]) == expected else 'held_transition_mismatch'
                if status != 'held_transition_consistent':
                    issue(step, actor, f'{status}:{event}')
                events.append({**base, 'event_id': f"{meta['trial_uid']}:{step}:{actor}:{ordinal}", 'event': event, 'event_ordinal': ordinal, 'evidence_source': 'inventory_transition' if event not in side['events'] else 'replay_trial.events', 'interaction_target': target, 'validation_status': status, 'action': side['action'], 'held_pre': side['held_pre'], 'held_post': side['held_post'], 'position_pre': side['pos'], 'direction_pre': side['dir'], 'productive': None, 'task_stage': None})
    if 'total_steps' in meta and len(decoded['steps']) != int(meta['total_steps']):
        issue(None, 'team', 'replay_length_mismatch')
    if decoded['steps'] and decoded['steps'][0]['step'] != 0:
        issue(None, 'team', 'nonzero_first_step')
    if not decoded['steps']:
        issue(None, 'team', 'empty_trial')
    return (observations, events, issues)

def synthetic_side(events=(), pre='empty', post='empty'):
    return dict(action='synthetic', events=list(events), held_pre=pre, held_post=post, pos=[1, 1], dir=[0, 1], movement_outcome=None, reward=0)

def synthetic_step(n, agent=None, clone=None):
    return dict(step=n, agent=agent or synthetic_side(), clone=clone or synthetic_side(), collision=False)

def write_csv(path, rows, columns):
    with path.open('w', newline='', encoding='utf-8') as f:
        writer = csv.DictWriter(f, fieldnames=columns)
        writer.writeheader()
        for row in rows:
            writer.writerow({k: json.dumps(v, ensure_ascii=False) if isinstance(v, (dict, list, tuple)) else v for k, v in row.items()})

BASE_TASK_GROUPS = {'pick_onion': 'onion', 'place_onion_pot': 'onion', 'pick_plate': 'plate', 'fill_plate': 'plate', 'deliver_soup': 'soup'}

LAYOUT_RULES = {'counter_circuit': {'place_onion_counter': 'onion'}, 'forced_coord': {'place_onion_counter': 'onion', 'place_plate_counter': 'plate'}}

def task_group(event, layout):
    return LAYOUT_RULES.get(layout, {}).get(event, BASE_TASK_GROUPS.get(event))

JOIN_FIELDS = ['script', 'agent_name', 'clone_name', 'clone_role', 'agent_score_input', 'agent_group', 'trial', 'agent_seat', 'aligned', 'total_steps']

def handling_diagnostics(decoded, uid):
    held = {a: None for a in ('agent', 'clone')}
    counters, items, flags = ({}, {}, [])

    def flag(step, actor, reason, item=None):
        flags.append(dict(trial_uid=uid, step=step, actor=actor, reason=reason, item_id=item))

    def new_item(kind):
        identity = f'{uid}:item:{len(items)}'
        items[identity] = {'kind': kind, 'retrievals': 0}
        return identity
    for rec in decoded['steps']:
        step = rec['step']
        targets = {a: tuple((rec[a]['pos'][j] + rec[a]['dir'][j] for j in (0, 1))) for a in held if rec[a]['action'] == INTERACT_ACTION}
        ambiguous = len(targets) == 2 and len(set(targets.values())) == 1
        if ambiguous:
            tile = next(iter(targets.values()))
            counters.pop(tile, None)
            flag(step, 'team', 'ambiguous_simultaneous_interactions')
            for a in targets:
                held[a] = None
            continue
        for actor in held:
            side = rec[actor]
            pre, post = (side['held_pre'], side['held_post'])
            identity = held[actor]
            if pre == post:
                continue
            if side['action'] != INTERACT_ACTION:
                held[actor] = None
                flag(step, actor, 'unexpected_inventory_change')
                continue
            tile = targets[actor]
            events = set(side['events'])
            if pre == 'empty' and post != 'empty':
                deposit = counters.pop(tile, None)
                if deposit is not None and items[deposit[0]]['kind'] == post:
                    identity, placer = deposit
                    items[identity]['retrievals'] += 1
                    if placer == actor:
                        flag(step, actor, 'same_chef_retrieval', identity)
                    if items[identity]['retrievals'] >= 2:
                        flag(step, actor, 'repeated_item_handling', identity)
                else:
                    identity = new_item(post)
                    stations = decoded.get('_stations', {})
                    source_key = {'onion': 'onion_dispensers', 'plate': 'plate_dispensers'}.get(post)
                    known_supply = source_key and tile in {tuple(x) for x in stations.get(source_key, [])}
                    if not known_supply:
                        flag(step, actor, 'unmatched_pickup', identity)
                held[actor] = identity
            elif events & {'place_onion_counter', 'place_plate_counter', 'place_soup_counter'}:
                if identity is None:
                    identity = new_item(pre)
                    flag(step, actor, 'unknown_item_history', identity)
                if tile in counters:
                    flag(step, actor, 'overwritten_counter_history', identity)
                counters[tile] = (identity, actor)
                held[actor] = None
            elif 'fill_plate' in events:
                held[actor] = new_item('soup')
            elif events & {'place_onion_pot', 'deliver_soup'}:
                held[actor] = None
            else:
                held[actor] = None
                flag(step, actor, 'unclassified_item_transition', identity)
    return flags

def calculate_task_shares(metadata_rows, events, flags):
    counts = Counter()
    diagnostics = Counter(((f['trial_uid'], f['reason']) for f in flags))
    for event in events:
        if event['validation_status'] != 'held_transition_consistent':
            raise ValueError('Resolve unvalidated events before calculating task shares')
        category = task_group(event['event'], event['layout'])
        if category:
            counts[event['trial_uid'], event['actor'], category] += 1
    rows = []
    for meta in metadata_rows:
        uid = meta['trial_uid']
        row = {**meta, 'same_chef_retrievals': diagnostics[uid, 'same_chef_retrieval'], 'repeated_handling_flags': diagnostics[uid, 'repeated_item_handling'], 'handling_review_flags': sum((v for (trial, reason), v in diagnostics.items() if trial == uid)), 'count_policy': 'qualifying_actions_no_handling_exclusions'}
        for category in ('total', 'onion', 'plate', 'soup'):
            groups = ('onion', 'plate', 'soup') if category == 'total' else (category,)
            a = sum((counts[uid, 'agent', g] for g in groups))
            c = sum((counts[uid, 'clone', g] for g in groups))
            row.update({f'agent_{category}_task_count': a, f'clone_{category}_task_count': c, f'combined_{category}_task_count': a + c, f'agent_{category}_task_share': a / (a + c) if a + c else None, f'zero_combined_{category}_actions': a + c == 0})
        for actor in ('agent', 'clone'):
            denominator = row[f'{actor}_total_task_count']
            for category in ('onion', 'plate', 'soup'):
                row[f'{actor}_{category}_task_proportion'] = row[f'{actor}_{category}_task_count'] / denominator if denominator else None
        rows.append(row)
    return rows

CONFLICT_TYPES = ('agent_bumps_clone', 'clone_bumps_agent', 'same_tile_collision', 'attempted_swap')

def summarize_conflicts(steps, uid):
    episodes, evidence = ([], [])
    start = None
    for i, step in enumerate(steps):
        if step['step'] != i:
            raise ValueError('Conflict episodes require complete, zero-based contiguous timesteps')
        flags = step['movement_conflict']
        assert sum((bool(flags[k]) for k in CONFLICT_TYPES)) == int(flags['conflict'])
        evidence.append({'trial_uid': uid, 'step': i, **flags})
        if flags['conflict'] and start is None:
            start = i
        if start is not None and (not flags['conflict'] or i == len(steps) - 1):
            end = i if flags['conflict'] else i - 1
            episodes.append({'trial_uid': uid, 'episode': len(episodes), 'start_step': start, 'end_step': end, 'duration': end - start + 1, 'ongoing_at_end': bool(flags['conflict'] and i == len(steps) - 1)})
            start = None
    durations = [e['duration'] for e in episodes]
    count = sum((bool(e['conflict']) for e in evidence))
    summary = {f'{k}_steps': sum((bool(e[k]) for e in evidence)) for k in CONFLICT_TYPES}
    summary.update(conflict_steps=count, conflict_rate=count / len(steps) if steps else None, conflict_episode_count=len(episodes), conflict_duration_mean=sum(durations) / len(durations) if durations else None, conflict_duration_max=max(durations) if durations else None, conflict_ongoing_at_end=bool(episodes and episodes[-1]['ongoing_at_end']))
    assert sum(durations) == count == sum((summary[f'{k}_steps'] for k in CONFLICT_TYPES))
    return (summary, episodes, evidence)

def conflict_fixture(types):
    return [{'step': i, 'movement_conflict': {**{k: k == kind for k in CONFLICT_TYPES}, 'conflict': kind is not None}} for i, kind in enumerate(types)]

CLEARANCE_WINDOWS = (1, 3, 5, 10)

CLEARANCE_DIRECTIONS = ('agent_clears_for_clone', 'clone_clears_for_agent')

def clearance_encounters(steps, uid, windows=CLEARANCE_WINDOWS):
    if not windows or any((not isinstance(w, int) or isinstance(w, bool) or w <= 0 for w in windows)):
        raise ValueError('Clearance windows must be positive integers')
    horizon = max(windows)
    encounters = []
    for blocked, blocker, bump, direction in [(1, 0, 'clone_bumps_agent', 'agent_clears_for_clone'), (0, 1, 'agent_bumps_clone', 'clone_clears_for_agent')]:
        active, previous_bump_tile = (None, None)
        for i, step in enumerate(steps):
            if step['step'] != i:
                raise ValueError('Clearance requires contiguous zero-based timesteps')
            ev = step['movement_evidence']
            pre, post = (ev['pre_positions'], ev['post_positions'])
            bump_tile = tuple(pre[blocker]) if step['movement_conflict'][bump] else None
            if active is not None:
                tile = active['target_tile']
                elapsed = i - active['start_step']
                if elapsed <= horizon:
                    if active['vacated_step'] is None and tuple(pre[blocker]) == tile and (tuple(post[blocker]) != tile):
                        active['vacated_step'] = i
                    if active['vacated_step'] is not None and tuple(post[blocker]) == tile:
                        active['outcome'] = 'blocker_returned'
                    elif tuple(post[blocked]) == tile and active['vacated_step'] is not None:
                        active['passage_step'] = i
                        active['passage_latency'] = elapsed
                        active['outcome'] = 'passage'
                    elif tuple(post[blocked]) != tuple(pre[blocked]):
                        active['outcome'] = 'blocked_chef_moved_elsewhere'
                    elif elapsed == horizon:
                        active['outcome'] = 'window_expired'
                else:
                    active['outcome'] = 'window_expired'
                if active['outcome'] != 'pending':
                    active['end_step'] = i
                    active = None
            if active is None and bump_tile is not None and (bump_tile != previous_bump_tile):
                active = {'trial_uid': uid, 'direction': direction, 'start_step': i, 'target_tile': bump_tile, 'vacated_step': None, 'passage_step': None, 'passage_latency': None, 'outcome': 'pending', 'end_step': None, 'followup_steps': len(steps) - 1 - i}
                encounters.append(active)
            previous_bump_tile = bump_tile
        if active is not None:
            active['outcome'] = 'trial_ended'
            active['end_step'] = len(steps) - 1
    return sorted(encounters, key=lambda x: (x['start_step'], x['direction']))

def summarize_clearance(encounters, windows=CLEARANCE_WINDOWS):
    result = {}
    for direction in CLEARANCE_DIRECTIONS:
        selected = [e for e in encounters if e['direction'] == direction]
        result[f'{direction}_opportunities'] = len(selected)
        for window in windows:
            eligible = [e for e in selected if e['followup_steps'] >= window]
            successes = [e for e in eligible if e['passage_latency'] is not None and e['passage_latency'] <= window]
            prefix = f'{direction}_w{window}'
            result.update({f'{prefix}_eligible': len(eligible), f'{prefix}_successes': len(successes), f'{prefix}_rate': len(successes) / len(eligible) if eligible else None, f'{prefix}_latency_mean': sum((e['passage_latency'] for e in successes)) / len(successes) if successes else None, f'{prefix}_censored': len(selected) - len(eligible)})
    return result

def clearance_fixture(transitions):
    records = []
    for i, (pre, candidate, post) in enumerate(transitions):
        records.append({'step': i, 'movement_conflict': movement_conflict(pre, candidate), 'movement_evidence': {'pre_positions': pre, 'candidate_positions': candidate, 'post_positions': post}})
    return records

def team_effectiveness(metadata_rows, events):
    deliveries = {}
    for event in events:
        if event['event'] != 'deliver_soup':
            continue
        if event['validation_status'] != 'held_transition_consistent':
            raise ValueError('Unvalidated delivery event')
        deliveries.setdefault(event['trial_uid'], []).append(event)
    summaries = {}
    for meta in metadata_rows:
        uid = meta['trial_uid']
        if uid in summaries:
            raise ValueError('Duplicate trial metadata')
        selected = deliveries.get(uid, [])
        duration = meta['observed_steps']
        seen = set()
        counts = {'agent': 0, 'clone': 0}
        for event in selected:
            key = (event['step'], event['actor'])
            if key in seen or event['actor'] not in counts or (not 0 <= event['step'] < duration):
                raise ValueError('Duplicate, invalid actor, or out-of-range delivery')
            seen.add(key)
            counts[event['actor']] += 1
        first = min((e['step'] for e in selected), default=None)
        summaries[uid] = {'agent_deliveries': counts['agent'], 'clone_deliveries': counts['clone'], 'team_deliveries': len(selected), 'first_delivery_step': first, 'first_delivery_latency_steps': first + 1 if first is not None else None, 'no_delivery': not selected, 'effectiveness_duration_not_400': duration != 400}
        assert summaries[uid]['team_deliveries'] == sum(counts.values())
    if set(deliveries) - set(summaries):
        raise ValueError('Delivery events without matching trial metadata')
    return summaries

def delivery_fixture(step, actor='agent'):
    return dict(trial_uid='test', event='deliver_soup', step=step, actor=actor, validation_status='held_transition_consistent')

def handling_step(step, pre, post, events):
    side = synthetic_side(events, pre, post)
    side['action'] = INTERACT_ACTION
    return synthetic_step(step, agent=side)
