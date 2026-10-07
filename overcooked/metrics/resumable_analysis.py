"""Filtered, memory-bounded trial analysis with transactional checkpoints."""
from pathlib import Path
import base64, csv, hashlib, json, sqlite3, sys, time, zlib
from collections import Counter
import metric_core as core

AGENTS = ('Baseline', 'CF', 'CCM', 'CCM+CF')

def raw_key(raw):
    return hashlib.sha256(json.dumps(raw, sort_keys=True).encode()).hexdigest()

def connect(path):
    path = Path(path); path.parent.mkdir(parents=True, exist_ok=True)
    db = sqlite3.connect(path)
    db.execute('PRAGMA journal_mode=WAL')
    db.execute('PRAGMA synchronous=FULL')
    return db

class FastReplay:
    def __init__(self, root):
        sys.path.insert(0, str(root))
        import jax, jax.numpy as jnp
        import numpy as np
        from jaxmarl import make
        from jaxmarl.environments.overcooked.layouts import overcooked_layouts
        from jaxmarl.environments.overcooked.overcooked import DIR_TO_VEC, Actions
        from simulator_adapter import _get_layout_stations
        from simulator_adapter import LAYOUT_NAME_MAP, detect_events_from_inv
        self.jax, self.jnp, self.np = jax, jnp, np
        self.make, self.layouts, self.stations_fn = make, overcooked_layouts, _get_layout_stations
        self.dirs, self.Actions, self.detect = DIR_TO_VEC, Actions, detect_events_from_inv
        core.LAYOUT_NAME_MAP = LAYOUT_NAME_MAP
        self.cache = {}
        self.batch_cache = {}

    def bundle(self, layout):
        if layout in self.cache: return self.cache[layout]
        jax, jnp = self.jax, self.jnp
        env = self.make('overcooked', layout=self.layouts[layout])
        stations = self.stations_fn(env)
        dirs, Actions = self.dirs, self.Actions
        def run(seed, actions):
            key = jax.random.PRNGKey(seed)
            key, reset_key = jax.random.split(key)
            _, state = env.reset(reset_key)
            def step(carry, action_ids):
                key, state = carry
                key, _, _, _ = jax.random.split(key, 4)
                key, step_key = jax.random.split(key)
                transition_key, _ = jax.random.split(step_key)  # env.step's internal split
                action = env.action_set.take(indices=action_ids)
                moving = ((action != Actions.stay) & (action != Actions.interact))[:, None]
                target = jnp.minimum(jnp.maximum(state.agent_pos + moving * dirs[jnp.minimum(action, 3)]
                    + ~moving * state.agent_dir, 0), jnp.array((env.width-1, env.height-1), dtype=jnp.uint32))
                walls = state.wall_map[target[:,1], target[:,0]]
                goals = jnp.any(jnp.all(target[:,None,:] == state.goal_pos[None,:,:], axis=2), axis=1)
                bounced = (walls | goals)[:,None] | ~moving
                candidates = (bounced * state.agent_pos + ~bounced * target).astype(jnp.uint32)
                _, post, reward, done, _ = env.step_env(transition_key, state,
                    {'agent_0': action_ids[0], 'agent_1': action_ids[1]})
                result = (state.agent_pos, state.agent_dir, state.agent_inv, post.agent_inv,
                    post.agent_pos, candidates, jnp.array([reward['agent_0'],reward['agent_1']]), done['__all__'])
                return (key, post), result
            return jax.lax.scan(step, (key, state), actions)[1]
        bundle = (env, stations, jax.jit(run))
        self.cache[layout] = bundle
        return bundle

    def decode_batch(self, raws, layout, batch_size=8):
        # Fixed-size padding avoids recompiling the last short batch.
        if not raws: return []
        if batch_size == 1: return [self.decode(raw,layout) for raw in raws]
        if len(raws)>batch_size: raise ValueError('Batch exceeds configured size')
        np=self.np
        arrays=[];seeds=[]
        for raw in raws:
            codes=np.frombuffer(zlib.decompress(base64.b64decode(raw['actions_b64'],validate=True)),dtype=np.uint8)
            if len(codes)!=int(raw['total_steps']) or np.any(codes>=36):raise ValueError('Invalid action stream')
            seat=int(raw['agent_seat']); actions=np.empty((len(codes),2),dtype=np.uint32)
            actions[:,seat],actions[:,1-seat]=codes//6,codes%6
            arrays.append(actions);seeds.append(int(raw['episode_seed']))
        while len(arrays)<batch_size:arrays.append(arrays[-1]);seeds.append(seeds[-1])
        if (layout,batch_size) not in self.batch_cache:
            _,_,single=self.bundle(layout)
            self.batch_cache[layout,batch_size]=self.jax.jit(self.jax.vmap(single))
        output=self.jax.device_get(self.batch_cache[layout,batch_size](np.array(seeds,dtype=np.uint32),np.stack(arrays)))
        return [self.decode(raw,layout,tuple(a[i] for a in output)) for i,raw in enumerate(raws)]

    def decode(self, raw, layout, precomputed=None):
        np = self.np
        packed = zlib.decompress(base64.b64decode(raw['actions_b64'], validate=True))
        if len(packed) != int(raw['total_steps']) or any(x >= 36 for x in packed):
            raise ValueError('Invalid packed actions')
        seat = int(raw['agent_seat'])
        if seat not in (0,1): raise ValueError('Invalid agent seat')
        codes = np.frombuffer(packed, dtype=np.uint8)
        actions = np.empty((len(codes),2), dtype=np.uint32)
        actions[:,seat], actions[:,1-seat] = codes // 6, codes % 6
        env, stations, run = self.bundle(layout)
        arrays = precomputed if precomputed is not None else self.jax.device_get(run(np.uint32(int(raw['episode_seed'])), actions))
        positions, directions, held_pre, held_post, post_positions, candidates, rewards, dones = arrays
        held_names = {1:'empty',3:'onion',5:'plate',9:'soup'}
        steps = []
        for i in range(len(codes)):
            semantic = {k:[a[i,seat].tolist(), a[i,1-seat].tolist()] for k,a in [
                ('pre_positions',positions),('post_positions',post_positions),('candidate_positions',candidates)]}
            flags = core.movement_conflict(semantic['pre_positions'],semantic['candidate_positions'])
            if semantic['post_positions'] != (semantic['pre_positions'] if flags['conflict'] else semantic['candidate_positions']):
                raise ValueError('Movement reconstruction disagrees with simulator')
            side = {}
            for actor,s in [('agent',seat),('clone',1-seat)]:
                pre, post = held_names[int(held_pre[i,s])], held_names[int(held_post[i,s])]
                pos, direction = positions[i,s].tolist(), directions[i,s].tolist()
                ev = self.detect(act=int(actions[i,s]),pre_inv=pre,post_inv=post,reward=float(rewards[i,s]),
                    agent_pos=pos,agent_dir=direction,pot_positions=stations['pots'],serving_positions=stations['serving_counters'])
                # Rewards are shared: the partner may deliver while this chef
                # deposits soup on a counter. Classify by interaction target.
                target = tuple(pos[j] + direction[j] for j in (0,1))
                serving = {tuple(p) for p in stations['serving_counters']}
                if int(actions[i,s]) == 5 and pre == 'soup' and post == 'empty' and target not in serving:
                    ev['place_soup_counter'] = True
                side[actor] = dict(action=int(actions[i,s]),pos=pos,dir=direction,held_pre=pre,held_post=post,
                    reward=float(rewards[i,s]),events=[k for k,v in ev.items() if v])
            # Record observed movement alongside intended-movement evidence.
            prepos, postpos = positions[i].tolist(),post_positions[i].tolist()
            tried = [int(x) < 4 for x in actions[i]]
            swap = all(tried) and postpos == list(reversed(prepos)) and postpos != prepos
            mutual = all(tried) and postpos == prepos and sum(abs(a-b) for a,b in zip(*prepos)) == 1
            steps.append(dict(step=i,**side,collision=bool(swap or mutual),movement_evidence=semantic,movement_conflict=flags))
            if bool(dones[i]): break
        if len(steps) != len(codes): raise ValueError('Early terminal transition: logged duration does not match replay')
        return {'trial_meta':{k:v for k,v in raw.items() if k != 'actions_b64'}, 'steps':steps}


def inventory(root):
    records, sources = [], []
    for path in sorted(Path(root).glob('evaluation/agents_competence_with_logs/*_trial_logs.csv')):
        with path.open(newline='') as f:
            for number, raw in enumerate(csv.DictReader(f),2):
                records.append(dict(raw=raw,source_file=str(path.relative_to(root)),source_row=number))
        sources.append({'path':str(path),'sha256':core.digest(path)})
    return records, sources


def result_metadata(root, records):
    index, sources = {}, []
    wanted = {(r['source_file'],)+tuple(r['raw'][k] for k in core.JOIN_FIELDS) for r in records}
    for source in sorted({r['source_file'] for r in records}):
        path = Path(root)/source.replace('_trial_logs.csv','_results.csv')
        with path.open(newline='') as f:
            for r in csv.DictReader(f):
                key=(source,)+tuple(r[k] for k in core.JOIN_FIELDS)
                if key not in wanted:continue
                value=(r['clone_score'],r['clone_group'],r['layout'],r)
                if key in index and index[key]!=value:raise ValueError('Conflicting actual competence metadata')
                index[key]=value
        sources.append({'path':str(path),'sha256':core.digest(path)})
    if set(index)!=wanted:raise ValueError('Missing results matches')
    return index,sources


def features(record, decoded, results, replay):
    meta=core.metadata(record); raw=record['raw'];uid=meta['trial_uid']
    key=(record['source_file'],)+tuple(raw[k] for k in core.JOIN_FIELDS)
    score,group,layout,*recorded=results[key]
    if layout!=meta['layout'] or not score.strip():raise ValueError('Invalid competence join')
    meta.update(clone_score=float(score),clone_group=group,agent_score_input=float(meta['agent_score_input']),
        observed_steps=len(decoded['steps']),terminal_state_corrections=int(bool(decoded['steps']) and len(decoded['steps'])==int(raw['total_steps'])))
    _, events, issues=core.reconstruct(decoded,meta)
    if issues:raise ValueError(f'Reconstruction issues: {issues[:3]}')
    _,stations,_=replay.bundle(layout)
    flags=core.handling_diagnostics(dict(decoded,_stations=stations),uid)
    row=core.calculate_task_shares([meta],events,flags)[0]
    summary,episodes,_=core.summarize_conflicts(decoded['steps'],uid);row.update(summary)
    row.update(core.team_effectiveness([meta],events)[uid])
    return row,dict(handling_flags=flags,conflict_episodes=episodes,replay_evidence=decoded)


def export_csv(db,path):
    path=Path(path);temp=path.with_suffix('.csv.tmp')
    cursor=db.execute('SELECT payload FROM trials ORDER BY ordinal')
    first=cursor.fetchone()
    if first is None:return
    row=json.loads(first[0]);fields=list(row)
    with temp.open('w',newline='') as f:
        writer=csv.DictWriter(f,fieldnames=fields);writer.writeheader();writer.writerow(row)
        for (payload,) in cursor:writer.writerow(json.loads(payload))
        f.flush()
        import os;os.fsync(f.fileno())
    temp.replace(path)


def run(root, output_dir, agents=AGENTS, batch_size=1000, limit_per_agent_layout=None, max_new_trials=None, replay_batch_size=1):
    """Never accumulates trial trajectories; commit each finished trial and export per batch."""
    import os, platform
    root=Path(root).resolve();out=Path(output_dir).resolve();out.mkdir(parents=True,exist_ok=True)
    replay=FastReplay(root)
    all_records,source_fingerprints=inventory(root)
    records=[];strata=Counter()
    for r in all_records:
        label=core.standardize_agent_name(r['raw']['agent_name'])
        if label not in agents:continue
        layout=core.resolve_layout(r['raw']);key=(label,layout)
        if limit_per_agent_layout is not None and strata[key]>=limit_per_agent_layout:continue
        strata[key]+=1;records.append(r)
    print(f'Selected {len(records):,}/{len(all_records):,} trials: {dict(Counter(core.standardize_agent_name(r["raw"]["agent_name"]) for r in records))}',flush=True)
    results,result_fingerprints=result_metadata(root,records)
    fingerprint_data={'inputs':source_fingerprints,'results':result_fingerprints,'agents':list(agents),
        'limit':limit_per_agent_layout,'python':platform.python_version(),'jax':replay.jax.__version__,
        'code':{str(p):core.digest(p) for p in [Path(__file__),Path(core.__file__),Path(__file__).with_name('simulator_adapter.py')]}}
    import inspect
    fingerprint_data['environment_sha256']=core.digest(inspect.getfile(type(replay.bundle(core.resolve_layout(records[0]['raw']))[0]))) if records else None
    fingerprint=hashlib.sha256(json.dumps(fingerprint_data,sort_keys=True).encode()).hexdigest()
    run_dir=out/'checkpoint_runs'/fingerprint[:20];run_dir.mkdir(parents=True,exist_ok=True)
    db=connect(run_dir/'trials.sqlite')
    db.execute('CREATE TABLE IF NOT EXISTS trials (trial_uid TEXT PRIMARY KEY, ordinal INTEGER UNIQUE, raw_key TEXT, payload TEXT, audit BLOB)')
    db.execute('CREATE TABLE IF NOT EXISTS errors (trial_uid TEXT, error TEXT, created_at TEXT DEFAULT CURRENT_TIMESTAMP)')
    completed={r[0] for r in db.execute('SELECT trial_uid FROM trials')}
    manifest=dict(fingerprint=fingerprint,configuration=fingerprint_data,selected_trials=len(records),status='running')
    (run_dir/'manifest.json').write_text(json.dumps(manifest,indent=2))
    recovery_path=out/'recovered_replays.sqlite'
    recovery=sqlite3.connect(f'file:{recovery_path}?mode=ro',uri=True) if recovery_path.exists() else None
    print(f'Resuming {len(completed):,} saved trials. Checkpoints: {run_dir}',flush=True)
    start=time.perf_counter();new=0;reused=0;prefetched={}
    if batch_size<1 or replay_batch_size<1:raise ValueError("Batch sizes must be positive")
    try:
        for ordinal,record in enumerate(records):
            meta=core.metadata(record);uid=meta['trial_uid']
            if uid in completed:continue
            if max_new_trials is not None and new>=max_new_trials:break
            raw=record['raw'];key=raw_key(raw)
            try:
                saved=recovery.execute('SELECT payload FROM replays WHERE raw_key=?',(key,)).fetchone() if recovery else None
                if saved:
                    decoded=json.loads(zlib.decompress(saved[0]));reused+=1
                    # Only reuse compatible, complete state evidence. Never guess missing fields.
                    if len(decoded['steps'])!=int(raw['total_steps']) or any('movement_conflict' not in s or 'movement_evidence' not in s for s in decoded['steps']):
                        raise ValueError('Recovered replay lacks required movement evidence')
                    codes=zlib.decompress(base64.b64decode(raw['actions_b64']))
                    for s,code in zip(decoded['steps'],codes):
                        if (s['agent']['action'],s['clone']['action'])!=divmod(code,6):raise ValueError('Recovered action mismatch')
                else:
                    if uid not in prefetched:
                        fresh=[]
                        for candidate in records[ordinal:]:
                            cm=core.metadata(candidate)
                            if cm['trial_uid'] in completed:continue
                            if cm['layout']!=meta['layout'] or candidate['raw']['total_steps']!=raw['total_steps']:break
                            if recovery and recovery.execute('SELECT 1 FROM replays WHERE raw_key=?',(raw_key(candidate['raw']),)).fetchone():continue
                            fresh.append(candidate)
                            if len(fresh)>=replay_batch_size or (max_new_trials is not None and new+len(fresh)>=max_new_trials):break
                        decoded_batch=replay.decode_batch([c['raw'] for c in fresh],meta['layout'],replay_batch_size)
                        prefetched.update({core.metadata(c)['trial_uid']:d for c,d in zip(fresh,decoded_batch)})
                    decoded=prefetched.pop(uid)
                row,audit=features(record,decoded,results,replay)
                db.execute('INSERT INTO trials VALUES (?,?,?,?,?)',(uid,ordinal,key,json.dumps(row),zlib.compress(json.dumps(audit).encode(),3)))
                db.commit()  # Each completed trial is durable, not just the batch.
                completed.add(uid);new+=1
                del decoded,audit,row
            except Exception as exc:
                db.rollback();db.execute('INSERT INTO errors(trial_uid,error) VALUES (?,?)',(uid,repr(exc)));db.commit();raise
            if new%batch_size==0:
                export_csv(db,run_dir/'trial_features_partial.csv')
            if new%100==0 or new%batch_size==0:
                elapsed=time.perf_counter()-start
                print(f'{len(completed):,}/{len(records):,} saved; {elapsed/new:.3f} sec/new trial; reused {reused:,}; remaining estimate {(len(records)-len(completed))*elapsed/new/3600:.2f} h',flush=True)
        complete=len(completed)==len(records)
        export_csv(db,run_dir/('trial_features.csv' if complete else 'trial_features_partial.csv'))
        manifest.update(status='complete' if complete else 'partial',completed_trials=len(completed),new_trials=new,recovered_trials=reused,elapsed_seconds=time.perf_counter()-start)
    except BaseException:
        export_csv(db,run_dir/'trial_features_partial.csv')
        manifest.update(status='interrupted_or_failed',completed_trials=len(completed),elapsed_seconds=time.perf_counter()-start)
        raise
    finally:
        (run_dir/'manifest.json').write_text(json.dumps(manifest,indent=2));db.close()
        if recovery:recovery.close()
    print(json.dumps(manifest|{'configuration':'see manifest.json','output_directory':str(run_dir)},indent=2),flush=True)
    return run_dir
