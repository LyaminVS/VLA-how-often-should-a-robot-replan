#!/usr/bin/env python3
"""Offline, paired command-discontinuity audit of the frozen 270-episode cohort.

No policy calls, simulator steps, or changes to existing evidence. Distances are
in controller-command coordinates, not metres or physical jerk. Each episode
is one observation for aggregation/bootstrap; calls are never iid replicates.
"""
from __future__ import annotations
import argparse,csv,hashlib,json,math
from collections import defaultdict
from pathlib import Path
import numpy as np

METHODS=['fixed_1','fixed_5','fixed_10','fixed_25','fixed_50','dvac_cap_5','dvac_cap_10','dvac_cap_25','dvac_cap_50']
LABELS=['K=1','K=5','K=10','K=25','K=50','DVAC C=5','DVAC C=10','DVAC C=25','DVAC C=50']
PREFIX=60
SEED=20261024
BOOTSTRAPS=10000

def sha(p): return hashlib.sha256(p.read_bytes()).hexdigest()
def read_jsonl(p):
    with p.open() as f:
        for line in f:
            yield json.loads(line)
def mean(x): return float(np.mean(x)) if len(x) else None
def csvwrite(p,rows):
    if not rows:return
    fields=list(dict.fromkeys(k for r in rows for k in r))
    with p.open('w') as f:
        w=csv.DictWriter(f,fieldnames=fields);w.writeheader();w.writerows(rows)
def boot_ci(x, rng):
    x=np.asarray(x,float);means=x[rng.integers(0,len(x),(BOOTSTRAPS,len(x)))].mean(1)
    return [float(v) for v in np.quantile(means,[.025,.975])]
def bootstrap_paired(a,b,key,rng):
    delta={r['pair_id']:r[key]-b[r['pair_id']][key] for r in a if r[key] is not None and b[r['pair_id']][key] is not None}
    groups=defaultdict(list)
    for r in a:
        if r['pair_id'] in delta:groups[r['task']].append(delta[r['pair_id']])
    draws=[]
    for x in groups.values():
        x=np.asarray(x);draws.append(x[rng.integers(0,len(x),(BOOTSTRAPS,len(x)))])
    ci=np.quantile(np.concatenate(draws,axis=1).mean(1),[.025,.975])
    return {'difference':mean(list(delta.values())),'ci_low':float(ci[0]),'ci_high':float(ci[1]),'pairs':len(delta)}

def run(repo,out):
    out.mkdir(parents=True,exist_ok=True)
    index=repo/'outputs/capped-report/episodes.csv'
    with index.open() as f:cohort=list(csv.DictReader(f))
    assert len(cohort)==270
    hashes={str(index.resolve()):sha(index)};episodes=[];boundaries=[];firsts={};tracechecks=0;caps_hashchecks=0
    for ri,record in enumerate(cohort):
        folder=repo/'outputs'/Path(record['raw_source_root']).name/'episodes'/record['episode_id']
        summary=json.loads((folder/'summary.json').read_text())
        assert summary['episode_id']==record['episode_id']
        assert int(record['steps'])==summary['steps'] and (record['success']=='True')==summary['success']
        for filename in ['summary.json','steps.jsonl','chunks.jsonl']:
            hashes[str((folder/filename).resolve())]=sha(folder/filename)
            expected=summary.get('evidence_sha256',{}).get(filename)
            if expected is not None:assert hashes[str((folder/filename).resolve())]==expected;caps_hashchecks+=1
        rows=list(read_jsonl(folder/'steps.jsonl'))
        # Ignore diagnostics in memory once a chunk has been parsed.
        chunks=[{k:v for k,v in row.items() if k in ('step','sampling_seed','actions','executed_steps')} for row in read_jsonl(folder/'chunks.jsonl')]
        assert len(rows)==summary['steps'] and len(chunks)==summary['model_calls']
        assert len(rows)>=PREFIX
        assert [r['step'] for r in rows]==list(range(len(rows)))
        assert [r['step'] for r in rows if r['replanned']]==[c['step'] for c in chunks]
        actions=np.asarray([r['action'] for r in rows],dtype=float)
        assert actions.shape==(len(rows),7) and np.isfinite(actions).all()
        starts=np.asarray([c['step'] for c in chunks]);plans=[np.asarray(c['actions'],float) for c in chunks]
        pair=summary['pair_id'];first=(summary['initial_state_sha256'],summary['initial_observation_sha256'],chunks[0]['sampling_seed'],plans[0].tolist())
        if pair in firsts:assert firsts[pair]==first
        else:firsts[pair]=first
        for ci,(c,plan) in enumerate(zip(chunks,plans)):
            assert plan.shape==(50,7) and np.isfinite(plan).all()
            stop=int(starts[ci+1]) if ci+1<len(starts) else len(rows)
            assert np.array_equal(actions[c['step']:stop],np.clip(plan[:stop-c['step']],-1,1))
            for step in range(c['step'],stop):assert rows[step]['chunk_index']==step-c['step']
            tracechecks+=stop-c['step']
        jumps=np.linalg.norm(np.diff(actions[:,:3],axis=0),axis=1)
        rots=np.linalg.norm(np.diff(actions[:,3:6],axis=0),axis=1)
        flips=np.diff(actions[:,6]>0)!=0
        replan=np.asarray([r['replanned'] for r in rows[1:]],bool)
        ep={k:record[k] for k in ['episode_id','method','task','state','policy_seed']}
        ep.update(pair_id=pair,success=summary['success'],steps=len(rows),model_calls=len(chunks),source_dir=str(folder.resolve()))
        for prefix,count in [('full',len(rows)),('first60',PREFIX)]:
            n=count-1
            ep.update({f'{prefix}_translation_tv100':float(jumps[:n].mean()*100),f'{prefix}_rotation_tv100':float(rots[:n].mean()*100),f'{prefix}_gripper_flips100':float(flips[:n].mean()*100),f'{prefix}_gripper_flips':int(flips[:n].sum()),f'{prefix}_boundary_translation_mean':mean(jumps[:n][replan[:n]]),f'{prefix}_within_translation_mean':mean(jumps[:n][~replan[:n]])})
        own=[]
        for ci in range(1,len(chunks)):
            step=int(starts[ci]);k=step-int(starts[ci-1]);old=plans[ci-1];new=plans[ci]
            b={'episode_id':record['episode_id'],'method':record['method'],'pair_id':pair,'task':record['task'],'state':record['state'],'success':summary['success'],'step':step,'elapsed_prefix':k,'overlap':max(0,50-k),'executed_translation_jump':float(jumps[step-1]),'executed_rotation_jump':float(rots[step-1]),'executed_gripper_flip':int(flips[step-1])}
            b.update(revision_translation_raw=None,revision_translation_clipped=None,revision_rotation_clipped=None,revision_gripper_flip=None,overlap10_translation_mean=None,old_planned_translation_jump=None,excess_boundary_translation_jump=None)
            if k<50:
                oldnext=np.clip(old[k],-1,1);newnext=np.clip(new[0],-1,1)
                rev=float(np.linalg.norm(new[0,:3]-old[k,:3]));oldjump=float(np.linalg.norm(oldnext[:3]-actions[step-1,:3]));L=min(10,50-k)
                b.update(revision_translation_raw=rev,revision_translation_clipped=float(np.linalg.norm(newnext[:3]-oldnext[:3])),revision_rotation_clipped=float(np.linalg.norm(newnext[3:6]-oldnext[3:6])),revision_gripper_flip=int((newnext[6]>0)!=(oldnext[6]>0)),overlap10_translation_mean=float(np.linalg.norm(np.clip(new[:L,:3],-1,1)-np.clip(old[k:k+L,:3],-1,1),axis=1).mean()),old_planned_translation_jump=oldjump,excess_boundary_translation_jump=float(jumps[step-1]-oldjump))
                saved=rows[step].get('plan_revision_translation_l2')
                assert saved is not None and math.isclose(saved,rev,abs_tol=1e-10)
            else:assert rows[step].get('plan_revision_translation_l2') is None
            own.append(b);boundaries.append(b)
        for prefix in ['full','first60']:
            select=[b for b in own if b['revision_translation_clipped'] is not None and (prefix=='full' or b['step']<PREFIX)]
            ep[f'{prefix}_overlap_boundaries']=len(select)
            for key in ['revision_translation_clipped','revision_rotation_clipped','revision_gripper_flip','overlap10_translation_mean','excess_boundary_translation_jump']:
                ep[f'{prefix}_{key}']=mean([b[key] for b in select])
        episodes.append(ep)
        if (ri+1)%30==0:print(f'Validated {ri+1}/270 episodes',flush=True)
    assert len(firsts)==30
    assert all(sum(e['method']==m for e in episodes)==30 for m in METHODS)
    csvwrite(out/'episodes.csv',episodes);csvwrite(out/'boundaries.csv',boundaries)
    rng=np.random.default_rng(SEED);groups=[]
    metrics=[k for k,v in episodes[0].items() if (k.startswith('full_') or k.startswith('first60_')) and not k.endswith('overlap_boundaries')]
    for method in METHODS:
        values=[e for e in episodes if e['method']==method];row={'method':method,'episodes':len(values),'successes':sum(e['success'] for e in values)}
        for key in metrics:
            valid=[e[key] for e in values if e[key] is not None];row[key]=mean(valid)
            row[key+'_n']=len(valid)
            if valid:row[key+'_ci_low'],row[key+'_ci_high']=boot_ci(valid,rng)
        groups.append(row)
    csvwrite(out/'methods.csv',groups)
    contrasts=[]
    for method,reference in [('fixed_1','fixed_5'),('fixed_1','fixed_10'),('fixed_5','fixed_10'),('dvac_cap_10','fixed_10'),('dvac_cap_50','fixed_50')]:
        a=[e for e in episodes if e['method']==method];b={e['pair_id']:e for e in episodes if e['method']==reference}
        for key in metrics:
            if any(e[key] is None for e in a) or any(e[key] is None for e in b.values()):continue
            contrasts.append({'method':method,'reference':reference,'metric':key,**bootstrap_paired(a,b,key,rng)})
    csvwrite(out/'paired_contrasts.csv',contrasts)
    cells=[]
    for method in METHODS:
        for task in sorted({e['task'] for e in episodes}):
            for success in [False,True]:
                vals=[e for e in episodes if e['method']==method and e['task']==task and e['success']==success]
                if vals:cells.append({'method':method,'task':task,'success':success,'episodes':len(vals),**{k:mean([e[k] for e in vals if e[k] is not None]) for k in metrics}})
    csvwrite(out/'task_outcome_groups.csv',cells)
    result={'design':{'episodes':270,'paired_initial_settings':30,'methods':METHODS,'common_prefix_actions':PREFIX,'shortest_episode_actions':min(e['steps'] for e in episodes),'control_hz':20,'bootstraps':BOOTSTRAPS,'bootstrap_seed':SEED,'paired_contrasts':'task-stratified paired resampling of 30 settings; descriptive 95% percentile intervals; no multiplicity adjustment','method_intervals':'episode bootstrap; pooled over the fixed 3 tasks; descriptive','units':'clipped controller-command coordinates, not Cartesian metres, radians, or physical jerk','overlap':'K=50 has no unexecuted tail; revision metrics are undefined, not zero','interpretation':'observational diagnostics, not causal attribution; trajectories and termination times differ'},'checks':{'all_270_episode_traces_validated':True,'executed_actions_matched_to_saved_predictions':tracechecks,'first_prediction_and_reset_matched_across_nine_methods':True,'capped_hash_matches':caps_hashchecks,'all_episodes_cover_first60':True},'methods':groups,'paired_contrasts':contrasts,'sources_sha256':hashes}
    (out/'analysis.json').write_text(json.dumps(result,indent=2,allow_nan=False))
    # Record analyzer identity independently of output file hashes.
    (out/'provenance.json').write_text(json.dumps({'analyzer':str(Path(__file__).resolve()),'analyzer_sha256':sha(Path(__file__)),'inputs':hashes,'outputs':{p.name:sha(p) for p in out.iterdir() if p.suffix in ('.csv','.json') and p.name!='provenance.json'}},indent=2))
    for row in groups:print(row['method'], 'TV100',round(row['full_translation_tv100'],2),'first60',round(row['first60_translation_tv100'],2),'flips100',round(row['full_gripper_flips100'],2),'revision',row['full_revision_translation_clipped'],flush=True)
    return result

if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__);parser.add_argument('--repo',type=Path,default=Path(__file__).resolve().parents[1]);parser.add_argument('--output',type=Path,required=True);args=parser.parse_args();run(args.repo.resolve(),args.output.resolve())
