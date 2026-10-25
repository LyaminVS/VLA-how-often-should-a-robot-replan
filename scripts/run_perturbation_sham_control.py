#!/usr/bin/env python3
"""Matched zero-amplitude kinematic controls: four starts and all five K values.

The object pose and velocity are held until first finger contact, exactly as in
the nonzero-amplitude sweep; no external force or model finetuning is used."""
from __future__ import annotations
import argparse,copy,hashlib,json,os,random,sys,time,traceback
from dataclasses import replace
from datetime import datetime,timezone
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
os.environ.setdefault('LIBERO_CONFIG_PATH',str(ROOT/'.cache/libero'))
os.environ.setdefault('MUJOCO_GL','glfw' if sys.platform=='darwin' else 'egl')
os.environ.setdefault('PYTORCH_ENABLE_MPS_FALLBACK','1')
from vla_replan.config import Experiment,Episode,digest
from vla_replan.storage import atomic_json,write_jsonl,exclusive_run,runtime_metadata
from moving_target.config import MovingEpisode,MotionSpec

def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def plan(config):
    states=list(config['base']['initial_states']);rng=random.Random(config['order_seed']);rng.shuffle(states);result=[]
    for state in states:
        pairs=[(k,a) for k in config['base']['intervals'] for a in config['amplitudes_cm']];rng.shuffle(pairs)
        for k,a in pairs:result.append({'state':state,'k':k,'amplitude_cm':a,'seed':config['base']['policy_seeds'][0]})
    return result

def verify_payload(folder,entry,fingerprint):
    ev=json.loads((folder/'evidence.json').read_text());assert ev['config_fingerprint']==fingerprint
    for name,expected in ev['files'].items():assert sha(folder/name)==expected
    s=json.loads((folder/'summary.json').read_text())
    for k in ['state','k','amplitude_cm']:assert s[k]==entry[k]
    return s

def run(path,out,device='mps',limit=None):
    cfg=json.loads(Path(path).read_text());assert cfg['amplitudes_cm']==[0];assert cfg['period_s']==4
    assert cfg['base']['intervals']==[1,5,10,25,50] and len(cfg['base']['initial_states'])==4
    assert cfg['base']['policy_seeds']==[11]
    from vla_replan.config import Task
    raw=copy.deepcopy(cfg['base']);raw['tasks']=tuple(Task(**t) for t in raw['tasks'])
    for field in ['intervals','initial_states','policy_seeds']:raw[field]=tuple(raw[field])
    base=Experiment(**raw);base.validate();entries=plan(cfg);assert len(entries)==20
    sources=[Path(__file__),ROOT/'vla_replan/backend.py',ROOT/'vla_replan/core.py',ROOT/'vla_replan/accelerator.py',ROOT/'moving_target/environment.py',ROOT/'moving_target/runner.py',ROOT/'moving_target/kinematic.py']
    identities={str(p.relative_to(ROOT)):sha(p) for p in sources};fp=digest(cfg);out=Path(out)
    with exclusive_run(out):
        old=json.loads((out/'manifest.json').read_text()) if (out/'manifest.json').exists() else None
        if old:assert old['config_fingerprint']==fp and old['sources']==identities and old['plan']==entries
        else:
            atomic_json(out/'manifest.json',{'data_kind':'paired_zero_amplitude_kinematic_control','created_utc':datetime.now(timezone.utc).isoformat(),'config':cfg,'config_fingerprint':fp,'sources':identities,'plan':entries,'status':'registered_before_policy_load','runtime':runtime_metadata()})
        from vla_replan.backend import SmolVLAPolicy
        from vla_replan.core import run_episode
        from moving_target.kinematic import KinematicTargetEnvironment
        from moving_target.runner import MovingEpisodeVideo,summarize_motion
        import numpy as np
        policy=SmolVLAPolicy(base,device);env=KinematicTargetEnvironment(base,MotionSpec())
        manifest=json.loads((out/'manifest.json').read_text())
        if 'hardware' in manifest:assert manifest['hardware']==policy.runtime
        manifest['hardware']=policy.runtime;manifest['status']='collecting';atomic_json(out/'manifest.json',manifest)
        firsts={};nnew=0;start_all=time.perf_counter()
        # Warm-up is outside the measured rollout and does not alter a treatment's reset.
        warm=Episode(base.tasks[0].suite,base.tasks[0].name,entries[0]['state'],11,10,base.tasks[0].max_steps)
        obs,_=env.reset(warm)
        for i in range(base.warmup_calls):policy.predict(obs,i)
        for i,entry in enumerate(entries):
            episode=Episode(base.tasks[0].suite,base.tasks[0].name,entry['state'],entry['seed'],entry['k'],base.tasks[0].max_steps)
            label=f"amp{entry['amplitude_cm']}cm";video_episode=MovingEpisode(episode,label);folder=out/'episodes'/video_episode.id
            if folder.exists():
                s=verify_payload(folder,entry,fp)
                with (folder/'chunks.jsonl').open() as f:first=json.loads(next(f))['actions']
                key=entry['state'];ident=(s['initial_state_sha256'],s['initial_observation_sha256'],first)
                if key in firsts:assert firsts[key]==ident
                firsts[key]=ident
                continue
            if limit is not None and nnew>=limit:break
            spec=MotionSpec(amplitude_m=.02,periods_s={'static':None,'slow':8.,'fast':4.},**cfg['force_controller']);spec.validate();spec=replace(spec,amplitude_m=0.);env.motion_spec=spec;env.set_condition('fast')
            attempt=out/'attempts'/f'{video_episode.id}__{time.time_ns()}';attempt.mkdir(parents=True);video=None;final={}
            def record(frame,row,outcome):
                nonlocal final
                row['motion']=copy.deepcopy(env.telemetry());row['amplitude_cm']=entry['amplitude_cm'];row['condition']=label
                if outcome is not None:final=row['motion']
                if video:video(frame,row,outcome)
            tick=time.perf_counter()
            try:
                video=MovingEpisodeVideo(attempt/'video.mp4',video_episode,base.control_hz)
                summary,rows,chunks=run_episode(env,policy,episode,chunk_size=50,control_hz=20,record=record)
                video.close();video=None
                identity=(summary['initial_state_sha256'],summary['initial_observation_sha256'],chunks[0]['actions'])
                if entry['state'] in firsts:assert firsts[entry['state']]==identity,'Unmatched physical reset or first prediction'
                firsts[entry['state']]=identity
                motion=summarize_motion(rows,final,base.control_hz)
                assert final['kinematic_writes_after_contact']==0
                assert final['max_imposed_tracking_error_m']<1e-7
                assert final['max_force_after_contact_n']<=1e-10
                assert final['peak_force_n']<=cfg['force_controller']['max_force_n']+1e-10
                if entry['amplitude_cm']==0:assert final['peak_force_n']==0
                # Verify executed commands are exactly the clipped saved plan.
                a=np.array([r['action'] for r in rows]);starts=[c['step'] for c in chunks]
                for j,c in enumerate(chunks):
                    end=starts[j+1] if j+1<len(starts) else len(rows)
                    assert np.array_equal(a[c['step']:end],np.clip(np.array(c['actions'])[:end-c['step']],-1,1))
                summary.update(episode_id=video_episode.id,condition=label,amplitude_cm=entry['amplitude_cm'],reference_period_s=None if entry['amplitude_cm']==0 else 4,config_fingerprint=fp,data_kind='paired_zero_amplitude_kinematic_control',episode_wall_s=time.perf_counter()-tick,motion_summary=motion,motion_final=final,reset_and_first_prediction_match=True)
                write_jsonl(attempt/'steps.jsonl',rows);write_jsonl(attempt/'chunks.jsonl',chunks);atomic_json(attempt/'summary.json',summary)
                atomic_json(attempt/'evidence.json',{'config_fingerprint':fp,'sources':identities,'files':{n:sha(attempt/n) for n in ['summary.json','steps.jsonl','chunks.jsonl','video.mp4']}})
                verify_payload(attempt,entry,fp);folder.parent.mkdir(parents=True,exist_ok=True);attempt.rename(folder)
                nnew+=1
                print(f"[{i+1}/20] state={entry['state']} A={entry['amplitude_cm']}cm K={entry['k']} success={summary['success']} steps={len(rows)} calls={len(chunks)} wall={summary['episode_wall_s']:.1f}s peak_force={final['peak_force_n']:.3f}N contact={final['first_contact_step']} elapsed={(time.perf_counter()-start_all)/60:.1f}min",flush=True)
                atomic_json(out/'progress.json',{'completed':i+1,'planned':20,'last_episode':video_episode.id,'elapsed_s':time.perf_counter()-start_all})
            except BaseException:
                atomic_json(attempt/'error.json',{'entry':entry,'traceback':traceback.format_exc()});raise
            finally:
                if video:video.close()
        env.close();completed=len(list((out/'episodes').glob('*/summary.json')))
        manifest['status']='complete' if completed==20 else 'partial';manifest['completed']=completed;atomic_json(out/'manifest.json',manifest)
        print(json.dumps({'completed':completed,'planned':20}),flush=True)

if __name__=='__main__':
    ap=argparse.ArgumentParser(description=__doc__);ap.add_argument('--config',required=True);ap.add_argument('--output',required=True);ap.add_argument('--device',default='mps');ap.add_argument('--limit',type=int);args=ap.parse_args();run(args.config,args.output,args.device,args.limit)
