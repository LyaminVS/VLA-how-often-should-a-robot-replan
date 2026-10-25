#!/usr/bin/env python3
"""Independent verification and cluster-paired analysis of continuous motion."""
import argparse,csv,hashlib,json
from pathlib import Path
import numpy as np

def save_csv(p,rs):
    with p.open('w',newline='') as f:
        w=csv.DictWriter(f,fieldnames=list(rs[0]));w.writeheader();w.writerows(rs)

def analyze(run,out):
    run,out=Path(run),Path(out);m=json.loads((run/'manifest.json').read_text());cfg=m['config']
    assert m['status']=='complete' and m['completed']==len(m['plan'])==200
    expected={(e['state'],e['k'],e['speed_cm_s']) for e in m['plan']}
    seen=set();ids={};prefixes={};episodes=[];actions_n=calls_n=0
    for folder in sorted((run/'episodes').iterdir()):
        s=json.loads((folder/'summary.json').read_text());ev=json.loads((folder/'evidence.json').read_text())
        assert ev['sources']==m['sources'] and ev['config_fingerprint']==s['config_fingerprint']==m['config_fingerprint']
        for name,h in ev['files'].items():assert hashlib.sha256((folder/name).read_bytes()).hexdigest()==h
        key=s['state'],s['k'],s['speed_cm_s'];assert key in expected and key not in seen;seen.add(key)
        rows=[json.loads(x) for x in (folder/'steps.jsonl').read_text().splitlines()]
        chunks=[json.loads(x) for x in (folder/'chunks.jsonl').read_text().splitlines()]
        assert len(rows)==s['steps'] and len(chunks)==s['model_calls']
        assert [c['step'] for c in chunks]==list(range(0,len(rows),s['k']))
        assert [r['step'] for r in rows]==list(range(len(rows)))
        assert all(r['replanned']==(r['step']%s['k']==0) for r in rows)
        actual=np.array([r['action'] for r in rows])
        for i,c in enumerate(chunks):
            end=chunks[i+1]['step'] if i+1<len(chunks) else len(rows)
            assert c['sampling_seed']==(s['policy_seed']*1000003+c['step']*97409)%(2**63-1)
            assert np.array_equal(actual[c['step']:end],np.clip(np.array(c['actions'])[:end-c['step']],-1,1))
        identity=(s['initial_state_sha256'],s['initial_observation_sha256'],chunks[0]['actions'])
        if s['state'] in ids:assert ids[s['state']]==identity
        ids[s['state']]=identity
        pre=([(r['action'],r['eef'],r['gripper'],r['motion']['object_qpos']) for r in rows if r['sim_time_s']<cfg['onset_s']],[(c['step'],c['actions']) for c in chunks if c['step']/20<cfg['onset_s']])
        if key[:2] in prefixes:assert prefixes[key[:2]]==pre
        prefixes[key[:2]]=pre
        final=s['motion_final'];motions=[r['motion'] for r in rows]+[final]
        assert all(r['pose_writes']==0 for r in motions)
        assert s['speed_cm_s'] or all(r['velocity_writes']==0 for r in motions)
        released=[r['velocity_writes'] for r in motions if r['drive_released']]
        assert not released or len(set(released))==1
        grasp=final['bilateral_grasp_sim_time'];touch=final['first_contact_sim_time']
        assert grasp is None or (touch is not None and grasp>=touch-1e-8)
        assert all(r['velocity_writes']==0 for r in motions if r['elapsed_sim_time_s']<cfg['onset_s']-1e-8)
        free=[r for r in rows if 1.1<=r['sim_time_s']<=2 and (touch is None or r['sim_time_s']<touch)]
        speed=float(np.polyfit([r['sim_time_s'] for r in free],[r['motion']['object_pos'][0]*100 for r in free],1)[0]) if len(free)>=3 else None
        episodes.append(dict(state=s['state'],k=s['k'],speed_cm_s=s['speed_cm_s'],success=int(s['success']),steps=s['steps'],model_calls=s['model_calls'],first_contact_s=touch,bilateral_grasp_s=grasp,actual_precontact_speed_cm_s=speed,final_displacement_cm=(final['object_pos'][0]-final['object_initial_pos'][0])*100,velocity_writes=final['velocity_writes'],source_dir=str(folder.resolve())))
        actions_n+=len(rows);calls_n+=len(chunks)
    assert seen==expected
    states=cfg['base']['initial_states'];ks=cfg['base']['intervals'];speeds=cfg['speeds_cm_s']
    lookup={(r['state'],r['k'],r['speed_cm_s']):r for r in episodes}
    draws=np.random.default_rng(20261025).integers(0,len(states),(10000,len(states)))
    methods=[];contrasts=[]
    for v in speeds:
        for k in ks:
            rs=[lookup[state,k,v] for state in states];y=np.array([r['success'] for r in rs]);base=np.array([lookup[state,k,0]['success'] for state in states]);delta=y-base
            actuals=[r['actual_precontact_speed_cm_s'] for r in rs if r['actual_precontact_speed_cm_s'] is not None]
            methods.append(dict(speed_cm_s=v,k=k,episodes=len(rs),successes=int(y.sum()),success_rate=float(y.mean()),outcomes=y.tolist(),states=states,mean_model_calls=float(np.mean([r['model_calls'] for r in rs])),success_delta_vs_zero=float(delta.mean()),paired_bootstrap95_vs_zero=np.quantile(delta[draws].mean(axis=1),[.025,.975]).tolist(),rescues_vs_zero=int(np.sum((base==0)&(y==1))),regressions_vs_zero=int(np.sum((base==1)&(y==0))),bilateral_grasps=sum(r['bilateral_grasp_s'] is not None for r in rs),median_actual_precontact_speed_cm_s=float(np.median(actuals)) if actuals else None))
            if k!=1:
                ref=np.array([lookup[state,1,v]['success'] for state in states]);d=y-ref
                contrasts.append(dict(speed_cm_s=v,k=k,reference_k=1,pairs=len(states),success_difference=float(d.mean()),paired_bootstrap95=np.quantile(d[draws].mean(axis=1),[.025,.975]).tolist(),k_only_successes=int(np.sum((ref==0)&(y==1))),k1_only_successes=int(np.sum((ref==1)&(y==0)))))
    checks=dict(validated_episodes=len(episodes),validated_action_steps=actions_n,validated_policy_calls=calls_n,matched_initial_states_images_first_predictions=True,matched_same_k_pre_motion_actions_and_predictions=True,all_executed_actions_equal_clipped_saved_predictions=True,seeds_tied_to_absolute_step=True,zero_position_writes=True,static_control_zero_velocity_writes=True,permanent_drive_release_after_bilateral_grasp=True)
    d=dict(complete=True,episodes=len(episodes),physical_starts=len(states),ks=ks,speeds_cm_s=speeds,onset_s=cfg['onset_s'],ramp_s=cfg['ramp_s'],methods=methods,contrasts_vs_k1=contrasts,checks=checks,limitations=cfg['limits']+['The eight registered starts vary robot joints and some distractor poses; the target can starts at the same x,y. They are not eight distinct can locations.', 'Original task clutter remains: physics-only calibration found a milk-object contact in state19 at +2cm/s drive, so directional effects can include scene geometry and cannot be attributed solely to visual feedback delay.'],demo_selection=dict(state=17,ks=ks,speeds_cm_s=[0,-1.5,1.5,-2,2],selection='Smallest registered state, all K and conditions; no outcome filtering'))
    out.mkdir(parents=True,exist_ok=True)
    save_csv(out/'episodes.csv',episodes);save_csv(out/'methods.csv',methods);save_csv(out/'contrasts_vs_k1.csv',contrasts)
    (out/'analysis.json').write_text(json.dumps(d,indent=2)+'\n');print(json.dumps(checks,indent=2));return d
if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--run',required=True);p.add_argument('--output',required=True);a=p.parse_args();analyze(a.run,a.output)
