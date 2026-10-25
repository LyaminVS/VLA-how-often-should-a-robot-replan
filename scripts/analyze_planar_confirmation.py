#!/usr/bin/env python3
"""Verify every held-out planar episode and preserve pairing across methods."""
import argparse,csv,hashlib,json
from pathlib import Path
import numpy as np

def csvwrite(path,rows):
 with path.open('w',newline='') as f:
  w=csv.DictWriter(f,fieldnames=list(rows[0]));w.writeheader();w.writerows(rows)

def analyze(run,out):
 run,out=Path(run),Path(out);m=json.loads((run/'manifest.json').read_text());cfg=m['config']
 assert m['status']=='complete' and m['completed']==len(m['plan'])==150
 expected={(e['state'],tuple(e['offset_cm']),e['k'],e['condition_id']) for e in m['plan']};seen=set();identities={};prefixes={};episodes=[];actions_n=calls_n=0
 for folder in sorted((run/'episodes').iterdir()):
  s=json.loads((folder/'summary.json').read_text());ev=json.loads((folder/'evidence.json').read_text());assert ev['sources']==m['sources']
  assert ev['config_fingerprint']==s['config_fingerprint']==m['config_fingerprint']
  for name,h in ev['files'].items():assert hashlib.sha256((folder/name).read_bytes()).hexdigest()==h
  pos=tuple(s['offset_cm']);key=(s['state'],pos,s['k'],s['condition_id']);assert key in expected and key not in seen;seen.add(key)
  rows=[json.loads(x) for x in (folder/'steps.jsonl').read_text().splitlines()];cs=[json.loads(x) for x in (folder/'chunks.jsonl').read_text().splitlines()]
  assert len(rows)==s['steps'] and len(cs)==s['model_calls'];assert [c['step'] for c in cs]==list(range(0,len(rows),s['k']))
  assert [r['step'] for r in rows]==list(range(len(rows)))
  aa=np.array([r['action'] for r in rows])
  for i,c in enumerate(cs):
   requested=(s['policy_seed']*1000003+c['step']*97409)%(2**63-1)
   actual_seed=(s['policy_seed']*1000003)%(2**63-1) if cfg['noise_mode']=='episode_common' else requested
   assert c['sampling_seed']==actual_seed and c['requested_sampling_seed']==requested
   end=cs[i+1]['step'] if i+1<len(cs) else len(rows)
   assert np.array_equal(aa[c['step']:end],np.clip(np.array(c['actions'])[:end-c['step']],-1,1))
  identity=(s['initial_scene_sha256'],s['initial_observation_sha256'],cs[0]['actions']);pair=key[:2]
  if pair in identities:assert identities[pair]==identity
  identities[pair]=identity
  pre=([(r['action'],r['eef'],r['gripper'],r['motion']['object_qpos']) for r in rows if r['sim_time_s']<.5],[(c['step'],c['actions']) for c in cs if c['step']/20<.5])
  prekey=key[:3]
  if prekey in prefixes:assert prefixes[prekey]==pre
  prefixes[prekey]=pre
  final=s['motion_final'];ms=[r['motion'] for r in rows]+[final]
  assert all(r['pose_writes']==0 for r in ms)
  if s['speed_cm_s']==0:assert all(r['velocity_writes']==0 for r in ms)
  released=[r['velocity_writes'] for r in ms if r['drive_released']];assert not released or len(set(released))==1
  touch=final['first_contact_sim_time'];free=[r for r in rows if 1.1<=r['sim_time_s']<=2 and (touch is None or r['sim_time_s']<touch)]
  speed=float(np.polyfit([r['sim_time_s'] for r in free],[r['motion']['object_pos'][s['axis']]*100 for r in free],1)[0]) if len(free)>2 else None
  episodes.append(dict(state=s['state'],position_id=cfg['positions_cm'].index(list(pos)),offset_x_cm=pos[0],offset_y_cm=pos[1],k=s['k'],condition_id=s['condition_id'],axis=s['axis'],drive_cm_s=s['speed_cm_s'],success=int(s['success']),steps=s['steps'],model_calls=s['model_calls'],first_contact_s=touch,bilateral_grasp_s=final['bilateral_grasp_sim_time'],actual_precontact_speed_cm_s=speed,initial_x_m=s['initial_can_position_m'][0],initial_y_m=s['initial_can_position_m'][1],source_dir=str(folder.resolve())))
  actions_n+=len(rows);calls_n+=len(cs)
 assert seen==expected and len(identities)==6
 assert len({identity[1] for identity in identities.values()})==6
 states=cfg['base']['initial_states'];ks=cfg['base']['intervals'];conds=cfg['motions'];lookup={(r['state'],r['position_id'],r['k'],r['condition_id']):r for r in episodes}
 # Three robot-state clusters, retaining both crossed placements per sampled cluster.
 draws=np.random.default_rng(20261025).integers(0,len(states),(10000,len(states)));methods=[];contrasts=[]
 for cond in conds:
  cid=cond['id']
  for k in ks:
   rs=[lookup[state,pos,k,cid] for state in states for pos in range(2)]
   y=np.array([r['success'] for r in rs]).reshape(3,2);control=np.array([lookup[state,pos,k,'static']['success'] for state in states for pos in range(2)]).reshape(3,2)
   diff=(y-control).mean(axis=1);real=[r['actual_precontact_speed_cm_s'] for r in rs if r['actual_precontact_speed_cm_s'] is not None]
   methods.append(dict(condition_id=cid,k=k,episodes=6,successes=int(y.sum()),success_rate=float(y.mean()),outcomes=y.reshape(-1).tolist(),mean_model_calls=float(np.mean([r['model_calls'] for r in rs])),success_delta_vs_zero=float(diff.mean()),paired_bootstrap95_vs_zero=np.quantile(diff[draws].mean(axis=1),[.025,.975]).tolist(),median_actual_speed_cm_s=float(np.median(real)) if real else None))
   if k!=1:
    ref=np.array([lookup[state,pos,1,cid]['success'] for state in states for pos in range(2)]).reshape(3,2);d=(y-ref).mean(axis=1)
    contrasts.append(dict(condition_id=cid,k=k,reference_k=1,success_difference=float(d.mean()),cluster_bootstrap95=np.quantile(d[draws].mean(axis=1),[.025,.975]).tolist()))
 positions=[dict(id=i,offset_cm=pos,actual_xy_m=[float(np.mean([r['initial_x_m'] for r in episodes if r['position_id']==i])),float(np.mean([r['initial_y_m'] for r in episodes if r['position_id']==i]))],units='m',coordinates='MuJoCo world horizontal plane') for i,pos in enumerate(cfg['positions_cm'])]
 pixels=json.loads((run/'initial_frame_verification/verification.json').read_text());assert pixels['all150_saved_initial_inputs_reproduced'] and pixels['same_state_different_placements_change_camera_pixels']
 checks=dict(distinct_raw_camera_frames=pixels['distinct_camera_frames'],raw_camera_pixels_verified_for_varied_placements=True,validated_episodes=150,validated_action_steps=actions_n,validated_policy_calls=calls_n,evidence_hashes=True,matched_actual_initial_scenes_images_first_predictions=True,six_distinct_initial_observations=True,matched_same_k_pre_motion_prefixes=True,all_actions_equal_clipped_predictions=True,noise_mode_recorded_and_verified=True,no_rollout_position_writes=True,zero_control_no_velocity_writes=True,permanent_bilateral_release=True)
 d=dict(complete=True,episodes=150,pilot_parameters_frozen=True,paired_scenarios=6,robot_states=states,can_positions=positions,ks=ks,conditions=conds,noise_mode=cfg['noise_mode'],methods=methods,contrasts_vs_k1=contrasts,checks=checks,limits=cfg['limits'],demo_selection=dict(state=30,position_id=0,condition_ids=[c['id'] for c in conds],ks=ks,rule='Smallest held-out state, first registered placement; all conditions and K, no filtering by outcome'))
 out.mkdir(parents=True,exist_ok=True);csvwrite(out/'episodes.csv',episodes);csvwrite(out/'methods.csv',methods)
 (out/'analysis.json').write_text(json.dumps(d,indent=2)+'\n');print(json.dumps(checks,indent=2));return d
if __name__=='__main__':
 p=argparse.ArgumentParser();p.add_argument('--run',required=True);p.add_argument('--output',required=True);a=p.parse_args();analyze(a.run,a.output)
