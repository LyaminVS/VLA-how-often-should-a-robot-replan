#!/usr/bin/env python3
"""Validate and summarize the complete paired kinematic-amplitude sweep."""
from __future__ import annotations
import argparse,csv,hashlib,json,math
from collections import defaultdict
from pathlib import Path
import numpy as np

def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()
def write_csv(p,rows):
 with p.open('w') as f:
  w=csv.DictWriter(f,fieldnames=list(dict.fromkeys(k for r in rows for k in r)));w.writeheader();w.writerows(rows)
def wilson(s,n):
 z=1.96;den=1+z*z/n;mid=(s/n+z*z/(2*n))/den;half=z*math.sqrt(s/n*(1-s/n)/n+z*z/(4*n*n))/den;return [mid-half,mid+half]
def analyze(root,output,partial=False):
 output.mkdir(parents=True,exist_ok=True);manifest=json.loads((root/'manifest.json').read_text());cfg=manifest['config'];summaries=sorted(root.glob('episodes/*/summary.json'))
 if not partial:assert len(summaries)==len(manifest['plan'])
 expected={(r['state'],r['k'],r['amplitude_cm']) for r in manifest['plan']};seen=set();reset={};out=[];inputs={str((root/'manifest.json').resolve()):sha(root/'manifest.json')};totalactions=totalcalls=0
 for p in summaries:
  s=json.loads(p.read_text());ev=json.loads((p.parent/'evidence.json').read_text());assert ev['config_fingerprint']==manifest['config_fingerprint'];assert ev['sources']==manifest['sources']
  for name,h in ev['files'].items():assert sha(p.parent/name)==h;inputs[str((p.parent/name).resolve())]=h
  key=(s['state'],s['k'],s['amplitude_cm']);assert key in expected and key not in seen;seen.add(key)
  rows=[json.loads(x) for x in (p.parent/'steps.jsonl').read_text().splitlines()];chunks=[json.loads(x) for x in (p.parent/'chunks.jsonl').read_text().splitlines()];assert len(rows)==s['steps'];assert len(chunks)==s['model_calls'];assert s['steps']>=60
  first=(s['initial_state_sha256'],s['initial_observation_sha256'],chunks[0]['actions'])
  if s['state'] in reset:assert reset[s['state']]==first
  reset[s['state']]=first;a=np.array([r['action'] for r in rows]);jumps=np.linalg.norm(np.diff(a[:,:3],axis=0),axis=1);rot=np.linalg.norm(np.diff(a[:,3:6],axis=0),axis=1);flips=np.diff(a[:,6]>0)!=0
  assert [c['step'] for c in chunks]==list(range(0,len(rows),s['k']))
  for i,c in enumerate(chunks):
   assert c['sampling_seed']==(s['policy_seed']*1_000_003+c['step']*97_409)%(2**63-1)
   end=chunks[i+1]['step'] if i+1<len(chunks) else len(rows);assert np.array_equal(a[c['step']:end],np.clip(np.asarray(c['actions'])[:end-c['step']],-1,1))
  eef=np.asarray([r['eef'] for r in rows]);velocity=np.diff(eef,axis=0)*20;acceleration=np.diff(velocity,axis=0)*20
  final=s['motion_final'];assert final['kinematic_writes_after_contact']==0 and final['peak_force_n']==0 and final['max_imposed_tracking_error_m']<1e-7
  if s['amplitude_cm']==0 and 'zero-amplitude' not in cfg['intervention']:assert final['kinematic_writes']==0
  # Independent reconstruction of the prescribed trajectory and its velocity.
  phase=float(cfg.get('trajectory_phase_rad',0.));amp=s['amplitude_cm']/100
  for sample in rows:
   motion=sample['motion'];tau=motion['elapsed_sim_time_s']-.5
   if tau<=0 or amp==0:dx=vel=0.
   else:
    u=min(tau,1.);ramp=u*u*(3-2*u);derivative=6*u*(1-u) if u<1 else 0.;omega=2*math.pi/4
    dx=amp*ramp*math.sin(omega*tau+phase);vel=amp*(derivative*math.sin(omega*tau+phase)+ramp*omega*math.cos(omega*tau+phase))
   assert abs(motion['target_pos'][0]-motion['object_initial_pos'][0]-dx)<1e-9
   assert abs(motion['reference_velocity_x_m_s']-vel)<1e-9
  contact=final['first_contact_step'];writes=[r['motion']['kinematic_writes'] for r in rows if contact is not None and r['step']>=contact]
  if writes:assert len(set(writes))==1,'Kinematic pose must remain released'
  row={k:s[k] for k in ['episode_id','state','policy_seed','k','amplitude_cm','success','steps','model_calls']};row.update(source_dir=str(p.parent.resolve()),first_contact_step=contact,full_translation_tv100=float(jumps.mean()*100),first60_translation_tv100=float(jumps[:59].mean()*100),full_rotation_tv100=float(rot.mean()*100),full_gripper_flips100=float(flips.mean()*100),precontact_x_range_cm=100*final['precontact_x_range_m'],precontact_y_range_cm=100*final['precontact_y_range_m'],precontact_z_range_cm=100*final['precontact_z_range_m'],precontact_tilt_deg=final['max_precontact_tilt_deg'],active_duration_s=final['active_sim_time_s'],kinematic_writes=final['kinematic_writes'],tracking_error_max_after_imposition_m=final['max_imposed_tracking_error_m'],object_contact_observed=contact is not None,model_inference_s=s['inference_s'],episode_wall_s=s['episode_wall_s']);row.update(first60_eef_accel_m_s2=float(np.linalg.norm(acceleration[:58],axis=1).mean()),full_eef_accel_m_s2=float(np.linalg.norm(acceleration,axis=1).mean()),first60_eef_speed_m_s=float(np.linalg.norm(velocity[:59],axis=1).mean()));out.append(row);totalactions+=len(rows);totalcalls+=len(chunks)
 if not partial:assert seen==expected
 groups=[]
 for amp in [0,2,4]:
  for k in [1,5,10,25,50]:
   rs=sorted([r for r in out if r['k']==k and r['amplitude_cm']==amp],key=lambda r:r['state'])
   if not rs:continue
   n=len(rs);wins=sum(r['success'] for r in rs);row={'amplitude_cm':amp,'k':k,'episodes':n,'successes':wins,'success_rate':wins/n,'states':','.join(str(r['state']) for r in rs),'outcomes':','.join(str(int(r['success'])) for r in rs)};row['wilson95_low'],row['wilson95_high']=wilson(wins,n)
   for field in ['model_calls','steps','first60_translation_tv100','full_translation_tv100','full_rotation_tv100','full_gripper_flips100','precontact_x_range_cm','precontact_y_range_cm','precontact_z_range_cm','precontact_tilt_deg','active_duration_s','first60_eef_accel_m_s2','full_eef_accel_m_s2','first60_eef_speed_m_s']:
    row[field+'_mean']=float(np.mean([r[field] for r in rs]))
   row['contacts']=sum(r['object_contact_observed'] for r in rs);groups.append(row)
 paired=[]
 for amp in [2,4]:
  for k in [1,5,10,25,50]:
   base={r['state']:r for r in out if r['k']==k and r['amplitude_cm']==0};treated={r['state']:r for r in out if r['k']==k and r['amplitude_cm']==amp};states=sorted(set(base)&set(treated))
   if not states:continue
   row={'amplitude_cm':amp,'k':k,'pairs':len(states),'rescues':sum(treated[s]['success'] and not base[s]['success'] for s in states),'regressions':sum(base[s]['success'] and not treated[s]['success'] for s in states),'success_difference':float(np.mean([float(treated[s]['success'])-float(base[s]['success']) for s in states]))}
   for field in ['model_calls','first60_translation_tv100','full_translation_tv100','precontact_x_range_cm']:row[field+'_difference']=float(np.mean([treated[s][field]-base[s][field] for s in states]))
   paired.append(row)
 result={'complete':seen==expected,'data_kind':'paired_kinematic_amplitude_sweep','config':cfg,'checks':{'episodes':len(out),'physical_starting_states':len(reset),'validated_action_steps':totalactions,'validated_policy_calls':totalcalls,'same_reset_and_first_prediction_per_state':True,'sampling_seed_tied_to_absolute_simulation_step':True,'prescribed_trajectory_independently_reconstructed':True,'zero_external_force':True,'kinematic_release_permanent':True,'tracking_error_after_pose_imposition_below_m':1e-7,'all_saved_payload_hashes_match':True},'methods':groups,'paired_to_static':paired,'limitations':['Only four physical starts and one policy seed; descriptive exploratory results.','One soup-can task; inference pauses simulator.','Prescribed free-joint pose/velocity until first finger contact; idealized moving-target test, not physically realistic disturbance.','Only amplitude changes, period and phase fixed; amplitude 0 uses free physics, nonzero conditions constrain pose before contact.','Contact is not proof of a successful grasp.','Four preliminary force-controller pilot episodes were excluded after user requested explicit object motion.','No pooling with previous CUDA moving-target or static 270-episode cohorts.'],'sources_sha256':inputs}
 write_csv(output/'episodes.csv',out);write_csv(output/'methods.csv',groups);write_csv(output/'paired_to_static.csv',paired);(output/'analysis.json').write_text(json.dumps(result,indent=2));print(json.dumps(result['checks'],indent=2));return result
if __name__=='__main__':
 p=argparse.ArgumentParser(description=__doc__);p.add_argument('--root',type=Path,required=True);p.add_argument('--output',type=Path,required=True);p.add_argument('--partial',action='store_true');a=p.parse_args();analyze(a.root.resolve(),a.output.resolve(),a.partial)
