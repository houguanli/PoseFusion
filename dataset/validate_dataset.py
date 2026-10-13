"""Portable validation for the released background dataset; requires numpy and Pillow."""
from pathlib import Path
import json,hashlib,sys
import numpy as np
from PIL import Image
BASE=Path(sys.argv[1]) if len(sys.argv)>1 else Path(__file__).resolve().parent
m=json.loads((BASE/'dataset_manifest.json').read_text(encoding='utf8'));errors=[];rows=[];images=0;gt_count=0
sha=lambda p:hashlib.file_digest(p.open('rb'),'sha256').hexdigest()
for c in m['cases']:
 cid=c['case_id'];b=BASE/cid
 if c['status']=='pending_original_background':
  assert not list(b.glob('pose*/image/*.png'));continue
 if c['status']!='complete_background':errors.append(f'{cid}: incomplete case')
 for p in c['poses']:
  d=b/f"pose{p['pose']}";z=np.load(d/'cameras_sphere.npz');ids=sorted(int(k[10:]) for k in z.files if k.startswith('world_mat_') and not k.startswith('world_mat_inv_'))
  assert ids==list(range(p['image_count'])),(cid,p['pose'],'camera count')
  assert sha(d/'cameras_sphere.npz')==p['camera_sha256']
  assert len(list((d/'image').glob('*.png')))==p['image_count'];assert len(list((d/'mask').glob('*.png')))==p['image_count']
  cj=json.loads((d/'camera_data.json').read_text(encoding='utf8')) if (d/'camera_data.json').exists() else None
  bg=[]
  for f in p['frames']:
   i=f['index'];im=Image.open(b/f['image']);mask=Image.open(b/f['mask']).convert('L');assert im.mode=='RGB';assert im.size==mask.size
   assert sha(b/f['image'])==f['image_sha256'];assert sha(b/f['mask'])==f['mask_sha256']
   a=np.array(im);mm=np.array(mask);outside=a[mm<10];assert len(outside)>100
   stat={'index':i,'outside_mask_mean':float(outside.mean()),'outside_mask_std':float(outside.std()),'outside_mask_nonzero_fraction':float((outside.max(axis=1)>5).mean())}
   assert stat['outside_mask_nonzero_fraction']>.1 and stat['outside_mask_std']>2,(cid,p['pose'],stat)
   W=z[f'world_mat_{i}'];S=z[f'scale_mat_{i}'];assert W.shape==S.shape==(4,4);assert np.isfinite(W).all() and np.isfinite(S).all();assert abs(np.linalg.det(W))>1e-12
   for key in ['world_mat','scale_mat','camera_mat']:
    if f'{key}_inv_{i}' in z.files:assert np.allclose(z[f'{key}_{i}']@z[f'{key}_inv_{i}'],np.eye(4),atol=.003),(cid,key,i)
   if cj:
    K=np.array(cj[f'{i}_K']);M=np.array(cj[f'{i}_M']);K4=np.eye(4);K4[:3,:3]=K;assert np.allclose(W,K4@np.linalg.inv(M),atol=.001,rtol=1e-5),(cid,'camera JSON',i)
   bg.append(stat);images+=1
  rows.append({'case_id':cid,'pose':p['pose'],'images':p['image_count'],'all_images_RGB_background':True,'frames':bg})
 if c['domain']=='synthetic':
  verification=b/'gt/verification.json'
  if verification.exists():
   for v in json.loads(verification.read_text(encoding='utf8'))['poses']:
    assert sha(b/v['mesh'])==v['GT_mesh_sha256']
    assert sha(b/f"pose{v['pose']}/cameras_sphere.npz")==v['camera_sha256']
  t=json.loads((b/'gt/transforms.json').read_text(encoding='utf8'));A={k:np.array(v) for k,v in t['object_to_pose'].items()}
  for name,v in t['relative_transforms'].items():
   s,d=name.split('_to_');T=np.array(v);assert np.allclose(T,A[d]@np.linalg.inv(A[s]),atol=1e-7);assert np.allclose(T[:3,:3].T@T[:3,:3],np.eye(3),atol=1e-5);assert abs(np.linalg.det(T[:3,:3])-1)<1e-5
  for p in [1,2,3]:
   f=b/f'gt/meshes/pose{p}.ply';assert f.is_file() and f.stat().st_size>100;assert f.open('rb').read(3)==b'ply';gt_count+=1
fm=BASE/'file_manifest.json'
sources=BASE/'provenance/GT_source_manifest.json'
if sources.exists():
 for v in json.loads(sources.read_text(encoding='utf8')):assert sha(BASE/v['dataset_mesh'])==v['sha256']
if fm.exists():
 for rel,h in json.loads(fm.read_text(encoding='utf8')).items():assert sha(BASE/rel)==h,rel
report={'status':'passed' if not errors else 'failed','errors':errors,'complete_cases':sum(c['status']=='complete_background' for c in m['cases']),'pending_cases':[c['case_id'] for c in m['cases'] if c['status']=='pending_original_background'],'images':images,'GT_meshes':gt_count,'scope':'File integrity; background content outside mask; RGB/mask dimensions; NeuS camera and camera JSON consistency; source-derived relative RT. Source render/source geometry audits are recorded separately.','poses':rows}
(BASE/'validation_report.json').write_text(json.dumps(report,indent=2),encoding='utf8');print({k:v for k,v in report.items() if k!='poses'},flush=True)
assert not errors
