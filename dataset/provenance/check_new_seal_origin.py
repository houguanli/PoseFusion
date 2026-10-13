import bpy,numpy as np,json
from pathlib import Path
bpy.ops.wm.open_mainfile(filepath='E:/PoseFusionExp/synthetic/seal_re_20260928/seal_background.blend',load_ui=False,use_scripts=False)
objs=[bpy.data.objects[f'_re_seal_pose{i}'] for i in [1,2,3]];vs=[np.array([v.co[:] for v in o.data.vertices]) for o in objs];can=vs[1];mat={};raw={};rows=[]
for i,(o,v) in enumerate(zip(objs,vs),1):
 assert v.shape==can.shape
 delta=v-can;d=delta.mean(axis=0);res=float(abs(delta-d).max());assert res<1e-6,res
 D=np.eye(4);D[:3,3]=d;A=np.array(o.matrix_world);mat[f'pose{i}']=(A@D).tolist();raw[f'pose{i}']=A.tolist();rows.append({'pose':i,'local_origin_translation':d.tolist(),'max_local_residual':res})
rel={}
checks={}
for i in [1,2,3]:
 for j in [1,2,3]:
  if i!=j:
   T=np.array(mat[f'pose{j}'])@np.linalg.inv(mat[f'pose{i}']);rel[f'pose{i}_to_pose{j}']=T.tolist()
   vi=np.c_[vs[i-1],np.ones(len(vs[i-1]))]@np.array(raw[f'pose{i}']).T
   vj=np.c_[vs[j-1],np.ones(len(vs[j-1]))]@np.array(raw[f'pose{j}']).T
   err=float(abs(vi@T.T-vj).max());assert err<1e-6
   checks[f'pose{i}_to_pose{j}']={'world_correspondence_max_abs_error':err,'method':'source-index local-origin verification only; no fitted world transform'}
out={'convention':'Column vectors. Common canonical local frame is original pose2 mesh. x_target = T @ x_source. Only local origin translation reconciled; rendered world geometry, source pose R/t, meshes and cameras unchanged.','object_to_pose':mat,'raw_blender_object_to_pose':raw,'origin_corrections':rows,'relative_transforms':rel}
out['relative_transform_checks']=checks
Path('E:/PoseFusionExp/synthetic/seal_re_20260928/canonical_transforms.json').write_text(json.dumps(out,indent=2),encoding='utf8');print('SEAL_ORIGIN',json.dumps(rows),flush=True)
