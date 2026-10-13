from pathlib import Path
import json,numpy as np,cv2
from PIL import Image
B=Path('F:/HGL_WORK_PASSED/posefusion/2public/dataset');O=Path('E:/PoseFusionExp/real_world/background_recovered_20260929');O.mkdir(exist_ok=True)
rows=[]
for c in json.loads((B/'dataset_manifest.json').read_text(encoding='utf8'))['cases']:
 if c['domain']!='real' or c['case_id'] in ['guardian','figurine_f']:continue
 for p in c['poses']:
  src=Path('E:/PoseFusionExp/real_world/All_dataset_calibrated')/(c['source_prefix']+'_pose'+str(p['pose'])); dst=B/c['case_id']/('pose'+str(p['pose']));out=O/c['case_id']/('pose'+str(p['pose']))/'image';out.mkdir(parents=True,exist_ok=True)
  old=np.load(dst/'cameras_sphere.npz');new=np.load(src/'cameras_sphere.npz');fr=[]
  for f in p['frames']:
   i=f['index'];a=np.array(Image.open(B/c['case_id']/f['image']).convert('RGB'));mask=np.array(Image.open(B/c['case_id']/f['mask']).convert('L'))>250;x=np.array(Image.open(src/'image'/f['source_image_name']).convert('RGB'))
   H=old[f'world_mat_{i}']@np.linalg.inv(new[f'world_mat_{i}']);assert np.allclose(H[:3,3],0,atol=.001)
   assert np.allclose(old[f'scale_mat_{i}'],new[f'scale_mat_{i}'])
   h=H[:3,:3];hr=np.round(h);assert np.allclose(h,hr,atol=.001),(c['case_id'],i,h)
   assert np.array_equal(hr[:2,:2],np.eye(2))
   off=-hr[:2,2].astype(int);xx,yy=off;z=x[yy:yy+a.shape[0],xx:xx+a.shape[1]];assert z.shape==a.shape
   delta=abs(z.astype(float)-a)[mask];err=float(delta.mean());assert err<.1,(c['case_id'],i,err)
   assert z[~mask].std()>5
   Image.fromarray(z).save(out/Path(f['image']).name)
   fr.append({'index':i,'source':str(src/'image'/f['source_image_name']),'crop_xy':off.tolist(),'foreground_mean_abs_error':err,'background_std':float(z[~mask].std())})
  rows.append({'case_id':c['case_id'],'pose':p['pose'],'frames':fr});print(c['case_id'],p['pose'],len(fr),'verified',flush=True)
  (O/'recovery_report.json').write_text(json.dumps(rows,indent=2),encoding='utf8')
