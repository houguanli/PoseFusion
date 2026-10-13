from pathlib import Path
from PIL import Image
import json,numpy as np,shutil
B=Path('F:/HGL_WORK_PASSED/posefusion/2public/dataset/guardian');S=Path('E:/PoseFusionExp/real_world/raw_video/realworld_datasets_2/sliced_images_sliced');O=Path('E:/PoseFusionExp/real_world/background_recovered_20260929/guardian');rows=[]
for p in [1,2,3]:
 src=S/f'Gardian_pose{p}';old=B/f'pose{p}';dest=O/f'pose{p}/image';dest.mkdir(parents=True,exist_ok=True);a=np.load(old/'cameras_sphere.npz');z=np.load(src/'cameras_sphere.npz');assert set(a.files)==set(z.files);assert all(np.array_equal(a[k],z[k]) for k in a.files),(p,'camera differs')
 fs=sorted((src/'image').glob('*.png'));assert len(fs)==60;fr=[]
 for i,f in enumerate(fs):
  x=np.array(Image.open(f).convert('RGB'));fg=np.array(Image.open(old/'image'/f.name).convert('RGB'));mask=np.array(Image.open(old/'mask'/f.name).convert('L'))>250;assert x.shape==fg.shape;err=float(abs(x.astype(float)-fg)[mask].mean());assert err==0,(p,i,err);assert x[~mask].std()>5
  shutil.copy2(f,dest/f.name);fr.append({'index':i,'source':str(f),'foreground_mean_abs_error':err,'background_std':float(x[~mask].std())})
 shutil.copy2(old/'cameras_sphere.npz',dest.parent/'cameras_sphere.npz');shutil.copytree(old/'mask',dest.parent/'mask',dirs_exist_ok=True)
 rows.append({'pose':p,'camera_exactly_equal':True,'frames':fr});print('Guardian',p,len(fr),'verified',flush=True)
(O/'recovery_report.json').write_text(json.dumps(rows,indent=2),encoding='utf8')
