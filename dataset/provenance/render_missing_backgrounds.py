"""Render backgrounds using unchanged source Blender objects and exact saved camera matrices."""
import bpy,json,sys,math,time
from pathlib import Path
import numpy as np
from mathutils import Matrix
BASE=Path('F:/HGL_WORK_PASSED/posefusion/2public/dataset');OUT=Path('E:/PoseFusionExp/synthetic/background_completion_20260929');OUT.mkdir(exist_ok=True)
DATA=json.loads((BASE/'dataset_manifest.json').read_text(encoding='utf-8'))
only_preview='--preview' in sys.argv
api={'__name__':'existing_render_export'}
exec(compile(Path('E:/PoseFusionExp/synthetic/seal_re_20260928/existing_render_export.py').read_text(encoding='utf-8'),'existing_render_export.py','exec'),api)
rows=[]
for case in DATA['cases']:
 cid=case['case_id']
 if case['domain']!='synthetic' or cid=='seal':continue
 if '--only-existing' in sys.argv and cid not in ['bunny','book']:continue
 bpy.ops.wm.open_mainfile(filepath=case['source_blend'],load_ui=False,use_scripts=False)
 # Source viewport-hidden camera sampling sphere is not part of the room or GT.
 if cid in ['tank','monster']:
  bpy.data.objects['Sphere'].hide_render=True
 for img in bpy.data.images:
  if img.packed_file or not img.filepath:continue
  if Path(bpy.path.abspath(img.filepath)).is_file():continue
  candidates=list(Path(case['source_blend']).parent.rglob(Path(img.filepath.replace('\\','/')).name))
  candidates=[p for p in candidates if p.is_file()]
  if len(candidates)==1:
   img.filepath=str(candidates[0]);img.reload()
 spec=json.loads((BASE/cid/'gt/source_blender_objects.json').read_text(encoding='utf-8'))
 scene=bpy.context.scene;camera=bpy.data.objects['Camera'];scene.camera=camera
 api['detach_object_preserve_world'](camera,clear_constraints=True)
 scene.render.engine='CYCLES';scene.cycles.samples=32;scene.cycles.use_denoising=True
 prefs=bpy.context.preferences.addons['cycles'].preferences;prefs.compute_device_type='OPTIX';prefs.get_devices()
 for device in prefs.devices:device.use=device.type=='OPTIX'
 scene.cycles.device='GPU';scene.render.film_transparent=False;scene.render.image_settings.file_format='PNG';scene.render.image_settings.color_mode='RGB';scene.use_nodes=False
 scene.render.resolution_percentage=100;scene.render.pixel_aspect_x=scene.render.pixel_aspect_y=1
 all_names=[n for v in spec['poses'].values() for n in v['source_object_parts']]
 case_rows=[]
 for p in case['poses']:
  pose=p['pose'];pose_spec=spec['poses'][str(pose)];active=list(pose_spec['source_object_parts'])
  api['set_scene_for_background_render'](active,all_names)
  for n in active:assert np.allclose(np.array(bpy.data.objects[n].matrix_world),pose_spec['source_object_parts'][n]['matrix_world'],atol=1e-6,rtol=0),(cid,n,'GT mismatch')
  # Only foreground materials are mandatory here; unused hidden archive assets can remain unresolved.
  missing=[]
  for n in active:
   obj=bpy.data.objects[n]
   if not hasattr(obj.data,'materials'):continue
   for mat in obj.data.materials:
    if mat and mat.use_nodes:
     for node in mat.node_tree.nodes:
      if node.type=='TEX_IMAGE' and node.image and any(o.is_linked for o in node.outputs) and not node.image.packed_file:
       resolved=Path(bpy.path.abspath(node.image.filepath))
       if not resolved.is_file():missing.append(str(resolved))
  if missing:raise RuntimeError((cid,pose,'missing foreground textures',missing))
  cameras=json.loads((BASE/cid/f'pose{pose}/camera_data.json').read_text())
  from_size=next(iter(p['dimensions'])).split()[0];w,h=map(int,from_size.split('x'))
  scene.render.resolution_x=w;scene.render.resolution_y=h
  dest=OUT/cid/f'pose{pose}/image';dest.mkdir(parents=True,exist_ok=True)
  frames=[0,14] if only_preview and '--only-existing' in sys.argv else ([0] if only_preview else range(p['image_count']))
  for i in frames:
   K=np.array(cameras[f'{i}_K']);M=np.array(cameras[f'{i}_M']);M[:,1:3]*=-1
   camera.data.sensor_fit='HORIZONTAL';camera.data.sensor_width=36;camera.data.lens=K[0,0]*36/w
   camera.data.shift_x=.5-K[0,2]/w;camera.data.shift_y=K[1,2]/h-.5
   camera.matrix_world=Matrix(M.tolist());bpy.context.view_layer.update()
   assert np.allclose(api['compute_K_from_blender_camera'](scene,camera),K,atol=1e-5,rtol=0)
   assert np.allclose(np.array(camera.matrix_world),M,atol=1e-6,rtol=0)
   path=dest/f'{i:03d}.png'
   if path.exists() and not only_preview:continue
   scene.render.filepath=str(path);bpy.ops.render.render(write_still=True)
   print('BACKGROUND FRAME',cid,pose,i,flush=True)
  case_rows.append({'pose':pose,'frames_requested':len(frames),'source_cameras':str(BASE/cid/f'pose{pose}/camera_data.json'),'source_GT_unchanged':True})
 rows.append({'case_id':cid,'source_blend':case['source_blend'],'poses':case_rows})
 report_name=('preview_report' if only_preview else 'render_report')+('_bunny_book' if '--only-existing' in sys.argv else '')+'.json'
 (OUT/report_name).write_text(json.dumps(rows,ensure_ascii=False,indent=2),encoding='utf-8')
 print('CASE COMPLETE',cid,flush=True)
print('ALL REQUESTED BACKGROUNDS COMPLETE',flush=True)
