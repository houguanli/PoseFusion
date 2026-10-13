import bpy,json,numpy as np,sys
from pathlib import Path
from mathutils import Matrix
S=Path('E:/PoseFusionExp/synthetic/seal_re_20260928');O=Path('E:/PoseFusionExp/synthetic/background_completion_20260929/seal');O.mkdir(exist_ok=True)
bpy.ops.wm.open_mainfile(filepath=str(S/'seal_background.blend'),load_ui=False,use_scripts=False)
api={'__name__':'existing_render_export'};exec(compile((S/'existing_render_export.py').read_text(encoding='utf8'),'existing_render_export.py','exec'),api)
scene=bpy.context.scene;camera=bpy.data.objects['Camera'];scene.camera=camera;api['detach_object_preserve_world'](camera,clear_constraints=True)
scene.render.engine='CYCLES';scene.cycles.samples=32;scene.cycles.use_denoising=True;scene.cycles.device='GPU'
prefs=bpy.context.preferences.addons['cycles'].preferences;prefs.compute_device_type='OPTIX';prefs.get_devices()
for device in prefs.devices:device.use=device.type=='OPTIX'
print('SCENE ORIGINAL',scene.use_nodes,scene.render.film_transparent,'floor shadow catcher',bpy.data.objects['Obj3d66-734029-1-584'].is_shadow_catcher,flush=True)
scene.use_nodes=False;scene.render.film_transparent=False;scene.render.image_settings.file_format='PNG';scene.render.image_settings.color_mode='RGB';scene.render.resolution_x=scene.render.resolution_y=800;scene.render.resolution_percentage=100;scene.render.pixel_aspect_x=scene.render.pixel_aspect_y=1
names=[f'_re_seal_pose{i}' for i in [1,2,3]];raw=json.loads((S/'transforms.json').read_text(encoding='utf8'))['object_to_pose'];rows=[]
for p in [1,2,3]:
 api['set_scene_for_background_render']([names[p-1]],names);assert np.allclose(np.array(bpy.data.objects[names[p-1]].matrix_world),raw[f'pose{p}'],rtol=0,atol=1e-7)
 cams=json.loads((S/f'rendered/_re_seal_pose{p}/camera_data.json').read_text());out=O/f'pose{p}/image';out.mkdir(parents=True,exist_ok=True)
 frames=[36] if '--preview' in sys.argv else range(40)
 for i in frames:
  K=np.array(cams[f'{i}_K']);M=np.array(cams[f'{i}_M']);M[:,1:3]*=-1;camera.data.sensor_fit='HORIZONTAL';camera.data.sensor_width=36;camera.data.lens=K[0,0]*36/800;camera.data.shift_x=.5-K[0,2]/800;camera.data.shift_y=K[1,2]/800-.5;camera.matrix_world=Matrix(M.tolist());bpy.context.view_layer.update()
  assert np.allclose(api['compute_K_from_blender_camera'](scene,camera),K,atol=1e-5,rtol=0);assert np.allclose(np.array(camera.matrix_world),M,atol=1e-6,rtol=0)
  file=out/f'{i:03d}.png'
  if file.exists() and '--preview' not in sys.argv:continue
  scene.render.filepath=str(file);bpy.ops.render.render(write_still=True);print('SEAL BACKGROUND',p,i,flush=True)
 rows.append({'pose':p,'frames_requested':len(frames),'source_cameras':str(S/f'rendered/_re_seal_pose{p}/camera_data.json'),'source_GT_unchanged':True})
(O/('preview_report.json' if '--preview' in sys.argv else 'render_report.json')).write_text(json.dumps({'source_blend':str(S/'seal_background.blend'),'compositor_disabled':True,'poses':rows},indent=2),encoding='utf8')
print('SEAL BACKGROUNDS COMPLETE',flush=True)
