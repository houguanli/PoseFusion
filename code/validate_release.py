"""One executable check for release inputs/configs and camera-coordinate regressions."""
import ast
import gc
import hashlib
import json
import tempfile
from pathlib import Path

import cv2
import mmengine
import numpy as np
from PIL import Image

from lib.load_dtu import load_dtu_data
from project_paths import ROOT, dataset_root, project_path, load_registration_config, resolve_network_config

def sha(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as f:
        for chunk in iter(lambda:f.read(1024*1024), b''):
            h.update(chunk)
    return h.hexdigest()

def require(condition, message):
    if not condition:
        raise AssertionError(message)

def coordinate_regression_checks():
    import torch
    import time
    import fuse_registrations as fusion
    # Real sampling implementations must support independently calibrated frames.
    for model in ['dvgo_ori','voxurf_coarse','voxurf_fine','voxurf_womask_coarse','voxurf_womask_fine']:
        model_tree = ast.parse((ROOT/'lib'/f'{model}.py').read_text(encoding='utf-8'))
        names = ['get_rays', 'get_rays_of_a_view', 'get_training_rays']
        nodes = [n for n in model_tree.body if isinstance(n,ast.FunctionDef) and n.name in names]
        scope = {'torch':torch,'np':np,'time':time}
        exec(compile(ast.Module(body=nodes,type_ignores=[]),model,'exec'),scope)
        intrinsics = np.stack([np.eye(3),np.diag([2.,2.,1.])])
        _,_,directions,_,_ = scope['get_training_rays'](
            torch.zeros(2,2,2,3),torch.eye(4).repeat(2,1,1),np.array([[2,2],[2,2]]),
            intrinsics,False,True,False,False)
        require(not torch.allclose(directions[0],directions[1]), f'{model}: second frame intrinsic ignored')
    # Test the actual registration projection without importing its CUDA models.
    tree = ast.parse((ROOT / 'mp_reg.py').read_text(encoding='utf-8'))
    func = next(n for n in tree.body if isinstance(n,ast.FunctionDef) and n.name=='_project_points_to_image')
    namespace = {'np':np, 'torch':torch}
    exec(compile(ast.Module(body=[func],type_ignores=[]), 'mp_reg.py', 'exec'), namespace)
    proj = namespace['_project_points_to_image']
    k = np.array([[8.,0,7.],[0,8.,5.],[0,0,1.]])
    u,v,visible = proj(np.array([[0.,0,1.],[0.,0,-1.]]), np.eye(4), k,10,14)
    require(visible.tolist()==[True,False] and (u[0],v[0])==(7,5), 'OpenCV front/back projection is inverted')
    scratch = ROOT / 'outputs/validation_tmp'
    scratch.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(prefix='coordinate_', dir=scratch) as tmp:
        base = Path(tmp).resolve()
        require(base.is_relative_to(scratch.resolve()), 'Scratch directory escaped validation folder')
        s_scale = np.diag([2.,2.,2.,1.]); s_scale[:3,3]=[.1,.2,.3]
        t_scale = np.diag([3.,3.,3.,1.]); t_scale[:3,3]=[-.2,.1,.4]
        for name,scale in [('source',s_scale),('target',t_scale)]:
            root = base / name
            (root/'image').mkdir(parents=True); (root/'mask').mkdir()
            camera = {}
            for i in reversed(range(12)):
                rgb = np.full((10,14,3), [10+i,70,120], dtype=np.uint8)
                silhouette = np.zeros((10,14), dtype=np.uint8); silhouette[2:8,3:10]=255
                Image.fromarray(rgb).save(root/'image'/f'{i:03d}.png')
                Image.fromarray(silhouette).save(root/'mask'/f'{i:03d}.png')
                c2w = np.eye(4); c2w[0,3]=i*.1
                w = np.eye(4); w[:3,:4]=k @ np.linalg.inv(c2w)[:3,:4]
                camera[f'world_mat_{i}']=w; camera[f'scale_mat_{i}']=scale
                camera[f'world_mat_inv_{i}']=np.linalg.inv(w)
                camera[f'scale_mat_inv_{i}']=np.linalg.inv(scale)
            np.savez(root/'cameras_sphere.npz', **camera)
        images, poses, _, hwf, intrinsics, _, scale, masks = load_dtu_data(base/'source', reso_level=3, mask=False)
        require(images.shape==(12,3,4,3) and masks.shape==(12,3,4,1), 'Nondivisible dimensions were resized incorrectly')
        require(np.allclose(images[0,0,0], np.array([10,70,120])/255), 'Background RGB was replaced')
        require(np.allclose(intrinsics[0,0,0], 8*4/14) and np.allclose(intrinsics[0,1,1],8*3/10), 'Intrinsic resizing mismatch')
        ordered = fusion.get_poses_from_npz(base/'source/cameras_sphere.npz')
        require(np.allclose([p[0,3] for p in ordered], (np.arange(12)*.1-.1)/2), 'Camera suffixes were sorted lexicographically')
        angle = np.deg2rad(30)
        transform = np.eye(4); transform[:2,:2]=[[np.cos(angle),-np.sin(angle)],[np.sin(angle),np.cos(angle)]]
        transform[:3,3]=[.3,-.2,.1]
        cfg = {key:str(base/'unused') for key in ['source_exp_dir','target_exp_dir','source_net_config','target_net_config','source_mesh_dir','target_mesh_dir']}
        cfg.update(source_datadir=str(base/'source'), target_datadir=str(base/'target'),fine_rt=transform.tolist())
        path = base/'pair.json'; path.write_text(json.dumps(cfg), encoding='utf-8')
        original_root = fusion.ROOT
        fusion.ROOT = base
        try:
            output, _ = fusion.fusion_dataset([path], output_name='test', similarity_threshold=-1)
            with np.load(output/'cameras_sphere.npz') as result, np.load(base/'source/cameras_sphere.npz') as source:
                require(np.allclose(result['world_mat_12'] @ result['scale_mat_12'], source['world_mat_0'] @ s_scale @ np.linalg.inv(transform), atol=1e-5), 'Fusion transform convention is incorrect')
            try:
                fusion.fusion_dataset([path], output_name='test')
            except FileExistsError:
                pass
            else:
                raise AssertionError('Fusion silently replaced an existing result')
        finally:
            fusion.ROOT = original_root
        (base/'source/mask/011.png').rename(base/'source/mask/099.png')
        try:
            load_dtu_data(base/'source', mask=False)
        except ValueError:
            pass
        else:
            raise AssertionError('Mispaired masks were accepted')
    return ['per-frame ray intrinsics in all five samplers', 'OpenCV projection/front-facing convention', 'background retention',
            'nondivisible resize/intrinsics', 'numeric camera index order',
            'fusion with nonidentity scales and nontrivial R/t',
            'refuse overwrite', 'reject mispaired masks']

def validate(cases, checksums=False):
    manifest = json.loads((dataset_root()/'dataset_manifest.json').read_text(encoding='utf-8-sig'))
    reference = {c['case_id']:c for c in manifest['cases']}
    index = json.loads((ROOT/'configs/dataset.json').read_text(encoding='utf-8'))
    require(index['dataset_version']==manifest['dataset_version'], 'Dataset version changed; regenerate configs')
    report = {'dataset_version':manifest['dataset_version'], 'dataset_root':str(dataset_root()),
              'cases':[], 'checksums_checked':checksums, 'regression_checks':coordinate_regression_checks()}
    for case in cases:
        cid = case['case_id']; expected = reference[cid]
        record = {'case_id':cid, 'domain':case['domain'], 'poses':[], 'registration_configs':0}
        require({p['pose'] for p in case['poses']}=={1,2,3}, f'{cid}: missing pose config')
        if case['domain']=='synthetic':
            gt = json.loads(project_path(case['gt_transforms']).read_text(encoding='utf-8-sig'))
            for pair in case['registration_configs']:
                config = load_registration_config(pair)
                source, target = f"pose{config['source_pose']}", f"pose{config['target_pose']}"
                a,b = [np.asarray(gt['object_to_pose'][p]) for p in [source,target]]
                require(np.allclose(np.asarray(gt['relative_transforms'][f'{source}_to_{target}']),b @ np.linalg.inv(a),atol=1e-5), f'{cid}: GT convention mismatch')
        for item in case['poses']:
            p = next(p for p in expected['poses'] if p['pose']==item['pose'])
            root = project_path(item['datadir'])
            require(item['image_count']==p['image_count'], f'{root}: count mismatch')
            for stage in ['coarse','fine']:
                cfg = resolve_network_config(mmengine.Config.fromfile(str(project_path(item[f'{stage}_config']))))
                require(Path(cfg.data.datadir)==root and cfg.data.inverse_y and cfg.data.mode.wmask, f'{root}: incorrect camera/mask config')
                require(list(cfg.WH)==[item['width'],item['height']] and cfg.expected_image_count==p['image_count'], f'{root}: incorrect dimensions/count')
                require(Path(cfg.basedir)/cfg.expname/'fine'/'surf_last.tar'==project_path(item['checkpoint']), f'{root}: checkpoint path mismatch')
            # Actual runtime reader loads every input frame and every camera at quarter size.
            images, poses, _, hwf, ks, _, scale, masks = load_dtu_data(root, reso_level=4, mask=False)
            n,h,w,_ = images.shape
            require((n,h,w)==(p['image_count'],item['height']//4,item['width']//4), f'{root}: loaded shape mismatch')
            require(masks.shape==(n,h,w,1) and poses.shape==(n,4,4) and ks.shape==(n,4,4), f'{root}: invalid runtime tensors')
            require(np.all(np.isfinite(images)) and np.all((masks>0).sum(axis=(1,2,3))>0), f'{root}: empty mask/invalid pixels')
            max_error=0.
            with np.load(root/'cameras_sphere.npz') as cameras:
                require(sha(root/'cameras_sphere.npz')==p['camera_sha256'],f'{root}: camera hash mismatch')
                for i in range(n):
                    point = poses[i] @ np.array([.03,-.02,1.,1.])
                    direct = (cameras[f'world_mat_{i}'] @ cameras[f'scale_mat_{i}'])[:3,:4] @ point
                    direct = direct[:2]/direct[2] * [w/item['width'],h/item['height']]
                    decomposed = ks[i,:3,:3] @ (np.linalg.inv(poses[i]) @ point)[:3]
                    decomposed = decomposed[:2]/decomposed[2]
                    max_error = max(max_error,float(np.max(np.abs(direct-decomposed))))
            require(max_error<.02, f'{root}: NeuS projection differs by {max_error} pixels')
            frame = p['frames'][0]
            original = np.asarray(Image.open(dataset_root()/cid/frame['image']).convert('RGB'))
            resized = cv2.resize(original.astype(np.float32)/255,(w,h),interpolation=cv2.INTER_AREA)
            require(np.allclose(images[0],resized,atol=1e-6),f'{root}: actual background RGB changed')
            if checksums:
                for frame in p['frames']:
                    for key in ['image','mask']:
                        require(sha(dataset_root()/cid/frame[key])==frame[f'{key}_sha256'],f'{cid}: {frame[key]} hash changed')
            record['poses'].append({'pose':item['pose'],'frames_loaded':n,'resolution':[item['width'],item['height']],
                                    'max_projection_error_px':max_error})
            del images, poses, masks, ks, original, resized
            gc.collect()
        for rel in case['registration_configs']:
            config = load_registration_config(rel)
            require(not any(k in config for k in ['raw_reg','fine_rt','fine_translation','fine_quaternion']),f'{rel}: stale transform cache')
            require(config['case_id']==cid and config['dataset_version']==manifest['dataset_version'],f'{rel}: wrong case/version')
            for role in ['source','target']:
                item = next(p for p in case['poses'] if p['pose']==config[f'{role}_pose'])
                for key,expected_path in [('datadir',item['datadir']),('net_config',item['fine_config']),('mesh_dir',item['mesh'])]:
                    require(Path(config[f'{role}_{key}'])==project_path(expected_path),f'{rel}: {role} {key} mismatch')
                require('gt' not in Path(config[f'{role}_mesh_dir']).parts,f'{rel}: using GT as reconstructed mesh')
            require((config['dataset_W'],config['dataset_H'])==(case['poses'][0]['width'],case['poses'][0]['height']),f'{rel}: wrong matching resolution')
            record['registration_configs']+=1
        require([(load_registration_config(p)['source_pose'],load_registration_config(p)['target_pose']) for p in case['fusion_configs']]==[(1,2),(3,2)],f'{cid}: fusion target mismatch')
        report['cases'].append(record)
        print(f"PASS {cid}: {sum(p['frames_loaded'] for p in record['poses'])} frames / 3 poses / 3 registrations",flush=True)
    report['summary'] = {'cases':len(report['cases']), 'poses':sum(len(c['poses']) for c in report['cases']),
                         'images':sum(p['frames_loaded'] for c in report['cases'] for p in c['poses']),
                         'reconstruction_configs':sum(len(c['poses'])*2 for c in report['cases']),
                         'registration_configs':sum(c['registration_configs'] for c in report['cases']), 'passed':True}
    path = ROOT/'outputs/validation/release_validation.json'; path.parent.mkdir(parents=True,exist_ok=True)
    path.write_text(json.dumps(report,ensure_ascii=False,indent=2)+'\n',encoding='utf-8')
    print(json.dumps(report['summary'],ensure_ascii=False),flush=True)
    print(f'Report: {path}',flush=True)
    return report
