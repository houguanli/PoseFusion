"""Generate release configs directly from the current dataset manifest."""
import json
import struct
from pathlib import Path

from project_paths import ROOT, dataset_root
DATA = dataset_root()
manifest = json.loads((DATA / 'dataset_manifest.json').read_text(encoding='utf-8-sig'))
index = {'dataset_version': manifest['dataset_version'], 'dataset_root': '../dataset',
         'camera_format': 'NeuS/IDR', 'transform_convention': 'x_target = T @ x_source',
         'registration_coordinate_space': 'normalized NeuS field; current scale matrices are identity',
         'cases': []}
for case in manifest['cases']:
    cid = case['case_id']
    entry = {k:case[k] for k in ['case_id', 'domain', 'paper_label']}
    entry['legacy_source_prefix'] = case.get('source_prefix')
    entry['poses'], entry['registration_configs'] = [], []
    for pose in case['poses']:
        name = f"pose{pose['pose']}"
        png = DATA / cid / pose['frames'][0]['image']
        with png.open('rb') as f:
            f.seek(16)
            width, height = struct.unpack('>II', f.read(8))
        item = {'pose': pose['pose'], 'name': name, 'image_count': pose['image_count'],
                'width': width, 'height': height, 'datadir': f'../dataset/{cid}/{name}',
                'camera': f'../dataset/{cid}/{name}/cameras_sphere.npz',
                'coarse_config': f'configs/cases/{cid}/{name}/coarse.py',
                'fine_config': f'configs/cases/{cid}/{name}/fine.py',
                'checkpoint': f'outputs/reconstruction/{cid}/{name}/fine/surf_last.tar',
                'mesh': f'outputs/reconstruction/{cid}/{name}/fine/meshes/surface_normalized.ply',
                'world_mesh': f'outputs/reconstruction/{cid}/{name}/fine/meshes/surface_world.ply'}
        for stage in ['coarse', 'fine']:
            path = ROOT / item[f'{stage}_config']
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_text(
                f"_base_ = '../../../custom_e2e/{stage}.py'\n"
                f"case_id = '{cid}'\npose_id = {pose['pose']}\n"
                "release_case_config = True\n"
                f"expname = '{cid}/{name}'\nbasedir = './outputs/reconstruction'\n"
                f"WH = [{width}, {height}]\nexpected_image_count = {pose['image_count']}\n"
                f"train_all = {stage == 'coarse'}\n"
                f"data = dict(datadir='../dataset/{cid}/{name}', mode=dict(wmask=True))\n",
                encoding='utf-8')
        entry['poses'].append(item)
    # Minimum common target for fusion, plus the remaining unordered pair.
    for source, target in [(1,2), (3,2), (1,3)]:
        s, t = entry['poses'][source-1], entry['poses'][target-1]
        config = {'case_id': cid, 'domain': case['domain'],
                  'dataset_version': manifest['dataset_version'],
                  'source_pose': source, 'target_pose': target,
                  'source_name': f'{cid}_pose{source}', 'target_name': f'{cid}_pose{target}',
                  'sdf_mode': 'voxurf_fine',
                  'source_exp_dir': str(Path(s['checkpoint']).parent).replace('\\','/'),
                  'target_exp_dir': str(Path(t['checkpoint']).parent).replace('\\','/'),
                  'source_net_config': s['fine_config'], 'target_net_config': t['fine_config'],
                  'source_datadir': s['datadir'], 'target_datadir': t['datadir'],
                  'source_mesh_dir': s['mesh'], 'target_mesh_dir': t['mesh'],
                  'dataset_W': s['width'], 'dataset_H': s['height'],
                  'trace_threshold_rotation': 50.0,
                  'trace_threshold_translation': .8 if cid == 'seal' else (.2 if cid=='bunny' else .5),
                  'coarse_obb_mode': 'pca' if cid in ['bunny','figurine_a','tank','figurine_b','seal','figurine_d','figurine_f','joystick'] else 'min_volume',
                  'coarse_drop_worst_faces': 2 if cid in ['rubik','figurine_d','figurine_f','joystick'] else 0,
                  'save_coarse_debug': True,
                  'coordinate_space': 'normalized_neus',
                  'transform_convention': 'x_target = T @ x_source',
                  'initialization': 'OBB from newly reconstructed meshes; no GT or cached transform'}
        rel = f'configs/reg/{cid}/pose{source}_to_pose{target}.json'
        path = ROOT / rel
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(config, ensure_ascii=False, indent=2)+'\n', encoding='utf-8')
        entry['registration_configs'].append(rel)
    entry['gt_transforms'] = f'../dataset/{cid}/gt/transforms.json' if case['domain']=='synthetic' else None
    entry['fusion_configs'] = entry['registration_configs'][:2]
    index['cases'].append(entry)
(ROOT / 'configs/dataset.json').write_text(json.dumps(index, ensure_ascii=False, indent=2)+'\n', encoding='utf-8')
print('Generated 108 reconstruction configs and 54 registration JSONs for 18 cases.')
