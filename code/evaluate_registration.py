"""Evaluate predicted R/t in the current dataset GT coordinate convention."""
import argparse
import json
import numpy as np
from project_paths import dataset_root, load_registration_config, project_path

def evaluate(config_path):
    config = load_registration_config(config_path)
    if config['domain'] != 'synthetic':
        raise ValueError('This real case has no rigid-transform GT')
    if 'fine_rt' not in config:
        raise ValueError('No predicted fine_rt; run registration first')
    gt_path = dataset_root() / config['case_id'] / 'gt/transforms.json'
    gt = json.loads(gt_path.read_text(encoding='utf-8-sig'))
    key = f"pose{config['source_pose']}_to_pose{config['target_pose']}"
    expected_world = np.asarray(gt['relative_transforms'][key], dtype=np.float64)
    with np.load(project_path(config['source_datadir']) / 'cameras_sphere.npz') as camera:
        source_scale = camera['scale_mat_0']
    with np.load(project_path(config['target_datadir']) / 'cameras_sphere.npz') as camera:
        target_scale = camera['scale_mat_0']
    prediction_world = target_scale @ np.asarray(config['fine_rt']) @ np.linalg.inv(source_scale)
    def so3(matrix):
        u, _, vt = np.linalg.svd(matrix)
        correction = np.eye(3)
        correction[2,2] = np.linalg.det(u @ vt)
        return u @ correction @ vt
    r = so3(prediction_world[:3,:3]) @ so3(expected_world[:3,:3]).T
    error = np.degrees(np.arccos(np.clip((np.trace(r)-1)/2, -1,1)))
    return {'case_id':config['case_id'], 'pair':key, 'rotation_error_deg':float(error),
            'translation_error_world':float(np.linalg.norm(prediction_world[:3,3]-expected_world[:3,3])),
            'translation_units':'current Blender/NeuS world coordinates', 'gt_file':str(gt_path)}

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--config', required=True)
    print(json.dumps(evaluate(parser.parse_args().config), indent=2, ensure_ascii=False))
