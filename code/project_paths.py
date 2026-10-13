"""Portable paths shared by release entry points (relative to this code folder)."""
import json
import os
from pathlib import Path

ROOT = Path(__file__).resolve().parent
REG_PATH_KEYS = ('source_datadir', 'target_datadir', 'source_exp_dir',
                 'target_exp_dir', 'source_net_config', 'target_net_config',
                 'source_mesh_dir', 'target_mesh_dir')

def dataset_root():
    value = os.environ.get('POSEFUSION_DATASET')
    return Path(value).expanduser().resolve() if value else ROOT.parent / 'dataset'

def project_path(value):
    path = Path(value)
    parts = path.parts
    if len(parts) >= 2 and parts[:2] == ('..', 'dataset'):
        return dataset_root().joinpath(*parts[2:])
    return path.resolve() if path.is_absolute() else (ROOT / path).resolve()

def load_registration_config(path):
    path = project_path(path)
    config = json.loads(path.read_text(encoding='utf-8-sig'))
    config['__path_fields__'] = {key:config[key] for key in REG_PATH_KEYS}
    for key in REG_PATH_KEYS:
        config[key] = str(project_path(config[key]))
    config['__config_path__'] = str(path)
    return config

def resolve_network_config(config):
    config.basedir = str(project_path(config.basedir))
    config.data.datadir = str(project_path(config.data.datadir))
    return config
