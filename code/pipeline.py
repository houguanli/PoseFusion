"""Release entry point. Never imports the CUDA training stack for plan/preflight."""
import argparse
import json
import os
import shlex
import shutil
import subprocess
import sys
from pathlib import Path
from project_paths import ROOT, dataset_root, project_path

def load_index():
    return json.loads((ROOT / 'configs/dataset.json').read_text(encoding='utf-8'))

def select_cases(names):
    cases = load_index()['cases']
    if not names or names == ['all']:
        return cases
    unknown = set(names) - {c['case_id'] for c in cases}
    if unknown:
        raise ValueError(f'Unknown cases: {sorted(unknown)}')
    return [c for c in cases if c['case_id'] in names]

def launch(command, dry_run):
    print(shlex.join(command), flush=True)
    if not dry_run:
        subprocess.run(command, cwd=ROOT, check=True, env=runtime_environment())

def runtime_environment():
    env = os.environ.copy()
    if Path('/usr/lib/wsl/lib').is_dir():
        env['LD_LIBRARY_PATH'] = '/usr/lib/wsl/lib:' + env.get('LD_LIBRARY_PATH','')
    if not shutil.which('nvcc', path=env.get('PATH','')) and Path('/usr/local/cuda/bin/nvcc').is_file():
        env.setdefault('CUDA_HOME','/usr/local/cuda')
        env['PATH'] = env['CUDA_HOME'] + '/bin:' + env.get('PATH','')
        env['LD_LIBRARY_PATH'] = env['CUDA_HOME'] + '/lib64:' + env.get('LD_LIBRARY_PATH','')
    return env

def runtime_config(template, dry_run=False):
    template = project_path(template)
    relative = template.relative_to(ROOT / 'configs/reg')
    result = ROOT / 'outputs/registration' / relative
    if not dry_run and not result.exists():
        result.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(template, result)
    return str(result)

def preflight(cases):
    ready, missing = [], []
    for c in cases:
        for p in c['poses']:
            root = project_path(p['datadir'])
            if not root.is_dir() or not (root / 'cameras_sphere.npz').is_file():
                raise FileNotFoundError(root)
            for key in ['checkpoint','mesh']:
                path = project_path(p[key])
                (ready if path.is_file() else missing).append(str(path))
    print(json.dumps({'dataset':str(dataset_root()), 'cases':len(cases),
                      'available_reconstruction_artifacts':len(ready),
                      'missing_reconstruction_artifacts':missing}, indent=2))
    return missing

def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('action', choices=['plan','validate','preflight','train','mesh','register','fuse','evaluate'])
    parser.add_argument('--case', nargs='+', default=['all'], help='case IDs in configs/dataset.json')
    parser.add_argument('--pose', type=int, nargs='+', choices=[1,2,3], default=[1,2,3])
    parser.add_argument('--stage', choices=['coarse','fine','both'], default='both')
    parser.add_argument('--dry-run', action='store_true')
    parser.add_argument('--checksums', action='store_true', help='validate all input SHA256 hashes')
    parser.add_argument('--require-results', action='store_true', help='preflight fails until checkpoints/meshes exist')
    args = parser.parse_args()
    cases = select_cases(args.case)
    if args.action == 'validate':
        from validate_release import validate
        validate(cases, checksums=args.checksums)
        return
    if args.action == 'preflight':
        missing = preflight(cases)
        if missing and args.require_results:
            raise SystemExit('Reconstruct and export meshes before registration.')
        return
    if args.action == 'plan':
        print(json.dumps(cases, ensure_ascii=False, indent=2))
        return
    py = sys.executable
    for case in cases:
        cid = case['case_id']
        if args.action in ['train','mesh']:
            for pose in case['poses']:
                if pose['pose'] not in args.pose:
                    continue
                stages = ['coarse','fine'] if args.stage=='both' else [args.stage]
                for stage in (stages if args.action=='train' else ['fine']):
                    command = [py, str(ROOT / 'run.py'), '--config', str(project_path(pose[f'{stage}_config'])),
                               '--sdf_mode', f'voxurf_{stage}']
                    if args.action == 'train' and stage == 'coarse':
                        command += ['--run_dvgo_init']
                    if args.action == 'mesh':
                        command += ['--render_only','--mesh_from_sdf']
                    launch(command, args.dry_run)
        elif args.action == 'register':
            if not args.dry_run and preflight([case]):
                raise SystemExit('Missing reconstruction artifacts; run train and mesh first.')
            for template in case['registration_configs']:
                launch([py, str(ROOT / 'mp_reg.py'), '--config', runtime_config(template, args.dry_run)], args.dry_run)
        elif args.action == 'fuse':
            configs = [runtime_config(p, args.dry_run) for p in case['fusion_configs']]
            launch([py, str(ROOT / 'fuse_registrations.py'), '--configs', *configs,
                    '--output_name', f'{cid}_fused_pose2'], args.dry_run)
        elif args.action == 'evaluate':
            if case['domain'] != 'synthetic':
                print(f'{cid}: no rigid-transform GT for real cases; skipped.')
                continue
            for config in case['registration_configs']:
                launch([py, str(ROOT / 'evaluate_registration.py'), '--config',
                        runtime_config(config, args.dry_run)], args.dry_run)

if __name__ == '__main__':
    main()
