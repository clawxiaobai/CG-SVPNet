"""Validate the exact case split, required assets and point correspondence sizes."""
import ast
import hashlib
import json
from pathlib import Path
import struct
import zipfile
from .settings import SPLITS, data_paths


def npy_header(z, name):
    with z.open(name+'.npy') as f:
        if f.read(6)!=b'\x93NUMPY':
            raise ValueError('Invalid numpy header')
        major,minor=f.read(2)
        count=struct.unpack('<H' if major==1 else '<I',f.read(2 if major==1 else 4))[0]
        return ast.literal_eval(f.read(count).decode('utf-8' if major==3 else 'latin1'))


def inspect_data(folder,kind='bundle',full_hash=False,progress=None):
    paths=data_paths(folder,kind)
    errors=[]
    warnings=[]
    details=[]
    expected=set(sum(SPLITS.values(),[]))
    for role,root in paths.items():
        if not root.is_dir():
            errors.append(f'Missing {role}: {root}')
    if errors:
        return dict(ok=False,errors=errors,warnings=warnings)
    actual={p.name for p in paths['data_root'].iterdir() if p.is_dir() and p.name.startswith('case_')}
    if actual!=expected:
        errors.append(f'Case folders mismatch. Missing={sorted(expected-actual)}, extra={sorted(actual-expected)}')
    states={k:0 for k in SPLITS}
    total_points={k:0 for k in SPLITS}
    for i,case in enumerate(sorted(expected),1):
        if progress: progress(f'检查算例 {i}/21：{case}')
        base=paths['data_root']/case
        split=next(k for k,v in SPLITS.items() if case in v)
        try:
            meta=json.loads((base/'metadata.json').read_text(encoding='utf-8-sig'))
            for key in ('world_center','world_scale','punch_translations'):
                if key not in meta: raise ValueError(f'metadata missing {key}')
            for part in ('die','punch'):
                with zipfile.ZipFile(base/f'{part}_contact_asset.npz') as z:
                    names=set(n.removesuffix('.npy') for n in z.namelist())
                    if not {'grad','curvature','center','scale'}.issubset(names) or not ({'distance','sdf'}&names):
                        raise ValueError(f'{part} contact fields incomplete')
            if not (paths['surface_root']/case/'surface_index.npz').is_file():
                raise ValueError('Missing surface_index.npz; original surface sampling cannot be reproduced')
            if not (paths['geometry_root']/case/'blank_features.npz').is_file():
                raise ValueError('Missing blank_features.npz')
            steps=sorted(base.glob('step_*.npz'),key=lambda p:int(p.stem.split('_')[-1]))
            counts=[]
            for p in steps:
                with zipfile.ZipFile(p) as z:
                    if not {'points.npy','effstn.npy','effsts.npy'}.issubset(z.namelist()):
                        raise ValueError(f'{p.name}: missing five-channel supervision')
                    n=npy_header(z,'points')['shape'][0]
                    for field in ('effstn','effsts'):
                        if npy_header(z,field)['shape'][0]!=n:
                            raise ValueError(f'{p.name}: {field} length mismatch')
                    counts.append(n)
            if len(counts)<2: raise ValueError('At least 2 frames required')
            compatible=sum(n==counts[0] for n in counts[1:])
            states[split]+=compatible
            total_points[split]+=counts[0]*compatible
            details.append(dict(case=case,split=split,frames=len(steps),particles=counts[0],target_states=compatible))
        except (OSError,ValueError,KeyError,zipfile.BadZipFile) as exc:
            errors.append(f'{case}: {exc}')
    if states!={'train':370,'val':75,'test':74}:
        errors.append(f'Paper target-state counts must be 370/75/74, got {states}')
    manifest_path=paths['data_root'].parent/'dataset_manifest.json'
    hashed=0
    if manifest_path.exists():
        manifest=json.loads(manifest_path.read_text(encoding='utf-8'))
        for i,r in enumerate(manifest['files'],1):
            rel=Path(r['path'])
            if rel.is_absolute() or '..' in rel.parts: raise ValueError('Unsafe manifest path')
            p=paths['data_root'].parent/rel
            if not p.is_file() or p.stat().st_size!=r['bytes']:
                errors.append(f'Manifest file missing or wrong size: {rel}')
            elif full_hash:
                if progress and i%20==0: progress(f'SHA-256 {i}/{len(manifest["files"])}')
                with p.open('rb') as f: value=hashlib.file_digest(f,'sha256').hexdigest()
                hashed+=1
                if value!=r['sha256']: errors.append(f'Hash mismatch: {rel}')
    else:
        warnings.append('No dataset_manifest.json: structure checked, not verified against packaged bytes.')
    return dict(ok=not errors,errors=errors,warnings=warnings,paths={k:str(v) for k,v in paths.items()},
                states=states,particle_state_counts=total_points,cases=details,
                hashed_files=hashed,full_hash_requested=full_hash)
