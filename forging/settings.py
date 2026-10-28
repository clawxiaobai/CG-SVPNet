"""Paths and fixed paper protocol, shared by GUI and CLI."""
import json
from pathlib import Path
from types import SimpleNamespace

PROJECT = Path(__file__).resolve().parents[1]
MODES = {'E0':'E0_direct','E1':'E1_no_local','E2':'E2_full_subvoxel'}
SPLITS = {
    'train':['case_01','case_02','case_03','case_04','case_07','case_08','case_10','case_11',
             'case_12','case_13','case_14','case_18','case_19','case_20','case_21'],
    'val':['case_09','case_06','case_15'],
    'test':['case_05','case_16','case_17'],
}


def data_paths(folder, kind='bundle'):
    folder=Path(folder).expanduser().resolve()
    root=folder if kind=='processed' else folder/'processed_mpm'
    return dict(data_root=root, surface_root=root.parent/'processed_mpm_surface',
                geometry_root=root.parent/'processed_mpm_geometry')


def make_args(model, folder, kind, output, profile='paper', device='auto'):
    if model not in MODES:
        raise ValueError('Only the five-output E0, E1 and E2 models are supported.')
    cfg=json.loads((PROJECT/'configs'/f'{model}.json').read_text(encoding='utf-8'))
    cfg.update(data_paths(folder,kind))
    out=Path(output).resolve()
    cfg.update(output_root=str(out.parent),exp_name=out.name,profile=profile,device=device,
               resume='',residual_gate_mode='shared')
    if profile=='smoke':
        cfg.update(res=16,width=8,hidden_dim=16,max_points=2048,eval_max_points=2048,
                   point_chunk=1024,epochs=1,patience=2,val_every=1,print_every=1)
    elif profile!='paper':
        raise ValueError('Unknown profile')
    return SimpleNamespace(**cfg)
