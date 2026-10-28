"""Background worker for the desktop GUI and command-line reproduction."""
import argparse
import json
import os
from pathlib import Path
import shutil
import sys
import traceback
from . import hooks
from .data_check import inspect_data
from .settings import PROJECT, make_args


def main(argv=None):
    for stream in (sys.stdout,sys.stderr):
        if hasattr(stream,'reconfigure'):stream.reconfigure(encoding='utf-8',errors='replace')
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('action',choices=['check','train','predict'])
    p.add_argument('--data-folder',type=Path,default=PROJECT/'data/closed_die')
    p.add_argument('--data-type',choices=['bundle','processed'],default='bundle')
    p.add_argument('--model',choices=['E0','E1','E2'],default='E0')
    p.add_argument('--output',type=Path,required=True)
    p.add_argument('--profile',choices=['paper','smoke'],default='paper')
    p.add_argument('--device',choices=['auto','cuda','cpu'],default='auto')
    p.add_argument('--split',choices=['val','test','both'],default='val')
    p.add_argument('--format',choices=['npz','vtu','both'],default='both')
    p.add_argument('--checkpoint',type=Path)
    p.add_argument('--resume',type=Path,help='Resume training from checkpoint_latest.pth, including optimizer and scheduler.')
    p.add_argument('--full-hash',action='store_true')
    p.add_argument('--max-states',type=int,default=0,help='Debug: export only the last N states; zero=all.')
    p.add_argument('--case',help='Debug: restrict export to one case; not full-split evaluation.')
    a=p.parse_args(argv)
    if a.max_states<0:p.error('--max-states must be non-negative.')
    if a.resume and a.action!='train':p.error('--resume is only valid with the train action.')
    out=a.output.resolve()
    out.mkdir(parents=True,exist_ok=True)
    # The GUI reserves the folder and creates worker.log only. Never overwrite results.
    if any((out/n).exists() for n in ['request.json','config.json','predictions','final_metrics.json','status.json']):
        raise FileExistsError('Output already contains a run; choose a new folder.')
    hooks.configure(out)
    hooks.atomic_json(out/'request.json',{k:str(v) if isinstance(v,Path) else v for k,v in vars(a).items()})
    try:
        hooks.event('checking')
        def progress(message):
            hooks.check_stop()
            print(message,flush=True)
            hooks.event('checking',message=message)
        report=inspect_data(a.data_folder,a.data_type,a.full_hash,progress)
        hooks.atomic_json(out/'data_check.json',report)
        for warning in report.get('warnings',[]):print('DATA WARNING: '+warning,flush=True)
        if not report['ok']: raise ValueError('\n'.join(report['errors']))
        if a.action=='check':
            hooks.event('completed',message='数据校验通过。'+('已核对全部文件哈希。' if a.full_hash else '本次为结构/大小校验，未重新计算全量哈希。'))
            return 0
        import torch
        if a.device=='cuda' and not torch.cuda.is_available():
            raise RuntimeError('CUDA is not available; install the matching GPU PyTorch build/driver.')
        args=make_args(a.model,a.data_folder,a.data_type,out,a.profile,a.device)
        if a.resume:
            resume=a.resume.expanduser().resolve()
            if not resume.is_file():raise FileNotFoundError(f'Resume checkpoint not found: {resume}')
            args.resume=str(resume)
            # Preserve the earlier best model and history in the new continuation
            # directory.  This lets early stopping and final prediction still use
            # the true best epoch even if no later epoch improves it.
            prior_best=resume.parent/'best_model.pth'
            prior_log=resume.parent/'train_log.csv'
            if prior_best.is_file():shutil.copy2(prior_best,out/'best_model.pth')
            if prior_log.is_file():shutil.copy2(prior_log,out/'train_log.csv')
        hooks.atomic_json(out/'environment.json',dict(python=sys.version,torch=torch.__version__,cuda=torch.version.cuda,
            cudnn=torch.backends.cudnn.version(),gpu=torch.cuda.get_device_name(0) if torch.cuda.is_available() else None))
        if a.profile=='smoke':
            print('SMOKE TEST: reduced grid/points/states, NOT paper results.',flush=True)
        checkpoint=a.checkpoint
        if a.action=='train':
            from .trainer import train
            hooks.event('loading_training_data')
            train(args)
            checkpoint=out/'best_model.pth'
        elif checkpoint is None:
            checkpoint=PROJECT/'pretrained'/a.model/'best_model.pth'
        from .prediction import predict
        hooks.event('loading_prediction_data')
        results=predict(args,checkpoint,a.split,a.format,a.max_states,a.case)
        hooks.atomic_json(out/'prediction_summary.json',results)
        hooks.event('completed',message='Training/prediction and exports completed.')
        return 0
    except hooks.Cancelled as exc:
        hooks.event('stopped',message=str(exc))
        print(str(exc),flush=True)
        return 2
    except Exception as exc:
        hooks.event('failed',message=str(exc))
        traceback.print_exc()
        return 1


if __name__=='__main__':
    raise SystemExit(main())
