"""Real-data end-to-end checks; NOT a repeat of full 500-epoch training."""
from datetime import datetime
import json
from pathlib import Path
import subprocess
import sys
import time

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
from forging.hooks import atomic_json


def main():
    stamp=datetime.now().strftime('%Y%m%d_%H%M%S')
    output=ROOT/'qa/integration'/stamp
    output.mkdir(parents=True)
    results=[]
    for model in ('E0','E1','E2'):
        for kind in ('smoke_train','paper_predict_case16_final'):
            run=output/f'{model}_{kind}'
            run.mkdir()
            cmd=[sys.executable,'-u','-m','forging.cli','train' if kind=='smoke_train' else 'predict',
                 '--model',model,'--output',str(run),'--format','both']
            if kind=='smoke_train':cmd+=['--profile','smoke','--split','val']
            else:cmd+=['--split','test','--case','case_16','--max-states','2']
            print(f'RUN {model} {kind}',flush=True)
            start=time.time()
            with (run/'worker.log').open('w',encoding='utf-8') as log:
                proc=subprocess.run(cmd,cwd=ROOT,stdout=log,stderr=subprocess.STDOUT)
            row=dict(model=model,test=kind,exit_code=proc.returncode,seconds=time.time()-start,
                     output=run.relative_to(ROOT).as_posix())
            if proc.returncode==0:
                summary=json.loads((run/'prediction_summary.json').read_text(encoding='utf-8'))
                row['prediction_states']={k:v['states'] for k,v in summary.items()}
                row['npz_files']=len(list(run.rglob('*.npz')))
                row['vtu_files']=len(list(run.rglob('*.vtu')))
                row['png_files']=len(list(run.rglob('*.png')))
                if kind!='smoke_train':
                    reference=json.loads((ROOT/'pretrained'/model/'final_metrics.json').read_text())['test_by_case']['case_16']
                    measured=summary['test']['metrics']
                    row['absolute_metric_differences']={k:abs(measured[k]-reference['final_'+k]) for k in ['disp_rmse','equiv_strain_rmse','effective_stress_rmse']}
                    row['reference_match']=all(v<1e-5 for v in row['absolute_metric_differences'].values())
            results.append(row)
            atomic_json(ROOT/'qa/integration_report.json',dict(full_retraining_performed=False,tests=results))
            print(row,flush=True)
            if proc.returncode or row.get('reference_match') is False:
                raise SystemExit(f'Integration failed; inspect {run}/worker.log')
    print('All integration checks passed.',flush=True)


if __name__=='__main__':main()
