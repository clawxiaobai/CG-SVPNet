"""Full-particle predictions, reference/error exports and central section plots."""
import csv
import json
from pathlib import Path
import shutil
import numpy as np
import torch
from . import hooks
from .dataset import load_case_assets_mpm
from .model import CoinEquivalentStateModel
from .protocol import (make_state_dataset,batch_one,move_state_batch,
                       _new_accumulator,_update_accumulator,_finish_accumulator)
from .vtk_io import write_vtu,write_pvd
from .checkpoints import load_checkpoint


def tool_sections(case_folder, punch_translation):
    """Intersect the saved contact triangles with the physical y=0 plane."""
    segments=[]
    for part,translation in [('die',np.zeros(3)),('punch',punch_translation)]:
        path=Path(case_folder)/f'{part}_contact_surface_meta.npz'
        if not path.exists():continue
        with np.load(path,allow_pickle=False) as z:
            points=z['points']+translation
            tri=points[z['triangles']]
        y=tri[:,:,1]
        tri=tri[(y.min(1)<=0)&(y.max(1)>=0)]
        for triangle in tri:
            hits=[]
            for a,b in [(0,1),(1,2),(2,0)]:
                va,vb=triangle[a],triangle[b]
                if abs(va[1])<1e-8:hits.append(va[[0,2]])
                elif va[1]*vb[1]<0:
                    hits.append((va+(-va[1]/(vb[1]-va[1]))*(vb-va))[[0,2]])
            if len(hits)>=2:
                unique=np.unique(np.round(hits,7),axis=0)
                if len(unique)>=2:segments.append(unique[:2])
    return segments


def plot_sections(data, folder, title, outlines=None):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    from matplotlib.collections import LineCollection
    folder=Path(folder)
    folder.mkdir(parents=True,exist_ok=True)
    ref,pred=data['points_reference_mm'],data['points_prediction_mm']
    mask=np.abs(ref[:,1])<=2.5
    if not mask.any(): return []
    ref,pred=ref[mask],pred[mask]
    utrue,upred=data['displacement_reference_mm'][mask],data['displacement_prediction_mm'][mask]
    fields=[('displacement',np.linalg.norm(utrue,axis=1),np.linalg.norm(upred,axis=1),np.linalg.norm(upred-utrue,axis=1),'Displacement / error (mm)'),
            ('strain',data['equivalent_strain_reference'][mask],data['equivalent_strain_prediction'][mask],None,'Equivalent strain / error (-)'),
            ('stress',data['equivalent_stress_reference_MPa'][mask],data['equivalent_stress_prediction_MPa'][mask],None,'Equivalent stress / error (MPa)')]
    xy=np.concatenate([ref[:,[0,2]],pred[:,[0,2]]])
    lo,hi=xy.min(0),xy.max(0)
    pad=np.maximum((hi-lo)*.03,.5)
    paths=[]
    with plt.rc_context({'font.family':'DejaVu Sans','font.size':10,'axes.titlesize':10,
                         'pdf.fonttype':42,'axes.linewidth':.7}):
        for name,true,estimate,err,label in fields:
            if err is None: err=np.abs(estimate-true)
            vmax=max(float(true.max()),float(estimate.max()),float(err.max()),1e-12)
            width=11.0
            left,gap,panel_width=.055,.035,.285
            panel_height=max(.45,width*panel_width*(hi[1]-lo[1]+2*pad[1])/(hi[0]-lo[0]+2*pad[0]))
            height=panel_height+1.60
            fig=plt.figure(figsize=(width,height))
            axes=[fig.add_axes([left+j*(panel_width+gap),1.10/height,panel_width,panel_height/height]) for j in range(3)]
            for ax,pts,values,heading in zip(axes,[ref,pred,ref],[true,estimate,err],['MPM reference','Prediction','Absolute error']):
                im=ax.scatter(pts[:,0],pts[:,2],c=values,s=.6,cmap='turbo',vmin=0,vmax=vmax,linewidths=0,rasterized=True)
                ax.set(title=heading,xlabel='x (mm)',xlim=(lo[0]-pad[0],hi[0]+pad[0]),ylim=(lo[1]-pad[1],hi[1]+pad[1]))
                ax.set_aspect('equal',adjustable='box')
                ax.set_facecolor('#f5f7fa')
                if outlines:ax.add_collection(LineCollection(outlines,colors='#687581',linewidths=.65))
            axes[0].set_ylabel('z (mm)')
            color_axis=fig.add_axes([.30,.45/height,.43,.085/height])
            fig.colorbar(im,cax=color_axis,orientation='horizontal',label=label)
            fig.suptitle(title+' | reference section: |y| <= 2.5 mm',fontsize=10,y=1-.025/height)
            path=folder/f'{name}.png'
            temporary=path.with_suffix('.png.tmp')
            fig.savefig(temporary,format='png',dpi=180,facecolor='white',bbox_inches='tight',pad_inches=.05)
            temporary.replace(path)
            temporary=path.with_suffix('.pdf.tmp')
            fig.savefig(temporary,format='pdf',dpi=180,facecolor='white',bbox_inches='tight',pad_inches=.05)
            temporary.replace(path.with_suffix('.pdf'))
            plt.close(fig)
            paths.append(str(path))
    return paths


def export_fields(path,data):
    disp_err=data['displacement_prediction_mm']-data['displacement_reference_mm']
    fields={
        'particle_row_id':data['particle_row_id'],
        'initial_position_mm':data['points_initial_mm'],
        'displacement_prediction_mm':data['displacement_prediction_mm'],
        'displacement_reference_mm':data['displacement_reference_mm'],
        'prediction_minus_reference_position_mm':disp_err,
        'displacement_magnitude_prediction_mm':np.linalg.norm(data['displacement_prediction_mm'],axis=1),
        'displacement_magnitude_reference_mm':np.linalg.norm(data['displacement_reference_mm'],axis=1),
        'displacement_error_magnitude_mm':np.linalg.norm(disp_err,axis=1),
    }
    for name in ('equivalent_strain','equivalent_stress'):
        unit='_MPa' if name=='equivalent_stress' else ''
        for source in ('prediction','reference'):
            fields[f'{name}_{source}{unit}']=data[f'{name}_{source}{unit}']
        fields[f'{name}_absolute_error{unit}']=np.abs(data[f'{name}_prediction{unit}']-data[f'{name}_reference{unit}'])
    write_vtu(path,data['points_reference_mm'],fields)


@torch.no_grad()
def predict(args, checkpoint, split='val', formats='both', max_states=0, case=None):
    device=torch.device(args.device if args.device!='auto' else ('cuda' if torch.cuda.is_available() else 'cpu'))
    saved=load_checkpoint(checkpoint)
    saved_args=saved.get('args',{})
    if saved_args.get('decoder_mode')!=args.decoder_mode:
        raise ValueError('Selected model and checkpoint decoder disagree.')
    for key in ('res','width','hidden_dim','point_chunk','mold_input_mode','friction_coefficient','state_init'):
        if key in saved_args: setattr(args,key,saved_args[key])
    # Explicit smoke or limited export is never reported as full evaluation.
    model=CoinEquivalentStateModel(res=args.res,width=args.width,decoder_mode=args.decoder_mode,
        hidden_dim=args.hidden_dim,point_chunk=args.point_chunk,mold_input_mode=args.mold_input_mode,
        friction_coefficient=args.friction_coefficient,state_init=args.state_init).to(device)
    model.load_state_dict(saved['model_state_dict'],strict=True)
    del saved
    model.eval()
    assets=load_case_assets_mpm(str(args.data_root),device,asset_variant='contact')
    root=Path(args.output_root)/args.exp_name/'predictions'
    splits=('val','test') if split=='both' else (split,)
    all_reports={}
    for mode in splits:
        hooks.check_stop()
        ds=make_state_dataset(mode,args,train=False)
        # The original dataset shuffles even evaluation samples. Export in a
        # stable case/time order for readable PVD animations; no training order changes.
        ds.samples.sort(key=lambda s:(s['case_name'],int(Path(s['target_path']).stem.split('_')[-1])))
        if case:
            ds.samples=[s for s in ds.samples if s['case_name']==case]
        if args.profile=='smoke':
            ds.samples=ds.samples[-1:]
        elif max_states:
            ds.samples=ds.samples[-max_states:]
        if not ds.samples: raise ValueError('Selected split/case has no samples.')
        out=root/mode
        out.mkdir(parents=True,exist_ok=True)
        if (out/'metrics.csv').exists(): raise FileExistsError('Prediction output already exists; choose a fresh output directory.')
        metrics=[]
        collections={}
        pool=_new_accumulator()
        temporal={'late_':_new_accumulator(),'final_':_new_accumulator()}
        case_pools={}
        for i in range(len(ds)):
            hooks.check_stop()
            hooks.event('predict',split=mode,step=i+1,total=len(ds))
            raw=ds[i]
            s=move_state_batch(batch_one(raw),device)
            n=s['pos'].shape[1]
            required=n*(280 if formats=='both' else 170)+500_000_000
            if shutil.disk_usage(out).free<required:
                raise OSError('Not enough free space for the next full-particle export.')
            prediction=model(s,assets)
            scale=s['target_state_scale'][:,None,:]
            state=prediction['state'].float()/scale
            disp=prediction['disp'].float()*(s['scale'][:,None,:]*.5)
            true_disp=s['target_disp']*(s['scale'][:,None,:]*.5)
            diff=disp-true_disp
            acc=_new_accumulator()
            _update_accumulator(acc,state,s['target_state_raw'],diff)
            _update_accumulator(pool,state,s['target_state_raw'],diff)
            row=_finish_accumulator(acc)
            progress=float(s['progress'].item())
            step=int(Path(ds.samples[i]['target_path']).stem.split('_')[-1])
            name=s['case']
            with np.load(ds.samples[i]['target_path'],allow_pickle=False) as target_file:
                solver_step=int(target_file.get('step_id',step))
            cp=case_pools.setdefault(name,{'':_new_accumulator(),'late_':_new_accumulator(),'final_':_new_accumulator()})
            _update_accumulator(cp[''],state,s['target_state_raw'],diff)
            for prefix,threshold in [('late_',.70),('final_',.999)]:
                if progress>=threshold:
                    _update_accumulator(temporal[prefix],state,s['target_state_raw'],diff)
                    _update_accumulator(cp[prefix],state,s['target_state_raw'],diff)
            row.update(case=name,step=step,solver_step_id=solver_step,progress=progress,particles=n)
            metrics.append(row)
            cpu=lambda t:t[0].detach().cpu().numpy()
            true=cpu(s['target_state_raw'])
            estimated=cpu(state)
            initial=cpu(s['p_pos_raw'])
            predicted_disp=cpu(disp)
            data=dict(particle_row_id=cpu(s['point_indices']),points_initial_mm=initial,
                points_reference_mm=cpu(s['p_pos_next']),points_prediction_mm=initial+predicted_disp,
                displacement_reference_mm=cpu(true_disp),displacement_prediction_mm=predicted_disp,
                equivalent_strain_reference=true[:,0],equivalent_strain_prediction=estimated[:,0],
                equivalent_stress_reference_MPa=true[:,1],equivalent_stress_prediction_MPa=estimated[:,1])
            d=out/name
            d.mkdir(exist_ok=True)
            stem=f'step_{step:06d}'
            if formats in ('npz','both'):
                temp=d/(stem+'.npz.tmp')
                with temp.open('wb') as f:
                    np.savez_compressed(f,**data,progress=np.float64(progress),target_step=np.int64(step),
                                        target_frame_index=np.int64(step),solver_step_id=np.int64(solver_step))
                temp.replace(d/(stem+'.npz'))
            if formats in ('vtu','both'):
                export_fields(d/(stem+'.vtu'),data)
                collections.setdefault(name,[]).append((progress,stem+'.vtu'))
                write_pvd(d/'sequence.pvd',collections[name])
            outlines=tool_sections(Path(args.data_root)/name,cpu(s['raw_target_punch']))
            plots=plot_sections(data,d/stem,f'{args.model_label} | {name} | step {step} | s={progress:.3f}',outlines)
            if plots: hooks.atomic_json(Path(args.output_root)/args.exp_name/'latest_preview.json',dict(images=plots))
            with (out/'metrics.csv').open('w',newline='',encoding='utf-8-sig') as f:
                writer=csv.DictWriter(f,fieldnames=list(metrics[0]))
                writer.writeheader();writer.writerows(metrics)
            print(f'PREDICT {mode} {i+1}/{len(ds)} {name} step={step} n={n} disp={row["disp_rmse"]:.6f}mm strain={row["equiv_strain_rmse"]:.6f} stress={row["effective_stress_rmse"]:.6f}MPa',flush=True)
            del raw,s,prediction,state,disp,true_disp,diff,data
        combined=_finish_accumulator(pool)
        for prefix,accumulator in temporal.items():
            if accumulator['count']:combined.update(_finish_accumulator(accumulator,prefix))
        by_case={}
        for name,prefixed in case_pools.items():
            by_case[name]={}
            for prefix,accumulator in prefixed.items():
                if accumulator['count']:by_case[name].update(_finish_accumulator(accumulator,prefix))
        report=dict(split=mode,states=len(metrics),particles=int(pool['count']),metrics=combined,by_case=by_case,
                    checkpoint=str(checkpoint),profile=args.profile,
                    full_split=args.profile=='paper' and not max_states and not case,
                    step_definition='step/target_step/target_frame_index: processed file index; solver_step_id: original solver step.',
                    final_definition='progress >= 0.999; can include more than the last frame',
                    geometry='reference particle positions; WarpByVector prediction_minus_reference_position_mm to show predicted shape',
                    slice_definition='abs(reference y) <= 2.5 mm',
                    color_scale='Each field, within each state: reference/prediction/error share zero-to-visible-maximum; no clipping.',
                    particle_id_note='particle_row_id is the preserved NPZ row index, not an invented solver identifier.')
        hooks.atomic_json(out/'summary.json',report)
        all_reports[mode]=report
    return all_reports
