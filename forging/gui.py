"""Tk desktop interface; workers run in independent Python processes."""
import csv
from datetime import datetime
import json
import os
from pathlib import Path
import subprocess
import sys
import time
import tkinter as tk
from tkinter import ttk,filedialog,messagebox
from .settings import PROJECT
from .gui_history import finite, read_history

DATA_TYPES={'整套数据包根目录':'bundle','直接选择 processed_mpm 目录':'processed'}
SPLIT_TYPES={'验证集：06 / 09 / 15（模型选择）':'val','测试集：05 / 16 / 17（论文结果）':'test','验证集 + 测试集':'both'}
PROFILE_TYPES={'论文原始设置（500轮，早停80）':'paper','快速流程测试（不用于论文结果）':'smoke'}
FORMAT_TYPES={'NPZ + ParaView VTU':'both','仅 NPZ':'npz','仅 ParaView VTU':'vtu'}
PHASES={'checking':'检查数据','loading_training_data':'载入训练数据','train':'训练中','evaluate':'验证/评价中',
        'loading_prediction_data':'载入预测数据','predict':'导出预测','completed':'已完成','failed':'失败','stopped':'已停止'}


class Application:
    def __init__(self,root):
        self.root=root
        self.process=None
        self.output=None
        self.log_handle=None
        self.log_offset=0
        self.last_csv=None
        self.closing=False
        self.preview_dirs=[]
        self.photo=None
        self.paraview=None
        self.preview_output=None
        self.monitor_only=False
        self.curve_signature=None
        self.last_curve_time=0
        self.resize_job=None
        self.preview_key=None
        self.curve_note=tk.StringVar(value='训练损失按批次更新；验证误差在完成验证后更新。')
        root.title('CG-SVPNet · 五通道 E0 / E1 / E2 复现实验')
        root.geometry('1180x860')
        root.minsize(1000,740)
        root.configure(bg='#edf2f7')
        root.protocol('WM_DELETE_WINDOW',self.close)
        style=ttk.Style(root)
        style.theme_use('clam')
        style.configure('.',font=('Microsoft YaHei UI',10))
        style.configure('TFrame',background='#edf2f7')
        style.configure('TLabelframe',background='#edf2f7')
        style.configure('TLabelframe.Label',font=('Microsoft YaHei UI',10,'bold'),foreground='#17324d')
        style.configure('TLabel',background='#edf2f7',foreground='#24384d')
        style.configure('Title.TLabel',font=('Microsoft YaHei UI',18,'bold'),foreground='#163552')
        style.configure('Accent.TButton',background='#176b91',foreground='white',padding=(15,8))
        style.configure('TButton',padding=(9,5))
        main=ttk.Frame(root,padding=16)
        main.pack(fill='both',expand=True)
        ttk.Label(main,text='五通道材料点预测 · 复现实验工作台',style='Title.TLabel').pack(anchor='w')
        ttk.Label(main,text='位移 (x, y, z) + 等效应变 + 等效应力 (MPa)   |   固定 15 / 3 / 3 算例划分').pack(anchor='w',pady=(3,12))
        setup=ttk.Frame(main);setup.pack(fill='x')
        setup.columnconfigure(0,weight=3);setup.columnconfigure(1,weight=2)
        data=ttk.LabelFrame(setup,text='1  数据与输出',padding=10);data.grid(row=0,column=0,sticky='nsew',padx=(0,10))
        data.columnconfigure(1,weight=1)
        self.data_folder=tk.StringVar(value=str(PROJECT/'data/closed_die'))
        self.data_type=tk.StringVar(value=next(iter(DATA_TYPES)))
        self.output_root=tk.StringVar(value=str(PROJECT/'runs'))
        ttk.Label(data,text='数据目录').grid(row=0,column=0,sticky='w')
        ttk.Entry(data,textvariable=self.data_folder).grid(row=0,column=1,sticky='ew',padx=8)
        ttk.Button(data,text='选择…',command=lambda:self.browse_folder(self.data_folder)).grid(row=0,column=2)
        ttk.Label(data,text='目录类型').grid(row=1,column=0,sticky='w',pady=7)
        ttk.Combobox(data,textvariable=self.data_type,values=list(DATA_TYPES),state='readonly').grid(row=1,column=1,columnspan=2,sticky='ew',padx=8)
        ttk.Label(data,text='输出目录').grid(row=2,column=0,sticky='w')
        ttk.Entry(data,textvariable=self.output_root).grid(row=2,column=1,sticky='ew',padx=8)
        ttk.Button(data,text='选择…',command=lambda:self.browse_folder(self.output_root)).grid(row=2,column=2)
        self.full_hash=tk.BooleanVar(value=False)
        ttk.Checkbutton(data,text='完整 SHA-256 数据校验（约 39 GB，耗时较长）',variable=self.full_hash).grid(row=3,column=0,columnspan=3,sticky='w',pady=(8,0))
        method=ttk.LabelFrame(setup,text='2  方法与预测范围',padding=10);method.grid(row=0,column=1,sticky='nsew')
        method.columnconfigure(1,weight=1)
        self.settings_panels=[data,method]
        self.model=tk.StringVar(value='E0')
        ttk.Label(method,text='训练方法').grid(row=0,column=0,sticky='w')
        line=ttk.Frame(method);line.grid(row=0,column=1,sticky='w',padx=8)
        for name in ['E0','E1','E2']:ttk.Radiobutton(line,text=name,variable=self.model,value=name,command=self.describe_model).pack(side='left',padx=(0,15))
        self.profile=tk.StringVar(value=next(iter(PROFILE_TYPES)))
        self.split=tk.StringVar(value=next(iter(SPLIT_TYPES)))
        self.formats=tk.StringVar(value=next(iter(FORMAT_TYPES)))
        for row,label,var,values in [(1,'训练配置',self.profile,PROFILE_TYPES),(2,'预测范围',self.split,SPLIT_TYPES),(3,'输出格式',self.formats,FORMAT_TYPES)]:
            ttk.Label(method,text=label).grid(row=row,column=0,sticky='w',pady=5)
            ttk.Combobox(method,textvariable=var,values=list(values),state='readonly',width=31).grid(row=row,column=1,sticky='ew',padx=(8,0))
        self.description=tk.StringVar()
        ttk.Label(main,textvariable=self.description,wraplength=1100).pack(anchor='w',pady=10)
        self.describe_model()
        actions=ttk.Frame(main);actions.pack(fill='x')
        self.check_button=ttk.Button(actions,text='检查数据',command=lambda:self.start('check'));self.check_button.pack(side='left')
        self.train_button=ttk.Button(actions,text='开始训练 → 自动预测',style='Accent.TButton',command=lambda:self.start('train'));self.train_button.pack(side='left',padx=8)
        self.resume_button=ttk.Button(actions,text='从断点继续训练…',command=self.choose_resume);self.resume_button.pack(side='left')
        self.predict_button=ttk.Button(actions,text='使用原论文权重预测',command=lambda:self.start('predict'));self.predict_button.pack(side='left')
        self.custom_button=ttk.Button(actions,text='选择已有权重…',command=self.choose_checkpoint);self.custom_button.pack(side='left',padx=8)
        self.stop_button=ttk.Button(actions,text='安全停止',command=self.stop,state='disabled');self.stop_button.pack(side='left')
        ttk.Button(actions,text='使用说明',command=lambda:self.open_path(PROJECT/'README_老师复现.md')).pack(side='right')
        status=ttk.Frame(main);status.pack(fill='x',pady=(10,7))
        self.status=tk.StringVar(value='就绪。先检查数据，再训练；也可以直接用附带的三个权重预测。')
        ttk.Label(status,textvariable=self.status).pack(anchor='w')
        self.progress=ttk.Progressbar(status,maximum=100);self.progress.pack(fill='x',pady=(5,0))
        ttk.Label(main,text='论文设置：64³网格 · 30万训练采样点/状态 · 验证与测试全粒子 · seed 20260805 · 测试集不参与模型选择',font=('Microsoft YaHei UI',9)).pack(side='bottom',anchor='w',pady=(7,0))
        self.notebook=ttk.Notebook(main);self.notebook.pack(fill='both',expand=True)
        logtab=ttk.Frame(self.notebook,padding=6);self.notebook.add(logtab,text='运行日志')
        self.log=tk.Text(logtab,wrap='word',height=10,font=('Consolas',10),bg='#102334',fg='#deebf5',relief='flat')
        scroll=ttk.Scrollbar(logtab,command=self.log.yview);self.log.configure(yscrollcommand=scroll.set)
        self.log.pack(side='left',fill='both',expand=True);scroll.pack(side='right',fill='y')
        self.curvetab=ttk.Frame(self.notebook);self.notebook.add(self.curvetab,text='训练曲线与验证误差')
        self.make_curves()
        self.viewtab=ttk.Frame(self.notebook,padding=8);self.notebook.add(self.viewtab,text='预测截面图')
        toolbar=ttk.Frame(self.viewtab);toolbar.pack(fill='x')
        self.field=tk.StringVar(value='位移')
        fieldbox=ttk.Combobox(toolbar,textvariable=self.field,values=['位移','等效应变','等效应力'],state='readonly',width=12)
        fieldbox.pack(side='left');fieldbox.bind('<<ComboboxSelected>>',lambda e:self.show_preview())
        ttk.Button(toolbar,text='打开已有结果…',command=self.browse_results).pack(side='left',padx=8)
        ttk.Button(toolbar,text='本次运行结果',command=lambda:self.set_preview_source(self.output)).pack(side='left')
        ttk.Button(toolbar,text='打开结果目录',command=lambda:self.open_path(self.preview_output or self.output)).pack(side='left',padx=8)
        ttk.Button(toolbar,text='用 ParaView 打开…',command=self.open_paraview).pack(side='left',padx=8)
        selectors=ttk.Frame(self.viewtab);selectors.pack(fill='x',pady=(5,0))
        ttk.Label(selectors,text='集合 / 算例').pack(side='left')
        self.case_choice=tk.StringVar()
        self.case_box=ttk.Combobox(selectors,textvariable=self.case_choice,state='readonly',width=19)
        self.case_box.pack(side='left',padx=6)
        self.case_box.bind('<<ComboboxSelected>>',lambda e:self.select_case())
        ttk.Label(selectors,text='时间步').pack(side='left')
        self.step_choice=tk.StringVar()
        self.step_box=ttk.Combobox(selectors,textvariable=self.step_choice,state='readonly',width=15)
        self.step_box.pack(side='left',padx=6)
        self.step_box.bind('<<ComboboxSelected>>',lambda e:self.select_step())
        self.prev_button=ttk.Button(selectors,text='上一时间步',command=lambda:self.move_step(-1))
        self.prev_button.pack(side='left')
        self.next_button=ttk.Button(selectors,text='下一时间步',command=lambda:self.move_step(1))
        self.next_button.pack(side='left',padx=6)
        self.preview_note=tk.StringVar(value='尚未选择预测结果。')
        ttk.Label(self.viewtab,textvariable=self.preview_note,font=('Microsoft YaHei UI',9)).pack(anchor='w',pady=(4,0))
        body=ttk.Frame(self.viewtab);body.pack(fill='both',expand=True,pady=(8,0))
        # Keep the full result index separate from the case/step selectors.
        self.result_list=tk.Listbox(body,width=27,font=('Consolas',10),exportselection=False)
        self.result_list.bind('<<ListboxSelect>>',lambda e:self.show_preview())
        body.pack_propagate(False)
        self.preview=ttk.Label(body,text='训练/预测完成后显示截面云图。\n参考值、预测值和误差采用相同色标。',anchor='center')
        self.preview.pack(fill='both',expand=True)
        self.preview.bind('<Configure>',self.schedule_preview)
        self.notebook.bind('<<NotebookTabChanged>>',self.schedule_preview)
        root.after(500,self.poll)

    def make_curves(self):
        from matplotlib.figure import Figure
        from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
        ttk.Label(self.curvetab,textvariable=self.curve_note,font=('Microsoft YaHei UI',9)).pack(anchor='w',padx=8,pady=3)
        self.fig=Figure(figsize=(10,4),dpi=90,layout='constrained',facecolor='#f8fafc')
        self.axes=self.fig.subplots(2,2).flatten()
        self.canvas=FigureCanvasTkAgg(self.fig,master=self.curvetab)
        self.canvas.get_tk_widget().pack(fill='both',expand=True)
        self.draw_curves([])

    def draw_curves(self,rows,epoch_losses=None,batch_losses=None):
        specs=[('train_total','Training weighted loss'),('disp_rmse','Validation displacement RMSE (mm)'),
               ('equiv_strain_rmse','Validation equivalent strain RMSE (-)'),('effective_stress_rmse','Validation equivalent stress RMSE (MPa)')]
        for ax,(key,title) in zip(self.axes,specs):
            ax.clear();ax.set_title(title,fontsize=10);ax.set_xlabel('Epoch',fontsize=9);ax.grid(alpha=.2)
            points=[(finite(r.get('epoch')),finite(r.get(key))) for r in rows]
            points=[(x,y) for x,y in points if x is not None and y is not None]
            if key=='train_total':
                if batch_losses:
                    ax.plot(*zip(*batch_losses),color='#9bbdce',lw=.8,marker='.',markersize=2,label='Batch loss (sampled)')
                if epoch_losses is not None:points=epoch_losses
            if points:
                ax.plot(*zip(*points),color='#176b91',lw=1.7,marker='o',markersize=3,label='Epoch mean' if key=='train_total' else None)
            if key=='train_total' and (points or batch_losses):ax.legend(fontsize=7,loc='upper right')
            if not points and not (key=='train_total' and batch_losses):
                ax.text(.5,.5,'Waiting for training data' if key=='train_total' else 'Waiting for validation',ha='center',va='center',transform=ax.transAxes,color='#64748b',fontsize=9)
            ax.tick_params(labelsize=8)
        self.canvas.draw_idle()

    def describe_model(self):
        self.description.set({'E0':'E0：网格直接插值，输出三维位移、等效应变与等效应力。',
            'E1':'E1：网格隐特征 + 材料点位置/进度 → 非线性五通道查询。',
            'E2':'E2：在 E1 基础上加入局部亚体素坐标、平方项和正余弦编码。'}[self.model.get()])

    def browse_folder(self,var):
        path=filedialog.askdirectory(initialdir=var.get() if Path(var.get()).exists() else str(PROJECT))
        if path:var.set(path)

    def choose_checkpoint(self):
        if self.process or self.monitor_only:return
        messagebox.showinfo('权重文件','请选择本项目或作者提供的可信权重。方法 E0/E1/E2 必须与权重一致。')
        path=filedialog.askopenfilename(initialdir=str(PROJECT/'pretrained'),filetypes=[('PyTorch checkpoint','*.pth')])
        if path:self.start('predict',Path(path))

    def choose_resume(self):
        if self.process or self.monitor_only:return
        path=filedialog.askopenfilename(initialdir=str(PROJECT/'runs'),title='选择 checkpoint_latest.pth',filetypes=[('Training checkpoint','*.pth')])
        if path:self.start('train',resume=Path(path))

    def start(self,action,checkpoint=None,resume=None):
        if self.process or self.monitor_only:return
        try:
            folder=Path(self.data_folder.get()).resolve()
            if not folder.is_dir():raise ValueError('请选择存在的数据目录。')
            output_base=Path(self.output_root.get()).resolve();output_base.mkdir(parents=True,exist_ok=True)
            profile=PROFILE_TYPES[self.profile.get()]
            if action=='predict' and checkpoint is None and profile=='smoke':
                raise ValueError('原论文权重请使用“论文原始设置”。快速流程测试仅用于新建的小规模训练。')
            if action=='train' and profile=='paper':
                if not messagebox.askyesno('确认开始完整训练','将按论文原设置训练，最多500轮，可能耗时较长。\n训练后自动导出所选集合的全部状态。\n每个预测集合建议预留20 GB空间。是否开始？'):return
            stamp=datetime.now().strftime('%Y%m%d_%H%M%S_%f')
            self.output=output_base/f'{action}_{self.model.get()}_{profile}_{stamp}'
            self.output.mkdir(exist_ok=False)
            cmd=[sys.executable,'-u','-m','forging.cli',action,'--data-folder',str(folder),'--data-type',DATA_TYPES[self.data_type.get()],
                '--model',self.model.get(),'--output',str(self.output),'--profile',profile,'--split',SPLIT_TYPES[self.split.get()],
                '--format',FORMAT_TYPES[self.formats.get()]]
            if checkpoint:cmd+=['--checkpoint',str(checkpoint)]
            if resume:cmd+=['--resume',str(resume)]
            if self.full_hash.get():cmd+=['--full-hash']
            self.log_handle=(self.output/'worker.log').open('w',encoding='utf-8')
            env=os.environ.copy();env.update(PYTHONUNBUFFERED='1',PYTHONIOENCODING='utf-8')
            self.process=subprocess.Popen(cmd,cwd=PROJECT,stdout=self.log_handle,stderr=subprocess.STDOUT,env=env,
                creationflags=subprocess.CREATE_NO_WINDOW if os.name=='nt' else 0)
            self.log.delete('1.0','end');self.log_offset=0;self.last_csv=None
            self.curve_signature=None;self.draw_curves([])
            self.set_preview_source(self.output)
            self.status.set('进程已启动，正在初始化…')
            self.toggle_buttons(True)
        except Exception as exc:messagebox.showerror('无法开始',str(exc))

    def toggle_buttons(self,active):
        for button in [self.train_button,self.resume_button,self.check_button,self.predict_button,self.custom_button]:button.configure(state='disabled' if active else 'normal')
        self.stop_button.configure(state='normal' if active else 'disabled')
        def visit(parent):
            for child in parent.winfo_children():
                if isinstance(child,(ttk.Entry,ttk.Combobox,ttk.Radiobutton,ttk.Checkbutton,ttk.Button)):
                    child.configure(state='disabled' if active else ('readonly' if isinstance(child,ttk.Combobox) else 'normal'))
                visit(child)
        for panel in self.settings_panels:visit(panel)

    def stop(self):
        if self.process and self.output:
            (self.output/'STOP').touch()
            self.status.set('已请求停止；等待当前批次/文件写入完成。已有最佳权重和完整导出文件会保留。')
            self.stop_button.configure(state='disabled')

    def poll(self):
        try:
            if self.output:
                logfile=self.output/'worker.log'
                if logfile.exists():
                    with logfile.open('r',encoding='utf-8',errors='replace') as f:
                        f.seek(self.log_offset);text=f.read();self.log_offset=f.tell()
                    if text:self.log.insert('end',text);self.log.see('end')
                statusfile=self.output/'status.json'
                if statusfile.exists():
                    try:
                        d=json.loads(statusfile.read_text(encoding='utf-8'))
                        phase=PHASES.get(d['stage'],d['stage'])
                        details=''
                        if 'epoch' in d:
                            details=f'  轮次 {d["epoch"]}/{d.get("epochs",500)}  批次 {d.get("batch",0)}/{d.get("batches","—")}'
                            self.progress['value']=100*d['epoch']/d.get('epochs',500)
                        elif 'step' in d:
                            details=f'  状态 {d["step"]}/{d["total"]}  {d.get("split","")}'
                            self.progress['value']=100*d['step']/d['total']
                        else:details='  '+d.get('message','')
                        if d['stage']=='completed':self.progress['value']=100
                        self.status.set(f'{phase}{details}   |   已用时 {d.get("elapsed_seconds",0)/60:.1f} 分钟')
                    except (OSError,ValueError):pass
                paths=[self.output/'train_log.csv',self.output/'worker.log']
                signature=tuple((p.stat().st_mtime_ns,p.stat().st_size) if p.exists() else None for p in paths)
                if signature!=self.curve_signature and time.monotonic()-self.last_curve_time>2:
                    try:
                        rows,epoch_losses,batch_losses=read_history(self.output)
                        self.draw_curves(rows,epoch_losses,batch_losses)
                        self.curve_signature=signature;self.last_curve_time=time.monotonic()
                        self.curve_note.set(f'训练：{len(epoch_losses)} 轮均值，{len(batch_losses)} 个批次采样；验证：{len(rows)} 次。第 1 轮及每 2 轮验证一次。')
                    except (ValueError,OSError,csv.Error) as exc:
                        self.curve_note.set('正在等待日志写入完成，将自动重试：'+str(exc))
            self.refresh_previews()
            if self.process and self.process.poll() is not None:
                code=self.process.returncode;self.process=None
                self.log_handle.close();self.log_handle=None;self.toggle_buttons(False)
                if code not in (0,2):self.status.set('任务失败；请查看运行日志。原数据和原论文权重未修改。')
                if self.closing:self.root.destroy();return
        finally:
            if self.root.winfo_exists():self.root.after(800,self.poll)

    def refresh_previews(self):
        source=self.preview_output or self.output
        if not source:return
        dirs=sorted(p.parent for p in (source/'predictions').glob('*/*/step_*/displacement.png'))
        if dirs!=self.preview_dirs:
            previous=self.selected_preview()
            self.preview_dirs=dirs;self.result_list.delete(0,'end')
            for p in dirs:self.result_list.insert('end',str(p.relative_to(source/'predictions')))
            groups=list(dict.fromkeys(self.preview_group(p) for p in dirs))
            self.case_box['values']=groups
            if dirs:
                if self.case_choice.get() not in groups:self.case_choice.set(groups[0])
                self.select_case(previous)
        if not dirs:
            self.clear_preview('此运行尚无可浏览的预测图片。\n完整训练完成后才会自动预测；也可“打开已有结果”查看其他运行。')
            self.preview_note.set(f'{source.name}  |  目前已导出 0 个时间步')
        else:
            self.show_preview()

    @staticmethod
    def preview_group(path):
        return f'{path.parent.parent.name} / {path.parent.name}'

    def selected_preview(self):
        selected=self.result_list.curselection()
        return self.preview_dirs[selected[0]] if selected and selected[0]<len(self.preview_dirs) else None

    def select_case(self,previous=None):
        choices=[p for p in self.preview_dirs if self.preview_group(p)==self.case_choice.get()]
        self.step_box['values']=[p.name for p in choices]
        if choices:
            self.step_choice.set(previous.name if previous in choices else choices[0].name)
            self.select_step()

    def select_step(self):
        for i,path in enumerate(self.preview_dirs):
            if self.preview_group(path)==self.case_choice.get() and path.name==self.step_choice.get():
                self.result_list.selection_clear(0,'end');self.result_list.selection_set(i)
                self.show_preview();return

    def move_step(self,offset):
        values=list(self.step_box['values'])
        if self.step_choice.get() in values:
            index=max(0,min(len(values)-1,values.index(self.step_choice.get())+offset))
            self.step_choice.set(values[index]);self.select_step()

    def clear_preview(self,message):
        self.preview.configure(image='',text=message);self.photo=None;self.preview_key=None
        self.prev_button.configure(state='disabled');self.next_button.configure(state='disabled')

    def set_preview_source(self,directory):
        if directory is None:return
        self.preview_output=Path(directory)
        self.preview_dirs=[];self.result_list.delete(0,'end')
        self.case_choice.set('');self.step_choice.set('')
        self.case_box['values']=[];self.step_box['values']=[]
        self.clear_preview('正在读取预测结果…')
        self.refresh_previews()

    def schedule_preview(self,event=None):
        if self.resize_job is not None:self.root.after_cancel(self.resize_job)
        self.resize_job=self.root.after(120,self.resized_preview)

    def resized_preview(self):
        self.resize_job=None;self.show_preview()

    def show_preview(self):
        from PIL import Image,ImageTk
        directory=self.selected_preview()
        if directory is None:return
        field={'位移':'displacement','等效应变':'strain','等效应力':'stress'}[self.field.get()]
        source=self.preview_output or self.output
        profile='';model=''
        try:
            request=json.loads((source/'request.json').read_text(encoding='utf-8'))
            profile=request.get('profile','');model=request.get('model','')
        except (OSError,ValueError):pass
        warning='快速流程测试（稀疏采样，非精度结果）' if profile=='smoke' else '完整配置' if profile=='paper' else '已有结果'
        detail=''
        try:
            with (directory.parent.parent/'metrics.csv').open(encoding='utf-8-sig',newline='') as stream:
                record=next((r for r in csv.DictReader(stream) if r.get('case')==directory.parent.name and int(r['step'])==int(directory.name.split('_')[1])),None)
            if record:detail=f' | solver step {record.get("solver_step_id","—")} | s={float(record["progress"]):.3f}'
        except (OSError,ValueError,KeyError):pass
        values=list(self.step_box['values'])
        self.preview_note.set(f'{source.name} | {model} {warning} | 本算例已导出 {len(values)} 步{detail}')
        index=values.index(directory.name) if directory.name in values else 0
        self.prev_button.configure(state='normal' if index>0 else 'disabled')
        self.next_button.configure(state='normal' if index+1<len(values) else 'disabled')
        path=directory/(field+'.png')
        if not path.is_file():
            self.clear_preview('此物理量的图片尚未导出。');return
        width=max(1,self.preview.winfo_width()-8);height=max(1,self.preview.winfo_height()-8)
        if width<20 or height<20:return
        key=(path,path.stat().st_mtime_ns,width,height)
        if key==self.preview_key:return
        try:
            with Image.open(path) as original:pic=original.copy()
            pic.thumbnail((width,height),Image.Resampling.LANCZOS)
            self.photo=ImageTk.PhotoImage(pic)
            self.preview.configure(image=self.photo,text='');self.preview_key=key
        except OSError:
            self.clear_preview('图片正在写入，请稍后切换时间步。')

    def browse_results(self):
        directory=filedialog.askdirectory(initialdir=str(PROJECT/'runs'),title='选择包含 predictions 的运行目录')
        if directory:
            self.set_preview_source(directory)

    def attach(self,directory):
        """Read-only monitor: never owns, restarts or cancels the training process."""
        self.output=Path(directory).resolve()
        if not self.output.is_dir():raise ValueError('运行目录不存在：'+str(self.output))
        self.monitor_only=True
        self.root.title('CG-SVPNet · 只读训练监视与结果浏览（关闭不会停止训练）')
        request=self.output/'request.json'
        action='train'
        if request.exists():
            saved=json.loads(request.read_text(encoding='utf-8'))
            action=saved.get('action','train')
            self.model.set(saved['model']);self.describe_model()
            for key,var,mapping in [('split',self.split,SPLIT_TYPES),('profile',self.profile,PROFILE_TYPES),('format',self.formats,FORMAT_TYPES)]:
                match=next((label for label,value in mapping.items() if value==saved.get(key)),None)
                if match:var.set(match)
        self.toggle_buttons(True);self.stop_button.configure(state='disabled')
        self.set_preview_source(self.output)
        self.notebook.select(2 if action=='predict' else 1)

    def open_path(self,path):
        if path is None or not Path(path).exists():return
        if Path(path).suffix=='.md':
            window=tk.Toplevel(self.root);window.title('老师复现 · 使用说明');window.geometry('950x700')
            text=tk.Text(window,wrap='word',font=('Microsoft YaHei UI',11),padx=15,pady=15)
            scroll=ttk.Scrollbar(window,command=text.yview);text.configure(yscrollcommand=scroll.set)
            scroll.pack(side='right',fill='y');text.pack(fill='both',expand=True)
            text.insert('1.0',Path(path).read_text(encoding='utf-8'));text.configure(state='disabled')
            return
        if os.name=='nt':os.startfile(str(path))
        else:subprocess.Popen(['open' if sys.platform=='darwin' else 'xdg-open',str(path)])

    def open_paraview(self):
        source=self.preview_output or self.output
        if not source:return
        pvd=filedialog.askopenfilename(initialdir=str(source/'predictions'),title='选择 sequence.pvd 或单帧 VTU',filetypes=[('ParaView data','*.pvd *.vtu')])
        if not pvd:return
        if not self.paraview:
            self.paraview=filedialog.askopenfilename(title='选择 ParaView 程序 paraview.exe',filetypes=[('ParaView','paraview.exe'),('All','*')])
        if self.paraview:subprocess.Popen([self.paraview,'--data='+pvd])

    def close(self):
        if self.process:
            if messagebox.askyesno('任务正在运行','是否请求安全停止，并在当前操作结束后关闭界面？'):
                self.closing=True;self.stop()
        else:self.root.destroy()


def main():
    import argparse
    parser=argparse.ArgumentParser()
    parser.add_argument('--attach',type=Path,help='Read-only monitor of an existing run; never stops its worker.')
    parser.add_argument('--monitor-latest',action='store_true',help='Monitor the newest training run.')
    args=parser.parse_args()
    root=tk.Tk()
    app=Application(root)
    if args.monitor_latest:
        candidates=list((PROJECT/'runs').glob('train_*'))
        if not candidates:raise ValueError('尚无训练运行目录。')
        args.attach=max(candidates,key=lambda p:p.stat().st_mtime_ns)
    if args.attach:app.attach(args.attach)
    root.mainloop()
