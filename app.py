"""Launch the desktop application from this project, with no external paths."""
if __name__=='__main__':
    try:
        from forging.gui import main
        main()
    except Exception as exc:
        import traceback
        from pathlib import Path
        error=traceback.format_exc()
        (Path(__file__).parent/'startup_error.log').write_text(error,encoding='utf-8')
        try:
            import tkinter as tk
            from tkinter import messagebox
            root=tk.Tk();root.withdraw()
            messagebox.showerror('启动失败','请先运行 setup_environment.cmd 安装依赖。\n\n'+str(exc)+'\n\n详情见 startup_error.log')
            root.destroy()
        finally:
            print(error)
        raise SystemExit(1)
