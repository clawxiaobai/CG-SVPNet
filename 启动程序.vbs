Option Explicit
Dim shell, fs, folder, executable
Set shell = CreateObject("WScript.Shell")
Set fs = CreateObject("Scripting.FileSystemObject")
folder = fs.GetParentFolderName(WScript.ScriptFullName)
shell.CurrentDirectory = folder
executable = folder & "\.venv\Scripts\pythonw.exe"
If Not fs.FileExists(executable) Then
  MsgBox "Please run setup_environment.cmd first, then open this launcher again.", 48, "CG-SVPNet setup required"
  WScript.Quit 1
End If
shell.Run Chr(34) & executable & Chr(34) & " " & Chr(34) & folder & "\app.py" & Chr(34), 1, False
