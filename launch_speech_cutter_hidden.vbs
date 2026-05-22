Set shell = CreateObject("WScript.Shell")
Set filesystem = CreateObject("Scripting.FileSystemObject")

appDirectory = filesystem.GetParentFolderName(WScript.ScriptFullName)
shell.CurrentDirectory = appDirectory
shell.Run """" & appDirectory & "\run.bat" & """", 0, False
