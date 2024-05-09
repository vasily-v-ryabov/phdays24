powershell -Command "Invoke-WebRequest https://www.python.org/ftp/python/3.9.13/python-3.9.13-amd64.exe -OutFile python-3.9.13-amd64.exe"
start /W python-3.9.13-amd64.exe /quiet InstallAllUsers=1 TargetDir=C:\Python39_x64 CompileAll=1 Include_debug=1

:: XXX: LLVM installer doesn't contains MLIR libraries
::powershell -Command "Invoke-WebRequest https://github.com/llvm/llvm-project/releases/download/llvmorg-18.1.5/LLVM-18.1.5-win64.exe -OutFile LLVM-18.1.5-win64.exe"
::call LLVM-18.1.5-win64.exe /AllUsers /S /NCRC /D=C:\LLVM-18

C:\Python39_x64\Scripts\pip.exe install cmake
"C:\Program Files\Microsoft Visual Studio\2022\Community\VC\Auxiliary\Build\vcvars64.bat"
