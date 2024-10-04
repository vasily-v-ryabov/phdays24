:: install Python 3.9 and cmake
set PY39_SETUP_DIR=C:\Python39_x64
if not exist "%PY39_SETUP_DIR%" (
    powershell -Command "Invoke-WebRequest https://www.python.org/ftp/python/3.9.13/python-3.9.13-amd64.exe -OutFile python-3.9.13-amd64.exe"
    start /W python-3.9.13-amd64.exe /quiet InstallAllUsers=1 TargetDir=%PY39_SETUP_DIR% CompileAll=1 Include_debug=1
)
pip install cmake

:: XXX: LLVM installer doesn't contains MLIR libraries
::powershell -Command "Invoke-WebRequest https://github.com/llvm/llvm-project/releases/download/llvmorg-19.1.0/LLVM-19.1.0-win64.exe -OutFile LLVM-19.1.0-win64.exe"
::call LLVM-19.1.0-win64.exe /AllUsers /S /NCRC /D=C:\LLVM-19

:: set up Visual Studio build environment
call "C:\Program Files\Microsoft Visual Studio\2022\Community\VC\Auxiliary\Build\vcvars64.bat"

:: clone LLVM repository (tag "llvmorg-19.1.1")
set CLONE_DIR=C:\llvm-project
if not exist "%CLONE_DIR%" (
    git clone https://github.com/llvm/llvm-project.git --depth=1 -b llvmorg-19.1.1 %CLONE_DIR%
)
cd /d %CLONE_DIR%
mkdir build
cd build

:: configure MLIR solution
cmake -S ..\llvm -DLLVM_ENABLE_PROJECTS="mlir" -DLLVM_TARGETS_TO_BUILD="Native" -DCMAKE_BUILD_TYPE=Debug -DLLVM_ENABLE_ASSERTIONS=ON

:: build MLIR & run tests which takes about 1 hour
cmake --build . --target check-mlir
