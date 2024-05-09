"C:\Program Files\Microsoft Visual Studio\2022\Community\VC\Auxiliary\Build\vcvars64.bat"

mkdir -p build
cd build
cmake -S .. -B . -G "Visual Studio 17 2022" -DCMAKE_BUILD_TYPE=Debug -DPython_ROOT_DIR=C:\Python39_x64
cmake --build . --target py39compiler
cd ..
