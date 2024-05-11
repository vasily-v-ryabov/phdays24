::"C:\Program Files\Microsoft Visual Studio\2022\Community\VC\Auxiliary\Build\vcvars64.bat"

mkdir build
cd build
cmake -S .. -B . -G "Visual Studio 17 2022" -DCMAKE_BUILD_TYPE=Debug
cmake --build . --target py39compiler
cd ..
