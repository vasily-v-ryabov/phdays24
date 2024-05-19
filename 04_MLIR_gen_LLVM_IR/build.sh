mkdir -p build
cd build

cmake -S .. -B . -DCMAKE_BUILD_TYPE=Debug
cmake --build . --target py39compiler
cd ..
