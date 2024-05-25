# Education Python 3.9 compiler for conference talk at PHDays 2024

This is a simplest set of examples of MLIR usage for very small subset of Python.
It is splitted into 5 chapters, but each chapter is much more compact than Toy language in MLIR examples.

## Requirements and environment setup

 1. First you need MLIR (get MLIR 18.x from Ubuntu/Debian apt repositories).
 2. Python 3.9 is also required since it contains AST parser functions in C API (in Python 3.10+ these functions are hidden).
 3. (Windows only) Visual Sttudio 2022 (Community Edition is enough minimum).

On Ubuntu MLIR 18.x can be installed by this Shell script:
```sh
./install.sh
```
On Windows MLIR can be built from source code by this batch script (VS 2022 needs to be installed manually):
```
./install.bat
```

## Chapters' structure

 1. Generate the most simple IR (in 22 lines of code).
 2. Generate IR for variables assignment using Python 3.9 frontend.
 3. Generate IR for 'if-else' statements, including simple type inference.
 4. Lower and translate to LLVM IR (with command line options).
 5. Run LLVM IR using JIT engine ORCv2 (ORC == On-Request Compilation).

Each chapter contains `build.sh` and `build.bat` scripts.
