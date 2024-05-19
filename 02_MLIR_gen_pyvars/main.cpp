//===- main.cpp - The Educational MLIR generator --------------------------===//
//
// This file implements the entry point which is mostly based on Toy example,
// but it is more compact and omits some chapters.
//
//===----------------------------------------------------------------------===//

// for MLIRContext and OpBuilder with method verify()
#include "mlir/IR/MLIRContext.h"
#include "mlir/IR/Verifier.h"

// dialects
#include "mlir/Dialect/LLVMIR/LLVMDialect.h"

#include "MLIRGen.h"
#include "py_ast.h"

static char *srcFilename = nullptr;

int main(int argc, char **argv) {
  [[maybe_unused]] PythonRuntime runtime;
  PyAST script;
  if (argc <= 1) {
    std::cerr << "Usage: py39compiler <filename.py>\n";
    return 1;
  }
  srcFilename = argv[1];
  if (!script.parse_file(srcFilename))
    return 2;

  mlir::MLIRContext context;
  context.getOrLoadDialect<mlir::LLVM::LLVMDialect>();

  MLIRGen gen(context, srcFilename);
  if (mlir::failed(gen.mlirGen(script.mod())))
    return 3;

  mlir::OwningOpRef<mlir::ModuleOp> module = gen.getModule();
  if (mlir::failed(mlir::verify(*module))) {
    module->emitError("Module verification failed!");
    module->dump(); // dump incorrect IR anyway
    return 4;
  }

  module->dump(); // dump MLIR to stderr
  return 0;
}
