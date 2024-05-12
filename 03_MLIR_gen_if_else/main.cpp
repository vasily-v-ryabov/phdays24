//===- main.cpp - The Educational MLIR generator --------------------------===//
//
// This file implements the entry point which is mostly based on Toy example,
// but it is more compact and omits some chapters.
//
//===----------------------------------------------------------------------===//

#include "mlir/IR/MLIRContext.h"
#include "mlir/IR/Verifier.h"

#include "mlir/Dialect/LLVMIR/LLVMDialect.h"
#include "mlir/Dialect/SCF/IR/SCF.h"
#include "mlir/Dialect/ControlFlow/IR/ControlFlow.h"

#include "mlir/Conversion/SCFToControlFlow/SCFToControlFlow.h"
#include "mlir/Conversion/ControlFlowToLLVM/ControlFlowToLLVM.h"
#include "mlir/Pass/Pass.h"
#include "mlir/Pass/PassManager.h"

#include "py_ast.h"
#include "MLIRGen.h"

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
  context.getOrLoadDialect<mlir::scf::SCFDialect>();
  context.getOrLoadDialect<mlir::cf::ControlFlowDialect>();

  MLIRGen gen(context, srcFilename);
  if (mlir::failed(gen.mlirGen(script.mod())))
    return 3;

  mlir::OwningOpRef<mlir::ModuleOp> module = gen.getModule();
  if (mlir::failed(mlir::verify(*module))) {
    module->emitError("Module verification failed!");
    module->dump(); // dump incorrect IR anyway
    return 4;
  }

  mlir::PassManager passes(&context);
  passes.addPass(mlir::createConvertSCFToCFPass());
  passes.addPass(mlir::createConvertControlFlowToLLVMPass());
  if (mlir::failed(passes.run(module.get())))
    return 5;

  module->dump(); // to stderr
  return 0;
}
