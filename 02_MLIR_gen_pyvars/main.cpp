//===- main.cpp - The Educational MLIR generator --------------------------===//
//
// This file implements the entry point which is mostly based on Toy example,
// but it is more compact and omits some chapters.
//
//===----------------------------------------------------------------------===//

#include "mlir/IR/MLIRContext.h"
#include "mlir/IR/Verifier.h"

#include "mlir/Dialect/LLVMIR/LLVMDialect.h"

#include "py_ast.h"


mlir::ModuleOp mlirGen(mlir::MLIRContext &context, mod_ty py_module, const char *src_name) {
  mlir::OpBuilder builder(&context);
  context.getOrLoadDialect<mlir::LLVM::LLVMDialect>();

  auto srcStringAttr = builder.getStringAttr(llvm::StringRef(src_name));
  mlir::Location loc = mlir::FileLineColLoc::get(srcStringAttr, 0, 0);

  auto module = mlir::ModuleOp::create(loc);
  builder.setInsertionPointToEnd(module.getBody());

  // create function main()
  auto mainFuncType = mlir::LLVM::LLVMFunctionType::get(builder.getI32Type(), {});
  auto mainFunc = builder.create<mlir::LLVM::LLVMFuncOp>(loc, "main", mainFuncType);
  mlir::Block *entryBlock = mainFunc.addEntryBlock();
  builder.setInsertionPointToStart(entryBlock);

  

  // return 0;
  auto constOp = builder.create<mlir::LLVM::ConstantOp>(loc, builder.getI32Type(), 0);
  builder.create<mlir::LLVM::ReturnOp>(loc, constOp->getResult(0));
  return module;
}

int main(int argc, char **argv) {
  [[maybe_unused]] PythonRuntime runtime;
  PyAST script;
  if (argc <= 1) {
    std::cerr << "Usage: py39compiler <filename.py>\n";
    return 1;
  }
  if (!script.parse_file(argv[1]))
    return 2;

  mlir::MLIRContext context;
  mlir::OwningOpRef<mlir::ModuleOp> module = mlirGen(context, script.mod(), argv[1]);
  if (!module)
    return 3;

  if (mlir::failed(mlir::verify(*module))) {
    module->emitError("Module verification failed!");
    module->dump(); // dump incorrect IR anyway
    return 4;
  }
  module->dump(); // to stderr
  return 0;
}
