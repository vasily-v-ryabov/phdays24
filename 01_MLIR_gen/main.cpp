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


mlir::ModuleOp mlirGen(mlir::MLIRContext &context) {
  mlir::OpBuilder builder(&context);
  context.getOrLoadDialect<mlir::LLVM::LLVMDialect>();
  context.getOrLoadDialect<mlir::scf::SCFDialect>();
  auto loc = builder.getUnknownLoc();

  auto module = mlir::ModuleOp::create(loc);
  builder.setInsertionPointToEnd(module.getBody());

  // create function main()
  auto mainFuncType = mlir::LLVM::LLVMFunctionType::get(builder.getI32Type(), {});
  auto mainFunc = builder.create<mlir::LLVM::LLVMFuncOp>(loc, "main", mainFuncType);
  mlir::Block *entryBlock = mainFunc.addEntryBlock();
  builder.setInsertionPointToStart(entryBlock);

  // function body
  auto constOp = builder.create<mlir::LLVM::ConstantOp>(loc, builder.getI32Type(), 0);
  builder.create<mlir::LLVM::ReturnOp>(loc, constOp->getResult(0));
  return module;
}

int main(int argc, char **argv) {
  mlir::MLIRContext context;
  mlir::OwningOpRef<mlir::ModuleOp> module = mlirGen(context);
  if (!module)
    return 1;

  if (mlir::failed(mlir::verify(*module))) {
    module->emitError("Module verification failed!");
    module->dump(); // dump incorrect IR anyway
    return 2;
  }
  module->dump(); // to stderr
  return 0;
}
