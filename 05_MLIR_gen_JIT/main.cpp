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
#include "mlir/Dialect/ControlFlow/IR/ControlFlow.h"
#include "mlir/Dialect/LLVMIR/LLVMDialect.h"
#include "mlir/Dialect/SCF/IR/SCF.h"

// for lowering to "llvm" dialect
#include "mlir/Conversion/ControlFlowToLLVM/ControlFlowToLLVM.h"
#include "mlir/Conversion/SCFToControlFlow/SCFToControlFlow.h"
#include "mlir/Pass/Pass.h"
#include "mlir/Pass/PassManager.h"

// for translation to LLVM IR
#include "mlir/Target/LLVMIR/Dialect/Builtin/BuiltinToLLVMIRTranslation.h"
#include "mlir/Target/LLVMIR/Dialect/LLVMIR/LLVMToLLVMIRTranslation.h"
#include "mlir/Target/LLVMIR/Export.h"

// for command line arguments parser
#include "llvm/Support/CommandLine.h"
#include <string>

// for JIT engine
#include "mlir/ExecutionEngine/ExecutionEngine.h"
#include "mlir/ExecutionEngine/OptUtils.h"
#include "llvm/ExecutionEngine/Orc/JITTargetMachineBuilder.h"
#include "llvm/Support/TargetSelect.h"

#include "MLIRGen.h"
#include "py_ast.h"

static char *srcFilename = nullptr;

enum Action {
  DumpMLIR,   // -emit=mlir
  DumpLLVM,   // -emit=llvm
  DumpLLVMIR, // -emit=llvm-ir
  RunJIT,     // -emit=jit
};

namespace cl = llvm::cl;
static cl::opt<enum Action> emitAction(
    "emit", cl::desc("Select the kind of output desired"),
    cl::values(clEnumValN(DumpMLIR, "mlir", "output the MLIR dump")),
    cl::values(clEnumValN(DumpLLVM, "llvm", "dump the MLIR \"llvm\" dialect")),
    cl::values(clEnumValN(DumpLLVMIR, "llvm-ir", "output the LLVM IR dump")),
    cl::values(clEnumValN(RunJIT, "jit", "JIT and run the main function")),
    cl::init(RunJIT));

static cl::opt<char>
    optLevel("O",
             cl::desc("Optimization level. [-O0, -O1, -O2, or -O3] "
                      "(default = '-O2')"),
             cl::Prefix, cl::init('2'));

static cl::opt<std::string> srcPy(cl::Positional, cl::desc("<input .py file>"),
                                  cl::init("-"), cl::value_desc("filename"));

int main(int argc, char **argv) {
  cl::ParseCommandLineOptions(argc, argv, "Python 3.9 demo compiler\n");

  // parse .py file
  [[maybe_unused]] PythonRuntime runtime;
  PyAST script;
  if (!script.parse_file(srcPy.c_str()))
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

  if (emitAction == Action::DumpMLIR) {
    module->dump(); // dump MLIR to stderr
    return 0;
  }

  // lower to "llvm" dialect
  mlir::PassManager passes(&context);
  passes.addPass(mlir::createConvertSCFToCFPass());
  passes.addPass(mlir::createConvertControlFlowToLLVMPass());
  if (mlir::failed(passes.run(module.get())))
    return 5;

  if (emitAction == Action::DumpLLVM) {
    module->dump(); // dump MLIR to stderr
    return 0;
  }

  // translate to LLVM IR
  mlir::registerBuiltinDialectTranslation(*module->getContext());
  mlir::registerLLVMDialectTranslation(*module->getContext());

  llvm::LLVMContext llvmContext;
  auto llvmModule = mlir::translateModuleToLLVMIR(*module, llvmContext);

  if (emitAction == Action::DumpLLVMIR) {
    llvm::errs() << *llvmModule << "\n"; // dump LLVM IR
    return 0;
  }

  // run JIT
  if (emitAction == Action::RunJIT) {
    llvm::InitializeNativeTarget();
    llvm::InitializeNativeTargetAsmPrinter();

    char optChar[2] = {optLevel.getValue(), '\0'};
    unsigned int level = atoi(optChar);
    auto optPipeline = mlir::makeOptimizingTransformer(
        /*optLevel=*/level, /*sizeLevel=*/0,
        /*targetMachine=*/nullptr);

    mlir::ExecutionEngineOptions engineOptions;
    engineOptions.transformer = optPipeline;
    auto maybeEngine = mlir::ExecutionEngine::create(*module, engineOptions);
    assert(maybeEngine && "failed to construct an execution engine");
    auto &engine = maybeEngine.get();

    llvm::SmallVector<void *> argsAndReturn;
    int32_t exitCode;
    argsAndReturn.push_back(&exitCode); // address for return value
    auto invocationResult = engine->invokePacked("main", argsAndReturn);
    if (invocationResult) {
      llvm::errs() << "JIT invocation failed\n";
      return 6;
    }
    llvm::errs() << "JIT is finished. Exit code: " << exitCode << "\n";
    return exitCode;
  }

  llvm::errs() << "Not supported -emit action!\n";
  return 7;
}
