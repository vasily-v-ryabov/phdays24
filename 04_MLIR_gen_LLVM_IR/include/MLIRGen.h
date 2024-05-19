//
// This file implements a simple IR generation targeting MLIR from a Python AST.
//
//===----------------------------------------------------------------------===//

#include <set>
#include <stack>

#include "mlir/IR/BuiltinOps.h"
#include "mlir/IR/BuiltinTypes.h"
#include "mlir/IR/Diagnostics.h"
#include "mlir/IR/MLIRContext.h"
#include "mlir/Support/LogicalResult.h"

#include "mlir/Dialect/LLVMIR/LLVMDialect.h"
#include "mlir/Dialect/SCF/IR/SCF.h"

#include "llvm/ADT/ScopedHashTable.h"
#include "llvm/ADT/StringRef.h"

#include "py_ast.h"

class MLIRGen {
public:
  MLIRGen(mlir::MLIRContext &context, const char *srcName)
      : builder(&context), srcFilename(srcName) {}

  /// Public API: convert the AST for a Python module (source file) to an MLIR
  /// Module operation.
  mlir::LogicalResult mlirGen(mod_ty pyModule) {
    mlir::LogicalResult genResult = mlir::success();
    auto srcStringAttr = builder.getStringAttr(srcFilename);
    mlir::Location loc = mlir::FileLineColLoc::get(srcStringAttr, 0, 0);

    module = mlir::ModuleOp::create(loc);
    builder.setInsertionPointToEnd(module.getBody());

    // create function main()
    auto mainFuncType =
        mlir::LLVM::LLVMFunctionType::get(builder.getI32Type(), {});
    auto mainFunc =
        builder.create<mlir::LLVM::LLVMFuncOp>(loc, "main", mainFuncType);
    mlir::Block *entryBlock = mainFunc.addEntryBlock();
    builder.setInsertionPointToEnd(entryBlock);

    // at least one scope is required
    llvm::ScopedHashTableScope<llvm::StringRef, mlir::Value> scope(symbolTable);

    switch (pyModule->kind) {
    case Module_kind:
      genResult = mlirGen(pyModule->v.Module.body);
      break;
    default:
      mlir::emitError(loc, "Not supported pyModule->kind = ")
          << pyModule->kind << "\n";
      return mlir::failure();
    }

    // return 0;
    auto constOp =
        builder.create<mlir::LLVM::ConstantOp>(loc, builder.getI32Type(), 0);
    builder.create<mlir::LLVM::ReturnOp>(loc, constOp->getResult(0));
    return genResult;
  }

  mlir::ModuleOp getModule() const { return module; }

private:
  mlir::ModuleOp module;
  mlir::OpBuilder builder;

  /// The symbol table maps a variable name to a value in the current scope.
  /// Entering a function creates a new scope, and the function arguments are
  /// added to the mapping. When the processing of a function is terminated, the
  /// scope is destroyed and the mappings created in this scope are dropped.
  llvm::ScopedHashTable<llvm::StringRef, mlir::Value> symbolTable;

  /// A Python source file name.
  llvm::StringRef srcFilename;

  llvm::MallocAllocator ma;
  std::stack<std::set<llvm::StringRef>> ifElseVariables;

  /// Helper conversion for a Python AST location to an MLIR location.
  template <typename T>
  mlir::Location location(T loc) {
    auto srcStringAttr = builder.getStringAttr(srcFilename);
    return mlir::FileLineColLoc::get(srcStringAttr, loc->lineno,
                                     loc->col_offset);
  }

  void defineVariable(llvm::StringRef name, mlir::Value value) {
    // llvm::outs() << "Add variable '" << name << "' = '" << value << "'\n";
    symbolTable.insert(name.copy(ma), value);
    if (!ifElseVariables.empty())
      ifElseVariables.top().insert(name.copy(ma));
  }

  mlir::FailureOr<mlir::Value> getVariable(mlir::Location loc,
                                           llvm::StringRef name) {
    auto value = symbolTable.lookup(name);
    if (value) // may be nullptr!
      return value;
    mlir::emitError(loc, "Variable '") << name << "' is not defined\n";
    return mlir::failure();
  }

  mlir::FailureOr<mlir::Value> mlirGen(expr_ty expr) {
    auto loc = location(expr);
    switch (expr->kind) {
    case Name_kind:
      switch (expr->v.Name.ctx) {
      case Store:
        mlir::emitError(loc, "Cannot store variable at the right side\n");
        return mlir::failure();
      case Load:
        return getVariable(loc,
                           llvm::StringRef(PyUnicode_AsUTF8(expr->v.Name.id)));
      case Del: {
        llvm::StringRef varName =
            llvm::StringRef(PyUnicode_AsUTF8(expr->v.Name.id));
        if (symbolTable.count(varName)) {
          // insert empty mlir::Value (== nullptr) to mark variable as deleted
          defineVariable(varName, mlir::Value());
          return mlir::success();
        } else {
          mlir::emitError(loc, "Variable is not defined! Cannot delete it!\n");
          return mlir::failure();
        }
      } // Del
      } // switch (expr->v.Name.ctx)
      mlir::emitError(loc, "Not supported expr->v.Name.ctx = ")
          << expr->v.Name.ctx << "\n";
      return mlir::failure(); // Name_kind
    case Constant_kind: {
      llvm::StringRef type = expr->v.Constant.value->ob_type->tp_name;
      if (type == "int") {
        long long intValue = PyLong_AsLongLong(expr->v.Constant.value);
        mlir::Value value = builder.create<mlir::LLVM::ConstantOp>(
            loc, builder.getI64Type(), intValue);
        return value;
      } else if (type == "float") {
        double floatValue = PyFloat_AsDouble(expr->v.Constant.value);
        mlir::Value value = builder.create<mlir::LLVM::ConstantOp>(
            loc, builder.getF64Type(), floatValue);
        return value;
      } else if (type == "bool") {
        long intValue = PyLong_AsLong(expr->v.Constant.value);
        mlir::Value value = builder.create<mlir::LLVM::ConstantOp>(
            loc, builder.getI1Type(), (intValue != 0));
        return value;
      }
      mlir::emitError(loc, "Not support constant type '") << type << "'\n";
      return mlir::failure();
    }
    // TODO: other kinds
    default:
      mlir::emitError(loc, "Not supported expr->kind = ") << expr->kind << "\n";
      return mlir::failure();
    }
  }

  mlir::LogicalResult mlirGen(stmt_ty statement) {
    auto loc = location(statement);
    switch (statement->kind) {
    case Assign_kind: {
      // right side expression
      auto valueOrError = mlirGen(statement->v.Assign.value);
      if (mlir::failed(valueOrError))
        return mlir::failure();
      auto rightValue = valueOrError.value();

      // left side
      expr_ty astTarget = (expr_ty)asdl_seq_GET(statement->v.Assign.targets, 0);
      if (astTarget->kind == Tuple_kind) {
        mlir::emitError(loc, "Tuple is not supported at left side\n");
        return mlir::failure();
      }
      switch (astTarget->kind) {
      case Name_kind: {
        llvm::StringRef varName =
            llvm::StringRef(PyUnicode_AsUTF8(astTarget->v.Name.id));
        switch (astTarget->v.Name.ctx) {
        case Store: {
          defineVariable(varName, rightValue);
          return mlir::success();
        }
        default:
          mlir::emitError(loc, "Loading or deleting variable at the left side "
                               "of assignment is not expected\n");
          return mlir::failure();
        }
      }
      default:
        mlir::emitError(loc, "Not supported astTarget->kind: ")
            << astTarget->kind << "\n";
        return mlir::failure();
      }
      return mlir::success();
    }
    case If_kind: {
      mlir::LogicalResult result = mlir::success();
      auto valueOrError = mlirGen(statement->v.If.test);
      if (mlir::failed(valueOrError))
        return mlir::failure();
      std::set<llvm::StringRef> varsSet;
      llvm::SmallVector<llvm::StringRef> resultVarsVector;
      auto condition = valueOrError.value();
      auto ifOp = builder.create<mlir::scf::IfOp>(
          loc, condition,
          /*thenBuilder=*/
          [&](mlir::OpBuilder &b, mlir::Location loc) {
            ifElseVariables.push(std::set<llvm::StringRef>());

            result = mlirGen(statement->v.If.body);

            varsSet = ifElseVariables.top();
            llvm::SmallVector<mlir::Value> returnValues;
            for (auto it = varsSet.begin(); it != varsSet.end(); it++) {
              auto value = symbolTable.lookup(*it);
              if (value) {
                returnValues.push_back(value);
                resultVarsVector.push_back(it->copy(ma));
              }
            }
            ifElseVariables.pop();
            b.create<mlir::scf::YieldOp>(loc, returnValues);
          },
          /*elseBuilder=*/
          [&](mlir::OpBuilder &b, mlir::Location loc) {
            ifElseVariables.push(std::set<llvm::StringRef>());

            result = mlirGen(statement->v.If.orelse);

            auto elseVars = ifElseVariables.top();
            if (varsSet != elseVars) {
              mlir::emitError(loc, "Assigned variables in 'if' region {")
                  << varsSet << "} are different than in 'else' region {"
                  << elseVars << "}\n";
              result = mlir::failure();
            }
            llvm::SmallVector<mlir::Value> returnValues;
            for (auto it = elseVars.begin(); it != elseVars.end(); it++) {
              auto value = symbolTable.lookup(*it);
              if (value)
                returnValues.push_back(value);
            }
            ifElseVariables.pop();
            b.create<mlir::scf::YieldOp>(loc, returnValues);
          }); // builder.create<mlir::scf::IfOp>
      if (mlir::failed(result))
        return mlir::failure();
      // write returned values to symbol table
      for (size_t i = 0; i < resultVarsVector.size(); i++) {
        defineVariable(resultVarsVector[i].copy(ma), ifOp.getResult(i));
      }
      return result;
    }
    default:
      mlir::emitError(loc, "Not supported statement->kind: ")
          << statement->kind << "\n";
      return mlir::failure();
    }
  }

  mlir::LogicalResult mlirGen(asdl_seq *statements) {
    for (int i = 0; i < asdl_seq_LEN(statements); i++) {
      stmt_ty statement = (stmt_ty)asdl_seq_GET(statements, i);
      if (mlir::failed(mlirGen(statement)))
        return mlir::failure();
    }
    return mlir::success();
  }
}; // class MLIRGen
