#ifndef _PY_AST_H_
#define _PY_AST_H_

#include <iostream>

#define PY_SSIZE_T_CLEAN
#include "Python.h" // PyParser_ASTFromFile, PyArena_New, PyArena_Free
#include "Python-ast.h" // mod_ty


class PythonRuntime {
public:
  PythonRuntime() { Py_Initialize(); }
  ~PythonRuntime() { Py_Finalize(); }
};

class PyAST {
private:
  PyArena *arena;
  mod_ty mod_pointer;

public:
  PyAST() : arena(PyArena_New()), mod_pointer(nullptr) {}

  ~PyAST() {
    if (arena != nullptr) {
      PyArena_Free(arena);
      arena = nullptr;
    }
  }

  mod_ty mod() { return mod_pointer; }

  bool parse_file(const char *name) {
    PyCompilerFlags flags = {0, PY_MINOR_VERSION};
    if (arena == nullptr) {
      std::cerr << "Python arena is not allocated!\n";
      return false;
    }
    auto f = fopen(name, "rt");
    if (f == nullptr) {
      std::cerr << "Cannot open file '" << name << "'\n";
      return false;
    }
    mod_pointer = PyParser_ASTFromFile(f, name, nullptr, Py_file_input,
                                       nullptr, nullptr, &flags, nullptr, arena);
    fclose(f);
    if (mod_pointer == nullptr) {
      std::cerr << "PyParser_ASTFromFile: failure!\n";
      return false;
    }
    return true;
  }

}; // class PyAST

#endif // _PY_AST_H_
