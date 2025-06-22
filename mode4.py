import os
from mode2 import parse_file, Program, FunctionDef, Declaration
from mode3 import TypeChecker

class CodeGenError(Exception):
    pass

class CodeGenerator:
    def __init__(self, filename, ast, typechecker):
        self.filename = filename
        self.ast = ast
        self.tc = typechecker
        self.lines = []
        self.classname = os.path.splitext(os.path.basename(filename))[0]
        self.local_vars = {}
        self.fields = {}

    def emit(self, line=""):
        self.lines.append(line)

    def header(self):
        self.emit(f".class public {self.classname}")
        self.emit(".super java/lang/Object")
        self.emit("")

    def gen_fields(self):
        for decl in self.ast.external_declarations:
            if isinstance(decl, Declaration):
                for d in decl.declarators:
                    if isinstance(d, tuple) and len(d) >= 2:
                        name = d[0]
                        base = self.to_descriptor(decl.type_spec)
                        desc = "[" + base if '[]' in decl.type_spec else base
                        self.fields[name] = desc
                        self.emit(f".field public static {name} {desc}")
        self.emit("")

    def to_descriptor(self, ctype):
        if ctype.endswith("[]"):
            return "[" + self.to_descriptor(ctype[:-2])
        return {"int":"I", "float":"F", "char":"C", "void":"V"}.get(ctype, "I")

    def gen_methods(self):
        for decl in self.ast.external_declarations:
            if isinstance(decl, FunctionDef):
                self.gen_function(decl)

        self.emit(".method <init> : ()V")
        self.emit("    .code stack 1 locals 1")
        self.emit("        aload_0")
        self.emit("        invokespecial Method java/lang/Object <init> ()V")
        self.emit("        return")
        self.emit("    .end code")
        self.emit(".end method\n")

        self.emit(".method public static main : ([Ljava/lang/String;)V")
        self.emit("    .code stack 1 locals 1")
        self.emit(f"        invokestatic Method {self.classname} main ()I")
        self.emit("        invokestatic Method java/lang/System exit (I)V")
        self.emit("        return")
        self.emit("    .end code")
        self.emit(".end method")

    def gen_function(self, fn):
        self.local_vars.clear()
        local_idx = 0
        for i, param in enumerate(fn.parameters):
            self.local_vars[param[0]] = local_idx
            local_idx += 1

        for decl in fn.body:
            if isinstance(decl, Declaration):
                for d in decl.declarators:
                    self.local_vars[d[0]] = local_idx
                    local_idx += 1

        self.emit(f".method public static {fn.name} : ()I")
        self.emit(f"    .code stack 10 locals {local_idx}")

        for stmt in fn.body:
            self.gen_expr(stmt)

        self.emit("        iconst_0")
        self.emit("        ireturn")
        self.emit("    .end code")
        self.emit(".end method\n")

    def gen_expr(self, expr):
        from mode2 import Expression, FunctionCall, BinaryOperation

        if isinstance(expr, FunctionCall):
            for arg in expr.arguments:
                self.gen_expr(arg)
            name = expr.func_expr.value
            ret, params, _ = self.tc.function_table[name]
            desc = "(" + "".join(self.to_descriptor(t) for t in params) + ")" + self.to_descriptor(ret)
            self.emit(f"        invokestatic Method lib440 {name} {desc}")
            return

        if isinstance(expr, BinaryOperation):
            op = expr.operator

            if op == '=':
                self.gen_expr(expr.right)
                if isinstance(expr.left, Expression):
                    v = expr.left.value
                    if v in self.local_vars:
                        self.emit(f"        istore_{self.local_vars[v]} ; {v}")
                    else:
                        d = self.fields[v]
                        self.emit(f"        putstatic Field {self.classname} {v} {d}")
                return

            if op in ('+=', '*='):
                if isinstance(expr.left, Expression):
                    v = expr.left.value
                    if v in self.local_vars:
                        self.emit(f"        iload_{self.local_vars[v]}")
                    else:
                        d = self.fields[v]
                        self.emit(f"        getstatic Field {self.classname} {v} {d}")
                    self.gen_expr(expr.right)
                    self.emit("        iadd" if op == "+=" else "        imul")
                    self.emit("        dup")
                    if v in self.local_vars:
                        self.emit(f"        istore_{self.local_vars[v]}")
                    else:
                        self.emit(f"        putstatic Field {self.classname} {v} {d}")
                    return

            if op in ('+', '-', '*', '/', '%'):
                self.gen_expr(expr.left)
                self.gen_expr(expr.right)
                ops = {'+':'iadd','-':'isub','*':'imul','/':'idiv','%':'irem'}
                self.emit(f"        {ops[op]}")
                return

            raise CodeGenError(f"Unsupported binary operator: {op}")

        if isinstance(expr, Expression):
            v = expr.value
            if isinstance(v, str):
                if v.isdigit():
                    self.emit(f"        ldc {v}")
                    return
                if v in self.local_vars:
                    self.emit(f"        iload_{self.local_vars[v]} ; {v}")
                    return
                d = self.fields.get(v)
                if d:
                    self.emit(f"        getstatic Field {self.classname} {v} {d}")
                    return
            self.emit(f"        ldc {v}")
            return

    def write(self):
        with open(self.classname + ".j", "w") as f:
            f.write("\n".join(self.lines))
        print(f"Generated: {self.classname}.j")

def codegen_file(fn):
    try:
        ast = parse_file(fn, write_file=False)
        tc = TypeChecker(fn)
        tc.check(ast)
        cg = CodeGenerator(fn, ast, tc)
        cg.header()
        cg.gen_fields()
        cg.gen_methods()
        cg.write()
    except Exception as e:
        print(f"Code generation error in file {fn}:")
        print(f"  {e}")

def main(input_filename):
    codegen_file(input_filename)

if __name__ == '__main__':
    import sys
    if len(sys.argv) < 2:
        print("Usage: python mode4.py <input_file>")
        sys.exit(1)
    main(sys.argv[1])
