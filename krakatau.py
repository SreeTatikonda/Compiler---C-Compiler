from pycparser import parse_file, c_ast
import argparse, os, json, sys

class KrakatauEmitter(c_ast.NodeVisitor):
    def __init__(self, class_name):
        self.continue_label_stack = []
        self.break_label_stack = []
        self.label_counter = 0
        self.func_body_buffer = []
        self.expecting_value = False
        self.inside_expr = False
        self.inside_return = False
        self.class_name = class_name
        self.lines = []
        self.global_vars = []  # (name, type, init_code)
        self.init_lines = []
        self.inside_function = False
        self.local_env = {}
        self.local_index = 0
        self.func_sigs = {}
        self.max_stack = 0
        self.lib440_funcs = {
            "getchar": ("()I", False),
            "putchar": ("(I)I", True),
            "getint": ("()I", False),
            "putint": ("(I)V", False),
            "getfloat": ("()F", False),
            "putfloat": ("(F)V", False),
            "putstring": ("([C)V", False),
            "java2c": ("(Ljava/lang/String;)[C", False)
        }

    def emit_short_circuit(self, node, true_label, false_label):
        """
        Emit optimized short-circuiting without producing booleans.
        Jump directly to true_label or false_label.
        """
        
        if isinstance(node, c_ast.BinaryOp) and node.op in ('&&', '||'):
            prev_inside_expr=self.inside_expr
            self.inside_expr=True
            if node.op == '&&':
                mid_label = self.new_label("L_and_next")
                self.emit_short_circuit(node.left, mid_label, false_label)
                self.lines.append(f"{mid_label}:")
                self.emit_short_circuit(node.right, true_label, false_label)
            elif node.op == '||':
                mid_label = self.new_label("L_or_next")
                self.emit_short_circuit(node.left, true_label, mid_label)
                self.lines.append(f"{mid_label}:")
                self.emit_short_circuit(node.right, true_label, false_label)
            self.inside_expr=prev_inside_expr
            return

        elif isinstance(node, c_ast.BinaryOp) and node.op in ['<', '>', '<=', '>=', '==', '!=']:
            prev_inside_expr = self.inside_expr
            self.inside_expr = True

            # Check if left is an assignment (like x = x + 2)
            if isinstance(node.left, c_ast.Assignment):
                # Evaluate right-hand side
                self.visit(node.left.rvalue)
                self.lines.append("        dup")

                # Store into variable
                index, _ = self.local_env[node.left.lvalue.name]
                self.lines.append(f"        istore {index}")
            else:
                self.visit(node.left)

            # Now do right side
            self.visit(node.right)

            op = node.op
            cmp_map = {
                '==': 'if_icmpeq',
                '!=': 'if_icmpne',
                '<':  'if_icmplt',
                '<=': 'if_icmple',
                '>':  'if_icmpgt',
                '>=': 'if_icmpge'
            }

            self.lines.append(f"        {cmp_map[op]} {true_label}")
            self.lines.append(f"        goto {false_label}")

            self.inside_expr = prev_inside_expr
            return



        # fallback – evaluate node and branch if non-zero
        self.visit(node)
        self.lines.append(f"        ifne {true_label}")
        self.lines.append(f"        goto {false_label}")
    
    def new_label(self, prefix):
        label = f"{prefix}_{self.label_counter}"
        self.label_counter += 1
        return label

    def is_numeric_type(self, typename):
        return typename in {"char", "int", "float"}
    
    def visit_If(self, node):
        label_true = self.new_label("L_if_true")
        label_false = self.new_label("L_if_false")
        label_end = self.new_label("L_if_end")

        # Type check
        cond_type = self.get_expr_type(node.cond)
        if not self.is_numeric_type(cond_type):
            lineno = getattr(node.cond, 'coord', None)
            lineno_str = str(lineno.line) if lineno else 'unknown'
            print(f"Code generation error in file {self.class_name}.c line {lineno_str}\n\tCondition in 'if' must be of numeric type", file=sys.stderr)

        self._check_subexpression_numeric(node.cond)

        self.emit_short_circuit(node.cond, label_true, label_false)

        self.lines.append(f"{label_true}:")
        self.visit(node.iftrue)
        self.lines.append(f"        goto {label_end}")
        self.lines.append(f"{label_false}:")

        if node.iffalse:
            self.visit(node.iffalse)

        self.lines.append(f"{label_end}:")

    def visit_Break(self, node):
        if not self.break_label_stack:
            lineno = getattr(node, 'coord', None)
            lineno_str = str(lineno.line) if lineno else 'unknown'
            print(f"Code generation error in file {self.class_name}.c line {lineno_str}\n\tbreak not inside a loop", file=sys.stderr)
            return
        break_label = self.break_label_stack[-1]
        self.lines.append(f"        goto {break_label}")

    
    def visit_Continue(self, node):
        if not self.continue_label_stack:
            lineno = getattr(node, 'coord', None)
            lineno_str = str(lineno.line) if lineno else 'unknown'
            print(f"Code generation error in file {self.class_name}.c line {lineno_str}\n\tcontinue not inside a loop", file=sys.stderr)
            return
        continue_label = self.continue_label_stack[-1]
        self.lines.append(f"        goto {continue_label}")


    def visit_While(self, node):
        cond_type = self.get_expr_type(node.cond)
        if not self.is_numeric_type(cond_type):
            lineno = getattr(node.cond, 'coord', None)
            lineno_str = str(lineno.line) if lineno else 'unknown'
            print(f"Code generation error in file {self.class_name}.c line {lineno_str}\n\tCondition in 'while' must be of numeric type", file=sys.stderr)

        self._check_subexpression_numeric(node.cond)

        label_start = self.new_label("L_loop_start")
        label_check = self.new_label("L_loop_check")
        label_end = self.new_label("L_loop_end")

        self.break_label_stack.append(label_end)
        self.continue_label_stack.append(label_check)
        # Jump to condition check first
        self.lines.append(f"        goto {label_check}")

        # Loop body
        self.lines.append(f"{label_start}:")
        self.visit(node.stmt)

        # Loop condition check
        self.lines.append(f"{label_check}:")
        self.emit_short_circuit(node.cond, label_start, label_end)

        # Loop exit
        self.break_label_stack.pop()
        self.continue_label_stack.pop()
        self.lines.append(f"{label_end}:")
    
    def visit_DoWhile(self, node):
        cond_type = self.get_expr_type(node.cond)
        if not self.is_numeric_type(cond_type):
            lineno = getattr(node.cond, 'coord', None)
            lineno_str = str(lineno.line) if lineno else 'unknown'
            print(f"Code generation error in file {self.class_name}.c line {lineno_str}\n\tCondition in 'do-while' must be of numeric type", file=sys.stderr)

        self._check_subexpression_numeric(node.cond)

        label_start = self.new_label("L_dowhile_start")
        label_check = self.new_label("L_dowhile_check")
        label_end = self.new_label("L_dowhile_end")

        self.break_label_stack.append(label_end)
        self.continue_label_stack.append(label_check)

        self.lines.append(f"{label_start}:")
        self.visit(node.stmt)

        self.lines.append(f"{label_check}:")
        self.emit_short_circuit(node.cond, label_start, label_end)

        self.break_label_stack.pop()
        self.continue_label_stack.pop()
        self.lines.append(f"{label_end}:")

    def visit_For(self, node):
        label_start = self.new_label("L_for_start")
        label_check = self.new_label("L_for_check")
        label_end = self.new_label("L_for_end")
        label_next = self.new_label("L_for_next")

        if node.init:
            self.visit(node.init)

        self.break_label_stack.append(label_end)
        self.continue_label_stack.append(label_next)

        self.lines.append(f"        goto {label_check}")

        self.lines.append(f"{label_start}:")
        self.visit(node.stmt)

        self.lines.append(f"{label_next}:")
        if node.next:
            self.visit(node.next)
            rtype = self.get_expr_type(node.next)
            if not self.is_numeric_type(rtype):
                lineno = getattr(node.next, 'coord', None)
                lineno_str = str(lineno.line) if lineno else 'unknown'
                print(f"Code generation error in file {self.class_name}.c line {lineno_str}\n\tUpdate expression in 'for' must be of numeric type", file=sys.stderr)

        self.lines.append(f"{label_check}:")
        if node.cond:
            cond_type = self.get_expr_type(node.cond)
            if not self.is_numeric_type(cond_type):
                lineno = getattr(node.cond, 'coord', None)
                lineno_str = str(lineno.line) if lineno else 'unknown'
                print(f"Code generation error in file {self.class_name}.c line {lineno_str}\n\tCondition in 'for' must be of numeric type", file=sys.stderr)

            self._check_subexpression_numeric(node.cond)
            self.emit_short_circuit(node.cond, label_start, label_end)
        else:
            self.lines.append(f"        goto {label_start}")

        self.break_label_stack.pop()
        self.continue_label_stack.pop()
        self.lines.append(f"{label_end}:")

    def visit_TernaryOp(self, node):
        # Type check the condition
        cond_type = self.get_expr_type(node.cond)
        if not self.is_numeric_type(cond_type):
            lineno = getattr(node.cond, 'coord', None)
            lineno_str = str(lineno.line) if lineno else 'unknown'
            print(f"Code generation error in file {self.class_name}.c line {lineno_str}\n\tCondition in ternary expression must be numeric", file=sys.stderr)

        label_true = self.new_label("L_tern_true")
        label_false = self.new_label("L_tern_false")
        label_end = self.new_label("L_tern_end")

        # Force expression context for both branches
        prev_inside_expr = self.inside_expr
        self.inside_expr = True

        # Emit condition check
        self.emit_short_circuit(node.cond, label_true, label_false)

        # True branch
        self.lines.append(f"{label_true}:")
        self.visit(node.iftrue)
        self.lines.append(f"        goto {label_end}")

        # False branch
        self.lines.append(f"{label_false}:")
        self.visit(node.iffalse)

        # End label
        self.lines.append(f"{label_end}:")

        # Restore previous expression context
        self.inside_expr = prev_inside_expr


    def _check_subexpression_numeric(self, node):
        """Recursively check all function calls and report if any return void in an expression."""
        if isinstance(node, c_ast.FuncCall):
            rettype = self.get_expr_type(node)
            if not self.is_numeric_type(rettype):
                lineno = getattr(node, 'coord', None)
                lineno_str = str(lineno.line) if lineno else 'unknown'
                print(f"Code generation error in file {self.class_name}.c line {lineno_str}\n\tFunction '{node.name.name}' returns void but is used in an expression", file=sys.stderr)

        for _, child in node.children():
            self._check_subexpression_numeric(child)


    def estimate_stack_size(self, body):
        self.stack_depth = 0
        self.max_stack = 0
        self._simulate_stack(body)
        # Add generous buffer to handle duplication (e.g., dup_x2) or array assignments
        return self.max_stack + 4  # Adjust this buffer if needed

    def _simulate_stack(self, node):
        if node is None:
            return

        if isinstance(node, c_ast.Compound):
            for stmt in node.block_items or []:
                self._simulate_stack(stmt)

        elif isinstance(node, c_ast.Constant):
            self._add_stack(1)

        elif isinstance(node, c_ast.ID):
            self._add_stack(1)

        elif isinstance(node, c_ast.BinaryOp):
            self._simulate_stack(node.left)
            self._simulate_stack(node.right)
            self._add_stack(-1)

        elif isinstance(node, c_ast.Assignment):
            self._simulate_stack(node.rvalue)
            self._add_stack(-1)

        elif isinstance(node, c_ast.Decl):
            if node.init:
                self._simulate_stack(node.init)
                self._add_stack(-1)

        elif isinstance(node, c_ast.UnaryOp):
            self._simulate_stack(node.expr)
            if node.op in ('++', '--', 'p++', 'p--'):
                self._add_stack(1)  # e.g., dup

        elif isinstance(node, c_ast.Return):
            if node.expr:
                self._simulate_stack(node.expr)

        elif isinstance(node, c_ast.FuncCall):
            if node.args:
                for arg in node.args.exprs:
                    self._simulate_stack(arg)
                self._add_stack(-len(node.args.exprs))
            self._add_stack(1)

        elif isinstance(node, c_ast.ArrayRef):
            self._simulate_stack(node.name)
            self._simulate_stack(node.subscript)
            self._add_stack(-1)  # array + index → value

        elif isinstance(node, c_ast.Cast):
            self._simulate_stack(node.expr)

        elif isinstance(node, c_ast.TernaryOp):
            self._simulate_stack(node.cond)
            self._simulate_stack(node.iftrue)
            self._simulate_stack(node.iffalse)
            # Worst-case: both branches use stack simultaneously before join
            self._add_stack(1)

        else:
            for _, child in node.children():
                self._simulate_stack(child)

    def _add_stack(self, delta):
        self.stack_depth += delta
        if self.stack_depth > self.max_stack:
            self.max_stack = self.stack_depth


    def emit(self):
        self.emit_header()
        self.emit_global_fields()
        self.emit_user_functions()
        self.emit_constructor()
        self.emit_clinit()
        self.emit_java_main()

    def emit_header(self):
        self.lines.append(f".class public {self.class_name}")
        self.lines.append(".super java/lang/Object\n")

    def emit_constructor(self):
        self.lines.append(".method <init> : ()V")
        self.lines.append("    .code stack 1 locals 1")
        self.lines.append("        aload_0")
        self.lines.append("        invokespecial Method java/lang/Object <init> ()V")
        self.lines.append("        return")
        self.lines.append("    .end code")
        self.lines.append(".end method\n")

    def emit_global_fields(self):
        for entry in self.global_vars:
            if len(entry) == 4:
                # entry is (name, ctype, init, arr_size)
                name, ctype, init, arr_size = entry
            else:
                # entry is (name, ctype, init) – primitive global
                name, ctype, init = entry
                arr_size = None

            if arr_size:
                # It's an array – emit array declaration and initialization
                jtype = self.get_jtype(ctype)
                self.lines.append(f".field public static {name} [{jtype}")
                self.init_lines.append(f"        ldc {arr_size}")
                self.init_lines.append(f"        newarray {self.get_newarray_type(ctype)}")
                self.init_lines.append(f"        putstatic Field {self.class_name} {name} [{jtype}")
            else:
                # Regular global variable
                jtype = self.get_jtype(ctype)
                self.lines.append(f".field public static {name} {jtype}")
                if init:
                    self.init_lines.append(f"        {init}")
                    self.init_lines.append(f"        putstatic Field {self.class_name} {name} {jtype}")

        if self.init_lines:
            self.lines.append("")


    def emit_clinit(self):
        if not self.init_lines:
            return
        self.lines.append(".method public static <clinit> : ()V")
        self.lines.append("    .code stack 1 locals 1")
        self.lines.extend(self.init_lines)
        self.lines.append("        return")
        self.lines.append("    .end code")
        self.lines.append(".end method\n")

    def emit_java_main(self):
        self.lines.append(".method public static main : ([Ljava/lang/String;)V")
        self.lines.append("    .code stack 1 locals 1")
        self.lines.append(f"        invokestatic Method {self.class_name} main ()I")
        self.lines.append("        invokestatic Method java/lang/System exit (I)V")
        self.lines.append("        return")
        self.lines.append("    .end code")
        self.lines.append(".end method\n")

    def emit_user_functions(self):
        for ext in self.ast.ext:
            if isinstance(ext, c_ast.FuncDef):
                try:
                    self.visit(ext)
                except Exception as e:
                    raise

    def visit_Decl(self, node):
        if isinstance(node.type, c_ast.TypeDecl):
            name = node.name
            ctype = node.type.type.names[0]
            if self.inside_function:
                self.local_env[name] = (self.local_index, ctype)
                if node.init:
                    self.visit(node.init)
                    op = "fstore" if ctype == "float" else "istore"
                    self.lines.append(f"        {op} {self.local_index}")
                self.local_index += 1
            else:
                # Global vars (arrays not supported globally here)
                init = None
                if node.init and isinstance(node.init, c_ast.Constant):
                    init = self.load_constant(node.init).strip()
                self.global_vars.append((name, ctype, init))

        elif isinstance(node.type, c_ast.ArrayDecl):
            name = node.name
            ctype = self.extract_ctype(node.type)
            if self.inside_function:
                # Local array
                self.local_env[name] = (self.local_index, ctype)
                jtype = self.get_jtype(ctype)
                self.lines.append(f"        ldc {node.type.dim.value}")
                self.lines.append(f"        newarray {self.get_newarray_type(ctype)}")
                self.lines.append(f"        astore {self.local_index}")
                self.local_index += 1
            else:
                # Global array
                arr_size = int(node.type.dim.value)
                self.global_vars.append((name, ctype, None, arr_size))

    def get_newarray_type(self, ctype):
        return {
            'int': 'int',
            'float': 'float',
            'char': 'char'
        }.get(ctype, 'int')  # default to int


    def visit_FuncDef(self, node):
        self.inside_function = True
        self.local_env = {}
        self.local_index = 0
        self.func_body_buffer = []

        name = node.decl.name
        sig = self.get_func_signature(node.decl.type)
        self.func_sigs[name] = sig

        # Get return type and store
        rettype_node = node.decl.type.type
        if isinstance(rettype_node, c_ast.TypeDecl) and isinstance(rettype_node.type, c_ast.IdentifierType):
            self.current_func_return_type = rettype_node.type.names[0]
        else:
            self.current_func_return_type = 'int'  # fallback

        # Track parameters
        if node.decl.type.args:
            for param in node.decl.type.args.params:
                pname = param.name
                if isinstance(param.type, c_ast.ArrayDecl):
                    base = self.extract_ctype(param.type)
                    self.local_env[pname] = (self.local_index, f"{base}[]")
                else:
                    base = self.extract_ctype(param.type)
                    self.local_env[pname] = (self.local_index, base)
                self.local_index += 1



        # Buffer body code
        old_lines = self.lines
        self.lines = self.func_body_buffer
        self.visit(node.body)
        self.lines = old_lines

        # Emit method header
        self.lines.append(f".method public static {name} : {sig}")
        stack_size = self.estimate_stack_size(node.body)
        self.lines.append(f"    .code stack {stack_size} locals {self.local_index}")

        # Emit buffered function body
        self.lines.extend(self.func_body_buffer)

        # Check for return
        has_return = self.contains_return(node.body)
        if not has_return:
            if self.current_func_return_type == "void":
                self.lines.append("        return")
            else:
                lineno = getattr(node.body, 'coord', None)
                lineno_str = str(lineno.line+2) if lineno else 'unknown'
                print(f"Code generation error in file {self.class_name}.c line {lineno_str}\n\tMissing return statement in non-void function '{name}'", file=sys.stderr)

        self.lines.append("    .end code")
        self.lines.append(".end method\n")
        self.inside_function = False


    def contains_return(self, node):
        if node is None:
            return False
        if isinstance(node, c_ast.Return):
            return True
        elif hasattr(node, 'block_items') and node.block_items:
            for stmt in node.block_items:
                if self.contains_return(stmt):
                    return True
        for _, child in node.children():
            if self.contains_return(child):
                return True
        return False

    def extract_ctype(self, type_node):
        """
        Recursively extracts the base type name from a declaration node,
        such as 'int', 'float', 'char', etc.
        """
        if isinstance(type_node, c_ast.TypeDecl):
            return self.extract_ctype(type_node.type)
        elif isinstance(type_node, c_ast.ArrayDecl):
            return self.extract_ctype(type_node.type)
        elif isinstance(type_node, c_ast.PtrDecl):
            return self.extract_ctype(type_node.type)
        elif isinstance(type_node, c_ast.IdentifierType):
            return type_node.names[0]
        else:
            return 'int'  # fallback


    def get_func_signature(self, func_decl):
        param_types = []
        if func_decl.args:
            for param in func_decl.args.params:
                if isinstance(param.type, c_ast.ArrayDecl):
                    base = self.extract_ctype(param.type)
                    param_types.append("[" + self.get_jtype(base))
                else:
                    ctype = self.extract_ctype(param.type)
                    param_types.append(self.get_jtype(ctype))
        ret_type = self.get_jtype(func_decl.type.type.names[0])
        return f"({''.join(param_types)}){ret_type}"


    def visit_Compound(self, node):
        for stmt in node.block_items or []:
            self.visit(stmt)
            if isinstance(stmt, c_ast.Return):
                break  # Stop visiting after return


    def visit_Return(self, node):
        self.inside_return = True

        if node.expr:
            self.expecting_value = True
            self.visit(node.expr)
            self.expecting_value = False

        rtype = self.current_func_return_type
        if rtype == 'char':
            self.lines.append("        i2c")
            self.lines.append("        ireturn")
        elif rtype == 'int':
            self.lines.append("        ireturn")
        elif rtype == 'float':
            self.lines.append("        freturn")
        elif rtype == 'void':
            self.lines.append("        return")
        else:
            raise NotImplementedError(f"Unsupported return type: {rtype}")
        self.inside_return = False


    def visit_Constant(self, node):
        self.lines.append(self.load_constant(node))

    def visit_ID(self, node):
        name = node.name
        if name in self.local_env:
            index, ctype = self.local_env[name]
            if ctype.endswith("[]"):
                self.lines.append(f"        aload {index}")
            else:
                load_op = "fload" if ctype == "float" else "iload"
                if index in range(4):
                    self.lines.append(f"        {load_op}_{index}")
                else:
                    self.lines.append(f"        {load_op} {index}")
            return

        for var in self.global_vars:
            name2 = var[0]
            ctype = var[1]
            is_array = len(var) == 4
            jtype = '[' + self.get_jtype(ctype) if is_array else self.get_jtype(ctype)
            if name == name2:
                self.lines.append(f"        getstatic Field {self.class_name} {name} {jtype}")
                return

        raise NotImplementedError(f"Undeclared variable: {node.name}")



    
    def visit_ArrayRef(self, node):
        # visit a[i]
        self.visit(node.name)
        self.visit(node.subscript)
        base_type = self.get_expr_type(node.name)

        if base_type.endswith("[]"):
            base_type = base_type[:-2]
        
        if base_type == 'char':
            self.lines.append("        caload")
        elif base_type == 'int':
            self.lines.append("        iaload")
        elif base_type == 'float':
            self.lines.append("        faload")
        else:
            raise NotImplementedError(f"Array load for type {base_type} not implemented")

    def visit_Assignment(self, node):
        # Check for compound assignment (e.g., x += y)
        is_compound = isinstance(node.op, str) and node.op.endswith('=') and node.op != '='
        base_op = node.op[0] if is_compound else None

        # Get type of LHS
        if isinstance(node.lvalue, c_ast.ID):
            name = node.lvalue.name
            is_global = name in [v[0] for v in self.global_vars]

            if name in self.local_env:
                ctype = self.local_env[name][1]
                index = self.local_env[name][0]
            else:
                for var in self.global_vars:
                    if var[0] == name:
                        ctype = var[1]
                        break
                else:
                    ctype = 'int'
                index = None  # not used for global

            store_op = "fstore" if ctype == "float" else "istore"
            load_op = "fload" if ctype == "float" else "iload"
            op_map = {
                '+': 'fadd' if ctype == 'float' else 'iadd',
                '-': 'fsub' if ctype == 'float' else 'isub',
                '*': 'fmul' if ctype == 'float' else 'imul',
                '/': 'fdiv' if ctype == 'float' else 'idiv',
                '%': 'irem'  # no fmod in JVM
            }

            if is_compound:
                # Load lhs
                if is_global:
                    jtype = self.get_jtype(ctype)
                    self.lines.append(f"        getstatic Field {self.class_name} {name} {jtype}")
                else:
                    if index < 4:
                        self.lines.append(f"        {load_op}_{index}")
                    else:
                        self.lines.append(f"        {load_op} {index}")

                # Load rhs
                self.visit(node.rvalue)

                # Apply operator
                self.lines.append(f"        {op_map[base_op]}")

                if ctype == "char":
                    self.lines.append("        i2c")

            else:
                # === Boolean short-circuit assignment support ===
                if isinstance(node.rvalue, c_ast.BinaryOp) and node.rvalue.op in ('&&', '||'):
                    label_true = self.new_label("L_bool_true")
                    label_false = self.new_label("L_bool_false")
                    label_end = self.new_label("L_bool_end")

                    self.emit_short_circuit(node.rvalue, label_true, label_false)

                    self.lines.append(f"{label_true}:")
                    self.lines.append("        iconst_1")
                    self.lines.append(f"        goto {label_end}")

                    self.lines.append(f"{label_false}:")
                    self.lines.append("        iconst_0")

                    self.lines.append(f"{label_end}:")
                else:
                    # Regular simple assignment
                    self.visit(node.rvalue)

            # If the result needs to be used, duplicate before storing
            if self.inside_return or self.expecting_value:
                self.lines.append("        dup")

            # Store back
            if is_global:
                jtype = self.get_jtype(ctype)
                self.lines.append(f"        putstatic Field {self.class_name} {name} {jtype}")
            else:
                if index < 4:
                    self.lines.append(f"        {store_op}_{index}")
                else:
                    self.lines.append(f"        {store_op} {index}")

        elif isinstance(node.lvalue, c_ast.ArrayRef):
            arr_type = self.get_expr_type(node.lvalue.name)
            op_map = {
                '+': 'iadd' if arr_type != 'float' else 'fadd',
                '-': 'isub' if arr_type != 'float' else 'fsub',
                '*': 'imul' if arr_type != 'float' else 'fmul',
                '/': 'idiv' if arr_type != 'float' else 'fdiv',
                '%': 'irem'  # no fmod in JVM
            }
            store_op = {
                'char': 'castore',
                'int': 'iastore',
                'float': 'fastore'
            }[arr_type]

            if is_compound:
                # Array compound assignment: A[i] += x
                self.visit(node.lvalue.name)       # aload array
                self.lines.append("        dup")   # duplicate array
                self.visit(node.lvalue.subscript)  # push index
                self.lines.append("        dup_x1")  # swap to get array under index
                load_op = {
                    'char': 'caload',
                    'int': 'iaload',
                    'float': 'faload'
                }[arr_type]
                self.lines.append(f"        {load_op}")
                self.visit(node.rvalue)
                self.lines.append(f"        {op_map[base_op]}")
                if arr_type == 'char':
                    self.lines.append("        i2c")
            else:
                # Simple array assignment: A[i] = x
                self.visit(node.lvalue.name)
                self.visit(node.lvalue.subscript)
                self.visit(node.rvalue)

            # If return/expecting value, preserve before storing
            if self.inside_return or self.expecting_value:
                self.lines.append("        dup_x2")
                self.lines.append("        dup")
            else:
                # Don't duplicate if not needed
                pass

            # Store
            self.lines.append(f"        {store_op}")

        else:
            raise NotImplementedError(f"Assignment to {type(node.lvalue)} not supported")


    def visit_UnaryOp(self, node):
        op = node.op
        expr_type = self.get_expr_type(node.expr)

        if op == '-':
            self.visit(node.expr)
            self.lines.append("        fneg" if expr_type in ('float', 'double') else "        ineg")

        elif op == '+':
            self.visit(node.expr)

        elif op == '!':
            self.visit(node.expr)
            label_id = self.label_counter
            self.label_counter += 1
            ltrue = f"L_true_{label_id}"
            lend = f"L_end_{label_id}"
            self.lines.append(f"        ifeq {ltrue}")
            self.lines.append("        iconst_0")
            self.lines.append(f"        goto {lend}")
            self.lines.append(f"{ltrue}:")
            self.lines.append("        iconst_1")
            self.lines.append(f"{lend}:")

        elif op == '~':
            self.visit(node.expr)
            self.lines.append("        iconst_m1")
            self.lines.append("        ixor")

        elif op in ('++', '--', 'p++', 'p--'):
            if isinstance(node.expr, c_ast.ArrayRef):
                is_post = op.startswith("p")
                is_inc = "++" in op
                array_type = self.get_expr_type(node.expr.name)
                if array_type != "int":
                    raise NotImplementedError("Only int arrays supported for ++/--")

                self.visit(node.expr.name)       # push array ref
                self.visit(node.expr.subscript)  # push index
                self.lines.append("        dup2")        # [arr, idx, arr, idx]
                self.lines.append("        iaload")      # [arr, idx, val]

                if is_post and (self.expecting_value or self.inside_expr or self.inside_return):
                    self.lines.append("        dup_x2")  # [val, arr, idx, val]

                self.lines.append("        iconst_1")
                self.lines.append("        iadd" if is_inc else "isub")
                self.lines.append("        iastore")

                if not is_post:
                    # Load updated value again
                    self.visit(node.expr.name)
                    self.visit(node.expr.subscript)
                    self.lines.append("        iaload")
                return
            
            if not isinstance(node.expr, c_ast.ID):
                raise NotImplementedError("++/-- only supported on identifiers")

            name = node.expr.name
            is_global = name in [v[0] for v in self.global_vars]

            if name in self.local_env:
                index, ctype = self.local_env[name]
            else:
                for var in self.global_vars:
                    if var[0] == name:
                        ctype = var[1]
                        break
                else:
                    raise NotImplementedError(f"Unknown identifier: {name}")
                index = None

            store_op = "fstore" if ctype == "float" else "istore"
            load_op = "fload" if ctype == "float" else "iload"
            const_op = "fconst_1" if ctype == "float" else "iconst_1"
            math_op = {
                '++': "fadd" if ctype == "float" else "iadd",
                '--': "fsub" if ctype == "float" else "isub",
                'p++': "fadd" if ctype == "float" else "iadd",
                'p--': "fsub" if ctype == "float" else "isub",
            }[op]

            is_post = op.startswith("p")
            is_inc = "++" in op

            # Special case: s[i++]
            is_array_index = (
                hasattr(node, 'parent') and isinstance(node.parent, c_ast.ArrayRef)
                and node.parent.subscript == node
            )

            if is_global:
                jtype = self.get_jtype(ctype)
                if is_post and self.expecting_value:
                    self.lines.append(f"        getstatic Field {self.class_name} {name} {jtype}")
                    self.lines.append("        dup")
                self.lines.append(f"        getstatic Field {self.class_name} {name} {jtype}")
                self.lines.append(f"        {const_op}")
                self.lines.append(f"        {math_op}")
                self.lines.append(f"        putstatic Field {self.class_name} {name} {jtype}")
            else:
                # Case: used as array index i++
                if is_post and is_array_index:
                    if index < 4:
                        self.lines.append(f"        iload_{index}")
                    else:
                        self.lines.append(f"        iload {index}")
                    self.lines.append("        dup")
                    self.lines.append("        iconst_1")
                    self.lines.append("        iadd")
                    if index < 4:
                        self.lines.append(f"        istore_{index}")
                    else:
                        self.lines.append(f"        istore {index}")
                    return  # Done!
                
                # Other cases
                if is_post:
                    # post-increment: use original value, then increment (no duplicate loads)
                    if index < 4:
                        self.lines.append(f"        {load_op}_{index}")
                    else:
                        self.lines.append(f"        {load_op} {index}")

                    # If the value is being used in an expression (e.g., x = y + x++;), keep it
                    if self.expecting_value or self.inside_expr or self.inside_return:
                        self.lines.append("        dup")

                    # Increment after use — no need to load again!
                    self.lines.append("        iconst_1")
                    self.lines.append(f"        {math_op}")
                    if index < 4:
                        self.lines.append(f"        {store_op}_{index}")
                    else:
                        self.lines.append(f"        {store_op} {index}")

                else:
                    # pre-increment/decrement (used in expressions like 40 - --i)
                    self.lines.append(f"        {load_op}_{index}" if index < 4 else f"        {load_op} {index}")
                    self.lines.append(f"        {const_op}")
                    self.lines.append(f"        {math_op}")
                    self.lines.append(f"        dup")
                    self.lines.append(f"        {store_op}_{index}" if index < 4 else f"        {store_op} {index}")

        else:
            raise NotImplementedError(f"Unary operator '{op}' not supported")


    def visit_BinaryOp(self, node):
        left_type = self.get_expr_type(node.left)
        right_type = self.get_expr_type(node.right)
        is_float = 'float' in (left_type, right_type)

        if node.op in ('&&', '||'):
            self.inside_expr=True
            label_id = getattr(self, "label_counter", 0)
            self.label_counter = label_id + 1
            l_short = f"L_short_{label_id}"
            l_end = f"L_end_{label_id}"

            if node.op == '&&':
                self.visit(node.left)
                self.lines.append(f"        ifeq {l_short}")  # if left is false, skip right
                self.visit(node.right)
                self.lines.append(f"        ifeq {l_short}")  # if right is false, jump
                self.lines.append("        iconst_1")          # both are true
                self.lines.append(f"        goto {l_end}")
                self.lines.append(f"{l_short}:")
                self.lines.append("        iconst_0")          # one is false
                self.lines.append(f"{l_end}:")
            else:  # '||'
                self.visit(node.left)
                self.lines.append(f"        ifne {l_short}")  # if left is true, skip right
                self.visit(node.right)
                self.lines.append(f"        ifne {l_short}")  # if right is true, jump
                self.lines.append("        iconst_0")          # both false
                self.lines.append(f"        goto {l_end}")
                self.lines.append(f"{l_short}:")
                self.lines.append("        iconst_1")          # one is true
                self.lines.append(f"{l_end}:")
            return


        # Arithmetic operators
        if node.op in ['+', '-', '*', '/', '%']:
            self.expecting_value = True
            self.visit(node.left)
            self.visit(node.right)
            ops = {
                '+': 'fadd' if is_float else 'iadd',
                '-': 'fsub' if is_float else 'isub',
                '*': 'fmul' if is_float else 'imul',
                '/': 'fdiv' if is_float else 'idiv',
                '%': None if is_float else 'irem'  # No fmod in JVM
            }
            op = ops.get(node.op)
            if not op:
                raise NotImplementedError(f"Unsupported binary operator or float mod: {node.op}")
            self.lines.append(f"        {op}")
            self.expecting_value = False
            return


        # Comparison operators
        if node.op in ['==', '!=', '<', '<=', '>', '>=']:
            label_id = getattr(self, "label_counter", 0)
            self.label_counter = label_id + 1
            ltrue = f"L_cmp_true_{label_id}"
            lend = f"L_cmp_end_{label_id}"

            if is_float:
                self.lines.append("        fcmpl")  # -1, 0, 1
                branch_map = {
                    '==': f"ifeq {ltrue}",
                    '!=': f"ifne {ltrue}",
                    '<':  f"iflt {ltrue}",
                    '<=': f"ifle {ltrue}",
                    '>':  f"ifgt {ltrue}",
                    '>=': f"ifge {ltrue}"
                }
            else:
                branch_map = {
                    '==': f"if_icmpeq {ltrue}",
                    '!=': f"if_icmpne {ltrue}",
                    '<':  f"if_icmplt {ltrue}",
                    '<=': f"if_icmple {ltrue}",
                    '>':  f"if_icmpgt {ltrue}",
                    '>=': f"if_icmpge {ltrue}"
                }

            if is_float:
                self.lines.append(f"        {branch_map[node.op]}")
            else:
                # try to reorder left and right if they were just emitted
                # this only works if both sides generated exactly one instruction
                # otherwise skip reordering (let JVM handle comparison anyway)
                if len(self.lines) >= 2 and all("load" in self.lines[-i] or "ldc" in self.lines[-i] or "iconst" in self.lines[-i] for i in (1, 2)):
                    self.lines[-2], self.lines[-1] = self.lines[-1], self.lines[-2]
                self.lines.append(f"        {branch_map[node.op]}")


            self.lines.append("        iconst_0")
            self.lines.append(f"        goto {lend}")
            self.lines.append(f"{ltrue}:")
            self.lines.append("        iconst_1")
            self.lines.append(f"{lend}:")
            return

        raise NotImplementedError(f"Unsupported binary operator: {node.op}")



    def visit_FuncCall(self, node):
        fname = node.name.name
        args = node.args.exprs if node.args else []
        if node.args:
            for arg in node.args.exprs:
                self.expecting_value = True
                self.visit(arg)
                self.expecting_value = False

        if fname == "putstring" and args and isinstance(args[0], c_ast.Constant):
            self.lines.append("        invokestatic Method lib440 java2c (Ljava/lang/String;)[C")

        if fname in self.lib440_funcs:
            sig, returns_value = self.lib440_funcs[fname]
            self.lines.append(f"        invokestatic Method lib440 {fname} {sig}")
            if returns_value and not self.expecting_value and not self.inside_expr:
                self.lines.append("        pop")

        else:
            sig = self.func_sigs.get(fname, "()I")
            self.lines.append(f"        invokestatic Method {self.class_name} {fname} {sig}")

    def _is_array_identifier(self, name):
        for ext in self.ast.ext:
            if isinstance(ext, c_ast.FuncDef):
                for decl in ext.decl.type.args.params if ext.decl.type.args else []:
                    if decl.name == name and isinstance(decl.type, c_ast.ArrayDecl):
                        return True
        return False


    def load_constant(self, node):
        if node.type == 'int':
            val = int(node.value)
            if 0 <= val <= 5:
                return f"        iconst_{val}"
            elif -128 <= val <= 127:
                return f"        bipush {val}"
            else:
                return f"        ldc {val}"
        elif node.type == 'char':
            val = ord(node.value.strip("'"))
            if 0 <= val <= 5:
                return f"        iconst_{val}"
            elif -128 <= val <= 127:
                return f"        bipush {val}"
            else:
                return f"        ldc {val}"
        elif node.type == 'float' or node.type == 'double':
            val = float(node.value)
            if val == 0.0:
                return "        fconst_0"
            elif val == 1.0:
                return "        fconst_1"
            elif val == 2.0:
                return "        fconst_2"
            else:
                return f"        ldc {val}f"
        elif node.type == 'string':
            s = node.value.strip('"')
            s = s.encode().decode('unicode_escape')
            s = json.dumps(s)[1:-1]
            return f'        ldc "{s}"\ninvokestatic Method lib440 java2c (Ljava/lang/String;)[C'
        else:
            raise NotImplementedError(f"Unsupported const type: {node.type}")


    def get_jtype(self, ctype):
        return {'int': 'I', 'float': 'F', 'void': 'V', 'char': 'C'}.get(ctype, 'I')


    def emit_to_file(self, path):
        with open(path, "w", newline="\n") as f:
            for line in self.lines:
                f.write(line + "\n")

    def _get_current_function_return_type(self):
        for line in reversed(self.lines):
            if line.startswith(".method public static"):
                sig = line.split(':')[-1].strip()
                ret_code = sig[sig.find(')')+1:]
                return {'I': 'int', 'F': 'float', 'V': 'void'}.get(ret_code, 'int')
        return 'int'
    
    def get_expr_type(self, node):
        if isinstance(node, c_ast.Constant):
            if node.type in ('float', 'double'):
                return 'float'
            return node.type  # 'int', 'char', etc.
        elif isinstance(node, c_ast.ID):
            name = node.name
            if name in self.local_env:
                return self.local_env[name][1]
            for var in self.global_vars:
                n = var[0]  # name
                t = var[1]  # ctype
                if n == name:
                    return t


        elif isinstance(node, c_ast.Cast):
            return node.to_type.type.type.names[0]

        elif isinstance(node, c_ast.ArrayRef):
            return self.get_expr_type(node.name)

        elif isinstance(node, c_ast.FuncCall):
            fname = node.name.name
            sig = self.func_sigs.get(fname, "()I")
            return_code = sig[sig.find(")") + 1:]
            return {
                'I': 'int',
                'F': 'float',
                'C': 'char',
                'V': 'void'
            }.get(return_code, 'int')  # default to int

        elif isinstance(node, c_ast.BinaryOp):
            left = self.get_expr_type(node.left)
            right = self.get_expr_type(node.right)
            if 'float' in (left, right):
                return 'float'
            elif 'char' in (left, right):
                return 'char'
            return 'int'
        
        elif isinstance(node, c_ast.UnaryOp):
            return self.get_expr_type(node.expr)

        elif isinstance(node, c_ast.TernaryOp):
            t1 = self.get_expr_type(node.iftrue)
            t2 = self.get_expr_type(node.iffalse)
            if 'float' in (t1, t2):
                return 'float'
            elif 'char' in (t1, t2):
                return 'char'
            return 'int'
        
        return 'int'  # fallback


    
    def visit_Cast(self, node):
        self.visit(node.expr)
        cast_type = node.to_type.type.type.names[0]
        expr_type = self.get_expr_type(node.expr)

        if cast_type == 'int' and expr_type == 'float':
            self.lines.append("        f2i")
        elif cast_type == 'float' and expr_type in ('int', 'char'):
            self.lines.append("        i2f")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("filename")
    args = parser.parse_args()

    base = os.path.splitext(os.path.basename(args.filename))[0]
    emitter = KrakatauEmitter(class_name=base)

    try:
        emitter.ast = parse_file(args.filename, use_cpp=True, cpp_path='gcc', cpp_args=['-E', '-P', '-w'])
    except Exception as e:
        print(f"Parsing failed for file {args.filename}")
        print(e)
        sys.exit(1)
    for ext in emitter.ast.ext:
        if isinstance(ext, c_ast.Decl):
            emitter.visit(ext)
    emitter.emit()
    emitter.emit_to_file(f"{base}.j")
