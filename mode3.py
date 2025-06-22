import sys
from mode2 import (
    Program, Declaration, FunctionDef, FunctionPrototype, ReturnStatement, StructDef, Expression, 
    ConditionalExpression, CastExpression, UnaryOperation,
    parse_file, BinaryOperation, ArrayIndex, FunctionCall, PostfixOperation
)

numeric_types = {'char', 'int', 'float'}
integer_types = {'char', 'int'}

class TypeChecker:
    def __init__(self, filename):
        self.filename = filename
        self.function_table = {
            'getchar':   ('int',   tuple(),           0),
            'putchar':   ('int',   ('int',),          0),
            'getint':    ('int',   tuple(),           0),
            'putint':    ('void',  ('int',),          0),
            'getfloat':  ('float', tuple(),           0),
            'putfloat':  ('void',  ('float',),        0),
            'putstring': ('void',  ('const char[]',), 0),
        }
        self.global_scope = {}    
        self.current_scope = None
        #self.function_table = {}  
        self.errors = []         
        self.expressions = []    
        self.current_function = None
        self.struct_scopes = [{}]

    def error(self, lineno, message):
        sys.stderr.write(f"Type checking error in file {self.filename} line {lineno}\n")
        sys.stderr.write(f"\t{message}\n")
        sys.exit(1)

    def widen_type(self, t1, t2):
        order = ['char', 'int', 'float']
        if t1 not in order or t2 not in order:
            return None
        return order[max(order.index(t1), order.index(t2))]

    def check(self, node):
        if isinstance(node, FunctionPrototype):
            func_name = node.name
            signature = (node.return_type, tuple(p[1] for p in node.parameters), node.lineno)
            if func_name in self.function_table:
                stored = self.function_table[func_name]
                if stored != signature:
                    actual_params = ", ".join(p[1] for p in node.parameters)  # Format actual parameters
                    self.error(node.lineno, f"Prototype {node.return_type} {func_name}({actual_params}) differs from\n\tprevious declaration at file {self.filename} line {stored[2]}")
            else:
                self.function_table[func_name] = signature
            return 
        if isinstance(node, Program):
            # Process global StructDef nodes.
            for ext_decl in node.external_declarations:
                if isinstance(ext_decl, StructDef):
                    self.check_struct_def(ext_decl)
            for ext_decl in node.external_declarations:
                self.check(ext_decl)
        elif isinstance(node, Declaration):
            # If a Declaration's type_spec is a StructDef and has no declarators
            if (not node.declarators) and isinstance(node.type_spec, StructDef):
                self.check_struct_def(node.type_spec)
            else:
                self.check_declaration(node)
        elif isinstance(node, FunctionDef):
            self.check_function_def(node)
        elif isinstance(node, StructDef):
            self.check_struct_def(node)
        elif isinstance(node, list):
            for item in node:
                self.check(item)
        else:
            et = self.evaluate_expression(node)
            # Do not record return statement evaluations
            from mode2 import ReturnStatement
            if not isinstance(node, ReturnStatement) and et is not None:
                lineno = getattr(node, 'lineno', 0)
                self.expressions.append((lineno, et))

    def check_declaration(self, node):
        type_spec = node.type_spec
        # If type_spec is a StructDef, flatten it.
        if isinstance(type_spec, StructDef):
            type_spec = "struct " + type_spec.name
        # Check if the type_spec indicates a struct type.
        if type_spec.startswith("struct "):
            parts = type_spec.split()
            if len(parts) >= 2:
                tag = parts[1]
                # Look for tag in all scopes (from inner to outer)
                found = False
                for scope in reversed(self.struct_scopes):
                    if tag in scope:
                        found = True
                        break
                if not found:
                    decl_lineno = node.declarators[0][1] if node.declarators else 0
                    self.error(decl_lineno, f"Unknown struct '{tag}'")
        for decl in node.declarators:
            var_name = decl[0]
            lineno = decl[1]
            declared_type = type_spec
            if len(decl) >= 3 and decl[2] == 'array':
                declared_type = type_spec + '[]'
            if type_spec == 'void':
                self.error(lineno, f"Variable '{var_name}' cannot be of type void")
                continue
            if self.current_scope is not None:
                if var_name in self.current_scope:
                    self.error(lineno, f"Redeclaration of local variable '{var_name}'")
                self.current_scope[var_name] = (declared_type, False)
            else:
                if var_name in self.global_scope:
                    self.error(lineno, f"Redeclaration of global variable '{var_name}'")
                self.global_scope[var_name] = (declared_type, False)

    def check_function_def(self, node):
        func_name = node.name
        return_type = node.return_type
        current_sig = (return_type, tuple(p[1] for p in node.parameters), node.lineno)
        if func_name in self.function_table:
            stored_sig = self.function_table[func_name]
            if (stored_sig[0] != return_type or stored_sig[1] != tuple(p[1] for p in node.parameters)):
                expected_params = ", ".join(stored_sig[1])  # Format expected parameters
                actual_params = ", ".join(p[1] for p in node.parameters)  # Format actual parameters
                self.error(node.lineno, f"Prototype {return_type} {func_name}({actual_params}) differs from\n\tprevious declaration at file {self.filename} line {stored_sig[2]}")
        else:
            self.function_table[func_name] = current_sig

        old_scope = self.current_scope
        self.current_scope = {}
        self.current_function = return_type
        param_types = []
        for param in node.parameters:
            param_name, param_type, param_lineno = param
            if param_name in self.current_scope:
                self.error(param_lineno, f"Duplicate parameter '{param_name}' in function '{func_name}'")
                continue
            self.current_scope[param_name] = (param_type, False)
            param_types.append(param_type)
        # Push a new struct scope for the function body, so any local struct declarations override the global.
        old_struct_scopes = self.struct_scopes
        # Start with a copy of the current top scope.
        self.struct_scopes = [dict(self.struct_scopes[-1])]
        
        self.check(node.body)
        if return_type != "void":
            if not self.contains_return(node.body):
                self.error(node.lineno, f"Return type mismatch: was void, expected {return_type}")
        self.struct_scopes = old_struct_scopes
        self.current_scope = old_scope
        self.current_function = None

    def check_struct_def(self, node):
        if not isinstance(node, StructDef):
            return
        member_dict = {}
        for decl in node.members:
            base_type = decl.type_spec
            if isinstance(base_type, StructDef):
                base_type = "struct " + base_type.name
            for d in decl.declarators:
                m_name = d[0]
                m_type = base_type
                if len(d) >= 3 and d[2] == 'array':
                    m_type = base_type + '[]'
                member_dict[m_name] = m_type
        # Store the member dictionary in the current (top) struct scope.
        self.struct_scopes[-1][node.name] = member_dict

    def evaluate_expression(self, node):
        if node is None:
            return None  
        if isinstance(node, UnaryOperation):
            operand_type = self.evaluate_expression(node.operand)
            op = node.operator
            if operand_type == 'void':
                self.error(node.lineno, f"Invalid operation: {op} {operand_type}")
                return None
            if op in ('-', '~'):
                if operand_type in numeric_types:
                    return operand_type
                else:
                    self.error(node.lineno, f"Cannot apply '{op}' to non-numeric type {operand_type}")
                    return None
            if op == '!':
                if operand_type in numeric_types:
                    return 'int'
                else:
                    self.error(node.lineno, f"Cannot apply '{op}' to non-numeric type {operand_type}")
                    return None
            if op in ('++', '--'):
                if operand_type in numeric_types:
                    return operand_type
                else:
                    self.error(node.lineno, f"Cannot apply '{op}' to non-numeric type {operand_type}")
                    return None
            return operand_type

        if isinstance(node, ConditionalExpression):
            cond_type = self.evaluate_expression(node.condition)
            true_type = self.evaluate_expression(node.true_expr)
            false_type = self.evaluate_expression(node.false_expr)
            if true_type != false_type:
                self.error(node.lineno, f"Type mismatch in conditional expression: {true_type} vs {false_type}")
                return None
            return true_type

        if isinstance(node, ReturnStatement):
            if node.expr is None:
                actual = "void"
            else:
                actual = self.evaluate_expression(node.expr)
            expected = self.current_function  
            if actual.startswith("const "):
                actual = actual[len("const "):]
            if expected.startswith("const "):
                expected = expected[len("const "):]
            if expected != actual:
                self.error(node.lineno, f"Return type mismatch: was {actual}, expected {expected}")
            return expected

        if isinstance(node, CastExpression):
            self.evaluate_expression(node.expr)
            return node.target_type

        if isinstance(node, BinaryOperation):
            left_type = self.evaluate_expression(node.left)
            right_type = self.evaluate_expression(node.right)
            compound_ops = {'+=', '-=', '*=', '/=', '%='}
            op = node.operator
            if op in {'+', '-', '*', '/', '%', '&', '|', '==', '!=', '>', '>=', '<', '<=', '&&', '||'} | compound_ops:
                left_stripped = left_type[len("const "):] if left_type.startswith("const ") else left_type
                right_stripped = right_type[len("const "):] if right_type.startswith("const ") else right_type
                if left_stripped == 'void' or right_stripped == 'void':
                    self.error(node.lineno, f"Invalid operation: {left_type} {op} {right_type}")
                    return None
                if left_stripped in numeric_types and right_stripped in numeric_types:
                    result = self.widen_type(left_stripped, right_stripped)
                    return result
                else:
                    self.error(node.lineno, f"Incompatible types for '{op}': {left_type} {op} {right_type}")
                    return None
            elif op in compound_ops:
                if left_type != right_type:
                    self.error(node.lineno, f"Type mismatch in compound assignment: {left_type} {op} {right_type}")
                    return None
                return left_type
            if op == '=':
                if left_type.endswith("[]"):
                    self.error(node.lineno, f"Cannot assign to {left_type} from {right_type}")
                    return None
                right_stripped = right_type[len("const "):] if right_type.startswith("const ") else right_type
                if left_type != right_stripped:
                    self.error(node.lineno, f"Type mismatch in assignment: {left_type} = {right_type}")
                    return None
                return left_type
            else:
                self.error(node.lineno, f"Unknown operator '{op}'")
                return None

        if isinstance(node, ArrayIndex):
            base_type = self.evaluate_expression(node.array_expr)
            index_type = self.evaluate_expression(node.index_expr)
            if base_type is None or not base_type.endswith('[]'):
                self.error(node.lineno, f"Cannot index non-array type '{base_type}'")
                return None
            if index_type != 'int':
                self.error(node.lineno, f"Array index must be int, got {index_type}")
                return None
            return base_type[:-2]

        if isinstance(node, FunctionCall):
            if isinstance(node.func_expr, Expression) and isinstance(node.func_expr.value, str):
                func_name = node.func_expr.value
                if func_name in self.function_table:
                    return_type, param_types, _ = self.function_table[func_name]
                    
                    # Check number of arguments
                    if len(node.arguments) != len(param_types):
                        self.error(node.lineno, f"In call to {return_type} {func_name}()\n\tToo many parameters given")
                        return None
                    
                    # Check each argument's type against the expected type.
                    for i, (arg, expected) in enumerate(zip(node.arguments, param_types), start=1):
                        arg_type = self.evaluate_expression(arg)
                        if arg_type != expected:
                            # Allow implicit conversion if both types are numeric.
                            if arg_type in numeric_types and expected in numeric_types:
                                if self.widen_type(arg_type, expected) == expected:
                                    continue  # Acceptable conversion.
                                else:
                                    self.error(node.lineno, f"In call to {return_type} {func_name}({expected})\n\tParameter #{i} should be {expected}, was {arg_type}")
                                    return None
                            # Allow implicit const-to-non-const array conversion, e.g., const char[] -> char[]
                            if arg_type.startswith("const ") and arg_type[6:] == expected:
                                continue
                            if arg_type.startswith("const ") and expected.endswith("[]"):
                                base = expected[:-2]
                                if arg_type == f"const {base}[]":
                                    continue
                            self.error(node.lineno, f"In call to {return_type} {func_name}({expected})\n\tParameter #{i} should be {expected}, was {arg_type}")
                            return None
                    
                            #else:
                            #   self.error(node.lineno, f"In call to {return_type} {func_name}({expected})\n\tParameter #{i} should be {expected}, was {arg_type}")
                            #   return None
                    return return_type
                else:
                    self.error(node.lineno, f"Call to undeclared function '{func_name}'")
                    return None

        if isinstance(node, PostfixOperation):
            et = self.evaluate_expression(node.operand)
            if et is None:
                self.error(node.lineno, f"Postfix operator '{node.operator}' applied to expression of unknown type")
                return None
            if et.startswith("const"):
                self.error(node.lineno, f"Invalid operation: {et} {node.operator}")
                return None
            if et not in numeric_types:
                self.error(node.lineno, f"Invalid operation: {et} {node.operator}")
                return None
            return et

        if isinstance(node, Expression) and isinstance(node.value, tuple):
            tup = node.value
            if len(tup)==3 and tup[1]=='.':
                base = tup[0]
                member_name = tup[2]
                base_type = self.evaluate_expression(base)
                if base_type is None:
                    self.error(node.lineno, "Base type is None in member selection")
                    return None
                temp = base_type
                if temp.startswith("const "):
                    temp = temp[len("const "):]
                if not temp.startswith("struct"):
                    self.error(node.lineno, f"Member selection on non-struct type '{base_type}'")
                    return None
                parts = temp.split()
                if len(parts) < 2:
                    self.error(node.lineno, f"Invalid struct type '{base_type}'")
                    return None
                struct_tag = parts[1]
                # Look for the struct tag in our global table.
                member_dict = None
                for scope in reversed(self.struct_scopes):
                    if struct_tag in scope:
                        member_dict = scope[struct_tag]
                        break
                if member_dict is None:
                    self.error(node.lineno, f"Struct '{struct_tag}' is not defined")
                    return None
                if member_name not in member_dict:
                    self.error(node.lineno, f"Unknown member '{member_name}' in struct {struct_tag}")
                    return None
                member_type = member_dict[member_name]
                if base_type.startswith("const"):
                    member_type = "const " + member_type
                return member_type

        if isinstance(node, Expression):
            value = node.value
            if isinstance(value, str):
                if value in self.current_scope:
                    t = self.current_scope[value][0]
                    if isinstance(t, StructDef):
                        return "struct " + t.name
                    return t
                elif value in self.global_scope:
                    t = self.global_scope[value][0]
                    if isinstance(t, StructDef):
                        return "struct " + t.name
                    return t
                elif value.isdigit():
                    return 'int'
                elif self.is_float_literal(value):
                    return 'float'
                elif self.is_char_literal(value):
                    return 'char'
                elif self.is_string_literal(value):
                    return 'const char[]'
                else:
                    self.error(node.lineno, f"Undeclared variable or unknown literal '{value}'")
                    return None
            self.error(node.lineno, "Unknown expression node")
            return None

        if isinstance(node, str):
            if node.isdigit():
                return 'int'
            elif self.is_float_literal(node):
                return 'float'
            elif self.is_char_literal(node):
                return 'char'
            elif self.is_string_literal(node):
                return 'const char[]'
        self.error(getattr(node, 'lineno', 0), "Unknown expression type")
        return None

    def contains_return(self, node):
        if isinstance(node, list):
            for child in node:
                if self.contains_return(child):
                    return True
            return False
        if isinstance(node, ReturnStatement):
            return True
        if hasattr(node, '__dict__'):
            for key, value in vars(node).items():
                if self.contains_return(value):
                    return True
        return False

    def is_float_literal(self, s):
        try:
            float(s)
            return '.' in s
        except ValueError:
            return False

    def is_char_literal(self, s):
        return isinstance(s, str) and len(s) >= 3 and s[0]=="'" and s[-1]=="'"

    def is_string_literal(self, s):
        return isinstance(s, str) and len(s) >= 2 and s[0]=='"' and s[-1]=='"'

    def report_types(self):
        if self.errors:
            return
        output_file = self.filename.rsplit('.', 1)[0] + ".types"
        with open(output_file, 'w') as f:
            for lineno, etype in sorted(self.expressions):
                f.write(f"File {self.filename} Line {lineno}: expression has type {etype}\n")

def typecheck_file(filename):
    ast_root = parse_file(filename, write_file=False)
    type_checker = TypeChecker(filename)
    type_checker.check(ast_root)
    type_checker.report_types()

