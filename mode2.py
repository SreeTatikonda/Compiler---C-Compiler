import sys
import ply.yacc as yacc
from mode1 import tokens, build_lexer

class ASTNode:
    pass

class Program(ASTNode):
    def __init__(self, external_declarations):
        self.external_declarations = external_declarations

class Declaration(ASTNode):
    def __init__(self, type_spec, declarators):
        self.type_spec = type_spec
        self.declarators = declarators

class FunctionDef(ASTNode):
    def __init__(self, return_type, name, parameters, body, lineno):
        self.return_type = return_type
        self.name = name
        self.parameters = parameters
        self.body = body
        self.lineno = lineno

class FunctionPrototype(ASTNode):
    def __init__(self, return_type, name, parameters, lineno):
        self.return_type = return_type
        self.name = name
        self.parameters = parameters
        self.lineno = lineno

class ReturnStatement(ASTNode):
    def __init__(self, expr, lineno):
        self.expr = expr
        self.lineno = lineno

class StructDef(ASTNode):
    def __init__(self, name, members, lineno):
        self.name = name
        self.members = members
        self.lineno = lineno

class Variable(ASTNode):
    def __init__(self, name, type_spec, lineno):
        self.name = name
        self.type_spec = type_spec
        self.lineno = lineno

class Expression:
    def __init__(self, value, lineno=None):
        self.value = value
        self.lineno = lineno

class BinaryOperation(ASTNode):
    def __init__(self, left, operator, right, lineno):
        self.left = left
        self.operator = operator
        self.right = right
        self.lineno = lineno
    
class ConditionalExpression(ASTNode):
    def __init__(self, condition, true_expr, false_expr, lineno):
        self.condition = condition
        self.true_expr = true_expr
        self.false_expr = false_expr
        self.lineno = lineno

class UnaryOperation(ASTNode):
    def __init__(self, operator, operand, lineno):
        self.operator = operator
        self.operand = operand
        self.lineno = lineno

class CastExpression(ASTNode):
    def __init__(self, target_type, expr, lineno):
        self.target_type = target_type
        self.expr = expr
        self.lineno = lineno

class ArrayIndex(ASTNode):
    def __init__(self, array_expr, index_expr, lineno):
        self.array_expr = array_expr
        self.index_expr = index_expr
        self.lineno = lineno

class FunctionCall(ASTNode):
    def __init__(self, func_expr, arguments, lineno):
        self.func_expr = func_expr
        self.arguments = arguments
        self.lineno = lineno

class PostfixOperation(ASTNode):
    def __init__(self, operand, operator, lineno):
        self.operand = operand
        self.operator = operator
        self.lineno = lineno

# List to store symbols
symbols = []
current_filename = ""

precedence = (
    ('left', 'COMMA'),
    ('right', 'ASSIGN', 'ADD_ASSIGN', 'SUB_ASSIGN', 'MUL_ASSIGN', 'DIV_ASSIGN'),
    ('right', 'QUESTION_MARK', 'COLON'),
    ('left', 'OR'),
    ('left', 'AND'),
    ('left', 'BITWISE_OR'),
    ('left', 'BITWISE_AND'),
    ('left', 'EQUAL', 'NEQUAL'),
    ('left', 'LESS', 'GREATER', 'LEQUAL', 'GEQUAL'),
    ('left', 'PLUS', 'MINUS'),
    ('left', 'STAR', 'SLASH', 'MODULO'),
    ('right', 'NOT', 'EXCLAMATION', 'TILDE', 'DECREMENT', 'INCREMENT'),
    ('left', 'LPAREN', 'RPAREN', 'LBRACKET', 'RBRACKET'),
    ('nonassoc', 'ELSE'),
)

def add_symbol(kind, ident, lineno, type_info=None):
    symbols.append((current_filename, lineno, kind, ident, type_info))

# Grammar Rules
def p_translation_unit(p):
    'translation_unit : external_declaration_list'
    p[0] = Program(p[1])

def p_external_declaration_list(p):
    '''external_declaration_list : external_declaration
                                  | external_declaration_list external_declaration'''
    if len(p)==2:
         p[0] = [p[1]]
    else:
         p[0] = p[1] + [p[2]]

# External Declaration
def p_external_declaration(p):
    '''external_declaration : declaration
                             | function_definition
                             | function_declaration
                             | struct_specifier SEMICOLON'''
    if len(p)==3:
         p[0] = p[1]
    else:
         p[0] = p[1]

def p_declaration(p):
    '''declaration : declaration_specifiers init_declarator_list SEMICOLON
                   | declaration_specifiers SEMICOLON'''
    if len(p)==3:
         p[0] = Declaration(p[1], [])
    else:
         for decl in p[2]:
              add_symbol("global variable", decl[0], decl[1], p[1])
         p[0] = Declaration(p[1], p[2])

def p_local_declaration(p):
    'local_declaration : declaration_specifiers init_declarator_list SEMICOLON'
    for decl in p[2]:
         add_symbol("local variable", decl[0], decl[1], p[1])
    p[0] = Declaration(p[1], p[2])

def p_declaration_specifiers(p):
    '''declaration_specifiers : type_specifier
                              | type_qualifier
                              | type_specifier declaration_specifiers
                              | type_qualifier declaration_specifiers'''
    if len(p)==2:
         p[0] = p[1]
    else:
         parts = []
         for i in range(1, len(p)):
              parts.extend(p[i].split())
         if "const" in parts:
              parts = ["const"] + [w for w in parts if w != "const"]
         p[0] = " ".join(parts)

def p_type_specifier(p):
    '''type_specifier : INT
                      | FLOAT
                      | CHAR
                      | DOUBLE
                      | LONG
                      | VOID
                      | struct_specifier'''
    p[0] = p[1]

def p_type_qualifier(p):
    'type_qualifier : CONST'
    p[0] = "const"

def p_init_declarator_list(p):
    '''init_declarator_list : init_declarator
                             | init_declarator_list COMMA init_declarator'''
    if len(p)==2:
         p[0] = [p[1]]
    else:
         p[0] = p[1] + [p[3]]

def p_init_declarator(p):
    '''init_declarator : declarator
                       | declarator ASSIGN initializer'''
    p[0] = p[1]

def p_declarator(p):
    '''declarator : IDENTIFIER declarator_tail_opt'''
    if p[2] is None:
         p[0] = (p[1], p.lineno(1))
    else:
         p[0] = (p[1], p.lineno(1)) + p[2]

def p_declarator_tail_opt(p):
    '''declarator_tail_opt : declarator_tail
                           | empty'''
    p[0] = p[1]

def p_declarator_tail(p):
    '''declarator_tail : LBRACKET constant_expression RBRACKET
                       | LBRACKET RBRACKET'''
    if len(p)==4:
         p[0] = ('array', p[2])
    else:
         p[0] = ('array', None)

def p_empty(p):
    'empty :'
    p[0] = None

def p_constant_expression(p):
    'constant_expression : INTEGER'
    p[0] = p[1]

def p_initializer(p):
    '''initializer : assignment_expression
                   | designated_initializer
                   | LBRACE initializer_list RBRACE
                   | LBRACE initializer_list COMMA RBRACE'''
    p[0] = p[1]

def p_designated_initializer(p):
    'designated_initializer : designator ASSIGN initializer'
    p[0] = ('designated', p[1], p[3])

def p_designator(p):
    'designator : DOT IDENTIFIER'
    p[0] = p[2]

def p_initializer_list(p):
    '''initializer_list : initializer
                         | initializer_list COMMA initializer'''
    if len(p)==2:
         p[0] = [p[1]]
    else:
         p[0] = p[1] + [p[3]]

def p_struct_specifier(p):
    '''struct_specifier : STRUCT IDENTIFIER
                        | STRUCT IDENTIFIER LBRACE struct_declaration_list_opt RBRACE'''
    if len(p)==3:
         p[0] = "struct " + p[2]
    else:
         p[0] = StructDef(p[2], p[4], p.lineno(1))

def p_struct_declaration_list_opt(p):
    '''struct_declaration_list_opt : 
                                   | struct_declaration_list'''
    p[0] = [] if len(p)==1 else p[1]

def p_struct_declaration_list(p):
    '''struct_declaration_list : struct_declaration
                               | struct_declaration_list struct_declaration'''
    if len(p)==2:
         p[0] = [p[1]]
    else:
         p[0] = p[1] + [p[2]]

def p_struct_declaration(p):
    'struct_declaration : declaration_specifiers declarator_list SEMICOLON'
    for decl in p[2]:
         add_symbol("member", decl[0], decl[1], p[1])
    p[0] = Declaration(p[1], p[2])

def p_local_struct_declaration(p):
    'local_struct_declaration : STRUCT IDENTIFIER LBRACE struct_declaration_list_opt RBRACE SEMICOLON'
    add_symbol("local struct", p[2], p.lineno(1))
    p[0] = StructDef(p[2], p[4], p.lineno(1))

def p_declarator_list(p):
    '''declarator_list : declarator
                       | declarator_list COMMA declarator'''
    if len(p)==2:
         p[0] = [p[1]]
    else:
         p[0] = p[1] + [p[3]]

def p_function_definition(p):
    'function_definition : declaration_specifiers function_declarator compound_statement'
    func_name, parameters, func_lineno = p[2]
    add_symbol("function", func_name, func_lineno, p[1])
    for param in parameters:
         add_symbol("parameter", param[0], func_lineno)
    p[0] = FunctionDef(p[1], func_name, parameters, p[3], func_lineno)

def p_function_declarator(p):
    'function_declarator : IDENTIFIER LPAREN parameter_list_opt RPAREN'
    p[0] = (p[1], p[3], p.lineno(1))

def p_function_declaration(p):
    'function_declaration : declaration_specifiers function_declarator SEMICOLON'
    func_name, parameters, func_lineno = p[2]
    add_symbol("function", func_name, func_lineno, p[1])
    for param in parameters:
         add_symbol("parameter", param[0], func_lineno)
    p[0] = FunctionPrototype(p[1], func_name, parameters, func_lineno)

def p_parameter_list_opt(p):
    '''parameter_list_opt : 
                           | parameter_list'''
    p[0] = [] if len(p)==1 else p[1]

def p_parameter_list(p):
    '''parameter_list : parameter_declaration
                      | parameter_list COMMA parameter_declaration'''
    if len(p)==2:
         p[0] = [p[1]]
    else:
         p[0] = p[1] + [p[3]]

def p_parameter_declaration(p):
    'parameter_declaration : declaration_specifiers declarator'
    param_name = p[2][0]
    param_type = p[1]
    if len(p[2])==4 and p[2][2]=='array':
         param_type = param_type + '[]'
    p[0] = (param_name, param_type, p.lineno(2))

def p_compound_statement(p):
    'compound_statement : LBRACE block_item_list_opt RBRACE'
    p[0] = [item for item in p[2] if item is not None]

def p_block_item_list_opt(p):
    '''block_item_list_opt : 
                           | block_item_list'''
    p[0] = [] if len(p)==1 else p[1]

def p_block_item_list(p):
    '''block_item_list : block_item
                       | block_item_list block_item'''
    if len(p)==2:
         p[0] = [p[1]]
    else:
         p[0] = p[1] + [p[2]]

def p_block_item(p):
    '''block_item : statement
                  | struct_specifier
                  | local_declaration
                  | local_struct_declaration'''
    p[0] = p[1]

def p_statement(p):
    '''statement : expression_statement
                 | compound_statement
                 | selection_statement
                 | iteration_statement
                 | jump_statement'''
    p[0] = p[1]

def p_expression_statement(p):
    'expression_statement : expression_opt SEMICOLON'
    p[0] = p[1]

def p_expression_opt(p):
    '''expression_opt : 
                      | expression'''
    p[0] = None if len(p)==1 else p[1]

def p_expression(p):
    '''expression : assignment_expression
                  | expression COMMA assignment_expression'''
    p[0] = p[1] if len(p)==2 else p[3]

def p_assignment_expression(p):
    '''assignment_expression : conditional_expression
                              | unary_expression ASSIGN assignment_expression
                              | unary_expression ADD_ASSIGN assignment_expression
                              | unary_expression SUB_ASSIGN assignment_expression
                              | unary_expression MUL_ASSIGN assignment_expression
                              | unary_expression DIV_ASSIGN assignment_expression'''
    if len(p)==2:
         p[0] = p[1]
    else:
         p[0] = BinaryOperation(p[1], p[2], p[3], p.lineno(2))

def p_bitwise_and_expression(p):
    '''bitwise_and_expression : additive_expression
                              | bitwise_and_expression BITWISE_AND additive_expression'''
    if len(p)==2:
         p[0] = p[1]
    else:
         p[0] = BinaryOperation(p[1], '&', p[3], p.lineno(2))

def p_bitwise_or_expression(p):
    '''bitwise_or_expression : bitwise_and_expression
                             | bitwise_or_expression BITWISE_OR bitwise_and_expression'''
    if len(p)==2:
         p[0] = p[1]
    else:
         p[0] = BinaryOperation(p[1], '|', p[3], p.lineno(2))

def p_conditional_expression(p):
    '''conditional_expression : bitwise_or_expression
                               | logical_or_expression
                               | logical_or_expression QUESTION_MARK expression COLON conditional_expression
                               | bitwise_or_expression QUESTION_MARK expression COLON conditional_expression'''
    if len(p) == 2:
         p[0] = p[1]
    else:
         p[0] = ConditionalExpression(p[1], p[3], p[5], p.lineno(2))

def p_logical_or_expression(p):
    '''logical_or_expression : logical_and_expression
                             | logical_or_expression OR logical_and_expression'''
    if len(p)==2:
        p[0] = p[1]
    else:
        p[0] = BinaryOperation(p[1], '||', p[3], p.lineno(2))

def p_logical_and_expression(p):
    '''logical_and_expression : equality_expression
                              | logical_and_expression AND equality_expression'''
    if len(p)==2:
        p[0] = p[1]
    else:
        p[0] = BinaryOperation(p[1], '&&', p[3], p.lineno(2))

def p_equality_expression(p):
    '''equality_expression : relational_expression
                           | equality_expression EQUAL relational_expression
                           | equality_expression NEQUAL relational_expression'''
    if len(p)==2:
        p[0] = p[1]
    else:
        p[0] = BinaryOperation(p[1], p[2], p[3], p.lineno(2))

def p_relational_expression(p):
    '''relational_expression : additive_expression
                             | relational_expression LESS additive_expression
                             | relational_expression GREATER additive_expression
                             | relational_expression LEQUAL additive_expression
                             | relational_expression GEQUAL additive_expression'''
    if len(p)==2:
        p[0] = p[1]
    else:
        p[0] = BinaryOperation(p[1], p[2], p[3], p.lineno(2))

def p_additive_expression(p):
    '''additive_expression : multiplicative_expression
                            | additive_expression PLUS multiplicative_expression
                            | additive_expression MINUS multiplicative_expression'''
    if len(p) == 2:
         p[0] = p[1]
    else:
         p[0] = BinaryOperation(p[1], p[2], p[3], p.lineno(2))


def p_multiplicative_expression(p):
    '''multiplicative_expression : unary_expression
                                  | multiplicative_expression STAR unary_expression
                                  | multiplicative_expression SLASH unary_expression
                                  | multiplicative_expression MODULO unary_expression'''
    if len(p) == 2:
         p[0] = p[1]
    else:
         p[0] = BinaryOperation(p[1], p[2], p[3], p.lineno(2))

def p_unary_expression(p):
    '''unary_expression : postfix_expression
                         | primary_expression
                         | MINUS unary_expression
                         | NOT unary_expression
                         | TILDE unary_expression
                         | INCREMENT unary_expression
                         | DECREMENT unary_expression
                         | cast_expression'''
    if len(p) == 2:
         p[0] = p[1]
    else:
         p[0] = UnaryOperation(p[1], p[2], p.lineno(1))

def p_postfix_expression(p):
    '''postfix_expression : primary_expression
                           | postfix_expression LPAREN argument_list_opt RPAREN
                           | postfix_expression DOT IDENTIFIER
                           | postfix_expression LBRACKET expression RBRACKET
                           | postfix_expression INCREMENT
                           | postfix_expression DECREMENT'''
    if len(p)==2:
         p[0] = p[1]
    elif p[2] == '(':
        p[0] = FunctionCall(p[1], p[3], p[1].lineno)
    elif p[2]=='.':
         p[0] = Expression((p[1], '.', p[3]), p.lineno(2))
    elif p[2]=='[':
         p[0] = ArrayIndex(p[1], p[3], p.lineno(2))
    elif p[2]=='++':
         p[0] = PostfixOperation(p[1], '++', p.lineno(2))
    elif p[2]=='--':
         p[0] = PostfixOperation(p[1], '--', p.lineno(2))

def p_cast_expression(p):
    'cast_expression : LPAREN type_name RPAREN unary_expression'
    p[0] = CastExpression(p[2], p[4], p.lineno(1))

def p_type_name(p):
    '''type_name : type_specifier
                 | type_specifier type_qualifier'''
    p[0] = p[1]

def p_primary_expression(p):
    '''primary_expression : IDENTIFIER
                           | INTEGER
                           | REAL
                           | STRING
                           | CHARACTER
                           | LPAREN expression RPAREN'''
    if len(p)==2:
         p[0] = Expression(p[1], p.lineno(1))
    else:
         p[0] = p[2]

def p_selection_statement(p):
    '''selection_statement : IF LPAREN expression RPAREN statement
                            | IF LPAREN expression RPAREN statement ELSE statement'''
    p[0] = None

def p_iteration_statement(p):
    '''iteration_statement : WHILE LPAREN expression RPAREN statement
                            | FOR LPAREN expression_opt SEMICOLON expression_opt SEMICOLON expression_opt RPAREN statement
                            | DO statement WHILE LPAREN expression RPAREN SEMICOLON'''
    p[0] = None

def p_jump_statement(p):
    '''jump_statement : RETURN expression_opt SEMICOLON
                      | RETURN SEMICOLON
                      | BREAK SEMICOLON
                      | CONTINUE SEMICOLON'''
    if p[1] == "return":
        if len(p) == 4:
            p[0] = ReturnStatement(p[2], p.lineno(1))
        else:
            p[0] = ReturnStatement(None, p.lineno(1))
    else:
        p[0] = None

def p_argument_list_opt(p):
    '''argument_list_opt : 
                         | argument_list'''
    p[0] = [] if len(p)==1 else p[1]

def p_argument_list(p):
    '''argument_list : assignment_expression
                     | argument_list COMMA assignment_expression'''
    if len(p)==2:
         p[0] = [p[1]]
    else:
         p[0] = p[1] + [p[3]]

def p_error(p):
    if p:
         sys.stderr.write(f"Parser error in file {current_filename} line {p.lineno} at text {p.value}\n")
         sys.stderr.write("    Syntax error\n")
    else:
         sys.stderr.write("Parser error: Unexpected end of file\n")
    sys.exit(1)

# Function to run the parser and generate the AST and output file
def parse_file(filename, write_file=True):
    parser = yacc.yacc(debug=False, write_tables=False, errorlog=yacc.NullLogger())
    global symbols, current_filename
    symbols = []
    current_filename = filename
    try:
         with open(filename, 'r') as f:
              data = f.read()
    except FileNotFoundError:
         sys.stderr.write(f"Error: File {filename} not found\n")
         sys.exit(1)
    lexer_instance = build_lexer()
    lexer_instance.filename = filename
    result = parser.parse(data, lexer=lexer_instance)
    
    if write_file:
        output_file = filename + ".parser"
        with open(output_file, 'w') as f:
             symbols.sort(key=lambda x: x[1])
             for sym in symbols:
                  f.write(f"File {sym[0]} Line {sym[1]}: {sym[2]} {sym[3]}\n")
        print_ast(result)
        
    return result

# Helper function to print the AST
def print_ast(node, indent=0):
    prefix = '  ' * indent
    if isinstance(node, list):
         for elem in node:
              print_ast(elem, indent)
    elif hasattr(node, '__dict__'):
         print(f"{prefix}{node.__class__.__name__}")
         for key, value in vars(node).items():
              print(f"{prefix}  {key}:")
              print_ast(value, indent+2)
    else:
         print(f"{prefix}{node}")
