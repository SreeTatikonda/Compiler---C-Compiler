import re
import os
import sys

# Global variables (to mimic the C globals)
filename = None
line_num = 1
output_file = None
write_tokens = 0

# Token patterns – note that comments are removed before tokenizing.
single_char_tokens = {
    '!': 33, '%': 37, '&': 38, '(': 40, ')': 41, '*': 42, '+': 43,
    ',': 44, '-': 45, '.': 46, '/': 47, ':': 58, ';': 59, '<': 60,
    '=': 61, '>': 62, '?': 63, '[': 91, ']': 93, '{': 123, '}': 125,
    '|': 124, '~': 126
}

# Patterns for keywords, literals, and multi-character operators.
token_patterns = [
    (r'\bconst\b', 401),
    (r'\bstruct\b', 402),
    (r'\bfor\b', 403),
    (r'\bwhile\b', 404),
    (r'\bdo\b', 405),
    (r'\bif\b', 406),
    (r'\belse\b', 407),
    (r'\bbreak\b', 408),
    (r'\bcontinue\b', 409),
    (r'\breturn\b', 410),
    (r'\bswitch\b', 411),
    (r'\bcase\b', 412),
    (r'\bdefault\b', 413),
    (r'\b(void|char|int|float)\b', 301),
    (r'\d+\.\d+([eE][+-]?\d+)?', 304),
    (r'0[xX][0-9a-fA-F]+', 307),
    (r'\d+', 303),
    (r"'(\\.|[^\\'])'", 302),
    (r'\"(\\.|[^\"])*\"', 305),
    (r'[a-zA-Z_]\w*', 306),
    (r'==|!=|>=|<=|\+\+|--|\|\||&&|\+=|-=|\*=|/=', 350)
]

def remove_comments(lines):
    """Remove both C-style (/* ... */) and C++-style (// ...) comments while preserving line numbers."""
    in_block_comment = False
    new_lines = []
    for line in lines:
        i = 0
        new_line = ""
        while i < len(line):
            if not in_block_comment and line[i:i+2] == "//":
                break  # skip rest of line
            elif not in_block_comment and line[i:i+2] == "/*":
                in_block_comment = True
                i += 2
                continue
            elif in_block_comment and line[i:i+2] == "*/":
                in_block_comment = False
                i += 2
                continue
            elif in_block_comment:
                i += 1
                continue
            else:
                new_line += line[i]
                i += 1
        new_lines.append(new_line)
    return new_lines

def tokenize_line(fname, ln, line):
    tokens = []
    index = 0
    while index < len(line):
        if line[index].isspace():
            index += 1
            continue
        match_found = False
        # First check for single-character tokens.
        if line[index] in single_char_tokens:
            lexeme = line[index]
            token_num = single_char_tokens[lexeme]
            tokens.append((fname, ln, token_num, lexeme))
            index += 1
            match_found = True
        if not match_found:
            for pattern, token_num in token_patterns:
                match = re.match(pattern, line[index:])
                if match:
                    lexeme = match.group(0)
                    tokens.append((fname, ln, token_num, lexeme))
                    index += len(lexeme)
                    match_found = True
                    break
        if not match_found:
            print(f"Lexer error in file {fname} line {ln} at text {line[index]}", file=sys.stderr)
            return None
    return tokens

def run_lexical_analysis(input_filename):
    global filename, line_num, output_file, write_tokens
    filename = input_filename
    try:
        with open(filename, 'r') as infile:
            lines = infile.readlines()
        processed_lines = remove_comments(lines)
        processed_lines = [re.sub(r'(?<=\d)\s+(?=\d)', '', line) for line in processed_lines]
        output_filename = filename.rsplit('.', 1)[0] + ".lexer"
        output_file = open(output_filename, 'w')
        write_tokens = 1
        for i, line in enumerate(processed_lines, start=1):
            line_num = i
            toks = tokenize_line(filename, i, line)
            if toks is None:
                output_file.close()
                os.remove(output_filename)
                sys.exit(1)
            for token in toks:
                output_file.write(f"File {filename} Line {i} Token {token[2]} Text {token[3]}\n")
        output_file.close()
        print("Lexical analysis completed")
    except FileNotFoundError:
        print(f"Error: Cannot open file {filename}", file=sys.stderr)
        sys.exit(1)
    except Exception as e:
        print(f"An error occurred: {e}", file=sys.stderr)
        sys.exit(1)

# --- Added for Parser (PLY lexer) ---
import ply.lex as lex

# Token names for PLY
tokens = [
    'IDENTIFIER','INTEGER','REAL','STRING','CHARACTER',
    'LPAREN','RPAREN','LBRACKET','RBRACKET','LBRACE','RBRACE',
    'SEMICOLON','COMMA','DOT',
    'ASSIGN','ADD_ASSIGN','SUB_ASSIGN','MUL_ASSIGN','DIV_ASSIGN',
    'PLUS','MINUS','STAR','SLASH','MODULO',
    'INCREMENT','DECREMENT',
    'EQUAL','NEQUAL','LESS','GREATER','LEQUAL','GEQUAL',
    'OR','AND',
    'NOT','TILDE',
    'IF','ELSE','WHILE','FOR','DO',
    'RETURN','BREAK','CONTINUE',
    'INT','FLOAT','CHAR','VOID','STRUCT'
]

reserved = {
    'const':   'CONST',
    'struct':  'STRUCT',
    'for':     'FOR',
    'while':   'WHILE',
    'do':      'DO',
    'if':      'IF',
    'else':    'ELSE',
    'break':   'BREAK',
    'continue':'CONTINUE',
    'return':  'RETURN',
    'switch':  'SWITCH',
    'case':    'CASE',
    'default': 'DEFAULT',
    'void':    'VOID',
    'char':    'CHAR',
    'int':     'INT',
    'float':   'FLOAT'
}

# <-- Add this so PLY will return '?' and ':' as literal tokens -->
literals = ['?', ':']

t_LPAREN     = r'\('
t_RPAREN     = r'\)'
t_LBRACKET   = r'\['
t_RBRACKET   = r'\]'
t_LBRACE     = r'\{'
t_RBRACE     = r'\}'
t_SEMICOLON  = r';'
t_COMMA      = r','
t_DOT        = r'\.'

t_ADD_ASSIGN = r'\+='
t_SUB_ASSIGN = r'-='
t_MUL_ASSIGN = r'\*='
t_DIV_ASSIGN = r'/='
t_INCREMENT  = r'\+\+'
t_DECREMENT  = r'--'
t_EQUAL      = r'=='
t_NEQUAL     = r'!='
t_LEQUAL     = r'<='
t_GEQUAL     = r'>='
t_LESS       = r'<'
t_GREATER    = r'>'
t_OR         = r'\|\|'
t_AND        = r'&&'
t_ASSIGN     = r'='
t_PLUS       = r'\+'
t_MINUS      = r'-'
t_STAR       = r'\*'
t_SLASH      = r'/'
t_MODULO     = r'%'
t_NOT        = r'!'
t_TILDE      = r'~'

def t_REAL(t):
    r'\d+\.\d+([eE][+-]?\d+)?'
    t.value = float(t.value)
    return t

def t_INTEGER(t):
    r'\d+'
    t.value = int(t.value)
    return t

def t_STRING(t):
    r'\"(\\.|[^\"])*\"'
    t.value = t.value[1:-1]
    return t

def t_CHARACTER(t):
    r'\'(\\.|[^\\\']?)\''
    t.value = t.value[1:-1]
    return t

def t_IDENTIFIER(t):
    r'[A-Za-z_]\w*'
    t.type = reserved.get(t.value, 'IDENTIFIER')
    return t

t_ignore = " \t"

def t_newline(t):
    r'\n+'
    t.lexer.lineno += len(t.value)

def t_error(t):
    print(f"Illegal character '{t.value[0]}'", file=sys.stderr)
    t.lexer.skip(1)

def build_lexer(**kwargs):
    return lex.lex(module=sys.modules[__name__], **kwargs)
