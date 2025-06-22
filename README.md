
##Compilation Phases

Each phase of the compiler is modular and invoked via `mycc.py` using command-line arguments.

###  `mode0.py` — Lexical Analyzer
- Performs tokenization of C source code using regular expressions.
- Outputs a list of tokens.

###  `mode1.py` — Parser & AST Generator
- Converts token stream into an Abstract Syntax Tree (AST).
- Uses grammar rules to build tree structures for functions, statements, and expressions.

###  `mode2.py` — Semantic Analyzer
- Walks the AST to verify semantic correctness.
- Builds symbol tables and checks variable/function declarations, types, and scopes.

###  `mode3.py` — Intermediate Representation (IR)
- Translates AST into IR (typically three-address code or a custom format).
- Optimizes IR for better code generation later.

###  `mode4.py` — JVM Assembly Generator
- Converts IR into `.j` files (Java bytecode in Krakatau assembly syntax).
- Ensures proper JVM-compliant function, control flow, and memory constructs.

### `mode5.py` — Full Pipeline
- Executes mode0 to mode4 in sequence.
- Used for end-to-end compilation from `.c` input to `.j` output.


## 🚀 How to Use

### Compile a C File:
```bash
python3 mycc.py -mode=5 INPUTS/hello.c
