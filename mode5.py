#!/usr/bin/env python3
"""
mode5.py – Phase 6 driver per Project Phase 6 spec:

• Invoked as mode "-5" in mycc.py
• Phase 1: Lexical analysis via mode1.run_lexical_analysis
• Phase 2: Syntax parse via mode2.parse_file (custom AST)
• Phase 3: Semantic type checking via mode3.typecheck_file
• Phase 4–5: Code generation by translating C AST to Krakatau Java assembly (via c_to_krakatau.KrakatauEmitter), supporting:
    – if/then, if/then/else
    – while, do/while, for loops (with type-checked init, cond, update)
    – break/continue inside loops
• Errors during codegen formatted:
    Code generation error in file <filename> line <lineno>
      <Description>

Generates <input_basename>.j when successful.
"""
import sys
import os
from mode1 import run_lexical_analysis
from mode2 import parse_file as parse_custom
from mode2 import Program, FunctionDef, Declaration
from mode3 import TypeChecker
from pycparser import parse_file as pyc_parse
from c_to_krakatau import KrakatauEmitter

def codegen_phase6(filename):
    # Phase 1–2: Lexical analysis and custom parse
    run_lexical_analysis(filename)
    ast_custom = parse_custom(filename)

    # Phase 3: Semantic type checking (exits on errors)
    typecheck_file(filename)

    # Prepare Krakatau emitter
    base = os.path.splitext(os.path.basename(filename))[0]
    emitter = KrakatauEmitter(class_name=base)

    # Build pycparser AST for emission
    try:
        emitter.ast = pyc_parse(filename, use_cpp=True)
    except Exception as parse_err:
        print(f"Parsing failed for file {filename}", file=sys.stderr)
        print(f"  {parse_err}", file=sys.stderr)
        sys.exit(1)

    # Emit Java assembly
    try:
        for ext in emitter.ast.ext:
            emitter.visit(ext)
        emitter.emit()
        out_file = f"{base}.j"
        emitter.emit_to_file(out_file)
        print(f"Generated {out_file}")
    except Exception as e:
        lineno = getattr(e, 'lineno', None)
        if lineno:
            print(f"Code generation error in file {filename} line {lineno}", file=sys.stderr)
        else:
            print(f"Code generation error in file {filename}", file=sys.stderr)
        print(f"  {e}", file=sys.stderr)
        sys.exit(1)

if __name__ == '__main__':
    if len(sys.argv) != 2:
        print("Usage: python mode5.py <input_file>", file=sys.stderr)
        sys.exit(1)
    codegen_phase6(sys.argv[1])
