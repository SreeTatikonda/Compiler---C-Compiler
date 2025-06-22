#!/usr/bin/env python3
import sys
from mode0 import display_version
from mode1 import run_lexical_analysis
from mode2 import parse_file
from mode3 import typecheck_file

def display_usage():
    print("Usage: mycc -mode [infile]", file=sys.stderr)
    print("Valid modes:", file=sys.stderr)
    print("\t-0: Version information only", file=sys.stderr)
    print("\t-1: Lexical Analysis", file=sys.stderr)
    print("\t-2: Syntax Analysis", file=sys.stderr)
    print("\t-3: Type Checking", file=sys.stderr)
    print("\t-4: Code Generation (Phase 5)", file=sys.stderr)

def main():
    print("mycc.py driver script started.")
    
    if len(sys.argv) < 2:
        display_usage()
        sys.exit(1)

    mode = sys.argv[1]

    # Mode -0 doesn't require an input file
    if mode == "-0":
        display_version()
    else:
        if len(sys.argv) != 3:
            display_usage()
            sys.exit(1)
        
        infile = sys.argv[2]

        if mode == "-1":
            run_lexical_analysis(infile)
        elif mode == "-2":
            parse_file(infile)
        elif mode == "-3":
            typecheck_file(infile)
        elif mode == "-5":
            command ="python3 krakatau.py " + infile
            import subprocess
            subprocess.run(command, shell=True)
        else:
            print(f"Error: Unknown mode {mode}", file=sys.stderr)
            display_usage()
            sys.exit(1)

if __name__ == "__main__":
    main()
