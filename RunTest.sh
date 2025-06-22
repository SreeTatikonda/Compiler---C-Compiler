#!/bin/bash

TIMEOUT="30"
SCRIPT=$0
ERRLINES=10
CONTEXT=2
MODE="-5"
EXT="j"
ERR="error"
FILES="files"
RMFILES=""
GCC="gcc"

KPATH="$HOME/Krakatau/target/release"
ASSEMBLE="$KPATH/krak2 asm"

bailout()
{
    echo
    echo "$1"
    echo "Terminating script"
    echo
    rm -f $RMFILES
    exit 1
}

ctrlc()
{
    bailout "Caught CTRL-C"
}

debug()
{
#    echo "$1" > /dev/stderr
    echo "$1" > /dev/null
}

usage()
{
    echo "Usage: $SCRIPT  [-G | executable [options]] infile infile ..."
    echo "Run compiler executable, in code generation mode, on specified"
    echo "input files.  Assemble the generated output, run it, and compare"
    echo "outputs with those of a gcc-generated executable."
    echo "Input files (sources and test inputs) should be in subdirectory INPUTS."
    echo "Oracle output should be in directory GCC."
    echo "Will not run on source files that produce errors (according to"
    echo "existence of error messages in OUTPUTS_B directory)."
    echo "Options: "
    echo "  -G  : generate test outputs with gcc"
    echo "  -t N: set timeout to N seconds (default is $TIMEOUT)"
    echo "        0 seconds uses no timeout"
    exit $1
}


green()
{
    printf "[1;32m%s[0;39m$2" "$1"
}

cyan()
{
    printf "[1;36m%s[0;39m$2" "$1"
}

yellow()
{
    printf "[1;33m%s[0;39m$2" "$1"
}

red()
{
    printf "[1;31m%s[0;39m$2" "$1"
}

# Arg1: file to clean
#
removeUnprintables()
{
    tr -d '\001\002\003\004\005\006\007\b\r' < $1 > $1.clean
    if diff -q $1 $1.clean > /dev/null; then
        cyan  "Already clean" "\n"
        rm $1.clean
    else
        red  "There were unprintable characters" "\n"
        mv -f $1.clean $1
    fi
}


# Arg1: input directory
# Arg2: base filename
# Arg3: timeout
timeoutCompile()
{
    local indir="$1"
    local inbase="$2"
    local timeout="$3"

    if [ "$indir" != "." ]; then
        cp $indir/$inbase.[ch] .
        RMFILES="$RMFILES $inbase.[ch]"
        if [ -f $indir/$inbase.files ]; then
            while read src; do
                cp $indir/$src .
                RMFILES="$RMFILES $src"
            done < $indir/$inbase.files
        fi
    fi

    RMFILES="$RMFILES $inbase.$ERR $inbase.$EXT $inbase.timeout"

    nice $EXE $MODE $inbase.c 1> /dev/null 2> $inbase.$ERR &
    local pid=$!

    if [ ! "$timeout" ]; then
        wait
    else
        disown -r
#       ^ avoid messages if we kill the process
        local killsignal=""
        SECONDS=0
        while true; do
            sleep 1
            if ps $pid > /dev/null; then
                # still running
                if [ $SECONDS -gt $TIMEOUT ]; then
                    touch $inbase.timeout
                        if [ $killsignal ]; then
                            kill $killsignal $pid
                        else
                            kill $pid
                            killsignal="-9"
                        fi
                fi
            else
                # completed
                break
            fi
        done
    fi
}

# Arg1: base 'executable' name
# Arg2: output file
# Arg3: error file
# Arg4: input file, or empty for none
#
timeoutRun()
{
    local classf="$1"
    local outf="$2"
    local errf="$3"
    local inf=""
    if [ "$4" ]; then
        inf="< $4"
    fi
    nice java $classf $inf > $outf 2> $errf &
    local pid=$!

        disown -r
#       ^ avoid messages if we kill the process
        local killsignal=""
        SECONDS=0
        while true; do
            sleep 1
            if ps $pid > /dev/null; then
                # still running
                if [ $SECONDS -gt $TIMEOUT ]; then
                    kill -9 $pid
                    sleep 1
                    echo "Timeout after $TIMEOUT seconds, killed." > $errf
                fi
            else
                # completed
                break
            fi
        done
}


# Arg1: Human description
# Arg2: file
# Arg3: max lines
showIndentedFile()
{
    local what="$1"
    local file="$2"
    local lines="$3"

    if [ ! -s "$file" ]; then
        return
    fi
    printf "%15s  $what (first $lines lines):\n" " "
    printf "%15s  ------------------------------------------------------------\n" " "
    awk "(NR<=$lines){print \"                 | \" \$0}" $file
    printf "%15s  ------------------------------------------------------------\n" " "
}

#############################################################

# Arg1: source directory
# Arg2: basename
#
# Build an executable using gcc, generate expected outputs
buildOracle()
{
    local indir="$1"
    local base="$2"
    local outfile=""

    mkdir -p GCC
    rm -f GCC/.$base GCC/$base.*

    #
    # Check if program has main(); skip if not
    #
    if ! grep "int main()" $indir/$base.c > /dev/null; then
        return
    fi
    if grep -w NORUN $indir/$base.c > /dev/null; then
        return
    fi

    #
    # Generate oracle program
    #

    echo "Generating oracle for $base.c"
    cat lib440.c $indir/$base.c > $base.c
    if $GCC $base.c -o GCC/_$base; then
        green "    gcc compiled no problems" "\n"
        rm -f $base.c
    else
        red   "    gcc had compile errors" "\n"
        rm -f GCC/_$base $base.c
        return 0
    fi

    #
    # Now run oracle program on input(s)
    #
    for infile in $indir/$base.input*; do
        local baseout=`basename $infile | sed 's/input/output/' `
        if [ -f $infile ]; then
            GCC/_$base < $infile > GCC/$baseout
        else
            baseout=`tr -d '*' <<< $baseout`
            GCC/_$base > GCC/$baseout
        fi
        echo "        $baseout"
    done
    rm -f GCC/_$base
    touch GCC/.$base
}


#############################################################

trap ctrlc INT

if [ $# -eq 0 ]; then
    usage 0
fi

EXE="$1"
shift

if [ "x-G" == "x$EXE" ]; then
    DASHG="y"
else
    DASHG=""
    echo Running tests using compiler:
    $EXE -0 | awk '{print "  | " $0}'
    echo " "
fi

TOnext=""
for arg; do

# check for options/switches

    if [ $TOnext ]; then
        TIMEOUT=$arg
        TOnext=""
        continue
    fi

    if [ "x-t" == "x-$arg" ]; then
        TOnext="y"
        continue
    fi

    argdir=`dirname $arg`
    argbase=`basename -s .c $arg`
    argout="$argbase.$EXT"
    argerr="$argbase.$ERR"

    if [ ! -f $argdir/$argbase.c ]; then
        continue
    fi

#
# Skip if there's an error file
#

    if [ -f OUTPUTS_B/$argerr ]; then
        continue
    fi

#
# If -G, run gcc and stuff here
#
    if [ "$DASHG" ]; then
        buildOracle $argdir $argbase
        continue
    fi

#
# Skip on missing oracle in GCC directory
#
    if [ ! -f GCC/.$argbase ]; then
        # echo "$argbase.c: missing oracle"
        continue
    fi

#
# Run student compiler
#
    printf "%-15s  " "$argbase.c"

    timeoutCompile $argdir $argbase $TIMEOUT

    if [ -f $argbase.timeout ]; then
        red "Timeout exceeded ($TIMEOUT seconds)" "\n"
    elif [ -s $argbase.error ]; then
        red "Unexpected error(s)" "\n"
        showIndentedFile "Student's error stream" $argbase.error $ERRLINES
    else
        RMFILES="$RMFILES $argbase.log $argbase.class $argbase.diff $argbase.jerr"
        cyan "Checking $argout" "\n"
	printf "%15s  " "Scrubbing: "
	removeUnprintables $argout

        printf "%15s  " "Assembling:"
        if $ASSEMBLE $argout --out $argbase.class > /dev/null 2> $argbase.log; then
            cyan "Success" "\n"
            for infile in $argdir/$argbase.input*; do
                basein=`basename $infile | awk -F. '{print $NF}' | tr -d '*' `
                baseout=`basename $infile | sed 's/input/output/' | tr -d '*' `
                if [ -f $infile ]; then
                    printf "%15s  " "$basein:"
                    timeoutRun $argbase $baseout $argbase.jerr $infile
                else
                    printf "%15s  " "no input"
                    timeoutRun $argbase $baseout $argbase.jerr
                fi
                if [ -s "$argbase.jerr" ]; then
                    red  "Java error" "\n"
                    showIndentedFile "java error stream" $argbase.jerr 20
                elif diff -q "$baseout" "GCC/$baseout" > /dev/null; then
                    green "Matches gcc" "\n"
                else
                    red   "Different output" "\n"
                    showIndentedFile "gcc output" GCC/$baseout 20
                    showIndentedFile "Student's output" $baseout 20
                fi
                rm -f $baseout
            done
        else
            red  "Error" "\n"
            showIndentedFile "Assembler log" $argbase.log $ERRLINES
        fi
    fi

#
# Cleanup
#

    rm -f $RMFILES
    RMFILES=""
done

