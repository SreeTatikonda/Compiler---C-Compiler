#!/bin/bash

TIMEOUT="30"
SCRIPT=$0
ERRLINES=10
CONTEXT=2
MODE="-5"
EXT="dots"
ERR="error"
FILES="files"
RMFILES=""

bailout()
{
    echo
    echo "Caught CTRL-C, terminating script"
    echo
    rm -f $RMFILES
    exit 1
}


usage()
{
    echo "Usage: $SCRIPT \"command to run\" [options] infile infile ..."
    echo "Run compiler executable, in code generation mode, on specified"
    echo "input files and compare with expected output files, but only"
    echo "on certain output lines that start with ."
    echo "Input files should be in a subdirectory; if so they will be"
    echo "copied into the working directory to run.  Expected outputs"
    echo "or error messages should be in directory OUTPUT_B.  Input files"
    echo "without corresponding output files will be skipped."
    echo "Options: "
    echo "  -g:  generate output files"
    echo "  -t N: set timeout to N seconds (default: $TIMEOUT)"
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

# Arg1: error file
firstError()
{
    awk '/^.*[Ee]rror in .*[Ll]ine *[0-9]*/ { errors++; if (errors>1) exit }\
                              { if (errors) print }' $1
}

# Arg1: error file
firstErrorLine()
{
    if [ -f $1 ]; then
        firstError $1 | head -n 1 | sed 's/.*[Ll]ine *\([0-9][0-9]*\).*/\1/'
    fi
}

# Arg1: line number or empty string
getAWKcontext()
{
    if [ "$1" ]; then
        local start=$[ $1 - $CONTEXT ]
        local stop=$[ $1 + $CONTEXT ]
        echo "(NR>=$start && NR<=$stop)"
    fi
}

# Arg1: input file
# Arg2: error line1 (or missing)
# Arg3: error line2 (or missing)
# Arg4: error line3 (or missing)
showContext()
{
  if [ ! "$2" ]; then
    return
  fi

  local inf="$1"

  local c1=`getAWKcontext $2`
  local c2=`getAWKcontext $3`
  local c3=`getAWKcontext $4`

  local acond=""

  if [ "$c1" ]; then
    acond="$acond $c1 ||"
  fi
  if [ "$c2" ]; then
    acond="$acond $c2 ||"
  fi
  if [ "$c3" ]; then
    acond="$acond $c3 ||"
  fi
  acond="($acond false)"

#   echo "$acond"

  printf "%15s  Input file relevant lines\n" " "
  printf "%15s  ------------------------------------------------------------\n" " "
  awk "$acond {printf(\"%15d  | %s\n\", NR, \$0);}" $inf
  printf "%15s  ------------------------------------------------------------\n" " "
}

# Arg1: whose (quoted string)
# Arg2: error file
showFirstError()
{
  if [ ! -f "$2" ]; then
    return 0
  fi
  printf "%15s  $1:\n" " "
  printf "%15s  ------------------------------------------------------------\n" " "
  firstError $2 | awk "(NR<=$ERRLINES){print \"                 | \" \$0}"
  printf "%15s  ------------------------------------------------------------\n" " "

}

# Arg1: error file
showRawError()
{
  if [ ! -s "$1" ]; then
    return 0
  fi
  printf "%15s  Student error stream (first $ERRLINES lines):\n" " "
  printf "%15s  ------------------------------------------------------------\n" " "
  awk "(NR<=$ERRLINES){print \"                 | \" \$0}" $1
  printf "%15s  ------------------------------------------------------------\n" " "

}


# Arg1: lineno
# Arg2: oracle line
# Arg3: student line
diffline()
{
  local strip1=`echo $2`
  local strip2=`echo $3`
  printf " %15s " "$1"
  if [ "x$strip1" == "x$strip2" ]; then
    cyan "| $strip1" "\n"
    return
  fi
  red "|" " "

  local oracle="$2"
  local student="$3"

  while [ "$student" ]; do
    read sfront srest <<< "$student"
    student="$srest"
    read ofront orest <<< "$oracle"
    oracle="$orest"

    if [ "x$sfront" == "x$ofront" ]; then
      cyan "$sfront" " "
    else
      red "$sfront" " "
    fi
  done

  if [ "$oracle" ]; then
    red "((eol))" " "
  fi
  printf "\n"
}

# Arg1: directory
# Arg2: output file
analyzeDifferences()
{
    local odir="$1"
    local ofile="$1/$2"
    local sfile="$2"

    # read oracle and student lines
    #
    local oracle=()
    local oln=0
    local student=()
    local sln=0
    while read line; do
        oracle+=("$line")
        oln=$[ oln + 1 ]
    done < $ofile
    while read line; do
        student+=("$line")
        sln=$[ sln + 1 ]
    done < $sfile

    #
    # Check: is the only difference the stack size?
    #
    local bigdiff=""
    local codediff=""
    local smallstack="y"
    local maxln="$oln"
    if [ "$sln" -gt "$oln" ]; then
        maxln="$sln"
    fi
    i=0
    while [ "$i" -lt "$oln" ]; do
        local oline=`echo ${oracle[$i]}`
        local sline=`echo ${student[$i]}`
        i=$[i+1]
        if [ "x$oline" == "x$sline" ]; then
            continue
        fi

        local ocode=`sed -n 's/\([.]code stack\) [0-9]* \(locals\) [0-9]*$/\1 \2/p' <<< "$oline"`
        local scode=`sed -n 's/\([.]code stack\) [0-9]* \(locals\) [0-9]*$/\1 \2/p' <<< "$sline"`

        if [ ! "$ocode" ]; then
            bigdiff="y"
            break
        fi
        if [ ! "$scode" ]; then
            bigdiff="y"
            break
        fi
        if [ "$ocode" != "$scode" ]; then
            bigdiff="y"
            break
        fi

        local olocal=`awk '{print $5}' <<< "$oline"`
        local slocal=`awk '{print $5}' <<< "$sline"`
        if [ "$olocal" != "$slocal" ]; then
            codediff="y"
            continue
        fi

        local ostack=`awk '{print $3}' <<< "$oline"`
        local sstack=`awk '{print $3}' <<< "$sline"`

        local delta=$[ sstack - ostack ]
        if [ "$delta" -lt "-2" ]; then
            codediff="y"
        elif [ "$delta" -gt "2" ]; then
            codediff="y"
        fi
    done

    if [ "$bigdiff" ]; then
        smallstack=""
        codediff=""
    fi
    if [ "$codediff" ]; then
        smallstack=""
    fi

    if [ "$smallstack" ]; then
        cyan "Small stack differences" "\n"
        return
    fi
    if [ "$codediff" ]; then
        yellow ".code stack/locals differences" "\n"

        # Show a summary: just the .method and .code lines,
        # and highlight them
        printf "%15s  ------------------------------------------------------------\n" " "
        i=0
        while [ "$i" -lt "$oln" ]; do
            local oline=`echo ${oracle[$i]}`
            local sline=`echo ${student[$i]}`
            i=$[i+1]
            local first=`awk '{print $1}' <<< "$oline"`
            if [ "$first" == ".method" ]; then
                printf "%15s  | $oline\n" " "
                continue
            fi
            if [ "$first" != ".code" ]; then
                continue
            fi
            printf "        Instruc: | $oline\n"
            if [ "$oline" == "$sline" ]; then
                printf "        Student: "
                green "| $sline" "\n"
            else
                diffline  "Student:" "$oline" "$sline"
            fi
            printf "                 |\n"
        done
        printf "%15s  ------------------------------------------------------------\n" " "
        return
    fi


    #
    # Different enough to dump output
    #
    red "Different output" "\n"

    #
    # Show oracle lines
    #
    i=0
    if [ "$odir" == "OUTPUTS_X" ]; then
        printf "%15s  Instructor extra credit output:\n" " "
    else
        printf "%15s  Instructor basic output:\n" " "
    fi
    printf "%15s  ------------------------------------------------------------\n" " "
    while [ "$i" -lt "$oln" ]; do
        printf "%10s  " " "
        local iii=`printf "%3d" "$i"`
        if [ "${oracle[$i]}" == "${student[$i]}" ]; then
            green " $iii | ${oracle[$i]}" "\n"
        else
            yellow " $iii | ${oracle[$i]}" "\n"
        fi
        i=$[i+1]
    done

    #
    # show student lines
    #
    i=0
    printf "%15s  ------------------------------------------------------------\n" " "
    printf "%15s  Student output:\n" " "
    printf "%15s  ------------------------------------------------------------\n" " "
    while [ "$i" -lt "$sln" ]; do
        diffline  $i "${oracle[$i]}" "${student[$i]}"
        i=$[i+1]
    done
    printf "%15s  ------------------------------------------------------------\n" " "
}



#
# Strip important . instructions from input stream,
# write to the output stream.
#
generate_dot()
{
    awk '($1==".class" || $1==".super" || $1==".method" || $1==".code" || $1==".end") { print }' | sed 's/ ;.*$//'
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

    RMFILES="$RMFILES $inbase.$ERR $inbase.j $inbase.$EXT $inbase.timeout"

    nice $EXE $MODE $inbase.c 1> /dev/null 2> $inbase.$ERR &
    local pid=$!

    if [ ! "$timeout" ]; then
        wait
    else
        disown -r
# ^ avoid messages if we kill the process
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
    if [ -f $inbase.j ]; then
        generate_dot < $inbase.j > $inbase.$EXT
    fi
}


# Arg1: input base file
# Arg2: oracle directory
#
# Writes:
#   : missing oracle
#   G output matches and empty error (Green)
#   Y output matches but nonempty error (Yellow)
#   R output different (Red)
#   N no output (Red)
#   m unexpected error (red)
#
compareValid()
{
  local out="$1.$EXT"
  local err="$1.$ERR"
  local dir="$2"

  if [ ! -f $dir/$out ]; then
    echo ":"
    return 0
  fi
  if [ ! -s $out ]; then
    if [ -s $err ]; then
      echo "m"
    else
      echo "N"
    fi
    return 0
  fi
  if diff -w -q $dir/$out $out > /dev/null; then
    if [ -s $err ]; then
      echo "Y"
    else
      echo "G"
    fi
  else
    echo "R"
  fi
}

# Arg1: input base file
# Arg2: oracle directory
#
# Writes:
#   : missing oracle
#   g first error matches perfectly (green)
#   c first line of error matches (cyan)
#   y first error line numbers match  (yellow)
#   r first errors different  (red)
#   e empty error stream      (red)
#   m unexpected error        (red)
compareInvalid()
{
  local out="$1.$EXT"
  local err="$1.$ERR"
  local dir="$2"

  if [ ! -f $dir/$err ]; then
    echo ":"
    return 0
  fi

  if [ ! -s $err ]; then
    echo "e"
    return 0
  fi

  firstoracle=`firstError $dir/$err`
  firststudent=`firstError $err`

  if [ ! "$firststudent" ]; then
    echo "m"
    return 0
  fi

  if [ "$firstoracle" == "$firststudent" ]; then
    echo "g"
    return 0
  fi

  linetextoracle=`head -n 1 <<< "$firstoracle"`
  linetextstudent=`head -n 1 <<< "$firststudent"`
  if [ "$linetextoracle" == "$linetextstudent" ]; then
    echo "c"
    return 0
  fi

  lineoracle=`sed 's/.*\([Ll]ine *[0-9]*\).*/\1/' <<< "$linetextoracle"`
  linestudent=`sed 's/.*\([Ll]ine *[0-9]*\).*/\1/' <<< "$linetextstudent"`

  if [ "$lineoracle" == "$linestudent" ]; then
    echo "y"
    return 0
  fi

  echo "r"
}



#############################################################

# Arg1: directory holding input file
# Arg2: base name of input file (without .c)
generate_oracle()
{
    local argdir="$1"
    local argbase="$2"
    local argout="$argbase.$EXT"
    local argerr="$argbase.$ERR"

    rm -f OUTPUTS_B/$argerr OUTPUTS_B/$argout

    if grep -w YESDOT $argdir/$argbase.c > /dev/null; then

        echo Generating outputs for $argbase.c in mode $MODE
        timeoutCompile $argdir $argbase

        if [ -s $argerr ]; then
            cp $argerr OUTPUTS_B
        else
            if [ ! -s $argout ]; then
                echo "Warning: empty output file $argout"
            fi
            cp $argout OUTPUTS_B
        fi

        rm -f $RMFILES
        RMFILES=""
    fi
}

#############################################################

trap bailout INT

if [ $# -eq 0 ]; then
  usage 0
fi

EXE="$1"
shift

GENERATE=""

echo Running tests using compiler:
$EXE -0 | awk '{print "  | " $0}'

echo " "

TOnext=""
for arg; do

# check for options/switches

  if [ $TOnext ]; then
    TIMEOUT=$arg
    TOnext=""
    continue
  fi

  case "$arg" in
    -g)
        GENERATE="y"
        mkdir -p OUTPUTS_B
        continue
        ;;

    -t)
        TOnext="y"
        continue
        ;;

  esac

  argdir=`dirname $arg`
  argbase=`basename -s .c $arg`
  argout="$argbase.$EXT"
  argerr="$argbase.$ERR"

  if [ ! -f $argdir/$argbase.c ]; then
      continue
  fi

#
# are we generating outputs?
#

  if [ $GENERATE ]; then
    generate_oracle $argdir $argbase
    continue
  fi

#
#
# Check for basic output/error files
#

  skip="y"
  if [ -f OUTPUTS_B/$argout ]; then
    skip=""
  fi
  if [ -f OUTPUTS_B/$argerr ]; then
    skip=""
  fi
  if [ "$skip" ]; then
    continue
  fi

#
# Run student compiler
#

  timeoutCompile $argdir $argbase $TIMEOUT
  if [ -f $argbase.timeout ]; then
    status_basic="T"
  else
    #
    # Compare with basic
    #
    if [ -f OUTPUTS_B/$argerr ]; then
      status_basic=`compareInvalid $argbase OUTPUTS_B`
    else
      status_basic=`compareValid $argbase OUTPUTS_B`
    fi
  fi

#
# Display best result
#

  printf "%-15s  " "$argbase.c"

  case "$status_basic" in

    g) green "1st Error matches" "\n"
          ;;

    c) cyan  "1st Error line matches" "\n"
          showContext $arg `firstErrorLine $argerr`
          showFirstError "Instructor error message" OUTPUTS_B/$argerr
          showFirstError "Student error message (first $ERRLINES lines)" $argerr
          ;;

    y) yellow "1st Error line number matches" "\n"
          showContext $arg `firstErrorLine $argerr`
          showFirstError "Instructor error message" OUTPUTS_B/$argerr
          showFirstError "Student error message (first $ERRLINES lines)" $argerr
          ;;


    G) green "Output matches" "\n"
          ;;

    Y) green "Output matches "
          yellow "but non-empty error stream" "\n"
          ;;

    r)
          red "1st Error is different" "\n"
          showContext $arg `firstErrorLine OUTPUTS_B/$argerr` `firstErrorLine $argerr`
          showFirstError "Instructor error message" OUTPUTS_B/$argerr
          showFirstError "Student error message (first $ERRLINES lines)" $argerr
          ;;

    R)
          analyzeDifferences OUTPUTS_B $argout
          ;;

    m)
          red "Unexpected error" "\n"
          showRawError $argerr
          ;;

    e)   red "Empty error stream" "\n"
          ;;

    N)   red "No output" "\n"
          ;;

    T)   red "Timeout exceeded ($TIMEOUT seconds)" "\n"
          ;;

    *)    red "Unexpected state: $status_basic" "\n"

  esac

  rm -f $RMFILES
  RMFILES=""
done

