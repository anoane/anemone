#!/bin/bash
# SPDX-License-Identifier: AGPL-3.0-or-later
# mma_forms_check.sh: compiles every instruction form of mma_forms_check.cu on its own, for each architecture, and
# prints one line per form and architecture: "compiles" or the compiler's message, whether that is what the
# documentation says (the words beside the form's symbol in the source), and for a form that compiles the mnemonics
# the disassembler shows with their counts, the tensor-core ones first. A form that became one instruction shows one.
#
# It launches nothing: no card is touched, no service is stopped, it is not a measurement batch and may run while the
# server serves. It does not stop at a form that is refused. Exit 0: every form was compiled and judged, whatever the
# verdicts. Exit 2: it could not do its work (no compiler, no source, a form line it cannot read, no log directory).
#
#   mma_forms_check.sh [log directory]                          default: $HOME/anemone/logs/mma_forms_check
#   ARCHITECTURES="sm_80 sm_120a"                               add sm_120 or sm_120f to see what those targets lose
#   NVCC=/usr/local/cuda/bin/nvcc   CUOBJDUMP=<beside nvcc>
set -u
MEASURE_DIRECTORY=$(cd "$(dirname "$0")" && pwd)
SOURCE=$MEASURE_DIRECTORY/mma_forms_check.cu
NVCC=${NVCC:-/usr/local/cuda/bin/nvcc}
CUOBJDUMP=${CUOBJDUMP:-$(dirname "$NVCC")/cuobjdump}
ARCHITECTURES=${ARCHITECTURES:-sm_80 sm_120a}
LOG_DIRECTORY=${1:-$HOME/anemone/logs/mma_forms_check}
SETUP_FAILURE=2
MESSAGE_COLUMNS=110              # of a compiler's message in the table; the whole message is in the form's log

[ -r "$SOURCE" ] || { echo "no source: $SOURCE" >&2; exit $SETUP_FAILURE; }
[ -x "$NVCC" ] || { echo "no compiler: $NVCC (set NVCC)" >&2; exit $SETUP_FAILURE; }
for architecture in $ARCHITECTURES; do
    mkdir -p "$LOG_DIRECTORY/$architecture" || { echo "no log directory: $LOG_DIRECTORY" >&2; exit $SETUP_FAILURE; }
done

# The forms and what the documentation says of each, "<symbol>|<targets>", from the lines that select a kernel:
#   #elif defined(CHECK_<FORM>)      /* documented for: <targets> (<a remark, or none>) */
SELECTING_LINE='^#(if|elif) defined\((CHECK_[A-Z0-9_]+)\)'
DOCUMENTED_FOR='[[:space:]]*/\* documented for: ([^(*]*[^(* ])[[:space:]]*(\(.*\))?[[:space:]]*\*/$'
FORMS=$(sed -En "s,$SELECTING_LINE$DOCUMENTED_FOR,\2|\3,p" "$SOURCE")
form_count=$(printf '%s\n' "$FORMS" | grep -c '^CHECK_')
selecting_lines=$(grep -c -E "$SELECTING_LINE" "$SOURCE")
if [ "$form_count" -eq 0 ] || [ "$form_count" -ne "$selecting_lines" ]; then
    echo "read $form_count forms from $selecting_lines selecting lines of $SOURCE: a line has another shape" >&2
    exit $SETUP_FAILURE
fi

# What the documentation says for one architecture: yes, no, or silent (the words are in the source's header).
documented_answer() {            # documented_answer "<targets>" <architecture>
    local targets=$1 architecture=$2 number token
    number=$(printf '%s' "$architecture" | tr -cd '0-9')
    set -f                       # a target may end in a question mark: it is a word here, not a file pattern
    for token in $targets; do
        case $token in
            all) echo yes; set +f; return ;;
            unstated) echo silent; set +f; return ;;
            from_*) [ "$number" -ge "${token#from_}" ] && { echo yes; set +f; return; } ;;
            "$architecture") echo yes; set +f; return ;;
            "$architecture?") echo silent; set +f; return ;;
        esac
    done
    set +f
    echo no
}

# The mnemonics of a disassembly with their counts, those of the tensor cores first: "HMMA.16816.F32 x1, LDG x7, ...".
mnemonic_counts() {              # mnemonic_counts <disassembly file>
    awk '
        match($0, /\/\*[0-9a-f]+\*\/[ \t]+/) {
            field_count = split(substr($0, RSTART + RLENGTH), fields, /[ \t;]+/)
            mnemonic = fields[1]
            if (substr(mnemonic, 1, 1) == "@" && field_count > 1) mnemonic = fields[2]      # a predicated instruction
            if (mnemonic != "") print mnemonic
        }' "$1" | sort | uniq -c | awk '
        { line = $2 " x" $1; if ($2 ~ /MMA/) first = first (first == "" ? "" : ", ") line; else rest = rest ", " line }
        END { if (first == "") first = "no tensor-core instruction"; print first rest }'
}

# One form for one architecture: compiled, disassembled, judged; one line of the table.
check_form() {                   # check_form <architecture> <symbol> "<targets>"
    local architecture=$1 form=$2 targets=$3 base result detail verdict compiler_status
    base=$LOG_DIRECTORY/$architecture/${form#CHECK_}
    rm -f "$base.cubin" "$base.sass"
    "$NVCC" -arch="$architecture" -cubin -std=c++17 -O3 "-D$form" -o "$base.cubin" "$SOURCE" \
        > "$base.log" 2>&1 < /dev/null
    compiler_status=$?
    if [ $compiler_status -eq 0 ] && [ -s "$base.cubin" ]; then
        result=compiles
        if [ -x "$CUOBJDUMP" ] && "$CUOBJDUMP" -sass "$base.cubin" > "$base.sass" 2>&1 < /dev/null; then
            detail=$(mnemonic_counts "$base.sass")
        else
            detail="not disassembled"
        fi
        if [ -s "$base.log" ]; then detail="$detail (the compiler printed $(wc -l < "$base.log" | tr -d ' ') lines)"; fi
    else
        result=refused
        detail=$(grep -i -m 1 'error' "$base.log" \
            | sed -E 's/^ptxas [^;]*; *//; s/^[^ ]*mma_forms_check\.cu[(:][0-9:)]* *//')
        [ -n "$detail" ] || detail=$(grep -m 1 '[^[:space:]]' "$base.log")
        [ -n "$detail" ] || detail="exit $compiler_status and no message"
        detail=$(printf '%s' "$detail" | cut -c "1-$MESSAGE_COLUMNS")
    fi
    case "$(documented_answer "$targets" "$architecture") $result" in
        "yes compiles" | "no refused") verdict="as documented" ;;
        "yes refused") verdict="DOCUMENTED, YET REFUSED" ;;
        "no compiles") verdict="NOT DOCUMENTED, YET COMPILES" ;;
        *) verdict="documentation silent" ;;
    esac
    printf '%-8s %-40s %-9s %-29s %s\n' "$architecture" "${form#CHECK_}" "$result" "$verdict" "$detail"
}

{
    echo "mma_forms_check $(date '+%F %T'): $form_count forms, architectures: $ARCHITECTURES"
    echo "compiler: $("$NVCC" --version 2>&1 | tail -n 1)"
    if [ -x "$CUOBJDUMP" ]; then
        echo "disassembler: $CUOBJDUMP"
    else
        echo "disassembler: none at $CUOBJDUMP (set CUOBJDUMP): mnemonics are not listed"
    fi
    echo "logs: $LOG_DIRECTORY/<architecture>/<form>.log, .cubin and .sass"
    echo
    for architecture in $ARCHITECTURES; do
        while IFS='|' read -r form targets; do
            check_form "$architecture" "$form" "$targets"
        done <<FORMS_END
$FORMS
FORMS_END
        echo
    done
} | tee "$LOG_DIRECTORY/summary.txt"
for architecture in $ARCHITECTURES; do   # a compiler that compiles nothing would make every negative "as documented"
    grep -q -E "^$architecture +[^ ]+ +compiles " "$LOG_DIRECTORY/summary.txt" \
        || { echo "NOTHING compiled for $architecture: its lines say nothing" >&2; exit $SETUP_FAILURE; }
done
echo "as documented: $(grep -c ' as documented ' "$LOG_DIRECTORY/summary.txt");" \
    "against the documentation: $(grep -c -E ' (DOCUMENTED, YET REFUSED|NOT DOCUMENTED, YET COMPILES) ' \
        "$LOG_DIRECTORY/summary.txt"); documentation silent: $(grep -c ' documentation silent ' \
        "$LOG_DIRECTORY/summary.txt")" | tee -a "$LOG_DIRECTORY/summary.txt"
