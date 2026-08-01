#!/bin/sh

set -eu

SCRIPT_DIR=$(CDPATH= cd -P "$(dirname "$0")" && pwd)
cd "$SCRIPT_DIR"

PROGRAM="${RAGTREE_BIN:-./test_ragtree_s}"
PYTHON_BIN="${PYTHON_BIN:-${PYTHON:-python3}}"
CHECKER="checker_ragtree_s.py"

usage() {
    cat >&2 <<EOF
Usage:
  $0 c <dataset>
  $0 f <dataset> <a|d>
  $0 1 <dataset> <a|d> <o|d> <topk> <ef_search>

Examples:
  $0 f laion a
  $0 1 laion a o 10 100
EOF
}

die() {
    printf 'Error: %s\n' "$*" >&2
    usage
    exit 1
}

require_program() {
    [ -x "$PROGRAM" ] || die "$PROGRAM is not executable; compile test_ragtree_s.cpp first"
}

require_python() {
    command -v "$PYTHON_BIN" >/dev/null 2>&1 || die "Python executable not found: $PYTHON_BIN"
}

is_positive_integer() {
    case "$1" in
        ""|0|0*|*[!0-9]*) return 1 ;;
        *) return 0 ;;
    esac
}

validate_ada_flag() {
    case "$1" in
        a|d) ;;
        *) die "index flag must be a or d" ;;
    esac
}

validate_opt_flag() {
    case "$1" in
        o|d) ;;
        *) die "query-plan flag must be o or d" ;;
    esac
}

[ "$#" -ge 1 ] || die "mode is required"
mode="$1"

case "$mode" in
    c)
        [ "$#" -eq 2 ] || die "check mode requires: c <dataset>"
        require_program
        "$PROGRAM" c "$2"
        ;;
    f)
        [ "$#" -eq 3 ] || die "build mode requires: f <dataset> <a|d>"
        validate_ada_flag "$3"
        require_program
        "$PROGRAM" f "$2" "$3"
        ;;
    1)
        [ "$#" -eq 6 ] || die "query mode requires: 1 <dataset> <a|d> <o|d> <topk> <ef_search>"
        dataset="$2"
        ada_flag="$3"
        opt_flag="$4"
        topk="$5"
        ef_search="$6"
        validate_ada_flag "$ada_flag"
        validate_opt_flag "$opt_flag"
        is_positive_integer "$topk" || die "topk must be a positive integer"
        is_positive_integer "$ef_search" || die "ef_search must be a positive integer"
        require_program
        require_python

        mkdir -p ../output
        result_path="../output/result_ragtree_s_${dataset}_ef${ef_search}_${ada_flag}${opt_flag}_selrandom.log"
        printf 'Running: %s 1 %s %s %s %s %s\n' \
            "$PROGRAM" "$dataset" "$ada_flag" "$opt_flag" "$topk" "$ef_search"
        "$PROGRAM" 1 "$dataset" "$ada_flag" "$opt_flag" "$topk" "$ef_search" >"$result_path"
        printf 'Result: %s\n' "$result_path"
        "$PYTHON_BIN" "$CHECKER" "$dataset" "$ef_search" "$ada_flag" "$opt_flag"
        ;;
    *)
        die "mode must be c, f, or 1"
        ;;
esac
