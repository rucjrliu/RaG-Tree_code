#!/bin/sh

set -eu

SCRIPT_DIR=$(CDPATH= cd -P "$(dirname "$0")" && pwd)
cd "$SCRIPT_DIR"

PROGRAM="${RAGTREE_UPDATE_ADA_BIN:-./test_ragtree_update_ada}"
PYTHON_BIN="${PYTHON_BIN:-${PYTHON:-python3}}"
CHECKER="checker_ragtree_update_ada.py"

usage() {
    cat >&2 <<EOF
Usage:
  $0 c <dataset> <updatename>
  $0 u <dataset> <updatename> <a|d> [updates_path updated_index_path max_updates]
  $0 1 <dataset> <updatename> <a|d> <o|d> <topk> <ef_search> [updates_path updated_index_path max_updates]

Examples:
  $0 u laion u_20 a
  $0 1 laion u_20 a o 10 100
EOF
}

die() {
    printf 'Error: %s\n' "$*" >&2
    usage
    exit 1
}

require_program() {
    [ -x "$PROGRAM" ] || die "$PROGRAM is not executable; compile test_ragtree_update_ada.cpp first"
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
        [ "$#" -eq 3 ] || die "check mode requires: c <dataset> <updatename>"
        require_program
        "$PROGRAM" c "$2" "$3"
        ;;
    u)
        if [ "$#" -lt 4 ] || [ "$#" -gt 7 ]; then
            die "update mode has an invalid argument count"
        fi
        validate_ada_flag "$4"
        require_program
        "$PROGRAM" "$@"
        ;;
    1)
        if [ "$#" -lt 7 ] || [ "$#" -gt 10 ]; then
            die "query mode has an invalid argument count"
        fi
        dataset="$2"
        update_name="$3"
        ada_flag="$4"
        opt_flag="$5"
        topk="$6"
        ef_search="$7"
        validate_ada_flag "$ada_flag"
        validate_opt_flag "$opt_flag"
        is_positive_integer "$topk" || die "topk must be a positive integer"
        is_positive_integer "$ef_search" || die "ef_search must be a positive integer"
        require_program
        require_python

        mkdir -p ../output
        result_path="../output/result_ragtree_update_ada_${dataset}_${update_name}_ef${ef_search}_${ada_flag}${opt_flag}.log"
        printf 'Running: %s %s\n' "$PROGRAM" "$*"
        "$PROGRAM" "$@" >"$result_path"
        printf 'Result: %s\n' "$result_path"
        "$PYTHON_BIN" "$CHECKER" "$dataset" "$update_name" "$ef_search" "$ada_flag" "$opt_flag"
        ;;
    *)
        die "mode must be c, u, or 1"
        ;;
esac
