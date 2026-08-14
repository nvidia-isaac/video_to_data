#!/usr/bin/env bash
# process_soma_sequence.sh
# ----------------------------------------------------------------------
# End-to-end driver for a single SOMA reconstruction sequence.
#
# Manual mode (default): run all 5 stages, pause for user confirmation
# between stages so each tool's GUI / viser server can be inspected.
# At each prompt: y=run (default), n=skip just this stage,
#                 s=skip every remaining stage, q=quit.
# Auto mode (--auto): non-interactively run only stage 2 (retarget +
# save) and stage 5 (Isaac Lab replay), plus stage 3 if --with-recon
# is passed.
#
# Stages
# ------
#   1. soma_loader_probe.py    --visualize           (manual only)
#   2. soma_to_g1.py            --visualize --save   (auto: --save only)
#   3. reconstruct_support_surfaces.py               (auto: only with --with-recon)
#   4. view_scene.py            (Isaac Lab static)   (manual only)
#   5. replay_motion.py         (Isaac Lab playback)
#
# Usage
# -----
#   scripts/retarget/process_soma_sequence.sh <sequence_id> [options]
#
#   Options:
#     --auto                  Non-interactive mode (runs 2, optionally 3, 5).
#     --with-recon            In auto mode, also run stage 3.
#     --reconstructed-root P  Override raw SOMA reconstructions root
#                             (default: reconstructed_data/).
#     --motion-root P         Override saved motion-data root
#                             (default:
#                              source/robotic_grounding/robotic_grounding/
#                              assets/human_motion_data).
#     --robot-name NAME       Robot config name (default: g1).
#     --soma-subdir DIR       Schema subfolder under <motion-root>/whole_body
#                             (default: soma).
#     -h, --help              Show this help and exit.
#
# Example
# -------
#   # Manual walkthrough:
#   scripts/retarget/process_soma_sequence.sh \
#       2026-03-06_10-24-18_snack_box_pick_and_place_01
#
#   # Hands-off retarget + replay:
#   scripts/retarget/process_soma_sequence.sh \
#       2026-03-06_10-24-18_snack_box_pick_and_place_01 --auto
#
#   # Same, but also rebuild support surfaces:
#   scripts/retarget/process_soma_sequence.sh \
#       2026-03-06_10-24-18_snack_box_pick_and_place_01 \
#       --auto --with-recon
# ----------------------------------------------------------------------

set -euo pipefail

# ---- locate repo root (script lives at <repo>/scripts/retarget/) ------
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/../.." && pwd)"

# ---- defaults ---------------------------------------------------------
AUTO=0
WITH_RECON=0
ROBOT_NAME="g1"
SOMA_SUBDIR="soma"
RECON_ROOT="${REPO_ROOT}/reconstructed_data"
MOTION_ROOT="${REPO_ROOT}/source/robotic_grounding/robotic_grounding/assets/human_motion_data"
SEQUENCE_ID=""

usage() {
    # Print the contiguous block of leading "# ..." comments after the
    # shebang. Stops at the first line that is not a comment, so help
    # stays in sync if the banner grows or shrinks.
    awk '
        NR == 1 { next }
        /^#/    { sub(/^# ?/, ""); print; next }
                { exit }
    ' "${BASH_SOURCE[0]}"
}

# ---- arg parsing ------------------------------------------------------
while [[ $# -gt 0 ]]; do
    case "$1" in
        -h|--help)
            usage
            exit 0
            ;;
        --auto)
            AUTO=1
            shift
            ;;
        --with-recon)
            WITH_RECON=1
            shift
            ;;
        --reconstructed-root)
            [[ $# -ge 2 ]] || { echo "ERROR: --reconstructed-root needs a value" >&2; exit 2; }
            RECON_ROOT="$2"
            shift 2
            ;;
        --motion-root)
            [[ $# -ge 2 ]] || { echo "ERROR: --motion-root needs a value" >&2; exit 2; }
            MOTION_ROOT="$2"
            shift 2
            ;;
        --robot-name)
            [[ $# -ge 2 ]] || { echo "ERROR: --robot-name needs a value" >&2; exit 2; }
            ROBOT_NAME="$2"
            shift 2
            ;;
        --soma-subdir)
            [[ $# -ge 2 ]] || { echo "ERROR: --soma-subdir needs a value" >&2; exit 2; }
            SOMA_SUBDIR="$2"
            shift 2
            ;;
        --)
            shift
            break
            ;;
        -*)
            echo "ERROR: unknown option: $1" >&2
            echo "Run with --help for usage." >&2
            exit 2
            ;;
        *)
            if [[ -z "${SEQUENCE_ID}" ]]; then
                SEQUENCE_ID="$1"
            else
                echo "ERROR: unexpected extra positional arg: $1" >&2
                exit 2
            fi
            shift
            ;;
    esac
done

if [[ -z "${SEQUENCE_ID}" ]]; then
    echo "ERROR: sequence_id is required." >&2
    echo "" >&2
    usage >&2
    exit 2
fi

# ---- derived paths ----------------------------------------------------
RECON_DIR="${RECON_ROOT}/${SEQUENCE_ID}"
SOMA_PARQUET_ROOT="${MOTION_ROOT}/whole_body/${SOMA_SUBDIR}"
MOTION_PARTITION="${SOMA_PARQUET_ROOT}/sequence_id=${SEQUENCE_ID}/robot_name=${ROBOT_NAME}"

# ---- pretty printing --------------------------------------------------
if [[ -t 1 ]]; then
    BOLD='\033[1m'
    DIM='\033[2m'
    GREEN='\033[32m'
    YELLOW='\033[33m'
    BLUE='\033[34m'
    RED='\033[31m'
    RESET='\033[0m'
else
    BOLD='' DIM='' GREEN='' YELLOW='' BLUE='' RED='' RESET=''
fi

banner() {
    local title="$1"
    echo
    printf "${BOLD}${BLUE}================================================================${RESET}\n"
    printf "${BOLD}${BLUE}  %s${RESET}\n" "$title"
    printf "${BOLD}${BLUE}================================================================${RESET}\n"
}

info()  { printf "${DIM}[info]${RESET} %s\n" "$*"; }
warn()  { printf "${YELLOW}[warn]${RESET} %s\n" "$*" >&2; }
error() { printf "${RED}[err ]${RESET} %s\n" "$*" >&2; }
ok()    { printf "${GREEN}[ ok ]${RESET} %s\n" "$*"; }

# ---- interactive helpers ---------------------------------------------
# SKIP_ALL is set by the "s" answer in confirm() and consumed by the
# manual-mode dispatcher to short-circuit every remaining stage.
SKIP_ALL=0

# Prompt user with [Y/n/s/q]:
#   y / yes / <Enter>  -> run the stage (return 0)
#   n / no             -> skip just this stage (return 1)
#   s / skip-all       -> skip this and every later stage (return 1, set SKIP_ALL=1)
#   q / quit / abort   -> exit the script with code 130
# In --auto mode this is bypassed entirely (we never call confirm()).
confirm() {
    local prompt="$1"
    local reply
    while true; do
        printf "${BOLD}%s${RESET} ${DIM}[Y/n/s/q]${RESET} " "$prompt"
        if ! read -r reply; then
            echo
            error "stdin closed; aborting."
            exit 130
        fi
        case "${reply,,}" in
            ""|y|yes)   return 0 ;;
            n|no)       return 1 ;;
            s|skip|skip-all|all)
                SKIP_ALL=1
                return 1
                ;;
            q|quit|abort)
                warn "Aborting at user request."
                exit 130
                ;;
            *)
                echo "  Please answer y, n, s, or q."
                ;;
        esac
    done
}

# Echo a command (with a leading "$") then run it. Used so the user
# can copy/paste the exact command if they want to rerun a stage.
run_cmd() {
    printf "${DIM}\$ %s${RESET}\n" "$*"
    "$@"
}

# ---- pre-flight -------------------------------------------------------
banner "SOMA pipeline driver"
info  "sequence_id        : ${SEQUENCE_ID}"
info  "mode               : $([[ "${AUTO}" -eq 1 ]] && echo 'auto' || echo 'manual')"
if [[ "${AUTO}" -eq 1 ]]; then
    info  "stage 3 (recon)    : $([[ "${WITH_RECON}" -eq 1 ]] && echo 'enabled' || echo 'skipped')"
fi
info  "robot_name         : ${ROBOT_NAME}"
info  "reconstructed root : ${RECON_ROOT}"
info  "motion root        : ${MOTION_ROOT}"
info  "raw SOMA dir       : ${RECON_DIR}"
info  "parquet partition  : ${MOTION_PARTITION}"

if [[ ! -d "${RECON_DIR}" ]]; then
    error "Raw SOMA reconstruction directory not found: ${RECON_DIR}"
    error "Pass --reconstructed-root if your data lives elsewhere."
    exit 1
fi

cd "${REPO_ROOT}"

# ---- pinocchio / cmeel loader-path fix --------------------------------
# The Isaac Lab container ships pinocchio compiled against urdfdom v4
# and tinyxml2 v10, but the system-installed cmeel-urdfdom is v6 and
# cmeel-tinyxml2 is v11. So ``import pinocchio`` dies with:
#   ImportError: liburdfdom_sensor.so.4.0: cannot open shared object file
# (verified with ldd on
#   .../cmeel.prefix/lib/python3.11/site-packages/pinocchio/pinocchio_pywrap_default*.so:
#   liburdfdom_{model,sensor,world}.so.4.0 => not found,
#   libtinyxml2.so.10 => not found).
# The fix is to install older cmeel wheels that ship the matching
# soversions into a separate prefix and prepend their lib/ directory
# to LD_LIBRARY_PATH so the loader finds them BEFORE walking into the
# v6/v11 prefix.
#
# Done once per host (cached at ${PINOCCHIO_DEPS_PREFIX}); subsequent
# runs just re-export LD_LIBRARY_PATH. If pinocchio already imports
# cleanly (different image, apt-installed pinocchio, system loader
# already configured), the whole step is skipped.
setup_pinocchio_ld_path() {
    # Fast path: pinocchio imports cleanly with the existing env.
    if python -c "import pinocchio" >/dev/null 2>&1; then
        info "pinocchio import OK (no LD_LIBRARY_PATH fix needed)"
        return 0
    fi

    local cache_dir="${PINOCCHIO_DEPS_PREFIX:-${HOME:-/tmp}/.cache/robotic_grounding/pinocchio_deps}"
    local lib_dir="${cache_dir}/cmeel.prefix/lib"

    if [[ ! -f "${lib_dir}/liburdfdom_sensor.so.4.0" || ! -f "${lib_dir}/libtinyxml2.so.10" ]]; then
        info "Pinocchio v4/v10 cmeel deps not cached; installing to ${cache_dir}"
        mkdir -p "${cache_dir}"
        # ``--no-deps`` keeps pip from upgrading cmeel-tinyxml2 to v11
        # (cmeel-urdfdom 4.0.1's metadata names tinyxml2 with a range
        # that picks v11 by default; we want v10 specifically).
        if ! python -m pip install --target "${cache_dir}" --no-deps \
                "cmeel-urdfdom==4.0.1" "cmeel-tinyxml2==10.0.0" >/dev/null 2>&1; then
            warn "pip install failed; stage 2 will hit the original ImportError."
            warn "Try running by hand:"
            warn "  python -m pip install --target '${cache_dir}' --no-deps cmeel-urdfdom==4.0.1 cmeel-tinyxml2==10.0.0"
            return 0
        fi
    fi

    export LD_LIBRARY_PATH="${lib_dir}${LD_LIBRARY_PATH:+:${LD_LIBRARY_PATH}}"
    info "Pinocchio v4 deps      : ${lib_dir} (prepended to LD_LIBRARY_PATH)"

    if ! python -c "import pinocchio" >/dev/null 2>&1; then
        warn "pinocchio still fails to import after prepending ${lib_dir}."
        warn "Run by hand to see the full traceback:"
        warn "  LD_LIBRARY_PATH='${lib_dir}':\$LD_LIBRARY_PATH python -c 'import pinocchio'"
    fi
}
setup_pinocchio_ld_path

# ---- stage runners ----------------------------------------------------
# Each stage is wrapped so it can be skipped (manual mode) or auto-skipped
# (auto mode for stages 1, 4, and conditionally 3).

stage1_probe() {
    banner "Stage 1 / 5 — Probe SOMA loader (visualize raw body+object)"
    info "Tip: open http://localhost:8080 once viser is up; Ctrl-C to continue."
    run_cmd python scripts/retarget/soma_loader_probe.py \
        "${RECON_DIR}" \
        --visualize || {
        local rc=$?
        # Ctrl-C inside the viser loop returns 130. That's the expected
        # way to leave stage 1, so don't treat it as a failure.
        if [[ ${rc} -ne 130 ]]; then
            warn "Stage 1 exited with code ${rc}."
        fi
    }
    ok "Stage 1 done."
}

stage2_retarget() {
    local extra=()
    if [[ "${AUTO}" -eq 0 ]]; then
        extra+=( "--visualize" )
    fi
    extra+=(
        "--save"
        "--robot-name" "${ROBOT_NAME}"
        "--motion-root" "${MOTION_ROOT}"
        "--soma-subdir" "${SOMA_SUBDIR}"
    )

    banner "Stage 2 / 5 — Retarget SOMA -> ${ROBOT_NAME} (save parquet)"
    if [[ "${AUTO}" -eq 0 ]]; then
        info "Tip: open http://localhost:8080 once viser is up; Ctrl-C to continue."
    fi
    run_cmd python scripts/retarget/soma_to_g1.py \
        "${RECON_DIR}" \
        "${extra[@]}" || {
        local rc=$?
        # Same Ctrl-C tolerance as stage 1 — only meaningful when we
        # asked for --visualize (i.e. manual mode).
        if [[ "${AUTO}" -eq 0 && ${rc} -eq 130 ]]; then
            :
        else
            error "Stage 2 (retarget) failed with code ${rc}."
            exit ${rc}
        fi
    }

    if [[ ! -d "${MOTION_PARTITION}" ]]; then
        error "Expected parquet partition was not produced: ${MOTION_PARTITION}"
        error "Check the soma_to_g1.py output above."
        exit 1
    fi
    ok "Stage 2 done. Parquet at: ${MOTION_PARTITION}"
}

stage3_recon() {
    banner "Stage 3 / 5 — Reconstruct support surfaces"
    run_cmd python scripts/reconstruct_support_surfaces.py \
        --input_dir "${SOMA_PARQUET_ROOT}" \
        --sequence_id "${SEQUENCE_ID}"
    ok "Stage 3 done."
}

stage4_view() {
    banner "Stage 4 / 5 — Static Isaac Lab scene viewer"
    info "Tip: this opens the Isaac Lab GUI. Close the window to continue."
    run_cmd python scripts/view_scene.py \
        --motion_file "${MOTION_PARTITION}"
    ok "Stage 4 done."
}

stage5_replay() {
    banner "Stage 5 / 5 — Isaac Lab motion replay"
    info "Tip: this opens the Isaac Lab GUI. Close the window to finish."
    run_cmd python scripts/replay_motion.py \
        --motion_file "${MOTION_PARTITION}"
    ok "Stage 5 done."
}

# ---- mode dispatch ----------------------------------------------------
if [[ "${AUTO}" -eq 1 ]]; then
    # Auto mode: 2 -> (3 if --with-recon) -> 5. No prompts, no GUI on stage 2.
    stage2_retarget
    if [[ "${WITH_RECON}" -eq 1 ]]; then
        stage3_recon
    fi
    stage5_replay
    banner "Auto pipeline complete."
    exit 0
fi

# Manual mode: each stage is opt-in via [Y/n/s/q]. Once SKIP_ALL has
# been raised by an "s" answer, every later stage is silently skipped
# (no further prompts).
manual_stage() {
    local label="$1"      # "1", "2", ...
    local prompt="$2"
    local fn="$3"
    local on_skip="${4-}" # optional shell snippet to run when skipped

    if [[ "${SKIP_ALL}" -eq 1 ]]; then
        info "Stage ${label} skipped (skip-all)."
        if [[ -n "${on_skip}" ]]; then eval "${on_skip}"; fi
        return 0
    fi

    # `confirm` returns 1 on n/s; treat that as a normal "skipped" path
    # rather than letting `set -e` abort the script. Real stage failures
    # still propagate because the stage functions exit on their own.
    if confirm "${prompt}"; then
        "${fn}"
    else
        info "Stage ${label} skipped."
        if [[ -n "${on_skip}" ]]; then eval "${on_skip}"; fi
    fi
    return 0
}

manual_stage 1 "Run stage 1 (probe + visualize raw SOMA)?" stage1_probe
manual_stage 2 "Run stage 2 (retarget + visualize + save parquet)?" stage2_retarget \
    'if [[ ! -d "${MOTION_PARTITION}" ]]; then warn "No parquet at ${MOTION_PARTITION}; later stages will fail."; fi'
manual_stage 3 "Run stage 3 (reconstruct support surfaces)?" stage3_recon
manual_stage 4 "Run stage 4 (static Isaac Lab scene view)?" stage4_view
manual_stage 5 "Run stage 5 (Isaac Lab motion replay)?" stage5_replay

banner "Manual pipeline complete."
