#!/usr/bin/env bash
# train_tpv_wholebody_workflow.sh
#
# !!! RUN THIS ON THE HOST, NOT INSIDE THE DOCKER CONTAINER !!!
#
# This script orchestrates Docker (`./workflow/run.sh exec`) and OSMO submission
# (`python .../run_experiment.py --osmo`), neither of which work from inside the
# container (osmo CLI is not installed there, and `docker exec` from a container
# would target the host's docker daemon). The script aborts at startup if it
# detects it's running inside the container.
#
# Driver for the recurring third-person-view (TPV) whole-body ReconBody training
# flow. There are two ways to use it:
#
# A) Interactive driver `run` (recommended for a new dataset):
#    Walks through all four stages below in order, prompting before each one
#    so any stage can be skipped or re-prompted. Detected state is shown so
#    the prompt defaults are sensible: e.g. init defaults to "skip" if a
#    config already exists, set-frames pre-fills the current frame range and
#    you can press Enter to keep it, etc. Resumable from any stage by
#    re-running `run <exp_id>`.
#      Examples:
#        ./scripts/train_tpv_wholebody_workflow.sh run                        # asks for input
#        ./scripts/train_tpv_wholebody_workflow.sh run <motion_file_path>     # new dataset
#        ./scripts/train_tpv_wholebody_workflow.sh run <exp_id>               # existing exp
#        ./scripts/train_tpv_wholebody_workflow.sh run --dry-run --yes ...    # auto-accept defaults, no exec
#
# B) Per-stage subcommands (for re-running just one step or scripting):
#
#   1. init         [host]         Create experiments/<exp_id>/config.yaml from
#                                  a template and add it to registry.yaml.
#   2. replay       [host->ctnr]   Calls `./workflow/run.sh exec ...` to run
#                                  scripts/replay_motion_viser.py inside the
#                                  already-running container.
#   3. set-frames   [host]         Edit the chosen frames into the YAML.
#   4. submit       [host]         Submit the OSMO job via run_experiment.py.
#
# This script is intentionally scoped to whole-body TPV ReconBody training only
# (see experiments/recon_body_*/config.yaml). It is NOT a general experiment
# runner.
#
# Usage:
#   ./scripts/train_tpv_wholebody_workflow.sh run [<exp_id_or_motion_file>] [--dry-run] [--yes]
#   ./scripts/train_tpv_wholebody_workflow.sh init <motion_file_path> [--exp-id NAME] [--force]
#   ./scripts/train_tpv_wholebody_workflow.sh replay <exp_id>
#   ./scripts/train_tpv_wholebody_workflow.sh set-frames <exp_id> <start> <end>
#   ./scripts/train_tpv_wholebody_workflow.sh submit <exp_id> [--build-image] [--run-name NAME] [--dry-run]
#
# Global flags (must come before the subcommand):
#   --dry-run    Print every command but never execute (works for `run` too).
#   --yes / -y   Auto-accept all interactive prompt defaults (CI-friendly).

set -euo pipefail

# --- Resolve repo paths ---
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"
EXPERIMENTS_DIR="${REPO_ROOT}/experiments"
REGISTRY_FILE="${EXPERIMENTS_DIR}/registry.yaml"
TEMPLATE_CONFIG="${EXPERIMENTS_DIR}/recon_body_snack_box_pick_and_place_01/config.yaml"
WORKFLOW_RUN_SH="${REPO_ROOT}/workflow/run.sh"
REPLAY_SCRIPT="scripts/replay_motion_viser.py"
RUN_EXPERIMENT_SCRIPT="experiments/run_experiment.py"
# Container target — must match what `./workflow/run.sh start` produces. Both
# values are also positionally required by `./workflow/run.sh exec` (see
# workflow/run.sh, the `exec` case `shift 3`s blindly), so we pass them
# explicitly when invoking exec to avoid the "robotic-grounding----gpupython"
# corruption when the placeholders are omitted.
CONTAINER_VERSION="${CONTAINER_VERSION:-latest}"
CONTAINER_GPU="${CONTAINER_GPU:-0}"
CONTAINER_NAME="robotic-grounding-${CONTAINER_VERSION}-gpu${CONTAINER_GPU}"

# --- Global flags (set by parse_global_flags) ---
DRY_RUN=0
ASSUME_YES=0

# --- Host-vs-container guard ---
# Returns 0 if running inside a Docker container, 1 otherwise. Used to refuse
# to run the workflow script from inside the container shell, where docker exec
# and the OSMO CLI both fail.
running_inside_container() {
    [[ -f /.dockerenv ]] && return 0
    if [[ -r /proc/1/cgroup ]] && grep -qE 'docker|containerd|kubepods' /proc/1/cgroup 2>/dev/null; then
        return 0
    fi
    return 1
}

# --- Pretty output helpers ---
_color() {
    if [[ -t 1 ]]; then printf '\033[%sm%s\033[0m' "$1" "$2"; else printf '%s' "$2"; fi
}
log_info()   { echo "[$(_color '36' INFO)] $*"; }
log_warn()   { echo "[$(_color '33' WARN)] $*" >&2; }
log_error()  { echo "[$(_color '31' ERROR)] $*" >&2; }
log_stage()  {
    echo
    echo "$(_color '1;34' "===== $* =====")"
}
log_cmd()    { echo "$(_color '90' '+') $*"; }

die() { log_error "$*"; exit 1; }

# Run a command, echoing it first. Honors --dry-run.
run_cmd() {
    log_cmd "$*"
    if (( DRY_RUN )); then
        log_info "(dry-run) skipping execution"
        return 0
    fi
    eval "$@"
}

# --- Prompt helpers (interactive) ---

# prompt_yn <question> <default y|n> -> echoes "y" or "n"
prompt_yn() {
    local question="$1" default="${2:-y}" answer suffix
    if [[ "${default}" == "y" ]]; then suffix="[Y/n]"; else suffix="[y/N]"; fi
    if (( ASSUME_YES )); then
        echo "${default}"
        return 0
    fi
    while true; do
        read -r -p "${question} ${suffix} " answer < /dev/tty || answer=""
        answer="${answer:-${default}}"
        case "${answer,,}" in
            y|yes) echo "y"; return 0 ;;
            n|no)  echo "n"; return 0 ;;
            q|quit) die "user quit" ;;
            *)     echo "Please answer y, n, or q." >&2 ;;
        esac
    done
}

# prompt_value <question> <default> -> echoes user value or default if empty.
prompt_value() {
    local question="$1" default="$2" answer
    if (( ASSUME_YES )); then
        echo "${default}"
        return 0
    fi
    read -r -p "${question} [${default}] " answer < /dev/tty || answer=""
    echo "${answer:-${default}}"
}

# --- Path / id derivation ---

# Strip trailing slash, then derive a short sequence name from a motion_file path
# Example: source/.../sequence_id=2026-02-27_16-25-23_sit_skinny_wood_chair_01/robot_name=g1
#          -> sit_skinny_wood_chair_01
derive_seq_name() {
    local p="${1%/}"
    # Drop trailing /robot_name=*
    p="${p%/robot_name=*}"
    local base="${p##*/}"
    # Strip leading sequence_id=
    base="${base#sequence_id=}"
    # Strip leading date/time prefix YYYY-MM-DD_HH-MM-SS_ if present
    base="$(echo "${base}" | sed -E 's/^[0-9]{4}-[0-9]{2}-[0-9]{2}_[0-9]{2}-[0-9]{2}-[0-9]{2}_//')"
    echo "${base}"
}

derive_exp_id_from_motion() {
    local motion_file="$1"
    echo "recon_body_$(derive_seq_name "${motion_file}")"
}

# Ensure a motion_file path points at a robot_name=* partition (a directory that
# pyarrow can read as a parquet dataset). If the input already contains
# /robot_name=*, return it unchanged. Otherwise look under the path for a
# single robot_name=* subdir and append it.
#
# Why: pq.read_table() walks every file under the given path. If the path is
# the parent sequence_id=* dir, pyarrow tries to parse object/material.mtl,
# textured_mesh.obj, etc. as parquet and dies.
normalize_motion_file_to_robot_partition() {
    local mf="${1%/}"
    if [[ "${mf}" == */robot_name=* ]]; then
        echo "${mf}"
        return 0
    fi
    local candidate_root=""
    if [[ -d "${mf}" ]]; then
        candidate_root="${mf}"
    elif [[ -d "${REPO_ROOT}/${mf}" ]]; then
        candidate_root="${REPO_ROOT}/${mf}"
    fi
    if [[ -z "${candidate_root}" ]]; then
        log_warn "motion_file does not contain '/robot_name=*' and the directory is not present locally; leaving as-is: ${mf}" >&2
        echo "${mf}"
        return 0
    fi
    local matches=()
    while IFS= read -r d; do
        [[ -n "${d}" ]] && matches+=("${d}")
    done < <(find "${candidate_root}" -mindepth 1 -maxdepth 1 -type d -name 'robot_name=*' -printf '%f\n' 2>/dev/null | sort)
    if (( ${#matches[@]} == 0 )); then
        die "motion_file '${mf}' has no robot_name=* subdirectory under ${candidate_root}; expected layout: <sequence_id=...>/robot_name=<robot>"
    fi
    if (( ${#matches[@]} > 1 )); then
        die "motion_file '${mf}' contains multiple robot_name=* subdirs (${matches[*]}); pass the specific one explicitly"
    fi
    echo "${mf}/${matches[0]}"
}

config_path_for() {
    echo "${EXPERIMENTS_DIR}/$1/config.yaml"
}

# --- YAML helpers (Python inline) ---

# Read a top-level scalar key (id, motion_file, run_name, ...) from a config.
yaml_get_top() {
    local config="$1" key="$2"
    python3 - "$config" "$key" <<'PY'
import sys, yaml
path, key = sys.argv[1], sys.argv[2]
with open(path) as f:
    data = yaml.safe_load(f) or {}
val = data.get(key, "")
print("" if val is None else val)
PY
}

# Read train_overrides.<dotted_key> as a string.
yaml_get_override() {
    local config="$1" key="$2"
    python3 - "$config" "$key" <<'PY'
import sys, yaml
path, key = sys.argv[1], sys.argv[2]
with open(path) as f:
    data = yaml.safe_load(f) or {}
overrides = data.get("train_overrides", {}) or {}
val = overrides.get(key, "")
print("" if val is None else val)
PY
}

# Set the start/end frames in a config, preserving comments and key order via
# a surgical line-by-line edit (only the two frame lines are rewritten).
# A value of "-" for start/end keeps the existing line untouched.
# If the key is missing entirely, it is appended under train_overrides.
yaml_set_frames() {
    local config="$1" start="$2" end="$3"
    python3 - "$config" "$start" "$end" <<'PY'
import re, sys
path, start, end = sys.argv[1], sys.argv[2], sys.argv[3]
with open(path) as f:
    lines = f.readlines()

KEY_START = "env.commands.motion.motion_start_frame"
KEY_END = "env.commands.motion.motion_end_frame"

def patch(lines, key, new_val):
    pat = re.compile(rf"^(\s*){re.escape(key)}\s*:\s*([^#\n]*)(#.*)?$")
    for i, line in enumerate(lines):
        m = pat.match(line.rstrip("\n"))
        if m:
            indent = m.group(1)
            trailing = m.group(3) or ""
            sep = "  " if trailing else ""
            lines[i] = f"{indent}{key}: {new_val}{sep}{trailing}\n"
            return True
    return False

def append_under_train_overrides(lines, key, new_val):
    for i, line in enumerate(lines):
        if line.startswith("train_overrides:"):
            j = i + 1
            while j < len(lines) and (lines[j].startswith(" ") or lines[j].startswith("\t") or lines[j].strip() == ""):
                j += 1
            insert_at = j
            lines.insert(insert_at, f"  {key}: {new_val}\n")
            return True
    lines.append("\ntrain_overrides:\n")
    lines.append(f"  {key}: {new_val}\n")
    return True

if start != "-":
    if not patch(lines, KEY_START, start):
        append_under_train_overrides(lines, KEY_START, start)
if end != "-":
    if not patch(lines, KEY_END, end):
        append_under_train_overrides(lines, KEY_END, end)

with open(path, "w") as f:
    f.writelines(lines)
PY
}

# --- Container detection ---

container_running() {
    docker ps --format '{{.Names}}' 2>/dev/null | grep -qx "${CONTAINER_NAME}"
}

# --- Stage: init ---

cmd_init() {
    local motion_file="" exp_id="" force=0
    while (( $# )); do
        case "$1" in
            --exp-id) exp_id="$2"; shift 2 ;;
            --force)  force=1; shift ;;
            -*)       die "init: unknown flag: $1" ;;
            *)
                if [[ -z "${motion_file}" ]]; then motion_file="$1"; else die "init: unexpected arg: $1"; fi
                shift
                ;;
        esac
    done

    [[ -n "${motion_file}" ]] || die "init: <motion_file_path> is required"
    [[ -f "${TEMPLATE_CONFIG}" ]] || die "init: template config not found: ${TEMPLATE_CONFIG}"

    if [[ -z "${exp_id}" ]]; then
        exp_id="$(derive_exp_id_from_motion "${motion_file}")"
    fi

    # Normalize to a robot_name=* partition; pyarrow chokes on the parent dir.
    local normalized_mf
    normalized_mf="$(normalize_motion_file_to_robot_partition "${motion_file}")"
    if [[ "${normalized_mf}" != "${motion_file}" ]]; then
        log_info "Normalized motion_file: ${motion_file} -> ${normalized_mf}"
        motion_file="${normalized_mf}"
    fi

    local target_dir="${EXPERIMENTS_DIR}/${exp_id}"
    local target_config="${target_dir}/config.yaml"
    log_info "Initializing experiment '${exp_id}'"
    log_info "  motion_file: ${motion_file}"
    log_info "  config:      ${target_config}"

    if [[ -e "${target_config}" && "${force}" -eq 0 ]]; then
        die "init: ${target_config} already exists (use --force to overwrite)"
    fi

    if (( DRY_RUN )); then
        log_info "(dry-run) would create ${target_config}"
        log_info "(dry-run) would ensure registry entry: ${exp_id}: ${exp_id}"
        return 0
    fi

    mkdir -p "${target_dir}"
    EXP_ID="${exp_id}" MOTION_FILE="${motion_file}" \
    python3 - "${TEMPLATE_CONFIG}" "${target_config}" <<'PY'
import os, sys, yaml
src, dst = sys.argv[1], sys.argv[2]
exp_id = os.environ["EXP_ID"]
motion_file = os.environ["MOTION_FILE"]
with open(src) as f:
    data = yaml.safe_load(f)
data["id"] = exp_id
data["run_name"] = exp_id
data["description"] = f"ReconBody {exp_id} with SONIC JOINT_RESIDUAL"
data["motion_file"] = motion_file
overrides = data.setdefault("train_overrides", {})
overrides["env.commands.motion.motion_start_frame"] = 0
overrides["env.commands.motion.motion_end_frame"] = -1
with open(dst, "w") as f:
    f.write("# ReconBody: " + exp_id + " (TPV whole-body)\n")
    f.write("#\n")
    f.write("# Usage:\n")
    f.write(f"#   python robotic_grounding/experiments/run_experiment.py {exp_id} --local\n")
    yaml.safe_dump(data, f, sort_keys=False, default_flow_style=False)
PY
    log_info "Wrote ${target_config}"

    EXP_ID="${exp_id}" python3 - "${REGISTRY_FILE}" <<'PY'
import os, sys
exp_id = os.environ["EXP_ID"]
path = sys.argv[1]
with open(path) as f:
    text = f.read()
needle = f"{exp_id}: {exp_id}"
if needle in text:
    print(f"[INFO] registry already contains '{needle}', skipping")
    sys.exit(0)
lines = text.splitlines()
inserted = False
out = []
for line in lines:
    out.append(line)
    if not inserted and line.startswith("# Whole-body experiments:"):
        out.append(needle)
        inserted = True
if not inserted:
    out.append("")
    out.append("# Whole-body experiments:")
    out.append(needle)
out.append("")
with open(path, "w") as f:
    f.write("\n".join(line for line in out if line is not None))
print(f"[INFO] added '{needle}' to {path}")
PY
}

# --- Stage: replay ---

cmd_replay() {
    local exp_id="${1:-}"
    [[ -n "${exp_id}" ]] || die "replay: <exp_id> is required"
    local config; config="$(config_path_for "${exp_id}")"
    [[ -f "${config}" ]] || die "replay: config not found: ${config}"

    local motion_file; motion_file="$(yaml_get_top "${config}" motion_file)"
    [[ -n "${motion_file}" ]] || die "replay: motion_file is empty in ${config}"

    log_info "exp_id      : ${exp_id}"
    log_info "motion_file : ${motion_file}"

    if ! container_running; then
        log_warn "container '${CONTAINER_NAME}' is not running."
        log_warn "Start it first with:  ${WORKFLOW_RUN_SH} start"
        if (( DRY_RUN )); then
            log_warn "(dry-run) continuing anyway to print the resolved command"
        else
            die "replay: container is not running"
        fi
    fi

    # NOTE: workflow/run.sh's `exec` case runs `shift 3` blindly, so the version
    # and gpu placeholders MUST be passed positionally even if they are the
    # defaults. Without them, $CONTAINER_NAME inside run.sh becomes
    # "robotic-grounding----gpupython" and `docker exec` fails.
    run_cmd "${WORKFLOW_RUN_SH} exec ${CONTAINER_VERSION} ${CONTAINER_GPU} -- python ${REPLAY_SCRIPT} --motion_file ${motion_file}"
}

# --- Stage: set-frames ---

cmd_set_frames() {
    local exp_id="${1:-}" start="${2:-}" end="${3:-}"
    [[ -n "${exp_id}" && -n "${start}" && -n "${end}" ]] || die "set-frames: usage: set-frames <exp_id> <start> <end>"
    local config; config="$(config_path_for "${exp_id}")"
    [[ -f "${config}" ]] || die "set-frames: config not found: ${config}"

    local cur_start cur_end
    cur_start="$(yaml_get_override "${config}" env.commands.motion.motion_start_frame)"
    cur_end="$(yaml_get_override "${config}" env.commands.motion.motion_end_frame)"
    log_info "current motion_start_frame=${cur_start:-<unset>} motion_end_frame=${cur_end:-<unset>}"
    log_info "new     motion_start_frame=${start} motion_end_frame=${end} (- means keep current)"

    if (( DRY_RUN )); then
        log_info "(dry-run) would update ${config}"
        return 0
    fi
    yaml_set_frames "${config}" "${start}" "${end}"
    log_info "Updated ${config}"
}

# --- Stage: submit ---

cmd_submit() {
    local exp_id="" run_name="" build_image=0 local_dry=0
    while (( $# )); do
        case "$1" in
            --build-image) build_image=1; shift ;;
            --run-name)    run_name="$2"; shift 2 ;;
            --dry-run)     local_dry=1; shift ;;
            -*)            die "submit: unknown flag: $1" ;;
            *)
                if [[ -z "${exp_id}" ]]; then exp_id="$1"; else die "submit: unexpected arg: $1"; fi
                shift
                ;;
        esac
    done
    [[ -n "${exp_id}" ]] || die "submit: <exp_id> is required"
    local config; config="$(config_path_for "${exp_id}")"
    if [[ ! -f "${config}" ]]; then
        if (( local_dry )) || (( DRY_RUN )); then
            log_warn "config not found: ${config} (dry-run; continuing anyway)"
        else
            die "submit: config not found: ${config}"
        fi
    fi

    [[ -n "${run_name}" ]] || run_name="${exp_id}_motion_data"

    local cmd="python ${RUN_EXPERIMENT_SCRIPT} ${exp_id} --osmo --run-name ${run_name}"
    if (( build_image )); then cmd+=" --build-image"; fi

    log_info "exp_id     : ${exp_id}"
    log_info "run_name   : ${run_name}"
    log_info "build_image: $(( build_image ))"

    if (( local_dry )) || (( DRY_RUN )); then
        log_cmd "${cmd}"
        log_info "(dry-run) skipping submission"
        return 0
    fi
    run_cmd "${cmd}"
}

# --- Interactive driver: run ---

cmd_run() {
    local arg=""
    while (( $# )); do
        case "$1" in
            -*) die "run: unknown flag: $1 (use --dry-run/--yes before the subcommand)" ;;
            *)
                if [[ -z "${arg}" ]]; then arg="$1"; else die "run: unexpected arg: $1"; fi
                shift
                ;;
        esac
    done

    if [[ -z "${arg}" ]]; then
        arg="$(prompt_value "Enter exp_id or motion_file path" "")"
        [[ -n "${arg}" ]] || die "run: nothing to do"
    fi

    # Decide whether arg is a motion file path or an exp_id.
    local exp_id="" motion_file=""
    if [[ "${arg}" == */robot_name=* || -d "${REPO_ROOT}/${arg}" || -d "${arg}" ]]; then
        motion_file="${arg}"
        exp_id="$(derive_exp_id_from_motion "${motion_file}")"
        log_info "Treating arg as motion_file path"
        log_info "  motion_file -> ${motion_file}"
        log_info "  derived exp_id -> ${exp_id}"
    else
        exp_id="${arg}"
        log_info "Treating arg as exp_id: ${exp_id}"
    fi

    local config; config="$(config_path_for "${exp_id}")"

    # ----- Stage 1: init -----
    log_stage "Stage 1/4: init  [host]"
    local needs_init=0 default_init="n"
    if [[ ! -f "${config}" ]]; then
        needs_init=1
        default_init="y"
        log_info "Config does not exist yet: ${config}"
        if [[ -z "${motion_file}" ]]; then
            motion_file="$(prompt_value "Enter motion_file path for ${exp_id}" "")"
            [[ -n "${motion_file}" ]] || die "init: motion_file is required when config does not exist"
        fi
    else
        log_info "Config already exists: ${config}"
    fi

    local ans; ans="$(prompt_yn "Run init for ${exp_id}?" "${default_init}")"
    if [[ "${ans}" == "y" ]]; then
        if [[ -z "${motion_file}" ]]; then
            motion_file="$(yaml_get_top "${config}" motion_file)"
            log_info "Using motion_file from existing config: ${motion_file}"
        fi
        local edited_exp_id; edited_exp_id="$(prompt_value "exp_id" "${exp_id}")"
        exp_id="${edited_exp_id}"
        config="$(config_path_for "${exp_id}")"
        local force_args=()
        if [[ -e "${config}" ]]; then
            local overwrite; overwrite="$(prompt_yn "Overwrite existing ${config}?" "n")"
            if [[ "${overwrite}" == "y" ]]; then force_args+=("--force"); else
                log_info "Keeping existing config; skipping init."
            fi
        fi
        if [[ ! -e "${config}" || ${#force_args[@]} -gt 0 ]]; then
            cmd_init "${motion_file}" --exp-id "${exp_id}" "${force_args[@]+"${force_args[@]}"}"
        fi
    else
        if (( needs_init )); then
            die "run: cannot continue without a config (init was skipped)"
        fi
        log_info "init skipped"
    fi

    # ----- Stage 2: replay -----
    log_stage "Stage 2/4: replay  [host -> container via ./workflow/run.sh exec]"
    local mfile=""
    if [[ -f "${config}" ]]; then
        mfile="$(yaml_get_top "${config}" motion_file)"
    elif (( DRY_RUN )) && [[ -n "${motion_file}" ]]; then
        mfile="${motion_file}"
        log_info "(dry-run) using in-memory motion_file (config not yet created)"
    fi
    log_info "motion_file: ${mfile:-<unknown>}"
    if container_running; then
        log_info "container '${CONTAINER_NAME}' is running"
    else
        log_warn "container '${CONTAINER_NAME}' is not running"
        local start_it; start_it="$(prompt_yn "Start container with '${WORKFLOW_RUN_SH} start'?" "n")"
        if [[ "${start_it}" == "y" ]]; then
            log_warn "'./workflow/run.sh start' is interactive and drops you into a shell."
            log_warn "Run it in another terminal, then come back and answer 'y' below."
        fi
    fi
    ans="$(prompt_yn "Run replay (viser) for ${exp_id}?" "y")"
    if [[ "${ans}" == "y" ]]; then
        if [[ ! -f "${config}" ]]; then
            log_warn "Skipping replay: ${config} not present (init was skipped or dry-run)"
        elif container_running; then
            cmd_replay "${exp_id}"
            log_info "Replay finished. Note the start/end frames you want."
        else
            log_warn "Skipping replay: container is still not running"
        fi
    else
        log_info "replay skipped — assuming frames are already chosen"
    fi

    # ----- Stage 3: set-frames -----
    log_stage "Stage 3/4: set-frames  [host]"
    local cur_start="" cur_end=""
    if [[ -f "${config}" ]]; then
        cur_start="$(yaml_get_override "${config}" env.commands.motion.motion_start_frame)"
        cur_end="$(yaml_get_override "${config}" env.commands.motion.motion_end_frame)"
        log_info "current motion_start_frame=${cur_start:-<unset>} motion_end_frame=${cur_end:-<unset>}"
    elif (( DRY_RUN )); then
        log_info "(dry-run) config not present yet; defaulting current frames to 0/-1"
        cur_start="0"
        cur_end="-1"
    fi
    ans="$(prompt_yn "Update frame range in ${config}?" "y")"
    if [[ "${ans}" == "y" ]]; then
        local new_start new_end
        new_start="$(prompt_value "motion_start_frame" "${cur_start:-0}")"
        new_end="$(prompt_value "motion_end_frame (-1 = end of motion)" "${cur_end:--1}")"
        if [[ -f "${config}" ]]; then
            if (( DRY_RUN )); then
                log_info "(dry-run) would update ${config} -> motion_start_frame=${new_start} motion_end_frame=${new_end}"
            else
                yaml_set_frames "${config}" "${new_start}" "${new_end}"
                log_info "Updated ${config}"
            fi
        else
            log_warn "Skipping set-frames: ${config} not present"
        fi
    else
        log_info "set-frames skipped"
    fi

    # ----- Stage 4: submit -----
    log_stage "Stage 4/4: submit  [host -> OSMO; osmo CLI not in container]"
    local default_run_name="${exp_id}_motion_data"
    local run_name; run_name="$(prompt_value "OSMO --run-name" "${default_run_name}")"
    local build_ans; build_ans="$(prompt_yn "Pass --build-image?" "n")"
    local submit_args=("${exp_id}" --run-name "${run_name}")
    if [[ "${build_ans}" == "y" ]]; then submit_args+=(--build-image); fi
    local final_cmd="python ${RUN_EXPERIMENT_SCRIPT} ${exp_id} --osmo --run-name ${run_name}"
    [[ "${build_ans}" == "y" ]] && final_cmd+=" --build-image"
    log_info "Resolved command:"
    log_cmd "${final_cmd}"
    ans="$(prompt_yn "Submit now?" "y")"
    if [[ "${ans}" == "y" ]]; then
        cmd_submit "${submit_args[@]}"
    else
        log_info "submit skipped"
    fi

    log_stage "Done"
}

# --- Top-level CLI ---

usage() {
    sed -n '2,51p' "$0"
}

# Banner shown at the top of every real invocation so it's obvious where the
# script runs. Suppressed for help/usage output to avoid duplicate text.
print_host_banner() {
    if running_inside_container; then
        echo "$(_color '1;31' '!!! Running INSIDE the container, but this script is host-only !!!')"
    else
        echo "$(_color '1;32' '[host]') train_tpv_wholebody_workflow.sh — running on host (not in container)"
    fi
}

# Abort if invoked from inside the container. Replay (docker exec) and submit
# (osmo CLI) cannot work from there.
require_host() {
    if running_inside_container; then
        log_error "This script must be run on the HOST, not inside the docker container."
        log_error "Detected container indicators: /.dockerenv or docker cgroup."
        log_error ""
        log_error "Why: 'replay' shells out to './workflow/run.sh exec' (host-only docker"
        log_error "command), and 'submit' calls the OSMO CLI which is not installed in"
        log_error "the container image."
        log_error ""
        log_error "Fix: open a host shell (exit the container) and run again from:"
        log_error "  ${REPO_ROOT}"
        exit 2
    fi
}

parse_global_flags() {
    local out=()
    for tok in "$@"; do
        case "${tok}" in
            --dry-run) DRY_RUN=1 ;;
            --yes|-y)  ASSUME_YES=1 ;;
            *)         out+=("${tok}") ;;
        esac
    done
    REMAINING_ARGS=("${out[@]+"${out[@]}"}")
}

main() {
    if (( $# == 0 )); then
        usage
        exit 1
    fi
    REMAINING_ARGS=()
    parse_global_flags "$@"
    set -- "${REMAINING_ARGS[@]+"${REMAINING_ARGS[@]}"}"
    if (( $# == 0 )); then
        usage
        exit 1
    fi
    local sub="$1"; shift
    case "${sub}" in
        -h|--help|help) usage; exit 0 ;;
    esac
    require_host
    print_host_banner
    case "${sub}" in
        run)        cmd_run "$@" ;;
        init)       cmd_init "$@" ;;
        replay)     cmd_replay "$@" ;;
        set-frames) cmd_set_frames "$@" ;;
        submit)     cmd_submit "$@" ;;
        *)          log_error "unknown subcommand: ${sub}"; usage; exit 1 ;;
    esac
}

main "$@"
