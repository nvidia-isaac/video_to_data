#!/usr/bin/env bash
set -euo pipefail

usage() {
    cat <<'EOF'
Run a GR00T server and an embodiment-parameterized IsaacLab closed-loop evaluation.

Required:
  --gr00t-dir PATH
  --model PATH
  --container NAME
  --expected-mount-source PATH
  --client-workdir PATH
  --task TASK_ID
  --contract PATH
  --task-profile PATH
  --motion-file PATH
  --human-motion-data-dir PATH
  --expected-sequence-id ID
  --expected-robot-name NAME
  --output-json PATH
  --episodes N
  --num-envs N
  --episode-horizon N

Optional:
  --port N                       Default: 5555
  --embodiment-tag TAG           Default: new_embodiment
  --execution-length N           Default: 4
  --model-seed N                 Default: 0
  --visual-warmup-steps N        Default: 60
  --visual-mode training|off     Default: training
  --success-video-dir PATH
  --success-video-camera NAME    Default: contract's first camera
  --max-success-videos N         Default: 3
  --server-log PATH              Default: /tmp/groot-server-PORT.log
  --client-extra-arg ARG         Repeat for additional client arguments
  --stop-container               Stop the supplied container during cleanup
  --dry-run                      Print escaped commands without executing
EOF
}

die() {
    echo "error: $*" >&2
    exit 2
}

require_value() {
    [[ $# -ge 2 ]] || die "$1 requires a value"
}

gr00t_dir=""
model=""
container=""
expected_mount_source=""
client_workdir=""
task=""
contract=""
task_profile=""
motion_file=""
human_motion_data_dir=""
expected_sequence_id=""
expected_robot_name=""
output_json=""
episodes=""
num_envs=""
episode_horizon=""
port="5555"
embodiment_tag="new_embodiment"
execution_length="4"
model_seed="0"
visual_warmup_steps="60"
visual_mode="training"
success_video_dir=""
success_video_camera=""
max_success_videos="3"
server_log=""
stop_container="0"
dry_run="0"
client_extra_args=()

while [[ $# -gt 0 ]]; do
    case "$1" in
        --gr00t-dir)
            require_value "$@"
            gr00t_dir="$2"
            shift 2
            ;;
        --model)
            require_value "$@"
            model="$2"
            shift 2
            ;;
        --container)
            require_value "$@"
            container="$2"
            shift 2
            ;;
        --expected-mount-source)
            require_value "$@"
            expected_mount_source="$2"
            shift 2
            ;;
        --client-workdir)
            require_value "$@"
            client_workdir="$2"
            shift 2
            ;;
        --task)
            require_value "$@"
            task="$2"
            shift 2
            ;;
        --contract)
            require_value "$@"
            contract="$2"
            shift 2
            ;;
        --task-profile)
            require_value "$@"
            task_profile="$2"
            shift 2
            ;;
        --motion-file)
            require_value "$@"
            motion_file="$2"
            shift 2
            ;;
        --human-motion-data-dir)
            require_value "$@"
            human_motion_data_dir="$2"
            shift 2
            ;;
        --expected-sequence-id)
            require_value "$@"
            expected_sequence_id="$2"
            shift 2
            ;;
        --expected-robot-name)
            require_value "$@"
            expected_robot_name="$2"
            shift 2
            ;;
        --output-json)
            require_value "$@"
            output_json="$2"
            shift 2
            ;;
        --episodes)
            require_value "$@"
            episodes="$2"
            shift 2
            ;;
        --num-envs)
            require_value "$@"
            num_envs="$2"
            shift 2
            ;;
        --episode-horizon)
            require_value "$@"
            episode_horizon="$2"
            shift 2
            ;;
        --port)
            require_value "$@"
            port="$2"
            shift 2
            ;;
        --embodiment-tag)
            require_value "$@"
            embodiment_tag="$2"
            shift 2
            ;;
        --execution-length)
            require_value "$@"
            execution_length="$2"
            shift 2
            ;;
        --model-seed)
            require_value "$@"
            model_seed="$2"
            shift 2
            ;;
        --visual-warmup-steps)
            require_value "$@"
            visual_warmup_steps="$2"
            shift 2
            ;;
        --visual-mode)
            require_value "$@"
            visual_mode="$2"
            shift 2
            ;;
        --success-video-dir)
            require_value "$@"
            success_video_dir="$2"
            shift 2
            ;;
        --success-video-camera)
            require_value "$@"
            success_video_camera="$2"
            shift 2
            ;;
        --max-success-videos)
            require_value "$@"
            max_success_videos="$2"
            shift 2
            ;;
        --server-log)
            require_value "$@"
            server_log="$2"
            shift 2
            ;;
        --client-extra-arg)
            require_value "$@"
            client_extra_args+=("$2")
            shift 2
            ;;
        --stop-container)
            stop_container="1"
            shift
            ;;
        --dry-run)
            dry_run="1"
            shift
            ;;
        --help|-h)
            usage
            exit 0
            ;;
        *)
            die "unknown argument: $1"
            ;;
    esac
done

for required_name in \
    gr00t_dir model container expected_mount_source client_workdir task contract task_profile \
    motion_file human_motion_data_dir expected_sequence_id expected_robot_name \
    output_json episodes num_envs episode_horizon
do
    [[ -n "${!required_name}" ]] || die "missing --${required_name//_/-}"
done

[[ "$episodes" =~ ^[1-9][0-9]*$ ]] || die "--episodes must be positive"
[[ "$num_envs" =~ ^[1-9][0-9]*$ ]] || die "--num-envs must be positive"
[[ "$episode_horizon" =~ ^[1-9][0-9]*$ ]] || die "--episode-horizon must be positive"
[[ "$port" =~ ^[1-9][0-9]*$ ]] || die "--port must be positive"
[[ "$model_seed" =~ ^[0-9]+$ ]] || die "--model-seed must be non-negative"
[[ "$visual_mode" == "training" || "$visual_mode" == "off" ]] \
    || die "--visual-mode must be training or off"

waves=$(( (episodes + num_envs - 1) / num_envs ))
num_steps=$(( waves * episode_horizon + 100 ))
[[ -n "$server_log" ]] || server_log="/tmp/groot-server-${port}.log"

gr00t_revision="$(git -C "$gr00t_dir" rev-parse HEAD 2>/dev/null || true)"
[[ -n "$gr00t_revision" ]] || gr00t_revision="unknown"
gr00t_dirty="false"
if [[ -n "$(git -C "$gr00t_dir" status --porcelain --untracked-files=no 2>/dev/null || true)" ]]; then
    gr00t_dirty="true"
fi
checkpoint_manifest_sha256=""
if [[ -d "$model" ]]; then
    checkpoint_manifest_material=""
    checkpoint_weight_count="0"
    for identity_file in \
        model.safetensors.index.json processor_config.json config.json \
        statistics.json embodiment_id.json
    do
        if [[ -f "$model/$identity_file" ]]; then
            identity_sha256="$(sha256sum "$model/$identity_file" | cut -d ' ' -f 1)"
            checkpoint_manifest_material+="${identity_file}:${identity_sha256}"$'\n'
        fi
    done
    shopt -s nullglob
    checkpoint_weight_files=("$model"/model*.safetensors)
    shopt -u nullglob
    for weight_file in "${checkpoint_weight_files[@]}"; do
        weight_name="$(basename "$weight_file")"
        weight_sha256="$(sha256sum "$weight_file" | cut -d ' ' -f 1)"
        checkpoint_manifest_material+="${weight_name}:${weight_sha256}"$'\n'
        checkpoint_weight_count=$((checkpoint_weight_count + 1))
    done
    [[ "$checkpoint_weight_count" -gt 0 ]] \
        || die "checkpoint contains no model*.safetensors weight shards: $model"
    checkpoint_manifest_sha256="$(printf '%s' "$checkpoint_manifest_material" | sha256sum | cut -d ' ' -f 1)"
elif [[ -f "$model" ]]; then
    checkpoint_manifest_sha256="$(sha256sum "$model" | cut -d ' ' -f 1)"
fi

script_dir="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd -P)"
seeded_server="${script_dir}/seeded_gr00t_server.py"
pending_output_json="${output_json}.pending-${port}"

server_cmd=(
    env PYTHONUNBUFFERED=1 uv run python "$seeded_server"
    --seed "$model_seed"
    --model-path "$model"
    --embodiment-tag "$embodiment_tag"
    --port "$port"
)

client_cmd=(
    env "HUMAN_MOTION_DATA_DIR=${human_motion_data_dir}"
    python scripts/rsl_rl/gr00t_infer.py
    --headless
    --task "$task"
    --contract "$contract"
    --task_profile "$task_profile"
    --motion_file "$motion_file"
    --require_partitioned_motion
    --require_support_surface
    --expected_sequence_id "$expected_sequence_id"
    --expected_robot_name "$expected_robot_name"
    --checkpoint_path "$model"
    --checkpoint_manifest_sha256 "$checkpoint_manifest_sha256"
    --model_seed "$model_seed"
    --gr00t_revision "$gr00t_revision"
    --gr00t_dirty "$gr00t_dirty"
    --num_envs "$num_envs"
    --num_steps "$num_steps"
    --eval_episodes "$episodes"
    --eval_episode_horizon "$episode_horizon"
    --eval_results "$pending_output_json"
    --gr00t_host localhost
    --gr00t_port "$port"
    --execution_length "$execution_length"
    --visual_warmup_steps "$visual_warmup_steps"
    --visual_mode "$visual_mode"
    --save_traj ""
    --save_cam_video ""
)

if [[ -n "$success_video_dir" ]]; then
    client_cmd+=(
        --save_success_videos_dir "$success_video_dir"
        --max_success_videos "$max_success_videos"
    )
    if [[ -n "$success_video_camera" ]]; then
        client_cmd+=(--success_video_camera "$success_video_camera")
    fi
fi
client_cmd+=("${client_extra_args[@]}")

printf -v escaped_server ' %q' "${server_cmd[@]}"
printf -v escaped_client ' %q' "${client_cmd[@]}"
printf -v escaped_gr00t_dir '%q' "$gr00t_dir"
printf -v escaped_client_workdir '%q' "$client_workdir"

if [[ "$dry_run" == "1" ]]; then
    echo "SERVER: cd ${escaped_gr00t_dir} &&${escaped_server}"
    echo "CLIENT: docker exec $(printf '%q' "$container") bash -lc $(printf '%q' "cd ${escaped_client_workdir} && HEADLESS=1${escaped_client}")"
    echo "PLAN: waves=${waves} num_steps=${num_steps} server_log=${server_log}"
    exit 0
fi

[[ -d "$gr00t_dir" ]] || die "GR00T directory does not exist: $gr00t_dir"
[[ -e "$model" ]] || die "model does not exist: $model"
[[ -f "$seeded_server" ]] || die "seeded GR00T server wrapper does not exist: $seeded_server"
command -v docker >/dev/null || die "docker is unavailable"
docker inspect "$container" >/dev/null 2>&1 || die "container does not exist: $container"
[[ "$(docker inspect -f '{{.State.Running}}' "$container")" == "true" ]] \
    || die "container is not running: $container"
[[ "$(docker inspect -f '{{.HostConfig.NetworkMode}}' "$container")" == "host" ]] \
    || die "container must use host networking"

mount_sources="$(docker inspect -f '{{range .Mounts}}{{println .Source}}{{end}}' "$container")"
grep -Fxq "$expected_mount_source" <<<"$mount_sources" \
    || die "container is not mounted from expected source: $expected_mount_source"

docker exec "$container" bash -lc \
    "cd ${escaped_client_workdir} && python -c 'import zmq, msgpack_numpy'" \
    >/dev/null \
    || die "closed-loop dependencies are missing in the IsaacLab interpreter"

# Never let a prior result satisfy the post-run check after an Isaac launcher masks a
# client exception. The evaluator writes a run-specific pending path which is promoted
# only after it completes.
docker exec "$container" rm -f "$pending_output_json"

server_pid=""
cleanup() {
    status=$?
    if [[ -n "$server_pid" ]] && kill -0 "$server_pid" 2>/dev/null; then
        kill -INT "$server_pid" 2>/dev/null || true
        wait "$server_pid" 2>/dev/null || true
    fi
    docker exec "$container" rm -f "$pending_output_json" >/dev/null 2>&1 || true
    if [[ "$stop_container" == "1" ]]; then
        docker stop "$container" >/dev/null 2>&1 || true
    fi
    exit "$status"
}
trap cleanup EXIT INT TERM

mkdir -p "$(dirname "$server_log")"
(
    cd "$gr00t_dir"
    exec "${server_cmd[@]}"
) >"$server_log" 2>&1 &
server_pid=$!

ready="0"
for _ in $(seq 1 180); do
    if ! kill -0 "$server_pid" 2>/dev/null; then
        tail -100 "$server_log" >&2 || true
        die "GR00T server exited before becoming ready"
    fi
    if grep -q "Server ready" "$server_log"; then
        ready="1"
        break
    fi
    sleep 1
done
[[ "$ready" == "1" ]] || die "GR00T server did not become ready within 180 seconds"

docker exec "$container" bash -lc \
    "cd ${escaped_client_workdir} && HEADLESS=1${escaped_client}"

# Isaac Sim's python launcher can mask an uncaught client exception with exit status 0.
# The evaluator writes this file only after it has completed and validated the requested
# episode count, so require that durable result before reporting success.
docker exec "$container" test -s "$pending_output_json" \
    || die "closed-loop client finished without a non-empty result: $pending_output_json"
docker exec "$container" mv "$pending_output_json" "$output_json"

echo "[INFO] closed-loop evaluation complete: $output_json"
