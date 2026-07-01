#!/bin/bash
# Re-evaluate ALL models on the CURRENT hammer success metric (one Isaac Sim at a time, GPU-gated so
# it never collides with a user collection). Appends "label<TAB>success%<TAB>group<TAB>ckpt" to a TSV.
set -u
cd /home/cning/simtoolreal_isaaclab
GROOT=/home/cning/simtoolreal_isaaclab/Isaac-GR00T/.venv/bin/activate
ISAAC=/home/cning/isaaclab/env_isaaclab/bin/activate
GS=/home/cning/simtoolreal_isaaclab/logs/gr00t_specialist
TSV=logs/all_eval_results.tsv
PROG=logs/eval_all_progress.log
: > "$TSV"; : > "$PROG"
ND="--headless --num_envs 25 --episodes 100 --replan 1"
log(){ echo "$(date +%H:%M) $*" >> "$PROG"; }

gpu_wait(){  # block until NO GPU compute app (i.e. no user collection); my own servers aren't up yet at call time
  for _ in $(seq 1 800); do
    [ -z "$(nvidia-smi --query-compute-apps=pid --format=csv,noheader 2>/dev/null)" ] && return 0
    sleep 15
  done
}
rate(){ grep -oE "success_rate [0-9.]+%" "$1" 2>/dev/null | tail -1 | grep -oE "[0-9.]+"; }

# generic server+client eval: $1 label  $2 group  $3 ckpt  $4 server.py  $5 client.py  $6 port
eval_srv(){
  gpu_wait; fuser -k "$6"/tcp 2>/dev/null; sleep 2
  ( source "$GROOT" && python "scripts/$4" --checkpoint "$3" --port "$6" > /tmp/ea_srv.log 2>&1 ) & SV=$!
  for _ in $(seq 1 120); do grep -q "listening on" /tmp/ea_srv.log 2>/dev/null && break; sleep 2; done
  ( source "$ISAAC" && cd /home/cning/isaaclab/IsaacLab && OMNI_KIT_ACCEPT_EULA=YES \
    ./isaaclab.sh -p "/home/cning/simtoolreal_isaaclab/scripts/$5" $ND --port "$6" > /tmp/ea_cli.log 2>&1 )
  R=$(rate /tmp/ea_cli.log); echo -e "$1\t${R:-NA}\t$2\t$3" >> "$TSV"; log "$1 = ${R:-NA}%"
  kill -9 $SV 2>/dev/null; fuser -k "$6"/tcp 2>/dev/null; sleep 3
}
eval_expert(){
  gpu_wait
  ( source "$ISAAC" && cd /home/cning/isaaclab/IsaacLab && OMNI_KIT_ACCEPT_EULA=YES \
    ./isaaclab.sh -p /home/cning/simtoolreal_isaaclab/scripts/eval_expert.py --headless --num_envs 25 --episodes 100 > /tmp/ea_cli.log 2>&1 )
  R=$(rate /tmp/ea_cli.log); echo -e "SAPG expert\t${R:-NA}\texpert\t(pretrained)" >> "$TSV"; log "expert = ${R:-NA}%"
}

log "=== eval-all start ==="
eval_expert
eval_srv "image (DINOv3)"      image      "$GS/hammer/hammer.pt"                     eval_specialist_server.py eval_specialist_client.py 5599
eval_srv "keypoint"            keypoint   "$GS/hammer_kp/hammer_kp.pt"               eval_keypoint_server.py   eval_keypoint_client.py   5601
eval_srv "simtoolreal-BC"      simtoolreal "$GS/hammer_str/hammer_str.pt"            eval_simtoolreal_server.py eval_simtoolreal_client.py 5602
eval_srv "50M+tele 50k"        teleport   "$GS/hammer_str_tele2/hammer_str_tele2_step20000.pt" eval_simtoolreal_server.py eval_simtoolreal_client.py 5602
eval_srv "50M+tele 60k"        teleport   "$GS/hammer_str_tele2/hammer_str_tele2_step30000.pt" eval_simtoolreal_server.py eval_simtoolreal_client.py 5602
eval_srv "50M+tele 70k"        teleport   "$GS/hammer_str_tele2/hammer_str_tele2.pt" eval_simtoolreal_server.py eval_simtoolreal_client.py 5602
eval_srv "150M+tele 20k"       teleport   "$GS/hammer_str_tele_150m/hammer_str_tele_150m_step20000.pt" eval_simtoolreal_server.py eval_simtoolreal_client.py 5602
eval_srv "150M+tele 30k"       teleport   "$GS/hammer_str_tele_150m/hammer_str_tele_150m_step30000.pt" eval_simtoolreal_server.py eval_simtoolreal_client.py 5602
eval_srv "150M+tele 40k"       teleport   "$GS/hammer_str_tele_150m/hammer_str_tele_150m_step40000.pt" eval_simtoolreal_server.py eval_simtoolreal_client.py 5602
log "=== EVAL-ALL DONE ==="
cat "$TSV"
