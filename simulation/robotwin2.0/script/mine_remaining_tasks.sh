#!/bin/bash
# Mine demo_clean eval seed lists for every RoboTwin task that lacks one.
#
# max_walk is raised from the 2000 default to 6000: the three short lists on disk
# (open_microwave 92, grab_roller 99, open_laptop 99) all ran out of walk rather than
# out of valid scenes, and a short list is a hard error at eval time -- a wasted arm.
# Accepted/walked is logged per task so a shortfall can be read as rate-limited
# (raise max_walk again) or asset-limited (nothing to raise).
set -u
# Absolute paths and an explicit cd: this script is launched detached (tmux), whose cwd
# is not the RoboTwin tree, and the cloud wrapper is addressed relatively.
RW=/data/vepfs/licheng.yang/DRRM-VODPP/simulation/robotwin2.0
cd "$RW" || { echo "cannot cd $RW"; exit 1; }
OUT=/DRoboticsAILab/users/licheng.yang/robotwin2.0/seedlists
LOGD=/DRoboticsAILab/users/licheng.yang/robotwin2.0/seedlists/_mining_logs
mkdir -p "$LOGD"
TASKS=(adjust_bottle blocks_ranking_size click_alarmclock handover_mic move_can_pot
       move_pillbottle_pad move_playingcard_away move_stapler_pad place_a2b_left
       place_a2b_right place_bread_skillet place_burger_fries place_can_basket
       place_cans_plasticbox place_container_plate place_empty_cup place_mouse_pad
       place_object_basket place_object_scale place_phone_stand place_shoe
       press_stapler put_bottles_dustbin shake_bottle_horizontally stack_bowls_two)
echo "=== mining ${#TASKS[@]} tasks, 100 seeds each, max_walk 6000 ==="
ok=0; short=0; failed=0
for t in "${TASKS[@]}"; do
  f="$OUT/${t}_demo_clean.txt"
  if [ -f "$f" ] && [ "$(wc -w < "$f")" -ge 100 ]; then
    echo "[skip] $t already has $(wc -w < "$f")"; ok=$((ok+1)); continue
  fi
  echo "--- $(date -u +%H:%M:%S) mining $t ---"
  bash "$RW/script/mine_eval_seeds_cloud.sh" --task_name "$t" --task_config demo_clean \
      --test_num 100 --max_walk 6000 --out "$f" > "$LOGD/${t}.log" 2>&1
  rc=$?
  n=0; [ -f "$f" ] && n=$(wc -w < "$f")
  if [ "$n" -ge 100 ]; then echo "  OK $t: $n"; ok=$((ok+1))
  elif [ "$n" -gt 0 ]; then echo "  SHORT $t: $n (rc=$rc)"; short=$((short+1))
  else echo "  FAILED $t (rc=$rc): $(tail -2 "$LOGD/${t}.log" | head -1 | cut -c1-90)"; failed=$((failed+1)); fi
done
echo "=== mining done: ok=$ok short=$short failed=$failed ==="
for t in "${TASKS[@]}"; do
  f="$OUT/${t}_demo_clean.txt"; [ -f "$f" ] && printf "  %-28s %s\n" "$t" "$(wc -w < "$f")"
done
