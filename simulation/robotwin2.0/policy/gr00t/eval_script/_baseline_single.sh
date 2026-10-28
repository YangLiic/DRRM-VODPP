#!/bin/bash
# 用你原本跑通的单卡路径(run_eval.sh)做对照, 同样 8 个 seed、原生 4 步、不注入调度。
set -u
V=/data/vepfs/licheng.yang/DRRM-VODPP/simulation/robotwin2.0
SITE=$V/.venv/lib/python3.10/site-packages
EXTRA=$(cat "$SITE"/__editable__*.pth 2>/dev/null | grep -E '^/' | paste -sd: -)
cd "$V"
export ROBOTWIN_SEED_LIST=/data/vepfs/licheng.yang/tmp/smoke8.txt
export TEST_NUM=8 ROBOTWIN_DESC_NUM=8
export GR00T_PY=/data/vepfs/licheng.yang/pythons/miniconda3/bin/python3.12
export PROBE_PY=$GR00T_PY
export PYTHON=/usr/bin/python3.10
export PYTHONPATH="$SITE:$EXTRA"
export EVAL_RESULT_ROOT=/data/vepfs/licheng.yang/tmp/groot_base
unset GR00T_TAU_NODES            # 原生 4 步, 不注入
export EVAL_RUN_NAME=native4
bash policy/gr00t/run_eval.sh blocks_ranking_rgb demo_clean n17_robotwin50_clean_bgr2rgb_h32_40000 0 0
