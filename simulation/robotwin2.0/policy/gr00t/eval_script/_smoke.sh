#!/bin/bash
set -u
V=/data/vepfs/licheng.yang/DRRM-VODPP/simulation/robotwin2.0
export ROBOTWIN_SEED_LIST=/data/vepfs/licheng.yang/tmp/smoke8.txt
export GR00T_PY=/data/vepfs/licheng.yang/pythons/miniconda3/bin/python3.12
export GR00T_GPUS=4 GR00T_PORT_BASE=5555
export EVAL_RESULT_ROOT=/data/vepfs/licheng.yang/tmp/groot_smoke
cd "$V"
bash policy/gr00t/eval_script/eval_gr00t_multi_gpu.sh \
  blocks_ranking_rgb demo_clean n17_robotwin50_clean_bgr2rgb_h32_40000 0 8 "smoke-u4:uniform4"
