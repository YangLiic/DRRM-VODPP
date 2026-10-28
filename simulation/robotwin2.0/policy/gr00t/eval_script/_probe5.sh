#!/bin/bash
V=/data/vepfs/licheng.yang/DRRM-VODPP/simulation/robotwin2.0
SITE=$V/.venv/lib/python3.10/site-packages
EXTRA=$(cat "$SITE"/__editable__*.pth 2>/dev/null | grep -E '^/' | paste -sd: -)
echo "editable 路径: $EXTRA"
cd "$V"
PYTHONPATH="$SITE:$EXTRA:$V" /usr/bin/python3.10 -c "
from envs.robot.planner import CuroboPlanner
import envs.blocks_ranking_rgb as m
print('  CuroboPlanner OK, 任务环境可导入 OK')" 2>&1 | tail -5
