#!/bin/bash
V=/data/vepfs/licheng.yang/DRRM-VODPP/simulation/robotwin2.0
SITE=$V/.venv/lib/python3.10/site-packages
cd "$V"
echo "=== 直接 import curobo, 看真实报错 ==="
PYTHONPATH="$SITE:$V" /usr/bin/python3.10 -c "import curobo; print('  curobo', curobo.__file__)" 2>&1 | tail -6
echo "=== planner.py 为什么没有 CuroboPlanner ==="
PYTHONPATH="$SITE:$V" /usr/bin/python3.10 -c "
import traceback
try:
    from envs.robot.planner import CuroboPlanner; print('  OK')
except Exception:
    traceback.print_exc()" 2>&1 | tail -12
echo "=== pi05 跑通时 PYTHONPATH 是什么(从它的 worker 日志) ==="
R=/DRoboticsAILab/users/licheng.yang/robotwin2.0/result
d=$(ls -dt $R/dump_bin_bigbin/pi05/demo_clean/*/*/ 2>/dev/null | head -1)
grep -ohE 'PYTHONPATH=[^ ]*' $d/_worker_1gpu.log 2>/dev/null | head -2
