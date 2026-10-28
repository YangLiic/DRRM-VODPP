#!/bin/bash
V=/data/vepfs/licheng.yang/DRRM-VODPP/simulation/robotwin2.0
echo "hostname: $(hostname)"
echo "robotwin venv python 可执行: $([ -x $V/.venv/bin/python ] && echo 是 || echo 否)"
ls -la $V/.venv/bin/python 2>/dev/null
echo "--- 目标是否存在 ---"
ls -la /root/miniconda3/envs/robotics_manipulation/bin/python 2>/dev/null || echo "  /root/miniconda3/envs/robotics_manipulation 不存在"
echo "--- 能否 import sapien/numpy ---"
$V/.venv/bin/python -c "import numpy, sapien; print('numpy', numpy.__version__, 'sapien ok')" 2>&1 | tail -2
