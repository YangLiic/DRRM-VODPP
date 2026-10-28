#!/bin/bash
V=/data/vepfs/licheng.yang/DRRM-VODPP/simulation/robotwin2.0
echo "=== 系统 python3.10 能否 import sapien(客户端需要) ==="
/usr/bin/python3.10 -c "import numpy, sapien; print('  numpy', numpy.__version__, '| sapien OK')" 2>&1 | tail -3
echo "=== pyzmq / msgpack-numpy (GR00T 客户端需要) ==="
/usr/bin/python3.10 -c "import zmq, msgpack_numpy; print('  zmq', zmq.__version__, '| msgpack_numpy OK')" 2>&1 | tail -3
echo "=== EFS python3.12 能否 import gr00t(服务端需要) ==="
PYTHONPATH=/DRoboticsAILab/vepfs_20260622/licheng.yang/Isaac-GR00T/.venv/lib/python3.12/site-packages:/DRoboticsAILab/vepfs_20260622/licheng.yang/Isaac-GR00T \
  /data/vepfs/licheng.yang/pythons/miniconda3/bin/python3.12 -c "import gr00t; print('  gr00t OK')" 2>&1 | tail -2
