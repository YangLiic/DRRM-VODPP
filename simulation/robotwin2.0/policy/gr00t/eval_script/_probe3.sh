#!/bin/bash
V=/data/vepfs/licheng.yang/DRRM-VODPP/simulation/robotwin2.0
SITE=$V/.venv/lib/python3.10/site-packages
echo "=== python3.10 + venv site-packages ==="
PYTHONPATH="$SITE" /usr/bin/python3.10 -c "
import numpy, sapien, zmq, msgpack_numpy
print('  numpy', numpy.__version__, '| sapien OK | zmq', zmq.__version__, '| msgpack_numpy OK')" 2>&1 | tail -3
echo "=== 系统上有哪些 python ==="
ls /usr/bin/python3* 2>/dev/null
