# pi0.5 FM tau-schedule 评测编排

默认 **并行**：一卡一个 schedule，各自起独立 openpi server（端口 `PI05_PORT_BASE + gpu_index`）。  
`--serial`：一卡串行（起 server → eval → 停 server）。

## 时间语义（重要）

| | VODPP | **pi0.5（本目录）** |
|--|--------|---------------------|
| 方向 | τ: 0 → 1 | **t: 1 → 0** |
| 默认 10 NFE | `0,0.1,...,1` | `uniform10` → `1,0.9,...,0` |
| 自定义示例 | `0,0.3,0.7,1` | **`1,0.7,0.3,0`** |

解析器：`../fm_schedule.py`。若把 VODPP/TrustEuler 的 0→1 节点原样传入会 **直接报错**；请先做 `τ_pi = 1 - τ_vodpp`。

## 脚本

| 文件 | 用途 |
|------|------|
| `eval_pi05_tau_sweep.sh` | 本地编排（默认并行；`--serial` 串行） |
| `eval_pi05_tau_sweep_cloud.sh` | 云端：source `scripts/eval/_robotwin_cloud_render_env.sh` 后调本地脚本 |

单次评测仍走 `../run_eval.sh`（内部 `serve.sh` 把 `PI05_FM_TAU_SCHEDULE` 解析成节点，传给 openpi `--fm-tau-schedule`）。

## 用法

本地并行（8 卡 × 8 网格）：

```bash
cd /data/vepfs/licheng.yang/DRRM-VODPP/simulation/robotwin2.0

PI05_CKPT=/DRoboticsAILab/users/licheng.yang/checkpoints/openpi/pi05_robotwin2.0_agilex/robotwin2.0_agilex_3task/35000 \
TEST_NUM=100 bash policy/pi05/eval_script/eval_pi05_tau_sweep.sh \
  adjust_bottle agilex_config 3task_35000 0 100 \
  sharp3:1,0.581,0.232,0 \
  sharp5:1,0.741,0.506,0.297,0.118,0 \
  sharp8:1,0.836,0.68,0.534,0.397,0.273,0.157,0.064,0 \
  sharp10:1,0.868,0.741,0.62,0.506,0.397,0.297,0.2,0.118,0.048,0 \
  uniform3:uniform3 \
  uniform5:uniform5 \
  uniform8:uniform8 \
  uniform10:uniform10
```

串行（一卡）：

```bash
bash policy/pi05/eval_script/eval_pi05_tau_sweep.sh --serial \
  adjust_bottle agilex_config 3task_35000 0 2 \
  uniform10:uniform10 \
  uniform3:uniform3
```

云 / Volcano（A800 八卡入口见 `dag_tgcontrolA800.py`）：

```bash
cd /data/vepfs/licheng.yang/DRRM-VODPP \
  && export PI05_CKPT=/DRoboticsAILab/users/licheng.yang/checkpoints/openpi/pi05_robotwin2.0_agilex/robotwin2.0_agilex_3task/35000 \
  && bash simulation/robotwin2.0/policy/pi05/eval_script/eval_pi05_tau_sweep_cloud.sh \
    adjust_bottle agilex_config 3task_35000 0 100 \
    ...
```

## 参数

```text
[--serial] <task> <task_config> <ckpt_setting> <seed> <test_num> <name:schedule> [...]
```

- `name` → `EVAL_RUN_NAME`（结果子目录名）
- `schedule` → `PI05_FM_TAU_SCHEDULE`（`uniformN` 或 `1,...,0`）
- 同 task 下所有 schedule **共用同一个 seed**（与 TrustEuler / VODPP `SEED_BASE` 逻辑一致；默认 `0` → `st_seed=100000`，`episode_k` 可比）
- 并行：GPU `i` → 端口 `PI05_PORT_BASE+i`（默认 8000+i），各自独立 server

## 环境变量

| 变量 | 说明 |
|------|------|
| `RESULT_ROOT` / `EVAL_RESULT_ROOT` | 结果根目录 |
| `GPU_ID` | 串行模式 GPU（默认 0） |
| `PI05_PORT` | 串行模式端口（默认 8000） |
| `PI05_PORT_BASE` | 并行模式端口基址（默认 8000） |
| `PI05_CKPT` / `OPENPI_ROOT` / `PI05_CONFIG` | 同 `../README.md` |

## 和 TrustEuler 网格的关系

OpenPI 校准产物在：

`/DRoboticsAILab/users/licheng.yang/robotwin2.0/TrustEuler_runs/pi05/<task>/`

TrustEuler `tau_nodes` 为 **0→1**；本目录 schedule 为 pi0.5 **1→0**，使用前做 `τ_pi = 1 - τ_te`。
说明见 TrustEuler `docs/ROBOTWIN_PI05_OPENPI_ZH.md`。

结果目录：

```text
$RESULT_ROOT/<task>/pi05/<task_config>/<run_name>/<timestamp>/
```

Launcher 日志：`$RESULT_ROOT/<task>/_launcher_logs/pi05_tau_sweep_*.log`
