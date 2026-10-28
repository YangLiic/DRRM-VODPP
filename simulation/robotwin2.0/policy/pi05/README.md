# pi05 — OpenPI π₀.₅ × RoboTwin 2.0 评测

Client–server（方案 A）：openpi 进程提供策略服务，RoboTwin 只做仿真 + websocket 推理，避免 Sapien 与 openpi 环境冲突。

支持任务（与三任务联合训练权重一致）：`adjust_bottle` / `scan_object` / `beat_block_hammer`。

## 架构

```
RoboTwin (.venv)                         openpi (.venv)
policy/pi05/deploy_policy.py  --ws-->  scripts/serve_policy.py
  encode_obs + replan                    config: pi05_robotwin2.0_agilex
  take_action(qpos)                      ckpt: .../25000/model.safetensors
```

- 观测：`head/left/right` RGB（uint8 HWC）+ 14 维 joint state  
- 动作：服务端输出绝对 **qpos 14 维**（训练时 delta，推理已还原）  
- 默认 `replan_steps=8`（训练 `action_horizon=32`）

## 文件

| 文件 | 作用 |
|------|------|
| `deploy_policy.py` | RoboTwin 评测接口：`get_model` / `encode_obs` / `eval` / `reset_model` |
| `deploy_policy.yml` | 默认 host/port/replan/prompt |
| `ws_client.py` | 轻量 websocket 客户端（msgpack），不依赖完整 openpi |
| `fm_schedule.py` | pi0.5 FM τ 解析（**1→0**）：`uniformN` / 显式节点 |
| `serve.sh` | 启动 openpi 策略服务（可绑 `PI05_FM_TAU_SCHEDULE`） |
| `eval.sh` | 仅客户端评测（需服务已就绪） |
| `run_eval.sh` | **单卡一键**：后台起 server → 等端口 → eval → 退出清 server |
| `eval_script/` | 串行 tau sweep（本地 / 云） |

云入口（DRRM-VODPP 根目录）：`scripts/eval/eval_robotwin2.0_pi05_cloud.sh`

**FM tau schedule 串行 sweep**：`policy/pi05/eval_script/`（本地 / 云各一脚本；时间语义为 pi0.5 的 **1→0**，见该目录 README）。

## 依赖

**服务端**（openpi 自带 `.venv`，一般不用再装）：

- `OPENPI_ROOT` 默认：`/DRoboticsAILab/vepfs_20260622/licheng.yang/openpi`

**客户端**（RoboTwin `.venv`，含 sapien；`eval.sh` 会自动选用该解释器）：

```bash
# 在 simulation/robotwin2.0 下
.venv/bin/pip install msgpack websockets
```

本机勿用 `(base)` conda 的 `python` 跑评测，会缺 `sapien`。云端通过 `PYTHON` / `_robotwin_cloud_render_env.sh` 覆盖即可。

## 默认权重

```
/DRoboticsAILab/users/licheng.yang/checkpoints/openpi/pi05_robotwin2.0_agilex/robotwin2.0_agilex_3task/25000
```

需含 `model.safetensors` 与  
`assets/adjust_bottle__scan_object__beat_block_hammer/norm_stats.json`。

预训练 base（`openpi/checkpoints/pi05_base*`）仅训练初始化用；评测微调模型时**不必**加载。

## 本机（1×L4）用法

L4 约 23GB，pi0.5 推理通常可放下；`run_eval.sh` 让 server 与仿真共用同一张卡。

```bash
cd /data/vepfs/licheng.yang/DRRM-VODPP/simulation/robotwin2.0

# 推荐：一键
bash policy/pi05/run_eval.sh adjust_bottle
bash policy/pi05/run_eval.sh scan_object
bash policy/pi05/run_eval.sh beat_block_hammer

# 完整参数
# bash run_eval.sh <task> [task_config] [ckpt_setting] [seed] [gpu_id]
bash policy/pi05/run_eval.sh adjust_bottle agilex_config 3task_25000 0 0
```

分两步（调试 server 时）：

```bash
# 终端 1：起服务（首次加载可能要几分钟）
bash policy/pi05/serve.sh

# 终端 2：评测
bash policy/pi05/eval.sh adjust_bottle agilex_config 3task_25000 0 0
```

结果目录（默认）：

```
/DRoboticsAILab/users/licheng.yang/robotwin2.0/result/<task>/pi05/<task_config>/<ckpt_setting>/<timestamp>/
```

可用 `EVAL_RESULT_ROOT` 覆盖；不设时与 VODPP 评测同一 `result` 根目录。

## 火山引擎（1×GPU job）

Job command 示例：

```bash
cd /data/vepfs/licheng.yang/DRRM-VODPP \
  && bash scripts/eval/eval_robotwin2.0_pi05_cloud.sh adjust_bottle agilex_config 3task_25000 0 0
```

该脚本会 source 云端 Sapien/EGL 环境，再调用本目录 `run_eval.sh`。

## 环境变量

| 变量 | 默认 | 说明 |
|------|------|------|
| `OPENPI_ROOT` | 见上 | openpi 仓库路径 |
| `PI05_CKPT` | `.../25000` | 微调权重目录 |
| `PI05_CONFIG` | `pi05_robotwin2.0_agilex` | openpi TrainConfig 名 |
| `PI05_HOST` / `PI05_PORT` | `127.0.0.1` / `8000` | 服务地址 |
| `ROBOTWIN_REPO_ID` | `adjust_bottle,scan_object,beat_block_hammer` | 须与训练一致（norm `asset_id`） |
| `PI05_REPLAN_STEPS` | `8` | 每段开环执行步数 |
| `PI05_PROMPT_MODE` | `train_short` | 见下 |
| `INSTRUCTION_TYPE` | `unseen` | 仅 `prompt_mode=env` 时影响 RoboTwin 生成指令 |
| `PYTHON` | `python` | 云端由 `_robotwin_cloud_render_env.sh` 注入 |
| `EVAL_RESULT_ROOT` | `/DRoboticsAILab/users/.../robotwin2.0/result` | 评测输出根目录 |
| `TEST_NUM` | `100` | 评测 episode 数 |
| `PI05_FM_TAU_SCHEDULE` | unset | 推理 τ 调度：`uniformN` 或 `1,0.7,0.3,0`（**1→0**）；启动 server 时绑定 |
| `PI05_NUM_STEPS` | unset | 仅当未设 schedule 时：均匀 NFE（默认模型内 10） |
| `EVAL_RUN_NAME` | `ckpt_setting` | 结果子目录名（sweep 用 schedule 的 name） |

## RoboTwin 侧常用参数在哪设

| 参数 | 位置 | 说明 |
|------|------|------|
| `test_num` | `deploy_policy.yml` / `TEST_NUM` / `--test_num` | episode 数（默认 100） |
| `task_config` | `run_eval.sh` 第 2 参 | 如 `agilex_config` → `task_config/*.yml`（相机、随机化等） |
| `seed` | `run_eval.sh` 第 4 参 | 评测 seed |
| `instruction_type` | `INSTRUCTION_TYPE` / yml | `seen` / `unseen`（仅 `prompt_mode=env`） |
| 单集步数上限 | `task_config/_eval_step_limit.yml` | 按任务名，如 adjust_bottle=400 |
| `replan_steps` / `prompt_mode` | yml 或 `PI05_*` | pi05 策略侧 |

冒烟示例（2 个 episode）：

```bash
TEST_NUM=2 bash policy/pi05/run_eval.sh adjust_bottle
```

带自定义 FM 调度（pi0.5 **1→0**）：

```bash
PI05_FM_TAU_SCHEDULE=uniform3 TEST_NUM=2 bash policy/pi05/run_eval.sh adjust_bottle
```

串行 schedule sweep 见 `eval_script/README.md`。


## 语言指令

| `prompt_mode` | 行为 |
|---------------|------|
| `train_short`（默认） | 固定短句，与 LeRobot `tasks.jsonl` 一致：`adjust bottle` / `scan object` / `beat block hammer` |
| `env` | 使用 RoboTwin `generate_episode_descriptions` 的 seen/unseen 长句 |

```bash
PI05_PROMPT_MODE=env bash policy/pi05/run_eval.sh adjust_bottle
```

## 注意

1. **不要改 openpi 也能评**：client 侧对齐观测即可；`--port` 须在 `policy:checkpoint` 之前（`serve.sh` 已处理）。  
2. Server 日志：`run_eval.sh` 写到 `$OPENPI_ROOT/logs/pi05_serve_<port>.log`。  
3. 正式评测默认 **100 episode**（`script/eval_policy.py`）；冒烟可临时改 `test_num`。  
4. 换权重时只改 `PI05_CKPT`，`ckpt_setting` 仅影响结果目录命名。
