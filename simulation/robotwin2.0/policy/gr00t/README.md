# gr00t — Isaac-GR00T N1.7 × RoboTwin 2.0 评测

Client–server：GR00T 进程提供策略服务，RoboTwin 只做仿真 + ZMQ 推理，避免 Sapien 与 GR00T 环境冲突。

权重：50 任务 × 50 demo_clean 联合微调，绝对 14 维关节，`action_horizon=32`。任意官方 RoboTwin 任务均可评（需 `agilex_config` 三相机）。

## 架构

```
    RoboTwin (.venv)                              Isaac-GR00T (.venv)
policy/gr00t/deploy_policy.py  --zmq-->  python -m gr00t.eval.run_gr00t_server
  encode_obs + replan                      embodiment: NEW_EMBODIMENT
  take_action(qpos)                        ckpt: .../checkpoint-40000
```

- 观测：`head/left/right` RGB（uint8 HWC → `(1,1,H,W,C)`）+ 14 维 joint state  
- 动作：服务端输出绝对 **qpos 14 维**（`left_arm(6)+gripper(1)+right_arm(6)+gripper(1)`）  
- 默认 `replan_steps=auto`（执行完整 32 步 chunk）  
- 默认 `prompt_mode=env`（与训练时 episode 自然语言指令一致）

## 文件

| 文件 | 作用 |
|------|------|
| `deploy_policy.py` | RoboTwin 评测接口：`get_model` / `encode_obs` / `eval` / `reset_model` |
| `deploy_policy.yml` | 默认 host/port/replan/prompt |
| `zmq_client.py` | 轻量 ZMQ 客户端（msgpack-numpy），不依赖完整 Isaac-GR00T |
| `serve.sh` | 启动 GR00T 策略服务 |
| `eval.sh` | 仅客户端评测（需服务已就绪） |
| `run_eval.sh` | **单卡一键**：后台起 server → 等端口 → eval → 退出清 server |

## 依赖

**服务端**（Isaac-GR00T `.venv`）：

- `GR00T_ROOT` 默认：`/DRoboticsAILab/vepfs_20260622/licheng.yang/Isaac-GR00T`

**客户端**（RoboTwin `.venv`，含 sapien）：

```bash
# 在 simulation/robotwin2.0 下
.venv/bin/pip install pyzmq msgpack-numpy
```

本机勿用 `(base)` conda 的 `python` 跑评测，会缺 `sapien`。

## 默认权重

```
/DRoboticsAILab/users/licheng.yang/checkpoints/gr00t/n17_robotwin50_clean_bgr2rgb_h32/n17_robotwin50_clean_bgr2rgb_h32/checkpoint-40000
```

预训练 base 在 `Isaac-GR00T/checkpoints/GR00T-N1.7-3B`，VLM backbone 在 `Isaac-GR00T/checkpoints/Cosmos-Reason2-2B`。评测微调权重时 **不必** 再加载 N1.7-3B，但 Cosmos tokenizer 需能本地解析（`serve.sh` 会 link HF snapshot）。

## 本机（1×L4）用法

L4 约 23GB，N1.7-3B bf16 推理通常可与仿真共用一张卡。

```bash
cd /data/vepfs/licheng.yang/DRRM-VODPP/simulation/robotwin2.0

# 推荐：一键
bash policy/gr00t/run_eval.sh adjust_bottle

# 完整参数
# bash run_eval.sh <task> [task_config] [ckpt_setting] [seed] [gpu_id]
bash policy/gr00t/run_eval.sh adjust_bottle agilex_config n17_robotwin50_clean_bgr2rgb_h32_40000 0 0
```

分两步（调试 server 时）：

```bash
# 终端 1：起服务（首次加载可能要几分钟）
bash policy/gr00t/serve.sh

# 终端 2：评测
bash policy/gr00t/eval.sh adjust_bottle agilex_config n17_robotwin50_clean_bgr2rgb_h32_40000 0 0
```

结果目录（默认）：

```
/DRoboticsAILab/users/licheng.yang/robotwin2.0/result/<task>/gr00t/<task_config>/<ckpt_setting>/<timestamp>/
```

可用 `EVAL_RESULT_ROOT` 覆盖。

## 环境变量

| 变量 | 默认 | 说明 |
|------|------|------|
| `GR00T_ROOT` | 见上 | Isaac-GR00T 仓库路径 |
| `GR00T_CKPT` | `.../checkpoint-40000` | 微调权重目录 |
| `GR00T_HOST` / `GR00T_PORT` | `127.0.0.1` / `5555` | 服务地址 |
| `GR00T_REPLAN_STEPS` | `auto` | 每段开环执行步数 |
| `GR00T_PROMPT_MODE` | `env` | 见下 |
| `INSTRUCTION_TYPE` | `unseen` | 仅 `prompt_mode=env` 时影响 RoboTwin 生成指令 |
| `PYTHON` | `python` | 云端可由 `_robotwin_cloud_render_env.sh` 注入 |
| `EVAL_RESULT_ROOT` | `/DRoboticsAILab/users/.../robotwin2.0/result` | 评测输出根目录 |
| `TEST_NUM` | `100` | 评测 episode 数 |
| `EVAL_RUN_NAME` | `ckpt_setting` | 结果子目录名 |

冒烟示例（5 个 episode）：

```bash
TEST_NUM=5 bash policy/gr00t/run_eval.sh adjust_bottle
```

## 语言指令

| `prompt_mode` | 行为 |
|---------------|------|
| `env`（默认） | 使用 RoboTwin `generate_episode_descriptions` 的 seen/unseen 长句（与训练 captions 同分布） |
| `train_short` | `task_name` 下划线改空格：`adjust bottle` |

```bash
GR00T_PROMPT_MODE=train_short bash policy/gr00t/run_eval.sh adjust_bottle
```

## 注意

1. **不要改 Isaac-GR00T 也能评**：client 侧对齐观测即可。  
2. Server 日志：`run_eval.sh` 写到 `$GR00T_ROOT/logs/gr00t_serve_<port>.log`。  
3. 正式评测默认 **100 episode**；冒烟可临时 `TEST_NUM=5`。  
4. 换权重时只改 `GR00T_CKPT`，`ckpt_setting` 仅影响结果目录命名。
