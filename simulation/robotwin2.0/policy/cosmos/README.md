# cosmos — Cosmos3-Edge × RoboTwin 2.0 闭环评测（客户端）

`GM717/cosmos3-edge-robotwin-aloha-agilex`（`nvidia/Cosmos3-Edge` 微调出的 RoboTwin
双臂动作策略）在 RoboTwin 2.0 里的评测适配器。和 `policy/pi05` 同构：
**client–server**，RoboTwin 只做仿真，模型跑在独立进程里，中间走 openpi 的
msgpack websocket。

服务端代码不在这里，在
`/DRoboticsAILab/vepfs_20260622/licheng.yang/cosmos-robotwin`
（cosmos-framework 和 sapien 装不进同一个 venv）。那边的 README 写了权重契约、
环境坑、embodiment domain id 是怎么定下来的，以及一处需要核对的不确定性，**先看那份**。

```
RoboTwin (.venv)                          cosmos-framework (.venv)
policy/cosmos/deploy_policy.py  --ws-->   cosmos_robotwin.server
  encode_obs: 3 路原始 RGB + 14 维 qpos      拼 mosaic + flow matching
  take_action(action, "qpos")               返回 16×14 绝对关节目标
```

## 文件

| 文件 | 作用 |
|------|------|
| `deploy_policy.py` | RoboTwin 评测接口：`get_model` / `encode_obs` / `eval` / `reset_model` |
| `deploy_policy.yml` | 默认 host/port/replan/prompt/instruction_type |
| `ws_client.py` | 轻量 websocket 客户端（msgpack），与 `policy/pi05/ws_client.py` 同一份 |
| `serve.sh` | 转发到 cosmos-robotwin 的 `scripts/serve.sh` |
| `eval.sh` | 仅客户端（服务已就绪时用） |
| `run_eval.sh` | **单卡一键**：后台起 server → 等端口 → eval → 退出清 server |

## 用法

```bash
cd /data/vepfs/licheng.yang/DRRM-VODPP/simulation/robotwin2.0

# 冒烟：3 个 episode
TEST_NUM=3 bash policy/cosmos/run_eval.sh place_empty_cup demo_clean

# 完整参数： <task> [task_config] [ckpt_setting] [seed] [gpu_id]
TEST_NUM=3 bash policy/cosmos/run_eval.sh beat_block_hammer demo_clean cosmos3_edge_robotwin 0 0
```

调试 server 时分两步：

```bash
# 终端 1
bash policy/cosmos/serve.sh
# 终端 2
bash policy/cosmos/eval.sh place_empty_cup demo_clean cosmos3_edge_robotwin 0 0
```

结果目录：
`${EVAL_RESULT_ROOT}/<task>/cosmos/<task_config>/<run_name>/<timestamp>/`

## 环境变量

| 变量 | 默认 | 说明 |
|------|------|------|
| `COSMOS_RW_CKPT` | `.../cosmos-checkpoints/cosmos3-edge-robotwin-aloha-agilex` | 微调权重目录 |
| `COSMOS_HOST` / `COSMOS_PORT` | `127.0.0.1` / `8000` | 服务地址 |
| `COSMOS_RW_NUM_STEPS` | `30` | flow matching 去噪步数；推理时定，不在权重里 |
| `COSMOS_RW_GUIDANCE` | `1.0` | guidance=1.0 时每步一次前向 |
| `COSMOS_REPLAN_STEPS` | `auto` | `auto` 执行整块 16 步；给整数则截断后重规划 |
| `COSMOS_PROMPT_MODE` | `env` | RoboTwin 生成的指令；训练用的就是这个池子 |
| `INSTRUCTION_TYPE` | `seen` | 模型卡：训练只用 `seen` |
| `TEST_NUM` | `100` | episode 数 |
| `EVAL_RESULT_ROOT` | `/DRoboticsAILab/users/licheng.yang/robotwin2.0/result` | 结果根目录 |
| `EVAL_RUN_NAME` | `ckpt_setting` | 结果子目录名 |

## 几个容易踩的点

1. **task_config 必须是 aloha-agilex 且开着 wrist 相机。** `demo_clean` /
   `demo_randomized` / `agilex_config` 都满足；策略要 head + 左右 wrist 三路 RGB，
   少一路 `encode_obs` 会直接报错而不是悄悄降级。
2. **`instruction_type` 用 `seen`。** 微调只在 seen 指令池上训过，跑 unseen 是在测
   泛化而不是在测复现。
3. **不要在这里翻夹爪。** 策略吃进和吐出的都是 RoboTwin 原生约定（`1.0` 开 / `0.0` 合），
   上游 DROID 的服务端会做 `1-g`，这个不做。
4. **一块动作 = 16 步 @ 16 Hz = 1 秒。** `step_lim=400` 的任务约 25 次推理。
5. **单卡上 server 和仿真共用一张 L4。** `run_eval.sh` 就是这么设计的；23G 显存放得下
   3.9B 的 bf16 推理加 Sapien（实测峰值约 13G）。
6. **怀疑策略不对时，先看 server 日志里的 `||action[0] - state||_2`。** 16 Hz 下第一个
   动作目标应该贴着当前位姿，这个数是零点几；如果是 4~5，说明条件化错了（第一版就
   踩过：embodiment domain id 用成了 17，正确的是 11），而不是任务太难。

## 实测开销（1×L4，num_steps=30）

| 项 | 值 |
|---|---|
| server 冷启动（读 3.9B 权重 + Wan VAE） | 约 50 s |
| 第一次推理（含 torch.compile） | 20~35 s |
| 稳定后单次推理 | 约 1.6 s |
| 一次推理 + 仿真 16 步 | 约 13 s（仿真渲染占大头） |
| 一个 400 步 episode | 约 5~6 min |
