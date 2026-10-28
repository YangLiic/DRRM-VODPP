# TrustEuler × RoboTwin 评测流水线

本目录把 **TrustEuler 校准（U100 dense Euler）** 与 **RoboTwin 评测** 拆开编排：先共享采集 dense traces，再在评测时按 `meanN` 感知推理网格；`uniform*` / 显式 τ 节点可直接评测。

## 1. 目标与边界

| 阶段 | 做什么 | 不做什么 |
|------|--------|----------|
| 校准 `trusteuler_calibrate*` | 建校准清单 + 跑一遍 U100 dense Euler，写出 traces | **不算** mean / scmean 推理网格 |
| 感知 `trusteuler_sense_grids` | 从已有 traces 按预算 NFE 冻结 Mean 网格（λ=0 → M0） | 不再跑 U100 |
| 评测 `eval_*_trusteuler_sweep*` | 解析调度 → 调用现有 `eval_robotwin2.0_parallel*` | 不改策略权重 |

算法含义（λ=0）：

```text
v_avg_gt(τ) = (GT - x(τ)) / (1 - τ)
M0(τ)       = RMS(v_model - v_avg_gt)
Q(τ)        ∝ 1 / (1 + smooth(M0) / median(M0))
逆 CDF(Q)   → budget+1 个 τ 节点（即 mean{budget}）
```

VODPP FM 语义是 **τ: 0→1（noise→action）**，`NFE = len(knots) - 1`，评测时原样喂给 `fm_tau_schedule`，不要做 `1-τ`。

## 2. 目录与产物

默认路径：

| 角色 | 路径 |
|------|------|
| 本编排脚本 | `DRRM-VODPP/scripts/eval/TrustEuler_eval/` |
| TrustEuler 源码 | `/data/vepfs/licheng.yang/TrustEuler` |
| RoboTwin 数据集 | `DRRM-VODPP/simulation/robotwin2.0/datasets/<task>` |
| TrustEuler 运行产物 | `/DRoboticsAILab/users/licheng.yang/robotwin2.0/TrustEuler_runs/<task>/` |
| RoboTwin 评测结果 | `/DRoboticsAILab/users/licheng.yang/robotwin2.0/result/<task>/` |

单个 task 的 TrustEuler 产物树：

```text
TrustEuler_runs/<task>/
├── configs/
│   ├── dataset.json          # 自动生成的校准清单配置
│   └── vodpp.json            # 自动生成的 VODPP 后端配置
├── calibration_dataset.json  # episode / chunk / noise_seed 清单
├── traces/*.npz           # U100 dense Flow traces（校准主产物）
└── calibration/
    ├── b3/mean_grid.json     # 评测时按预算感知（可并行写不同 bN）
    ├── b5/mean_grid.json
    ├── b8/mean_grid.json
    └── b10/mean_grid.json
```

## 3. 脚本一览

| 文件 | 作用 |
|------|------|
| `prepare_trusteuler_configs.py` | 读 ckpt `config.json` + 数据集 meta，写 `dataset.json` / `vodpp.json`（horizon、action_dim、`rgb_key`→cameras） |
| `trusteuler_calibrate.sh` | **仅** U100：prepare → `build-dataset` → `collect-vodpp` |
| `trusteuler_calibrate_cloud.sh` | 云端入口（Sapien/Python 环境）→ 调用上面 |
| `trusteuler_sense_grids.sh` | 从 traces 对给定 `--budget N` 跑 `trusteuler calibrate --lambda-seq 0` |
| `resolve_mean_schedule.py` | `mean3`→逗号分隔 τ；`uniform*` / `0,0.3,1` 原样透传 |
| `eval_robotwin2.0_trusteuler_sweep.sh` | 本地编排：U100（可选）→ sense → eval |
| `eval_robotwin2.0_trusteuler_sweep_cloud.sh` | 云端编排，`EVAL_SH` 指向 `eval_robotwin2.0_parallel_cloud.sh` |

## 4. 端到端流程

```text
任务名 + 权重路径 + 网格种类 (mean3 / uniform5 / 0,0.35,1 ...)
                │
                ▼
┌───────────────────────────────────────┐
│  Stage A  trusteuler_calibrate        │
│  （若有 mean* 且未 --skip-calibrate）  │
│  1. prepare configs                   │
│  2. build-dataset                     │
│  3. collect-vodpp  U100 dense Euler   │
│  → TrustEuler_runs/<task>/traces      │
└───────────────────────────────────────┘
                │  （serial / parallel 都必须先等 A 完成）
                ▼
┌───────────────────────────────────────┐
│  Stage B  每个评测 run                │
│  · meanN  → sense budget N → 读网格   │
│  · 其它   → 调度原样使用              │
│  · 调用 eval_robotwin2.0_parallel*    │
└───────────────────────────────────────┘
```

### 串行 `--serial`

1. 若需要 `mean*`：先完整跑完 Stage A（或 `--skip-calibrate` 复用 traces）  
2. 再按顺序：对每个 run，若是 `meanN` 先 sense，再评测

### 并行（不加 `--serial`）

1. **同样先等 Stage A 跑完**（只采一次 U100）  
2. 再对各 GPU worker 并行：各自对 `meanN` 写 `calibration/bN/`，再评测  
3. `uniform*` / 显式节点不走 sense

### `--skip-calibrate`

- **跳过 Stage A（U100）**，要求 `TrustEuler_runs/<task>/traces` 已存在  
- 评测时仍会按需对 `meanN` 做 Stage B sense

> 不要对同一 task 起多个「未 skip」的评测任务同时跑 Stage A，否则会重复 U100 甚至抢写 traces。推荐：**先单独校准，再多个评测任务都加 `--skip-calibrate`**。

## 5. 用法示例

### 5.1 只做校准（推荐先批量起单卡任务）

本地：

```bash
cd /data/vepfs/licheng.yang/DRRM-VODPP
bash scripts/eval/TrustEuler_eval/trusteuler_calibrate.sh \
  click_bell \
  /DRoboticsAILab/datasets/xianbao01.hou/robochallenge_checkpoints/click_bell_100/fm_dinov3cnnl_3v_cnn_l
```

云端：

```bash
bash scripts/eval/TrustEuler_eval/trusteuler_calibrate_cloud.sh \
  click_bell \
  /path/to/ckpt
```

`dag_tgcontrol_1GPU.py` 里的 `launch_robotwin_trusteuler_calibrate_jobs()` 即按 task 各起一张卡，只跑这一步。

### 5.2 校准已完成 → 评测（多网格）

```bash
cd /data/vepfs/licheng.yang/DRRM-VODPP

# 云端串行示例：先跳过 U100，再依次 mean + uniform
bash scripts/eval/TrustEuler_eval/eval_robotwin2.0_trusteuler_sweep_cloud.sh --serial --skip-calibrate \
  VODPP click_bell agilex_config \
  /DRoboticsAILab/datasets/xianbao01.hou/robochallenge_checkpoints/click_bell_100/fm_dinov3cnnl_3v_cnn_l \
  100 1 0 headless true 100 \
  mean3 mean5 mean8 mean10 \
  uniform3:uniform3 uniform5:uniform5 uniform8:uniform8 uniform10:uniform10
```

参数位置与 `eval_robotwin2.0_tau_sweep.sh` 一致：

```text
[--serial] [--skip-calibrate]
<policy> <task> <task_config> <ckpt>
<expert_n> <nproc> <seed> <mode> <save_video> <test_num>
<name[:schedule]> ...
```

调度写法：

| 写法 | 行为 |
|------|------|
| `mean3` 或 `mean3:mean3` | 从 traces sense budget=3，再评测 |
| `uniform5:uniform5` | 直接 uniform5 |
| `custom:0,0.35,1` | 显式节点 |
| `0,0.35,1`（无名时 name=token） | 显式节点 |

### 5.3 单次命令内自动校准再评测

不加 `--skip-calibrate` 时，sweep 会在评测前自动跑 Stage A（适合单任务一气呵成；多卡拆 mean 时不要这样重复起）：

```bash
bash scripts/eval/TrustEuler_eval/eval_robotwin2.0_trusteuler_sweep.sh --serial \
  VODPP click_bell agilex_config /path/to/ckpt \
  100 1 0 headless false 2 \
  mean3 mean5
```

## 6. 环境变量

| 变量 | 默认 | 说明 |
|------|------|------|
| `TRUSTEULER_ROOT` | `/data/vepfs/licheng.yang/TrustEuler` | TrustEuler 源码 |
| `TRUSTEULER_RUNS_ROOT` | `/DRoboticsAILab/users/.../TrustEuler_runs` | 校准产物根目录 |
| `ROBOTWIN_DATASETS` | `<DRRM>/simulation/robotwin2.0/datasets` | LeRobot 数据 |
| `DRRM_ROOT` | 本仓库根 | DINOv3 相对路径、PYTHONPATH |
| `RESULT_ROOT` | `/DRoboticsAILab/users/.../robotwin2.0/result` | 评测输出 |
| `EPISODE_COUNT` | `8` | 校准分层抽样 episode 数 |
| `DENSE_STEPS` | `100` | dense Euler 步数（U100） |
| `USE_EMA` | `false` | 是否加载 `ema/model.safetensors` |
| `DEVICE` / `GPU_ID` | `cuda:0` / `0` | 采集与评测 GPU |
| `FORCE_CALIBRATE` | `0` | `1` 时强制重采 traces |
| `EVAL_SH` | parallel / parallel_cloud | 评测入口（cloud 脚本会设置） |
| `PYTHON` / `TE_PYTHON` | robotwin `.venv` 或镜像 python | 解释器 |

云端还需 `_robotwin_cloud_render_env.sh`（`*_cloud.sh` 会自动 source）。

## 7. 换任务检查清单

1. `datasets/<task>` 存在，且 `meta/info.json` 中 `action`/`state` 维数正确  
2. ckpt 目录含 `config.json` + `model.safetensors`（或 ema）  
3. `prepare_trusteuler_configs` 打出的 `cameras` 覆盖 policy `rgb_key`（1v / 3v）  
4. `horizon` 与 `action_horizon` / `chunk_stride` 一致  
5. 先跑通 `trusteuler_calibrate*`，确认 `traces/*.npz` 数量 ≥ manifest `chunk_count`  
6. 再 `--skip-calibrate` 跑 mean / uniform 评测  

## 8. 推荐上云节奏

```text
① 每 task 一张卡：trusteuler_calibrate_cloud.sh     → 只产 U100 traces
② 确认 TrustEuler_runs/<task>/traces 齐全
③ 每 task × 每网格 起评测卡（或一卡串行多网格）：
     eval_*_trusteuler_sweep_cloud.sh --skip-calibrate ...
```

这样 **U100 只采一次**，`mean3/5/8/10` 只在评测侧各自（或串行）感知网格，避免重复 dense Euler。

## 评测任务清单

见 [`docs/eval_tasks_and_checkpoints.md`](docs/eval_tasks_and_checkpoints.md)。
