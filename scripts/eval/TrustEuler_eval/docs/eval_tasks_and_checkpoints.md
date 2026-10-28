# TrustEuler / RoboTwin 评测任务清单

记录当前已规划 / 已提交的评测：任务名、权重路径、调度种类、云端 job 命名约定。

结果根目录：

```text
/DRoboticsAILab/users/licheng.yang/robotwin2.0/result/<task>/VODPP/agilex_config/<run_name>/
```

TrustEuler U100 traces：

```text
/DRoboticsAILab/users/licheng.yang/robotwin2.0/TrustEuler_runs/<task>/traces/
```

权重根目录简写：

```text
CKPT_ROOT=/DRoboticsAILab/datasets/xianbao01.hou/robochallenge_checkpoints
```

---

## 1. 共用推理网格种类

| run_name | schedule（喂给 `fm_tau_schedule`） | 说明 |
|----------|-------------------------------------|------|
| `mean10` | 由 TrustEuler M0 / λ=0 从 traces 感知 | NFE=10 |
| `mean8` | 同上 | NFE=8 |
| `mean5` | 同上 | NFE=5 |
| `mean3` | 同上 | NFE=3 |
| `uniform10` | `uniform10` | 均匀 10 步 |
| `uniform8` | `uniform8` | 均匀 8 步 |
| `uniform5` | `uniform5` | 均匀 5 步 |
| `uniform3` | `uniform3` | 均匀 3 步 |
| `dense_jump` | `0,0.1,0.2,0.3,0.4,0.5,1` | 前半密、尾段大跳 |
| `sharp7` | `0,0.2,0.4,0.6,0.8,0.9,0.95,1` | 尾段加密 |

默认评测超参（除非另行注明）：`agilex_config`，`headless`，`save_video=true`，`test_num=100`，`nproc=1`，单卡 L4。

---

## 2. TrustEuler 批次（3v cnn_l）

编排：`dag_tgcontrol_1GPU.py` → `launch_robotwin_trusteuler_eval_jobs()`  
入口：`eval_robotwin2.0_trusteuler_sweep_cloud.sh --serial --skip-calibrate`  
Job 名：`lcyang-te-ev-<task>-<run>`（`_` → `-`）

| task | 权重 | expert_n | 评测数 |
|------|------|----------|--------|
| `adjust_bottle` | `$CKPT_ROOT/adjust_bottle_100/fm_dinov3cnnl_3v_cnn_l` | 100 | 10 |
| `click_bell` | `$CKPT_ROOT/click_bell_100/fm_dinov3cnnl_3v_cnn_l` | 100 | 10 |
| `move_playingcard_away` | `$CKPT_ROOT/move_playingcard_away_200/fm_dinov3cnnl_3v_cnn_l` | 200 | 10 |
| `move_stapler_pad` | `$CKPT_ROOT/move_stapler_pad_100/fm_dinov3cnnl_3v_cnn_l` | 100 | 10 |
| `open_laptop` | `$CKPT_ROOT/open_laptop_100/fm_dinov3cnnl_3v_cnn_l` | 100 | 10 |
| `place_a2b_left` | `$CKPT_ROOT/place_a2b_left_100/fm_dinov3cnnl_3v_cnn_l` | 100 | 10 |
| `place_can_basket` | `$CKPT_ROOT/place_can_basket_100/fm_dinov3cnnl_3v_cnn_l` | 100 | 10 |
| `place_fan` | `$CKPT_ROOT/place_fan_100/fm_dinov3cnnl_3v_cnn_l` | 100 | 10 |
| `place_phone_stand` | `$CKPT_ROOT/place_phone_stand_100/fm_dinov3cnnl_3v_cnn_l` | 100 | 10 |
| `press_stapler` | `$CKPT_ROOT/press_stapler_100/fm_dinov3cnnl_3v_cnn_l` | 100 | 10 |
| `stack_blocks_two` | `$CKPT_ROOT/stack_blocks_two_100/fm_dinov3cnnl_3v_cnn_l` | 100 | 10 |

合计：**11 × 10 = 110** 个单卡评测任务（每个 task 覆盖上表 10 种网格）。

校准（仅 U100，先于评测）：`lcyang-te-cal-<task>` → `trusteuler_calibrate_cloud.sh`。

每个 task 下所有 run 共用同一个 `seed_arg=0`（`st_seed=100000`），保证同 task 不同网格的 `episode_k` 场景一致。

### 2.1 `adjust_bottle` CRLS 显式网格（`λ_arr=0`）

校准产物：`TrustEuler_runs/adjust_bottle/calibration_crls/`（复用已有 150 条 U100 traces）  
编排：`launch_adjust_bottle_crls_eval_jobs()`  
`expert_n=100`，`test_num=100`，**全网格 `seed=0`**。

| run_name | schedule |
|----------|----------|
| `crls10` | `0,0.11812881,0.2495201,0.36970414,0.48342132,0.59252934,0.69010245,0.77472129,0.86358064,0.935,1` |
| `crls8` | `0,0.156138,0.31949333,0.46710489,0.6068803,0.72501338,0.83931822,0.935,1` |
| `crls5` | `0,0.28061523,0.5392604,0.75183335,0.935,1` |
| `crls3` | `0,0.5392604,0.935,1` |

Job 名：`lcyang-te-ev-adjust-bottle-crls{10,8,5,3}`。`τ_L*=0.935`（Anchor-only，与 B 无关）。

### 2.2 `adjust_bottle` SharpEuler 显式网格（`γ=0.5`）

校准产物：`TrustEuler_runs/adjust_bottle/calibration_sharp/`（复用同一批 U100 traces）  
编排：`launch_adjust_bottle_sharp_eval_jobs()`  
`expert_n=100`，`test_num=100`，**全网格 `seed=0`**（与 CRLS / 3v 批次对齐）。

| run_name | schedule |
|----------|----------|
| `sharp10` | `0,0.12868545,0.26831189,0.3929242,0.5143767,0.62743881,0.72404283,0.8190337,0.90538473,0.96647185,1` |
| `sharp8` | `0,0.16412261,0.33286366,0.48505662,0.62743881,0.74618395,0.86503345,0.95470826,1` |
| `sharp5` | `0,0.26831189,0.5143767,0.72404283,0.90538473,1` |
| `sharp3` | `0,0.43423165,0.78672224,1` |

Job 名：`lcyang-te-ev-adjust-bottle-sharp{10,8,5,3}`。

---

## 3. `scan_object` 批次（1v adapter128）

**权重：**

```text
$CKPT_ROOT/dit_fm_dinov3cnnl_1v_adapter128_dit1024_demo200/scan_object
```

### 3.1 uniform / SharpEuler / CRLS × {10,8,5,3}（同 seed，12 卡）

编排：`launch_scan_object_uniform_sharp_crls_eval_jobs()`  
本地网格：`TrustEuler/runs/scan_object/calibration_{sharp,crls}/`（复用 166 条 U100 traces）  
`expert_n=500`，`test_num=100`，**全 12 网格 `seed=0`**（episode_k 对齐）。

| run_name | schedule |
|----------|----------|
| `uniform10` | `uniform10` |
| `uniform8` | `uniform8` |
| `uniform5` | `uniform5` |
| `uniform3` | `uniform3` |
| `sharp10` | `0,0.12525588,0.25762859,0.38452631,0.50407018,0.61636699,0.71911434,0.81181359,0.89965835,0.96345052,1` |
| `sharp8` | `0,0.16005467,0.3210521,0.47463057,0.61636699,0.74249659,0.85783283,0.9503372,1` |
| `sharp5` | `0,0.25762859,0.50407018,0.71911434,0.89965835,1` |
| `sharp3` | `0,0.42505582,0.78017566,1` |
| `crls10` | `0,0.12171032,0.25230141,0.37702671,0.49483468,0.60638792,0.70822605,0.79948468,0.88796704,0.955,1` |
| `crls8` | `0,0.16079365,0.32344066,0.47836831,0.62097209,0.74767866,0.86417267,0.955,1` |
| `crls5` | `0,0.28261283,0.55340252,0.77597742,0.955,1` |
| `crls3` | `0,0.55340252,0.955,1` |

Job 名：`lcyang-te-ev-scan-object-{uniform,sharp,crls}{10,8,5,3}`。  
CRLS：`λ_arr=0`，`τ_L*=0.955`；SharpEuler：`γ=0.5`。

### 3.2 历史 mean / uniform / dense_jump / sharp7

**编排入口（历史串行 sweep）：** `eval_robotwin2.0_tau_sweep_cloud.sh --serial`  
（mean 节点为事先算好的显式 τ，不是评测时再 sense。）

| run_name | schedule | seed（历史） | 状态 |
|----------|----------|------|------|
| `mean3` | `0,0.27613568,0.56139327,1` | 0 | 已跑 |
| `mean5` | `0,0.16700648,0.33105311,0.50147707,0.69000495,1` | 1 | 已跑（历史串行按 index 偏移，与 mean3 不对齐） |
| `mean8` | `0,0.10528261,0.20790006,0.31038832,0.4149321,0.52366633,0.6401061,0.77126596,1` | 2 | 已跑（同上） |
| `mean10` | `0,0.08459036,0.16700648,0.24880404,0.33105311,0.4149321,0.50147707,0.59230189,0.69000495,0.80110218,1` | 3 | 已跑（同上） |
| `uniform3` | `uniform3` | 4 | 原串行 / 后续单卡补齐（**与 §3.1 seed=0 不同**） |
| `uniform5` | `uniform5` | 5 | 单卡补齐（同上） |
| `uniform8` | `uniform8` | 6 | 单卡补齐（同上） |
| `uniform10` | `uniform10` | 7 | 单卡补齐（同上） |
| `dense_jump` | `0,0.1,0.2,0.3,0.4,0.5,1` | 8（已提）/ 新逻辑应为 0 | 已提交；新编排默认 seed=0 |
| `sharp7` | `0,0.2,0.4,0.6,0.8,0.9,0.95,1` | 9（已提）/ 新逻辑应为 0 | 已提交；新编排默认 seed=0 |

> 新逻辑：同 task 所有网格 `seed_arg=0`。上表「历史」列保留当时实际使用的 seed，便于对照已跑结果。§3.1 的 uniform 是 **seed=0 重跑**，勿与历史 seed=4..7 结果混比。

Job 名（补跑）：`lcyang-scan-object-dense-jump`、`lcyang-scan-object-sharp7`。

原始一键串行命令（仅含 mean+uniform，缺 dense_jump / sharp7）：

```bash
cd /data/vepfs/licheng.yang/DRRM-VODPP/simulation/robotwin2.0 && source .venv/bin/activate \
&& cd /data/vepfs/licheng.yang/DRRM-VODPP \
&& export RESULT_ROOT=/DRoboticsAILab/users/licheng.yang/robotwin2.0/result \
&& export GPU_ID=0 \
&& bash scripts/eval/eval_robotwin2.0_tau_sweep_cloud.sh --serial \
  VODPP scan_object agilex_config \
  $CKPT_ROOT/dit_fm_dinov3cnnl_1v_adapter128_dit1024_demo200/scan_object \
  500 1 0 headless true 100 \
  mean3:0,0.27613568,0.56139327,1 \
  mean5:0,0.16700648,0.33105311,0.50147707,0.69000495,1 \
  mean8:0,0.10528261,0.20790006,0.31038832,0.4149321,0.52366633,0.6401061,0.77126596,1 \
  mean10:0,0.08459036,0.16700648,0.24880404,0.33105311,0.4149321,0.50147707,0.59230189,0.69000495,0.80110218,1 \
  uniform3:uniform3 uniform5:uniform5 uniform8:uniform8 uniform10:uniform10
```

补跑单卡命令等价于：

```bash
# dense_jump / sharp7（新逻辑统一 seed=0，与同 task 其它网格对齐）
bash scripts/eval/eval_robotwin2.0_tau_sweep_cloud.sh --serial \
  VODPP scan_object agilex_config \
  $CKPT_ROOT/dit_fm_dinov3cnnl_1v_adapter128_dit1024_demo200/scan_object \
  500 1 0 headless true 100 \
  dense_jump:0,0.1,0.2,0.3,0.4,0.5,1

bash scripts/eval/eval_robotwin2.0_tau_sweep_cloud.sh --serial \
  VODPP scan_object agilex_config \
  $CKPT_ROOT/dit_fm_dinov3cnnl_1v_adapter128_dit1024_demo200/scan_object \
  500 1 0 headless true 100 \
  sharp7:0,0.2,0.4,0.6,0.8,0.9,0.95,1
```


---

## 4. `beat_block_hammer` 批次（本机权重，test_num=50）

编排：`launch_beat_block_hammer_calibrate_job()` → `launch_beat_block_hammer_eval_jobs()`

**权重：**

```text
/DRoboticsAILab/users/licheng.yang/robotwin2.0/beat_block_hammer
```

| 项 | 值 |
|----|----|
| expert_n | 500 |
| test_num | 50 |
| seed | 全网格 `0` |
| 网格 | 同 §1 的 10 种 |
| 校准 job | `lcyang-te-cal-beat-block-hammer` |
| 评测 job | `lcyang-te-ev-beat-block-hammer-<run>` |

评测带 `--skip-calibrate`，需等 U100 traces 写到 `TrustEuler_runs/beat_block_hammer/traces/`。

### 4.1 CRLS 显式网格（`λ_arr=0` Anchor-only）

编排：`launch_beat_block_hammer_crls_eval_jobs()`  
网格来源：`TrustEuler/runs/beat_block_hammer/calibration_crls/crls_grid.json`  
`test_num=50`，**全网格 `seed=0`**（与 §4 对齐）。

| run_name | schedule |
|----------|----------|
| `crls10` | `0,0.11802962,0.2349025,0.3503867,0.46748501,0.5816336,0.68260336,0.77822086,0.86427141,0.935,1` |
| `crls8` | `0,0.15307626,0.30019751,0.45054662,0.59541963,0.72448055,0.8393047,0.935,1` |
| `crls5` | `0,0.26375829,0.52512607,0.7563449,0.935,1` |
| `crls3` | `0,0.52512607,0.935,1` |

Job 名：`lcyang-te-ev-beat-block-hammer-crls{10,8,5,3}`。

### 4.2 SharpEuler 显式网格（`γ=0.5`，全区间）

编排：`launch_beat_block_hammer_sharp_eval_jobs()`  
网格来源：`TrustEuler/runs/beat_block_hammer/calibration_sharp/sharp_grid.json`  
`test_num=50`，**全网格 `seed=0`**。

| run_name | schedule |
|----------|----------|
| `sharp10` | `0,0.1262834,0.24945788,0.37262663,0.49612852,0.61142488,0.71821791,0.81431959,0.90180102,0.96472897,1` |
| `sharp8` | `0,0.15808544,0.31054177,0.46521687,0.61142488,0.74499284,0.86051213,0.95165551,1` |
| `sharp5` | `0,0.24945788,0.49612852,0.71821791,0.90180102,1` |
| `sharp3` | `0,0.4129238,0.78431513,1` |

Job 名：`lcyang-te-ev-beat-block-hammer-sharp{10,8,5,3}`。

### 4.3 OptimizeLoss 显式网格（固定预算 Adam，无 U100）

校准命令：`trusteuler calibrate-optimize-loss --backend vodpp`  
产物：`TrustEuler/runs/beat_block_hammer/calibration_optimize_loss/optimize_loss_grid.json`  
设定：`λ0=1, λ1=λ2=0`，`max_steps=40`，`batch_size=4`，112 chunks / 8 episodes，**不采 dense traces**。  
目标为终端 `M_0^{term}=||X_1-Y||_F^2`（≠ Mean 路径 `M_q(g_y)`）。

| run_name | schedule |
|----------|----------|
| `ol3` | `0,0.29805863,0.62534210,1` |
| `ol5` | `0,0.17075178,0.34802507,0.53959809,0.78912857,1` |
| `ol8` | `0,0.12220225,0.24528968,0.36964213,0.49593081,0.62604921,0.76652003,0.93759360,1` |
| `ol10` | `0,0.08285173,0.16683939,0.25229521,0.33985306,0.43059719,0.52691294,0.63508447,0.78234347,0.95443068,1` |

Job 名：`lcyang-te-ev-beat-block-hammer-ol{10,8,5,3}`。  
编排：`launch_beat_block_hammer_optimize_loss_eval_jobs()`。

### 4.4 OptimizeLoss + 平滑项（`λ0=1, λ1=0.5, λ2=0.25`）

产物目录：`TrustEuler/runs/beat_block_hammer/calibration_optimize_loss_l1_0.5_l2_0.25/`  
除 `optimize_loss_grid.json` 外，每次校准固定写出 `hyperparameters.json`（完整超参快照）；
`schedules[<B>].hyperparameters` 也按 budget 复写一份。

| run_name | schedule |
|----------|----------|
| `ol_s3` | `0,0.33333334,0.66666669,1`（best_step=0，均匀） |
| `ol_s5` | `0,0.2,0.4,0.6,0.8,1`（best_step=0，均匀） |
| `ol_s8` | `0,0.09673814,0.19505878,0.29576232,0.40025418,0.51215362,0.64551889,0.95755406,1` |
| `ol_s10` | `0,0.08118506,0.16375852,0.24819351,0.33528238,0.42636600,0.52428401,0.63659593,0.78647001,0.95636076,1` |

> 实现已改为：**按固定 eval 子集选最优 checkpoint**（非最后一轮 minibatch），
> 且 VODPP 只对 `n_obs_steps` 之后的 action 槽位计 `M_q^{term}`。
> 旧版 `0,0.14,0.28,1` 是发散后错误返回最后一轮的结果，已作废。
>
> 另已修复 episode **双重加权**（现为 `p∝w` 采样 + batch 内无权平均），
> `eval_seed` 与 budget 解耦，λ=0 跳过对应 `M_q`，`eval_every/max/min_step`
> 进入 config 并写入 `hyperparameters.json`。VODPP 上 OptimizeLoss 在文档中
> 明确降级为对 `h_k` 的部分梯度启发式（离散 τ bucket）。

### 4.5 GT-inverse 显式网格（`lrsched_B5`，NFE=5）

编排：`launch_beat_block_hammer_gtinv_eval_jobs()`  
网格来源：`ChunkAwareODEs/src/GT-inverse/runs/beat_block_hammer/schedule_grid_lrsched_B5.json`  
方法：`gt-inverse-hybrid(α=1)`，`λ=[1,0.5,0.25]`，带 LR schedule。  
`test_num=100`，`seed=0`（与 §4 的 seed 对齐；episode 数升到 100）。

| run_name | schedule |
|----------|----------|
| `gti5` | `0,0.82896953,0.96696286,0.98985778,0.99660690,1` |

Job 名：`lcyang-te-ev-beat-block-hammer-gti5`。

### 4.6 GT-inverse L100_shift 显式网格（NFE=5）

编排：`launch_beat_block_hammer_l100shift_eval_jobs()`  
网格来源：`ChunkAwareODEs/src/GT-inverse/runs/beat_block_hammer/schedule_grid_L100_shift_B5.json`  
`test_num=100`，`seed=0`。

| run_name | schedule |
|----------|----------|
| `l100shift5` | `0,0.21045546,0.42813805,0.66176565,0.94747538,1` |

Job 名：`lcyang-te-ev-beat-block-hammer-l100shift5`。

---

## 5. 相关任务对照（非本 TrustEuler 3v 批，供检索）

| task | 权重族 | 备注 |
|------|--------|------|
| `scan_object` | `dit_fm_dinov3cnnl_1v_adapter128_dit1024_demo200/scan_object` | 见 §3 |
| `beat_block_hammer` | `/DRoboticsAILab/users/licheng.yang/robotwin2.0/beat_block_hammer` | 见 §4 |
| `click_alarmclock` | 同 `dit_fm_*_demo200/<task>` 或本地 TrustEuler `runs/` | 早期校准手册用 |

---

## 6. 快速统计

| 批次 | task 数 | 每 task 网格数 | 单卡 job 约数 |
|------|---------|----------------|---------------|
| TrustEuler 3v | 11 | 10 | 110 |
| scan_object 1v | 1 | 10（含补跑 2） | mean/uniform 已有 + dense_jump/sharp7 补 2 |
| beat_block_hammer | 1 | 10 | 1 校准 + 10 评测 |
