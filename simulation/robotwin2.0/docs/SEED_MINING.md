# Eval seed 的挖掘与固定(SEED MINING)

> 2026-08-10。动机:blocks_ranking_rgb 第一次闭环 bootstrap 用完整评测挖 seed,
> 2 小时后死在 ffmpeg 管道断裂上,一无所获。而 seed 合法性本来就不由策略决定——
> 由 curobo 专家规划器决定,可以单独跑,分钟级。

## 1. 为什么 seed 必须固定(pin)

`script/eval_policy.py` 默认**发现式**选集:从 `st_seed = 100000*(1+seed)` 逐个往前走,
专家解得动的 seed 才算合法 episode。专家(curobo)带随机性,两次运行只要一个 seed
判定不同,之后全部岔开——scan_object 实测 8 个 100-episode run 只共享 15 个 seed。
**不同 schedule 评在不同场景上,成功率不可比。**

解法:`ROBOTWIN_SEED_LIST` 环境变量 pin 一份清单,所有 run 评完全相同的场景。
canonical 清单放在(路径无空格,可直接进 `export`):

```
/DRoboticsAILab/users/licheng.yang/robotwin2.0/seedlists/<task>_<task_config>.txt
```

## 2. seed 合法性谓词:专家判定,与策略无关

`eval_policy.py` 发现模式接受一个 seed 当且仅当:

```python
TASK_ENV.setup_demo(seed=s, is_test=True, **args)   # args 来自 task_config yml
TASK_ENV.play_once()                                 # curobo 专家执行
TASK_ENV.plan_success and TASK_ENV.check_success()   # 规划成功 且 任务判成
```

策略(pi0.5 等)在这个判定**之后**才出场。所以用完整闭环评测挖 seed,是花
~2.5 h(权重加载 + 每 chunk 一次 VLM 前向 + ffmpeg 编码)买一个专家几分钟就能给的答案。

## 3. 快速挖法:`script/mine_eval_seeds.py`

只跑上面的谓词,不加载策略、不写视频:

```bash
# 云上(经 cloud wrapper,自动设 headless 渲染环境):
bash script/mine_eval_seeds_cloud.sh \
  --task_name blocks_ranking_rgb --task_config demo_clean \
  --test_num 100 \
  --out /DRoboticsAILab/users/licheng.yang/robotwin2.0/seedlists/blocks_ranking_rgb_demo_clean.txt
```

- seed 序列、task_config、`is_test=True` 与 eval 发现模式**逐项一致**;
- 输出格式与 `_eval_seeds.txt` 相同,直接喂 `ROBOTWIN_SEED_LIST`;
- 挖不满 `--test_num`(资产缺失等)时**照样写出已挖到的**,以退出码 1 标记不完整
  ——pin 到短清单仍然有效,只是 test_num 要按清单长度给;
- `--max_walk`(默认 2000)防止不可挖配置下无限走。

批量提交:`dag_tgcontrol_1GPU.py::launch_blocks_seed_mining_jobs()`(1×L4 每难度)。

## 4. 三个容易踩的坑

**`collect_data.py` 的 seed.txt 不是替代品。** 它也用专家挖 seed(数据集目录里的
`seed.txt` 就是它写的),但从 seed 0 起走、用采集配置、**不带 `is_test=True`**,
与 eval 谓词不等价。评测清单必须按 §2 的谓词在**目标 task_config** 下挖。

**专家有随机性,清单是"一份合法抽样"而非唯一真值。** 这不碍事:pin 模式下
专家复核失败的 seed 也照评(`eval_policy.py`:"expert failed on pinned seed,
evaluating anyway"),所以规划器噪声不会让下游死锁。碍事的是**换清单**——
一个 task_config 只挖一次,之后永远用它。

**hard 配置的清单可能偏短。** `cluttered_table: true` 撒杂物会撞缺失资产
(如 `objaverse/list.json` 列了 `pencil` 但磁盘上没有),这类 seed 被拒。
scan_object hard 只有 85 个即此因。属预期行为,按实际长度 pin 即可;
根治要补资产或改 list.json(影响所有 clutter 任务,需单独决策)。

## 5. 事故记录(为什么会有这份文档)

2026-08-10:blocks_ranking_rgb 首次闭环评测,用「完整评测顺便挖 seed」的
bootstrap 模式。demo_clean 跑到 ~48 集时 ffmpeg 编码进程无声死亡
(疑似 OOM,`-loglevel error` 下无输出),`BrokenPipeError` 从
`_write_eval_video_frame` 一路冒顶,整场作废,seed 清单一无所获,连带阻塞
所有等它 pin 的下游任务。两个修复:

1. `envs/_base_task.py`:视频管道死亡不再致命(catch + 继续无视频评测);
2. 本文档的挖法:seed 挖掘与策略评测解耦,bootstrap 模式废弃。
   uniform5 基线改为和其他 schedule 一样 pin 着跑——更干净,也不再有
   「基线 run 身兼挖矿职责」的耦合。
