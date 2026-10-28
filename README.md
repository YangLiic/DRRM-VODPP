# D-robotics Robotic Manipulation Platform
## 🔥 Latest Work: VO-DP
[![project page](https://img.shields.io/badge/Project%20Page-DRRM-orange)](https://d-robotics-ai-lab.github.io/vodp/)
[![arXiv paper](https://img.shields.io/badge/Paper-arXiv-red)](https://arxiv.org/abs/2510.15530)
[![dataset](https://img.shields.io/badge/Dataset-HuggingFace-yellow)](https://huggingface.co/datasets/D-Robotics/DRRM)

https://github.com/user-attachments/assets/fdca37aa-164b-4281-a446-3c909a3f1456

## 🌟 Update Log
- **2025-11**: xxxxxxxx

## ⚙️ Installation
### Basic Environment Setup
```bash
git clone https://github.com/D-Robotics-AI-Lab/DRRM.git
cd DRRM
conda create -n drrm python=3.10
conda activate drrm
pip install -e .
```

### VODP Environment Setup

```bash
mkdir -p pretrained
cd pretrained
git clone https://github.com/facebookresearch/vggt.git
cd vggt
pip install .
cd ../..
```

### VODPP Environment Setup

### Robotwin Environment Setup
See `simulation/robotwin/README.md` for detailed simulator installation and usage steps.

## 📊 Dataset Preparation
Create the `datasets/` directory and download one of the preprocessed dataset bundles from HuggingFace:

```bash
mkdir -p datasets
```

Preprocessed bundles available (choose one):
- [`drrm_robotwin1.0_D435_200_rgb`](https://huggingface.co/datasets/D-Robotics/DRRM/tree/main/drrm_robotwin1.0_D435_200_rgb) — RGB-only version (no point clouds)
- [`drrm_robotwin1.0_D435_200_pcd`](https://huggingface.co/datasets/D-Robotics/DRRM/tree/main/drrm_robotwin1.0_D435_200_pcd) — includes point clouds

Visit https://huggingface.co/datasets/D-Robotics/DRRM to download the dataset and place it under `datasets/`.

### Processing Raw Datasets (Optional)
If you are working with raw datasets from Robotwin 2.0 or real-world robots (D-robotics), you can use our provided shell scripts to convert them into LeRobot format.

#### Process Real-Robot (D-robotics) Data
You can process single or multiple tasks using the unified conversion script. The script automatically leverages GNU Parallel for multi-task runs.
```bash
# Usage: ./scripts/dataset/drobotics2lerobot.sh <source_data_root> <target_data_root> <task1> [task2 ...]

# Process a single task:
./scripts/dataset/drobotics2lerobot.sh ./datasets/drobotics_raw ./datasets/drrm_drobotics_processed task_1

# Process multiple tasks in parallel:
./scripts/dataset/drobotics2lerobot.sh ./datasets/drobotics_raw ./datasets/drrm_drobotics_processed task_1 task_2 task_3
```

#### Process Robotwin 2.0 Data
Similar to the real-world data, use the unified script to parse Robotwin 2.0 datasets.
```bash
# Process single or multiple tasks:
./scripts/dataset/robotwin2lerobot.sh ./datasets/robotwin2.0_D435_Agilex_200_rgb ./datasets/drrm_robotwin2.0_D435_Agilex_200_rgb adjust_bottle click_bell
```

To assemble multiple tasks into a unified structured dataset:
```bash
# Usage: ./scripts/dataset/robotwin2lerobot_unified.sh <source_data_root> <target_data_root>
./scripts/dataset/robotwin2lerobot_unified.sh datasets/robotwin2.0_D435_Agilex_200_rgb datasets/drrm_robotwin2.0_D435_Agilex_200_rgb
```

## 📑 Training
1. Modify the acceleration configuration file based on your training environment: [configs/accelerate_config.yaml](configs/accelerate_config.yaml)
2. In the training script [scripts/train_demo.sh](scripts/train_demo.sh), specify the following parameters:
   - `dataset`: Path to your training dataset
   - `task`: Your training task
   - `demo`: Number of demonstration samples to use (set to `null` for unlimited)
   - `config_dir`: Training configuration directory (refer to [configs/](configs/))

- Training VODP
```bash
accelerate launch\
    --config_file configs/accelerate_config.yaml \
    main.py \
    --config-path="configs/vodp_train" \
    --config-name="vodp_23d_1f.yaml" \
    train_dataset.path=datasets/lerobot_D435_200 \
    train_dataset.task=block_hammer_beat \
    train_dataset.demo=100
```

- Training VODPP
```bash
accelerate launch\
    --config_file configs/accelerate_config.yaml \
    main.py \
    --config-path="configs/vodpp_train" \
    --config-name="dit_ddpm_dinov3cnnl_1v" \
    train_dataset.path=datasets/lerobot_D435_200 \
    train_dataset.task=block_hammer_beat \
    train_dataset.demo=100
```

- Training DP
```bash
accelerate launch\
    --config_file configs/accelerate_config.yaml \
    main.py \
    --config-path="configs/dp_train" \
    --config-name="dp.yaml" \
    train_dataset.path=datasets/lerobot_D435_200 \
    train_dataset.task=block_hammer_beat \
    train_dataset.demo=100
```

- Training DP3
```bash
accelerate launch\
    --config_file configs/accelerate_config.yaml \
    main.py \
    --config-path="configs/dp3_train" \
    --config-name="dp3.yaml" \
    train_dataset.path=datasets/lerobot43d_D435_200 \
    train_dataset.task=block_hammer_beat \
    train_dataset.demo=100
```

### Advanced: Configuring VODPP for Custom Robots and Datasets

The VODPP configurations are modularized under `configs/vodpp_train/` into four directories: `dataset_config/`, `robot_config/`, `observation_config/`, and `policy_config/`.

1. **Configure Dataset (`dataset_config/`)**
   Create a yaml file here (e.g., `dataset_single.yaml`) specifying your local path and vision feature requirements:
   ```yaml
   dataset:
     _target_: drrm.datasets.drrm_dataset.DRRMDataset
     repo_id: dummy
     root: datasets/lerobot_D435_200
     task_list: [ adjust_bottle ]
   ```

2. **Configure Robot shape_meta (`robot_config/`)**
   Define your robot's observation space and action bounds (e.g., `robotwin2_agilex.yaml`):
   ```yaml
   shape_meta:
     action:
       shape: ${eval:'[${horizon}, 14]'} # Dimension matches your robot's arm Action
     obs:
       agent_pos:
         shape: [14]
       camera1_rgb:
         shape: [3, 224, 224]
         type: rgb
   ```

3. **Configure Observation Encoder (`observation_config/`)**
   Define the vision encoder backbone structure and pre-trained weights (e.g., `dinov3cnnl.yaml`):
   ```yaml
   model:
     obs_encoder:
       emvis_config:
         dim_2d: 1536
         vggt_target: 'dinov3cnn'
         vggt_model_path: pretrained/dinov3-cnnl
         # ... other vision encoder settings ...
   ```

4. **Assemble the Top-Level Configuration**
   In the top-level VODPP config (e.g., `configs/vodpp_train/dit_ddpm_dinov3cnnl_1v.yaml`), reference all your modular config names under `defaults`:
   ```yaml
   defaults:
     - policy_config: dit_ddpm
     - robot_config: robotwin2_agilex  # Your custom robot configuration
     - dataset_config: dataset_single  # Your custom dataset configuration
     - observation_config: dinov3cnnl  # Your chosen vision encoder
     - _self_
   ```

Run the model using your custom config setup by pointing to it via `--config-name`:
```bash
accelerate launch\
    --config_file configs/accelerate_config.yaml \
    main.py \
    --config-path="configs/vodpp_train" \
    --config-name="YOUR_CUSTOM_CONFIG_NAME"
```

## 🤖 Simulation Evaluation
DRRM is compatible with the Robotwin simulator — refer to the following files:

- `scripts/eval/eval_robotwin.sh` — convenience shell wrapper used for experiments.
- `scripts/eval/robotwin_exp/` — example experiment wrappers used in our paper.
- `simulation/robotwin/script/` — simulator-facing Python evaluation scripts (e.g. `eval_policy_vodp.py`, `eval_policy_dp.py`, `eval_policy_dp3.py`).

Supported benchmark tasks:

```
[
    'block_hammer_beat', 'bottle_adjust', 'container_place',
    'dual_bottles_pick_hard', 'put_apple_cabinet',
    'tool_adjust', 'pick_apple_messy', 'dual_bottles_pick_easy',
    'diverse_bottles_pick', 'empty_cup_place', 'shoe_place',
    'dual_shoes_place', 'blocks_stack_easy', 'block_handover'
]
```


### Basic usage
Replace `YOUR/CHECKPOINT/DIR` and `YOUR/SAVE/DIR` with the paths to your checkpoint directory and the directory where you want to store evaluation results. The `--num-process` flag controls parallel simulator workers.

```bash
# Evaluate VODP
python simulation/robotwin/script/eval_policy_vodp.py \
    --checkpoint-dir YOUR/CHECKPOINT/DIR \
    --save-dir YOUR/SAVE/DIR \
    --task-name TASK_NAME \
    --num-process 8 \
    --seed 0
```

```bash
# Evaluate DP
python simulation/robotwin/script/eval_policy_dp.py \
    --checkpoint-dir YOUR/CHECKPOINT/DIR \
    --save-dir YOUR/SAVE/DIR \
    --task-name TASK_NAME \
    --num-process 8 \
    --seed 0
```

```bash
# Evaluate DP3
python simulation/robotwin/script/eval_policy_dp3.py \
    --checkpoint-dir YOUR/CHECKPOINT/DIR \
    --save-dir YOUR/SAVE/DIR \
    --task-name TASK_NAME \
    --num-process 8 \
    --seed 0
```

### Robotwin 2.0 Evaluation
For the newer **Robotwin 2.0** environments, we provide parallelized batch testing scripts to evaluate models across multiple environment instances efficiently.

To evaluate a specific policy configuration against the 10 benchmark tasks, you can use the multi-task runner script. It accepts the policy name, your model checkpoint config, and optionally the number of parallel simulator processes (default is 8).

```bash
# General Usage:
# ./scripts/eval/robotwin2_exp/eval_robotwin2.0_10task.sh <policy_config_name> <checkpoint_dir_name> [num_processes]

# Example - Evaluating VODPP with 16 parallel processes:
./scripts/eval/robotwin2_exp/eval_robotwin2.0_10task.sh vodpp dit_ddpm_dinov3cnnl_1v 16
```

Under the hood, this iterates through tasks like `open_laptop`, `move_stapler_pad`, and `handover_block`, invoking `scripts/eval/eval_robotwin2.0_parallel.sh`, which loads the policy from `simulation/robotwin2.0/policy/<policy_name>/deploy_policy.yml`. Ensure you have properly synced your checkpoints before running.

## 👏 Citation
```
@article{ni2025vodp,
  title={VO-DP: Semantic-Geometric Adaptive Diffusion Policy for Vision-Only Robotic Manipulation},
  author={Zehao Ni and Yonghao He and Lingfeng Qian and Jilei Mao and Fa Fu and Wei Sui and Hu Su and Junran Peng and Zhipeng Wang and Bin He},
  journal={arXiv preprint arXiv:2510.15530},
  year={2025}
}
```
