#!/bin/bash
policy_config=$1
model_config=$2
train_demo=${3:-50}
num_processes=${4:-8}

task_list=(
    'open_laptop' 'move_stapler_pad' 'press_stapler'
    'place_a2b_left' 'stack_blocks_two' 
    'hanging_mug' 'beat_block_hammer' 'dump_bin_bigbin' 
    'handover_block' 'blocks_ranking_rgb'
)
config_list=(
    'agilex_config'
    # 'agilex_config_mid' 
    # 'agilex_config_hard'
)

policy="${config/_*/}"
for task in "${task_list[@]}"; do
    for config in "${config_list[@]}"; do
        echo $task $config $model_config
        ./scripts/eval/eval_robotwin2.0_parallel.sh $policy_config $task $config $model_config $train_demo $num_processes
    done
done