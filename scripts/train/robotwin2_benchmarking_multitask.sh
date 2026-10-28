#!/bin/bash
demo=100
positional_args=()

while [[ $# -gt 0 ]]; do
    case $1 in
        --config_dir=*) config_dir="${1#*=}"; shift ;;
        --task=*) task="${1#*=}"; shift ;;
        --num_processes=*) num_processes="${1#*=}"; shift ;;
        --demo=*) demo="${1#*=}"; shift ;;
        --*) echo "Unknown parameter: $1"; exit 1 ;;
        *) positional_args+=("$1"); shift ;;
    esac
done

if [ -z "$config_dir" ] && [ ${#positional_args[@]} -ge 1 ]; then config_dir="${positional_args[0]}"; fi
if [ -z "$task" ] && [ ${#positional_args[@]} -ge 2 ]; then task="${positional_args[1]}"; fi
if [ -z "$num_processes" ] && [ ${#positional_args[@]} -ge 3 ]; then num_processes="${positional_args[2]}"; fi
if [ ${#positional_args[@]} -ge 4 ]; then demo="${positional_args[3]}"; fi

if [ -n "$task" ]; then
    task_list=("$task")
else
    task_list=(
        'open_laptop' 'move_stapler_pad' 'press_stapler'
        'place_a2b_left' 'stack_blocks_two' 
        'hanging_mug' 'beat_block_hammer' 'dump_bin_bigbin' 
        'handover_block' 'blocks_ranking_rgb'
    )
fi

num_flag=""
if [ -n "$num_processes" ]; then
    num_flag="--num_processes=$num_processes"
fi

for task in "${task_list[@]}"; do
    echo "Training task: $task ($demo) with config: $config_dir"
    accelerate launch\
        --config_file configs/accelerate_config.yaml \
        $num_flag \
        main.py \
        --config-path="${config_dir%/*}" \
        --config-name="${config_dir##*/}" \
        train_dataset.path=datasets/drrm_robotwin2.0_D435_Agilex_200_rgb \
        train_dataset.task=$task \
        train_dataset.demo=$demo
    
    config_name="${config_dir##*/}"
    config="${config_name%.*}"
    ckpt_name="${config/_*/}/${task}_${demo}/${config#*_}"
    # ckpt_name="${config/_*/}"
    echo ./checkpoints/$ckpt_name

    if [ -n "$BACKUP_HOME" ] && [ -d "$BACKUP_HOME/checkpoints" ]; then
        cp -r "./checkpoints/$ckpt_name" "$BACKUP_HOME/checkpoints/$ckpt_name"
        echo "Backup completed: $ckpt_name has been copied to $BACKUP_HOME/checkpoints/"
    else
        echo "Warning: BACKUP_HOME is not set or checkpoints directory does not exist"
        exit 1
    fi
done

