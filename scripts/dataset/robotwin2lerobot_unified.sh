#!/bin/bash
echo "Current path: $(pwd)"

if [ "$#" -ne 2 ]; then
    echo "Usage: $0 <source_data_root> <target_data_root>"
    echo "Example: $0 datasets/robotwin2.0_D435_Agilex_200_rgb datasets/drrm_robotwin2.0_D435_Agilex_200_rgb"
    exit 1
fi

pkl_path=$1
echo "Source path: $pkl_path"
if [ -d "$pkl_path" ]; then
    abs_path="$(realpath "$pkl_path" 2>/dev/null || echo "Conversion failed")"
    echo "Absolute path: $abs_path"
else
    echo "Path does not exist"
    exit 1
fi

lerobot_path=$2
echo "Target path: $lerobot_path"
python tools/preprocess_robotwin2_unified.py \
    --src_dir "$pkl_path" \
    --dst_dir "$lerobot_path" \
    --repo D-robotics/drrm_robotwin2.0_D435_Agilex_200_rgb \
    --fps 30