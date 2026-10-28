import os
import pickle
import json
import random
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Any
import cv2
import h5py
import numpy as np
import argparse
from loguru import logger
import shutil
from PIL import Image
import io

from lerobot.datasets.lerobot_dataset import LeRobotDataset
from tqdm import tqdm

''' HDF5 structure of robotwin2.0 episode file:
endpose
    left_endpose
        <HDF5 dataset "left_endpose": shape (116, 7), type "<f8">
    left_gripper
        <HDF5 dataset "left_gripper": shape (116,), type "<f8">
    right_endpose
        <HDF5 dataset "right_endpose": shape (116, 7), type "<f8">
    right_gripper
        <HDF5 dataset "right_gripper": shape (116,), type "<f8">
joint_action
    left_arm
        <HDF5 dataset "left_arm": shape (116, 6), type "<f8">
    left_gripper
        <HDF5 dataset "left_gripper": shape (116,), type "<f8">
    right_arm
        <HDF5 dataset "right_arm": shape (116, 6), type "<f8">
    right_gripper
        <HDF5 dataset "right_gripper": shape (116,), type "<f8">
    vector
        <HDF5 dataset "vector": shape (116, 14), type "<f8">
observation
    front_camera
        cam2world_gl
            <HDF5 dataset "cam2world_gl": shape (116, 4, 4), type "<f4">
        extrinsic_cv
            <HDF5 dataset "extrinsic_cv": shape (116, 3, 4), type "<f4">
        intrinsic_cv
            <HDF5 dataset "intrinsic_cv": shape (116, 3, 3), type "<f4">
        rgb
            <HDF5 dataset "rgb": shape (116,), type "|S15034">
    head_camera
        cam2world_gl
            <HDF5 dataset "cam2world_gl": shape (116, 4, 4), type "<f4">
        extrinsic_cv
            <HDF5 dataset "extrinsic_cv": shape (116, 3, 4), type "<f4">
        intrinsic_cv
            <HDF5 dataset "intrinsic_cv": shape (116, 3, 3), type "<f4">
        rgb
            <HDF5 dataset "rgb": shape (116,), type "|S17891">
    left_camera
        cam2world_gl
            <HDF5 dataset "cam2world_gl": shape (116, 4, 4), type "<f4">
        extrinsic_cv
            <HDF5 dataset "extrinsic_cv": shape (116, 3, 4), type "<f4">
        intrinsic_cv
            <HDF5 dataset "intrinsic_cv": shape (116, 3, 3), type "<f4">
        rgb
            <HDF5 dataset "rgb": shape (116,), type "|S5958">
    right_camera
        cam2world_gl
            <HDF5 dataset "cam2world_gl": shape (116, 4, 4), type "<f4">
        extrinsic_cv
            <HDF5 dataset "extrinsic_cv": shape (116, 3, 4), type "<f4">
        intrinsic_cv
            <HDF5 dataset "intrinsic_cv": shape (116, 3, 3), type "<f4">
        rgb
            <HDF5 dataset "rgb": shape (116,), type "|S13549">
pointcloud
    <HDF5 dataset "pointcloud": shape (116, 0), type "<f8">
'''

# Constants
# TASK_STR = "dual bottles pick easy"
CAMERA_SHAPE = (240, 320, 3)
# Feature schema definition for the LeRobot dataset
FEATURES = {
    "head_cam": {"dtype": "image", "shape": CAMERA_SHAPE, "names": ["h", "w", "c"]},
    "front_cam": {"dtype": "image", "shape": CAMERA_SHAPE, "names": ["h", "w", "c"]},
    "left_cam": {"dtype": "image", "shape": CAMERA_SHAPE, "names": ["h", "w", "c"]},
    "right_cam": {"dtype": "image", "shape": CAMERA_SHAPE, "names": ["h", "w", "c"]},
    "endpose": {"dtype": "float32", "shape": (16,), "names": ["endpose"]},
    "state": {"dtype": "float32", "shape": (14,), "names": ["state"]},
    "action": {"dtype": "float32", "shape": (14,), "names": ["action"]},
}

TASK_LIST = [
    'adjust_bottle', 'click_bell', 'hanging_mug', 'move_stapler_pad', 'place_a2b_left',     
    'place_can_basket', 'place_fan', 'place_phone_stand', 'rotate_qrcode', 'stack_blocks_two',
    'beat_block_hammer', 'dump_bin_bigbin', 'lift_pot', 'open_laptop', 'place_a2b_right',
    'place_cans_plasticbox', 'place_mouse_pad', 'place_shoe', 'scan_object', 'stack_bowls_three',
    'blocks_ranking_rgb', 'grab_roller', 'move_can_pot', 'open_microwave', 'place_bread_basket', 
    'place_container_plate', 'place_object_basket', 'press_stapler', 'shake_bottle', 'stack_bowls_two',
    'blocks_ranking_size', 'handover_block', 'move_pillbottle_pad', 'pick_diverse_bottles', 'place_bread_skillet',
    'place_dual_shoes', 'place_object_scale', 'put_bottles_dustbin', 'shake_bottle_horizontally', 'stamp_seal',
    'click_alarmclock', 'handover_mic', 'move_playingcard_away', 'pick_dual_bottles', 'place_burger_fries',
    'place_empty_cup', 'place_object_stand', 'put_object_cabinet', 'stack_blocks_three', 'turn_switch'
]

def _extract_frame(data: Dict[str, Any], instructions: list) -> Dict[str, Any]:
    """Convert raw pickle dictionary to the format expected by LeRobot."""
    endpose = np.concatenate((
        data["endpose"]['left_endpose'], 
        [data["endpose"]['left_gripper']], 
        data["endpose"]['right_endpose'],
        [data["endpose"]['right_gripper']]
    )).astype(np.float32)
    head_cam = cv2.imdecode(np.frombuffer(data["observation"]["head_camera"]["rgb"], dtype=np.uint8), cv2.IMREAD_COLOR)
    front_cam = cv2.imdecode(np.frombuffer(data["observation"]["front_camera"]["rgb"], dtype=np.uint8), cv2.IMREAD_COLOR)
    left_cam = cv2.imdecode(np.frombuffer(data["observation"]["left_camera"]["rgb"], dtype=np.uint8), cv2.IMREAD_COLOR)
    right_cam = cv2.imdecode(np.frombuffer(data["observation"]["right_camera"]["rgb"], dtype=np.uint8), cv2.IMREAD_COLOR)
    return {
        # "instructions": instructions,
        "state": data["joint_action"]['vector'].astype(np.float32),
        "action": data["joint_action"]['vector'].astype(np.float32),
        "endpose": endpose,
        "head_cam": head_cam.astype(np.float32) / 255.0,
        "front_cam": front_cam.astype(np.float32) / 255.0,
        "left_cam": left_cam.astype(np.float32) / 255.0,
        "right_cam": right_cam.astype(np.float32) / 255.0,
    }

def get_i_hdf5(f, i):
    def get_items(f):
        result = {}
        for k,v in f.items():
            if isinstance(v, h5py._hl.group.Group):
                result[k] = get_items(v)
            elif isinstance(v, h5py._hl.dataset.Dataset):
                result[k] = v[i]
            else:
                result[k] = v[i]
        return result
    return get_items(f)

def _process_episode(dataset: LeRobotDataset, episode_dir: Path, task: str, instructions: list) -> None:
    """Convert a single episode and append its frames to *ds*."""

    # Sort pickle files numerically to preserve temporal order
    # pkl_files = sorted(episode_dir.glob("*.pkl"), key=lambda p: int(p.stem))
    with h5py.File(episode_dir, "r") as f:
        # keys = list(f.keys())
        # print(keys)
        
        for i in range(len(f['pointcloud'])):
            frame = _extract_frame(get_i_hdf5(f, i), instructions=instructions)
            dataset.add_frame(frame, task=task)

    # Persist episode to disk so that an unexpected crash does not lose work.
    dataset.save_episode()
    dataset.clear_episode_buffer()


def main(args):
    # Create (or open) the target dataset
    ds = LeRobotDataset.create(
        root=args.dst_dir,
        repo_id=args.repo,
        fps=args.fps,
        robot_type="AgileBot",
        features=FEATURES,
        image_writer_threads=4,  # Use async PNG writing to improve I/O performance
        use_videos=False,
    )

    for task in tqdm(TASK_LIST):
        src_dir = os.path.join(args.src_dir, f"{task}/agilex_config/data")
        instruction_dir = os.path.join(args.src_dir, f"{task}/agilex_config/instructions")
        for episode_idx in range(len([d for d in os.listdir(src_dir)])):
            episode_dir = Path(os.path.join(src_dir, f"episode{episode_idx}.hdf5"))
            assert os.path.exists(episode_dir), f"Episode directory {episode_dir} does not exist"
            
            instruction_file = Path(os.path.join(instruction_dir, f"episode{episode_idx}.json"))
            with open(instruction_file, 'r', encoding='utf-8') as f:
                instruction_data = json.load(f)
            instructions = instruction_data['seen']
            
            _process_episode(ds, episode_dir, task, instructions)

            img_dir = os.path.join(args.dst_dir, "images")
            if os.path.exists(img_dir):
                shutil.rmtree(img_dir)

    if args.push:
        logger.info("Pushing dataset to the HuggingFace Hub …")
        ds.push_to_hub(push_videos=True, tags=["emvis", "converted"])

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--src_dir", type=str, required=True, help="Source directory containing episode folders, robotwin format")
    parser.add_argument("--dst_dir", type=str, required=True, help="Destination directory for the dataset")
    parser.add_argument("--repo", type=str, required=True, help="HuggingFace repository ID")
    # parser.add_argument("--task", type=str, required=True, help="task name")
    parser.add_argument("--fps", type=int, default=30, help="Frames per second for the dataset")
    parser.add_argument("--push", type=bool, default=False, help="Whether to push the dataset to the HuggingFace Hub")
    args = parser.parse_args()

    if os.path.exists(args.dst_dir):
        shutil.rmtree(args.dst_dir)

    main(args)