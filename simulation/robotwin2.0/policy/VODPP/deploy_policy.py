import os
import numpy as np
from drrm.models.policy.vodpp.env_runner.base_runner import VODPPRunner
import yaml

ROOT_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), "../../../.."))

def encode_obs(observation):
    # Match training: float32 CHW in [0, 1]; this ckpt uses head_cam + front_cam + state.
    head_cam = (np.moveaxis(observation["observation"]["head_camera"]["rgb"], -1, 0).astype(np.float32) / 255.0)
    front_cam = (np.moveaxis(observation["observation"]["front_camera"]["rgb"], -1, 0).astype(np.float32) / 255.0)
    obs = dict(
        head_cam=head_cam,
        front_cam=front_cam,
    )
    # optional wrist cams if present (ignored by 2v policy via shape_meta / rgb_key)
    if "left_camera" in observation["observation"]:
        obs["left_cam"] = (np.moveaxis(observation["observation"]["left_camera"]["rgb"], -1, 0).astype(np.float32) / 255.0)
    if "right_camera" in observation["observation"]:
        obs["right_cam"] = (np.moveaxis(observation["observation"]["right_camera"]["rgb"], -1, 0).astype(np.float32) / 255.0)

    obs["state"] = np.asarray(observation["joint_action"]["vector"], dtype=np.float32)

    if "endpose" in observation:
        endpose = observation["endpose"]
        obs["endpose"] = np.concatenate([
            endpose["left_endpose"],
            [endpose["left_gripper"]],
            endpose["right_endpose"],
            [endpose["right_gripper"]],
        ]).astype(np.float32)

    return obs


def get_model(usr_args):
    setting = usr_args['ckpt_setting']
    # Prefer absolute / existing checkpoint path (mix5 multitask ckpt etc.)
    if os.path.isabs(setting) or os.path.exists(setting):
        ckpt_file = setting
    else:
        ckpt_file = f"{ROOT_DIR}/checkpoints/vodpp/{usr_args['task_name']}_{usr_args['expert_data_num']}/{setting}"
    if usr_args.get('checkpoint_num') is not None:
        ckpt_file = f"{ckpt_file}/checkpoint-{usr_args['checkpoint_num']}"
    kwargs = {'mixed_precision': 'fp32'}
    if 'device' in usr_args:
        kwargs['device'] = f"cuda:{usr_args['device']}"
    # Flow-Matching Euler tau knots, e.g. "0,0.25,0.5,1" or "uniform10"
    fm_tau = usr_args.get('fm_tau_schedule', usr_args.get('inference_tau_schedule'))
    if fm_tau is not None:
        kwargs['inference_tau_schedule'] = fm_tau
    return VODPPRunner(checkpoint_dir=ckpt_file, **kwargs)


def eval(TASK_ENV, model, observation):
    """
    TASK_ENV: Task Environment Class, you can use this class to interact with the environment
    model: The model from 'get_model()' function
    observation: The observation about the environment
    """
    obs = encode_obs(observation)
    instruction = TASK_ENV.get_instruction()
    # if all([(model.obs[-1][k] == obs[k]).all() for k in model.obs[-1]]):
    #     print("Warning: Redundant Observation Detected!")

    # ======== Get Action ========
    actions, action_type = model.get_action(obs)

    for action in actions:
        kwargs = {'action_type': action_type} if action_type else {}
        TASK_ENV.take_action(action, **kwargs)
        observation = TASK_ENV.get_obs()
        obs = encode_obs(observation)
        model.update_obs(obs)

def reset_model(model):
    model.reset_obs()