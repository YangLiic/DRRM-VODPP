import os
import numpy as np
from drrm.models.policy.var0.env_runner.base_runner import VAR0Runner
import yaml

ROOT_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), "../../../.."))

def encode_obs(observation):
    head_cam = (np.moveaxis(observation["observation"]["head_camera"]["rgb"], -1, 0) / 255)
    left_cam = (np.moveaxis(observation["observation"]["left_camera"]["rgb"], -1, 0) / 255)
    right_cam = (np.moveaxis(observation["observation"]["right_camera"]["rgb"], -1, 0) / 255)
    front_cam = (np.moveaxis(observation["observation"]["front_camera"]["rgb"], -1, 0) / 255)
    obs = dict(
        head_cam=head_cam,
        left_cam=left_cam,
        right_cam=right_cam,
        front_cam=front_cam,
    )
    obs["agent_pos"] = observation["joint_action"]["vector"]
    
    endpose = observation["endpose"]
    obs["endpose"] = np.concatenate([
        endpose["left_endpose"], 
        [endpose["left_gripper"]], 
        endpose["right_endpose"], 
        [endpose["right_gripper"]]
    ])
    
    return obs


def get_model(usr_args):
    ckpt_file = f"{ROOT_DIR}/checkpoints/var0/{usr_args['task_name']}_{usr_args['expert_data_num']}/{usr_args['ckpt_setting']}"
    if usr_args['checkpoint_num'] is not None:
        ckpt_file = f"{ckpt_file}/checkpoint-{usr_args['checkpoint_num']}"
    kwargs = {}
    if 'device' in usr_args:
        kwargs['device'] = f"cuda:{usr_args['device']}"
    return VAR0Runner(checkpoint_dir=ckpt_file, **kwargs)


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
