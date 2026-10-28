import torch
import os
import time
import numpy as np
import hydra
from pathlib import Path
from collections import deque

import yaml
from datetime import datetime
import importlib
import dill
from argparse import ArgumentParser
from drrm.common.pytorch_util import dict_apply
from drrm.models.base_policy import BasePolicy
from drrm.models.base_runner import BaseRunner, load_policy


class VODPPRunner(BaseRunner):

    def __init__(
        self,
        policy: BasePolicy = None,
        checkpoint_dir: str = None,
        mixed_precision: str = 'fp32',
        output_dir: str = None,
        eval_episodes: int = 20,
        max_steps: int = 300,
        # n_obs_steps = 3,
        # n_action_steps = 8,
        fps: int = 10,
        crf: int = 22,
        tqdm_interval_sec: float = 5.0,
        task_name: str = None,
        inference_tau_schedule=None,
        **kwargs,
    ):
        super().__init__(output_dir)
        device = kwargs.pop('device', None)
        # alias
        if inference_tau_schedule is None:
            inference_tau_schedule = kwargs.pop('fm_tau_schedule', None)
        else:
            kwargs.pop('fm_tau_schedule', None)

        if policy is not None:
            self.policy = policy
        else:
            assert checkpoint_dir is not None, "Either policy or checkpoint_dir must be provided"
            self.policy = load_policy(checkpoint_dir, use_ckp_code=False)
            # Checkpoint may store DiT in bf16 while visual/normalizer are fp32.
            # Cast everything to fp32 for stable sim eval (avoids TimestepEmbedder
            # emitting bf16 into fp32 Linear when autocast is off / mixed).
            self.policy.float()
            self.policy.eval()
            if device is not None:
                self.policy.to(device)
            else:
                self.policy.to('cuda')
        if inference_tau_schedule is not None and hasattr(self.policy, 'set_inference_tau_schedule'):
            self.policy.set_inference_tau_schedule(inference_tau_schedule)
            print(f"[FM] inference tau schedule: {self.policy.inference_tau_schedule} "
                  f"(NFE={len(self.policy.inference_tau_schedule)-1})")
        elif hasattr(self.policy, 'inference_tau_schedule'):
            print(f"[FM] inference tau schedule: {self.policy.inference_tau_schedule} "
                  f"(NFE={len(self.policy.inference_tau_schedule)-1})")
        if mixed_precision == 'bf16':
            self.dtype = torch.bfloat16
        elif mixed_precision == 'fp16':
            self.dtype = torch.float16
        else:
            self.dtype = torch.float32

        self.task_name = task_name
        self.eval_episodes = eval_episodes
        self.fps = fps
        self.crf = crf
        self.n_obs_steps = self.policy.n_obs_steps
        self.n_action_steps = self.policy.n_action_steps
        self.max_steps = max_steps
        self.tqdm_interval_sec = tqdm_interval_sec

        self.obs = deque(maxlen=self.n_obs_steps + 1)
        self.env = None
        # Per-call policy inference latency (seconds), reset each episode via reset_obs()
        self.infer_latencies = []
        self.last_infer_s = None

    def _sync_policy_device(self):
        device = getattr(self.policy, "device", None)
        if device is None:
            return
        if torch.cuda.is_available() and "cuda" in str(device):
            torch.cuda.synchronize(device)

    def stack_last_n_obs(self, all_obs, n_steps):
        assert len(all_obs) > 0
        all_obs = list(all_obs)
        if isinstance(all_obs[0], np.ndarray):
            result = np.zeros((n_steps, ) + all_obs[-1].shape, dtype=all_obs[-1].dtype)
            start_idx = -min(n_steps, len(all_obs))
            result[start_idx:] = np.array(all_obs[start_idx:])
            if n_steps > len(all_obs):
                # pad
                result[:start_idx] = result[start_idx]
        elif isinstance(all_obs[0], torch.Tensor):
            result = torch.zeros((n_steps, ) + all_obs[-1].shape, dtype=all_obs[-1].dtype)
            start_idx = -min(n_steps, len(all_obs))
            result[start_idx:] = torch.stack(all_obs[start_idx:])
            if n_steps > len(all_obs):
                # pad
                result[:start_idx] = result[start_idx]
        else:
            raise RuntimeError(f"Unsupported obs type {type(all_obs[0])}")
        return result

    def reset_obs(self):
        self.obs.clear()
        self.infer_latencies = []
        self.last_infer_s = None

    def get_infer_stats(self):
        """Return inference latency stats for the current episode."""
        times = list(self.infer_latencies)
        if not times:
            return {
                "infer_cnt_timed": 0,
                "infer_ms_mean": None,
                "infer_ms_max": None,
                "infer_time_s": 0.0,
            }
        arr = np.asarray(times, dtype=np.float64)
        return {
            "infer_cnt_timed": int(arr.size),
            "infer_ms_mean": float(arr.mean() * 1000.0),
            "infer_ms_max": float(arr.max() * 1000.0),
            "infer_time_s": float(arr.sum()),
        }

    def update_obs(self, current_obs):
        self.obs.append(current_obs)

    def get_n_steps_obs(self):
        assert len(self.obs) > 0, "no observation is recorded, please update obs first"

        result = dict()
        for key in self.obs[0].keys():
            result[key] = self.stack_last_n_obs([obs[key] for obs in self.obs], self.n_obs_steps)

        return result

    def get_action(self, observaton=None):
        policy: BasePolicy = self.policy
        if observaton != None:
            self.obs.append(observaton)  # update
        device, dtype = policy.device, self.dtype
        obs = self.get_n_steps_obs()

        # create obs dict
        np_obs_dict = dict(obs)
        # device transfer (force float32 — deploy /255 often yields float64)
        obs_dict = dict_apply(np_obs_dict, lambda x: torch.from_numpy(x).float().to(device=device))
        # run policy — only feed keys declared in shape_meta.obs
        with torch.no_grad():
            obs_dict_input = {}
            shape_meta_keys = set(policy.config.shape_meta.get('obs', {}).keys())
            for key in obs_dict:
                if key in shape_meta_keys:
                    obs_dict_input[key] = obs_dict[key].unsqueeze(0)
            # always ensure action_key state is present even if naming drifts
            if policy.action_key not in obs_dict_input and policy.action_key in obs_dict:
                obs_dict_input[policy.action_key] = obs_dict[policy.action_key].unsqueeze(0)

            device_type = str(device).split(':')[0]
            # Measure policy forward + D2H (H2D already done above).
            self._sync_policy_device()
            t0 = time.perf_counter()
            if dtype == torch.float32:
                action_dict = policy.predict_action(obs_dict_input)
            else:
                with torch.autocast(device_type=device_type, dtype=dtype):
                    action_dict = policy.predict_action(obs_dict_input)
            np_action_dict = dict_apply(action_dict, lambda x: x.detach().float().to("cpu").numpy())
            self._sync_policy_device()
            infer_s = time.perf_counter() - t0
            self.last_infer_s = infer_s
            self.infer_latencies.append(infer_s)

        action = np_action_dict["action"].squeeze(0)[:self.n_action_steps]
        return action, 'ee' if 'endpose' in str(policy.action_key) else None
    
    def get_last_obs(self):
        return self.obs[-1]

    def run(self, policy: BasePolicy):
        pass

if __name__ == '__main__':
    test = VODPPRunner('./')
    print('ready')