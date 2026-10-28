import re
from typing import Dict
import hydra
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from einops import rearrange, reduce
from diffusers.schedulers.scheduling_ddpm import DDPMScheduler
from diffusers.schedulers.scheduling_dpmsolver_multistep import \
    DPMSolverMultistepScheduler

from drrm.models.base_policy import BasePolicy
# from .module.obs_encoder import SceneEncoder as VODPPlusEncoder
from .emvis.embodied_vision import VisualEncoder
from .common.normalizer import LinearNormalizer
from .common.pytorch_util import dict_apply
from .common.module_attr_mixin import ModuleAttrMixin
from .common.fm_schedule import resolve_inference_tau_plan, tau_to_timestep_bucket

import yaml
import json
from dataclasses import dataclass
from typing import Optional
from transformers import PretrainedConfig, PreTrainedModel

@dataclass
class VODPPlusDitFlowMatchingConfig(PretrainedConfig):
    shape_meta: dict
    noise_scheduler: dict
    obs_encoder: dict
    hidden_size: int
    depth: int
    num_heads: int
    horizon: int
    n_action_steps: int
    n_obs_steps: int
    action_key: str
    obs_as_global_cond: bool = True
    diffusion_step_embed_dim: int = 256
    down_dims: tuple = (256,512,1024)
    kernel_size: int = 5
    n_groups: int = 8
    cond_predict_scale: bool = True
    self_attn_first: bool = True
    block_type: str = ''
    # Inference tau knots, e.g. [0, 0.25, 0.5, 1] (NFE=3) or "uniform10".
    # None -> uniform with noise_scheduler.num_inference_timesteps.
    inference_tau_schedule: Optional[list] = None
    # VODPP FM is 0->1; set True only for 1->0 formulations (applies 1-tau).
    inference_tau_invert: bool = False
    

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        
        self.auto_map = {}
        self.pkg_map = {}
        for key, value in kwargs.items():
            setattr(self, key, value)
    
    @classmethod
    def from_customed_yaml(cls, yaml_path: str):
        with open(yaml_path, 'r') as f:
            config_dict = yaml.safe_load(f)
        return cls(**config_dict)
    
    @classmethod
    def from_customed_json(cls, json_path: str):
        with open(json_path, 'r') as f:
            config_dict = json.load(f)
        return cls(**config_dict)
    
    @classmethod
    def from_customed_dict(cls, config_dict):
        return cls(**config_dict)


def create_dit_model(
    config: VODPPlusDitFlowMatchingConfig,
    action_dim: int,
    scene_cond_len: int,
    scene_pos_embed_config: list
):
    horizon = config.horizon
    n_obs_steps = config.n_obs_steps
    hidden_size = config.hidden_size
    
    if config.block_type.lower() == 'invblock':
        from .module.conditional_dit_decouple_head import DecoupleDiT
        return DecoupleDiT(
            output_dim=action_dim,
            horizon=horizon,
            n_obs_steps=n_obs_steps,
            hidden_size=hidden_size,
            depth=config.depth,
            num_heads=config.num_heads,
            self_attn_first=config.self_attn_first,
            scene_cond_len=scene_cond_len,
            scene_pos_embed_config=scene_pos_embed_config
        )
    elif config.block_type.lower() == 'largeblock':
        from .module.conditional_largedit_head import LargeDiT
        return LargeDiT(
            output_dim=action_dim,
            horizon=horizon,
            n_obs_steps=n_obs_steps,
            hidden_size=hidden_size,
            depth=config.depth,
            num_heads=config.num_heads,
            action_only=False,
            scene_cond_len=scene_cond_len,
            scene_pos_embed_config=scene_pos_embed_config
        )
    elif config.block_type.lower() == 'nostateblock':
        from .module.conditional_largedit_head import LargeDiT
        return LargeDiT(
            output_dim=action_dim,
            horizon=horizon,
            n_obs_steps=n_obs_steps,
            hidden_size=hidden_size,
            depth=config.depth,
            num_heads=config.num_heads,
            action_only=True,
            scene_cond_len=scene_cond_len,
            scene_pos_embed_config=scene_pos_embed_config
        )
    elif config.block_type.lower() == 'vablock':
        from .module.conditional_vadit_head import VADiT
        return VADiT(
            output_dim=action_dim,
            horizon=horizon,
            n_obs_steps=n_obs_steps,
            hidden_size=hidden_size,
            depth=config.depth,
            num_heads=config.num_heads,
            self_attn_first=False,
            scene_cond_len=scene_cond_len,
            scene_pos_embed_config=scene_pos_embed_config
        )
    else:
        from .module.conditional_dit_head import DiT
        return DiT(
            output_dim=action_dim,
            horizon=horizon,
            hidden_size=hidden_size,
            depth=config.depth,
            num_heads=config.num_heads,
            self_attn_first=config.self_attn_first,
            scene_cond_len=scene_cond_len,
            scene_pos_embed_config=scene_pos_embed_config
        )

class VODPPlusDitFlowMatching(BasePolicy, PreTrainedModel, ModuleAttrMixin):
    config_class = VODPPlusDitFlowMatchingConfig
    action_keys = ['state', 'delta_state', 'endpose', 'delta_endpose']

    def __init__(self, config: VODPPlusDitFlowMatchingConfig):
        super().__init__(config)
        self.num_timestep_buckets = config.noise_scheduler['num_train_timesteps']
        self.num_inference_timesteps = config.noise_scheduler['num_inference_timesteps']
        self.beta_dist = torch.distributions.Beta(
            config.noise_scheduler['noise_beta_alpha'], 
            config.noise_scheduler['noise_beta_beta']
        )
        self.noise_s = config.noise_scheduler['noise_s']
        self.inference_tau_invert = bool(getattr(config, 'inference_tau_invert', False))
        self.set_inference_tau_schedule(
            getattr(config, 'inference_tau_schedule', None)
        )
        # self.obs_encoder = hydra.utils.instantiate(config.obs_encoder)
        # EncoderClass = hydra.utils.get_class(config.obs_encoder['_target_'])
        # self.obs_encoder = EncoderClass(**config.obs_encoder)
        self.obs_encoder = VisualEncoder(**config.obs_encoder)

        # Compat: older dumps use {"action":{"shape":[D]}}; newer use {"action":{"state":{"shape":[D]}, ...}}
        action_meta = config.shape_meta['action']
        if isinstance(action_meta, dict) and 'shape' in action_meta:
            action_shape = action_meta['shape']
        else:
            action_shape = action_meta[config.action_key]['shape']
        horizon = config.horizon
        hidden_size = config.hidden_size
        n_action_steps = config.n_action_steps
        n_obs_steps = config.n_obs_steps
        # parse shapes
        assert len(action_shape) == 1
        action_dim = action_shape[0]
        # get feature dim
        V, H, W, dim = self.obs_encoder.output_shape_meta() # V, H, W, D
        obs_feature_dim = dim
        action_mask = torch.zeros(1, horizon, action_dim).bool()
        action_mask[:,n_obs_steps:,:] = True

        # create diffusion model
        num_patches = H * W
        scene_cond_len = (config.n_obs_steps * V * num_patches)
        scene_pos_embed_config = [("image", (config.n_obs_steps, V, num_patches)),]
        block_type = (config.block_type or '').lower()
        self_attn_first = True if config.self_attn_first is None else config.self_attn_first

        if block_type == 'invblock':
            from .module.conditional_dit_decouple_head import DecoupleDiT
            self.model = DecoupleDiT(
                output_dim=action_dim,
                horizon=horizon,
                n_obs_steps=n_obs_steps,
                hidden_size=hidden_size,
                depth=config.depth,
                num_heads=config.num_heads,
                self_attn_first=self_attn_first,
                scene_cond_len=scene_cond_len,
                scene_pos_embed_config=scene_pos_embed_config
            )
        elif block_type == 'largeblock':
            from .module.conditional_largedit_head import LargeDiT
            self.model = LargeDiT(
                output_dim=action_dim,
                horizon=horizon,
                n_obs_steps=n_obs_steps,
                hidden_size=hidden_size,
                depth=config.depth,
                num_heads=config.num_heads,
                action_only=False,
                scene_cond_len=scene_cond_len,
                scene_pos_embed_config=scene_pos_embed_config
            )
        elif block_type == 'nostateblock':
            from .module.conditional_largedit_head import LargeDiT
            self.model = LargeDiT(
                output_dim=action_dim,
                horizon=horizon,
                n_obs_steps=n_obs_steps,
                hidden_size=hidden_size,
                depth=config.depth,
                num_heads=config.num_heads,
                action_only=True,
                scene_cond_len=scene_cond_len,
                scene_pos_embed_config=scene_pos_embed_config
            )
        elif block_type == 'vablock':
            from .module.conditional_vadit_head import VADiT
            self.model = VADiT(
                output_dim=action_dim,
                horizon=horizon,
                n_obs_steps=n_obs_steps,
                hidden_size=hidden_size,
                depth=config.depth,
                num_heads=config.num_heads,
                self_attn_first=False,
                scene_cond_len=scene_cond_len,
                scene_pos_embed_config=scene_pos_embed_config
            )
        else:
            from .module.conditional_dit_head import DiT
            self.model = DiT(
                output_dim=action_dim,
                horizon=horizon,
                hidden_size=hidden_size,
                depth=config.depth,
                num_heads=config.num_heads,
                self_attn_first=self_attn_first,
                scene_cond_len=scene_cond_len,
                scene_pos_embed_config=scene_pos_embed_config
            )

        self.scene_adaptor = self.build_condition_adapter(
            config.scene_adaptor, 
            in_features=obs_feature_dim, 
            out_features=hidden_size
        )

        # A `state` refers to an action or a proprioception vector
        self.state_adaptor = self.build_condition_adapter(
            config.state_adaptor, 
            in_features=action_dim,    # state + state mask (indicator)
            out_features=hidden_size
        )

        # self.mask_generator = LowdimMaskGenerator(
        #     action_dim=action_dim,
        #     obs_dim=0 if obs_as_global_cond else obs_feature_dim,
        #     max_n_obs_steps=n_obs_steps,
        #     fix_obs_steps=True,
        #     action_visible=False
        # )
        self.normalizer = LinearNormalizer()
        # self.normalizer = None
        self.horizon = horizon
        self.obs_feature_dim = obs_feature_dim
        self.action_dim = action_dim
        self.n_action_steps = n_action_steps
        self.n_obs_steps = n_obs_steps
        self.action_mask = action_mask
        self.action_key = config.action_key
        # self.kwargs = kwargs
        self.kwargs = {} #

    def build_condition_adapter(
        self, projector_type, in_features, out_features):
        projector = None
        if projector_type == 'linear':
            projector = nn.Linear(in_features, out_features)
        else:
            mlp_gelu_match = re.match(r'^mlp(\d+)x_gelu$', projector_type)
            if mlp_gelu_match:
                mlp_depth = int(mlp_gelu_match.group(1))
                modules = [nn.Linear(in_features, out_features)]
                for _ in range(1, mlp_depth):
                    modules.append(nn.GELU(approximate="tanh"))
                    modules.append(nn.Linear(out_features, out_features))
                projector = nn.Sequential(*modules)

        if projector is None:
            raise ValueError(f'Unknown projector type: {projector_type}')

        return projector
    
    def adapt_conditions(self, scene_tokens):
        '''
        scene_cond: (batch_size, patch_num, img_token_dim)
        
        return: adpated (..., hidden_size) for all input tokens
        '''
        adpated_scene = self.scene_adaptor(scene_tokens)
        return adpated_scene
    
    # ========= inference  ============
    def set_inference_tau_schedule(self, schedule=None, invert: bool = None):
        """Set Euler tau knots. NFE = len(knots) - 1.

        Examples:
            [0, 0.25, 0.5, 1] -> 3 NFE
            "uniform10" / None with num_inference_timesteps=10 -> 10 NFE
        """
        if invert is not None:
            self.inference_tau_invert = bool(invert)
        # last_query is None for every plain Euler grid, which is why the
        # sampler below can branch on it instead of on a separate mode flag.
        self.inference_tau_schedule, self.inference_last_query = (
            resolve_inference_tau_plan(
                schedule,
                num_inference_timesteps=self.num_inference_timesteps,
                invert=self.inference_tau_invert,
            )
        )

    def conditional_sample(
            self, scene_cond, state_traj, action_mask, return_flow:bool = False, **kwargs
        ) -> torch.Tensor:
        '''
        scene_cond: image conditional data, (batch_size, patch_num, hidden_size).
        state_traj: (batch_size, 1, hidden_size), state trajectory.
        
        return: (batch_size, horizon, action_dim)
        '''
        device = state_traj.device
        dtype = state_traj.dtype
        batch_size = state_traj.shape[0]
        noisy_action = torch.randn(
            size=(batch_size, self.horizon, self.action_dim), 
            dtype=dtype, device=device
        )
        flow = []
        if return_flow:
            flow.append(noisy_action.detach().cpu().numpy())

        # Tau-space Euler: evaluate at knot[i], step with dt = knot[i+1]-knot[i].
        # VODPP FM is 0->1 (noise->data); knots like [0,0.25,0.5,1] => 3 NFE.
        tau_knots = self.inference_tau_schedule
        state_mask = ~action_mask
        last_query = getattr(self, "inference_last_query", None)
        for i in range(len(tau_knots) - 1):
            tau_i = float(tau_knots[i])
            dt = float(tau_knots[i + 1] - tau_knots[i])
            # Identity 2: the final step may evaluate the field at a tau other
            # than the knot it rides from.  x + dt * v(x, q) is exact for
            # any (dt, q), so decoupling the query changes which velocity is
            # used -- not the integrator.  Every earlier step is untouched.
            if last_query is not None and i == len(tau_knots) - 2:
                tau_i = float(last_query)
            t_discretized = tau_to_timestep_bucket(tau_i, self.num_timestep_buckets)
            timesteps = torch.full(
                size=(batch_size,), fill_value=t_discretized, device=device
            )
            # Prepare state-action trajectory
            noisy_action[state_mask] = state_traj[state_mask]
            state_action_traj = self.state_adaptor(noisy_action)
            
            # Predict the model output
            pred_velocity = self.model(state_action_traj, timesteps, scene_cond)
            
            # Integrate along tau: x <- x + dt * v_theta(x, tau_i)
            noisy_action = noisy_action + dt * pred_velocity
            if return_flow:
                flow.append(noisy_action.detach().cpu().numpy())
        
        # Finally apply the action mask to mask invalid action dimensions
        noisy_action[state_mask] = state_traj[state_mask]

        return noisy_action, flow

    @torch.no_grad()
    def predict_action(
        self, 
        obs_dict: Dict[str, torch.Tensor], 
        return_flow:bool = False, 
        gt: Dict[str, torch.Tensor] = None
    ) -> Dict[str, torch.Tensor]:
        """
        obs_dict: must include "obs" key
        result: must include "action" key
        """
        device = self.device
        dtype = self.dtype
        assert 'past_action' not in obs_dict # not implemented yet
        # normalize input
        nobs = {k: v.to(device=device, dtype=dtype) for k, v in obs_dict.items()}
        # filtered_obs_dict = {key: value for key, value in obs_dict.items() 
        #         if key in self.normalizer.params_dict}
        filtered_obs_dict = {
            key: value for key, value in obs_dict.items() 
            if key in self.action_keys
        }
        nobs.update(self.normalizer.normalize(filtered_obs_dict))
        value = next(iter(nobs.values()))
        B, To = value.shape[:2]
        T = self.horizon
        Da = self.action_dim
        Do = self.obs_feature_dim
        To = self.n_obs_steps

        # handle different ways of passing observation
        # condition through global feature
        this_nobs = dict_apply(nobs, lambda x: x[:,:To,...].reshape(-1,*x.shape[2:]))
        nobs_features = self.obs_encoder(this_nobs) # (BS, VP, Do)
        scene_cond = self.adapt_conditions(nobs_features) # (BS, VP, hidden_size)
        state_traj = torch.zeros(size=(B, T, Da), device=device, dtype=dtype)
        state_traj[:,:To,:] = this_nobs[self.action_key].view(B,To,-1)
        action_mask = self.action_mask.expand(B, -1, -1).to(device=device)

        # run sampling
        nsample, flow = self.conditional_sample(scene_cond, state_traj, action_mask, return_flow)
        
        # unnormalize prediction
        naction_pred = nsample[...,:Da]
        action_pred = self.normalizer[self.action_key].unnormalize(naction_pred)

        # get action
        start = To
        end = start + self.n_action_steps
        if gt is not None:
            action_loss = F.mse_loss(naction_pred[:,start:end], nobs[self.action_key][:,start:end], reduction="mean")
            mse_loss = F.mse_loss(action_pred[:,start:end], gt[self.action_key][:,start:end], reduction="mean")
            return {
                'loss': action_loss.detach(),
                'action loss': action_loss.detach(),
                'mse loss': mse_loss.detach()
            }
        
        action = action_pred[:,start:end]
        if 'delta' in self.action_key:
            key = self.action_key.replace('delta_', '')
            result = {
                'action': action + obs_dict[key][:,0:1,...],
                'delta_action': action,
                'predicted_raw': action_pred,
            }
        else:
            result = {
                'action': action,
                'predicted_raw': action_pred,
            }

        if return_flow:
            flow = [smp[:,start:end] for smp in flow]
            result.update({
                'flow': flow,
                'gt': nobs[self.action_key][:,start:end]
            })
        
        return result

    # ========= training  ============
    def set_normalizer(self, normalizer: LinearNormalizer):
        self.normalizer.load_state_dict(normalizer.state_dict())

    def sample_time(self, batch_size, device, dtype):
        sample = self.beta_dist.sample([batch_size]).to(device, dtype=dtype)
        return (self.noise_s - sample) / self.noise_s

    def compute_loss(self, batch):
        # normalize input
        assert 'valid_mask' not in batch
        nobs = {**batch['obs']}
        nobs.update(self.normalizer.normalize(
            {key: nobs[key] for key in self.action_keys if key in nobs}
        ))
        state_traj = nobs[self.action_key]
        actions = self.normalizer.normalize(batch['action'])[self.action_key]
        batch_size = actions.shape[0]

        # handle different ways of passing observation
        # reshape B, T, ... to B*T
        this_nobs = dict_apply(nobs, lambda x: x[:,:self.n_obs_steps,...].reshape(-1,*x.shape[2:]))
        nobs_features = self.obs_encoder(this_nobs) # (BS, VP, Do)
        scene_cond = self.adapt_conditions(nobs_features) # (BS, VP, hidden_size)
        scene_cond = scene_cond.view(batch_size, -1, scene_cond.shape[-1]) # (B, SVP, hidden_size)
        action_mask = self.action_mask.expand(batch_size, -1, -1).to(device=actions.device)

        # Sample noise that we'll add to the images
        noise = torch.randn(actions.shape, device=actions.device)
        # Sample a random timestep for each image
        t = self.sample_time(
            actions.shape[0],
            device=actions.device,
            dtype=actions.dtype
        )
        t = t[:, None, None]  # shape (B,1,1) for broadcast
        # Convert (continuous) t -> discrete if needed
        timesteps = (t[:, 0, 0] * self.num_timestep_buckets).long() # shape (B,)
        
        # Add noise to the clean images according to the noise magnitude at each timestep
        # (this is the forward diffusion process)
        noisy_traj = (1 - t) * noise + t * actions
        velocity = actions - noise
        
        # compute loss mask
        cond_mask = ~action_mask
        # apply conditioning
        noisy_traj[cond_mask] = state_traj[cond_mask]
        state_noisy_tokens = self.state_adaptor(noisy_traj)
        
        # Predict the noise residual
        pred = self.model(state_noisy_tokens, timesteps, scene_cond)

        loss = F.mse_loss(pred, velocity, reduction='none')
        loss = loss * action_mask.type(loss.dtype)
        loss = reduce(loss, 'b ... -> b (...)', 'mean')
        loss = loss.mean()
        return loss