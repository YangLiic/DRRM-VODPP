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
from .module.blocks import SIGReg
from .module.action_codec import MultiEmbodimentActionEncoder, MultiEmbodimentActionDecoder
from .module.entangled_dit_model import EntangledDiT
from .emvis.embodied_vision import VisualEncoder
from .common.normalizer import LinearNormalizer
from .common.pytorch_util import dict_apply
from .common.module_attr_mixin import ModuleAttrMixin

import yaml
import json
from dataclasses import dataclass
from typing import Optional
from transformers import PretrainedConfig, PreTrainedModel

@dataclass
class VODPPlusDitFlowMatchingWAMConfig(PretrainedConfig):
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
    visual_adaptor: str = "mlp2x_gelu"
    timestep_mode: str = "concat"
    cond_pos_embed_mode: str = "s_v_hw"
    pretraining: bool = False
    sigreg: dict = None
    freeze_keys: list = None
    loss_weights: dict = None

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

class VODPPlusDitFlowMatchingWAM(BasePolicy, PreTrainedModel, ModuleAttrMixin):
    config_class = VODPPlusDitFlowMatchingWAMConfig
    action_keys = ['state', 'delta_state', 'endpose', 'delta_endpose']

    @staticmethod
    def build_cond_pos_embed_config(mode, n_obs_steps, num_views, height, width):
        num_patches = height * width
        if mode == "s_vhw":
            return [("image", (n_obs_steps, num_views * num_patches))]
        if mode == "s_v_hw":
            return [("image", (n_obs_steps, num_views, num_patches))]
        if mode == "s_v_h_w":
            return [("image", (n_obs_steps, num_views, height, width))]
        raise ValueError(
            f"Unknown cond_pos_embed_mode: {mode}. "
            "Expected one of: s_vhw, s_v_hw, s_v_h_w"
        )

    def __init__(self, config: VODPPlusDitFlowMatchingWAMConfig):
        super().__init__(config)
        self.num_timestep_buckets = config.noise_scheduler['num_train_timesteps']
        self.num_inference_timesteps = config.noise_scheduler['num_inference_timesteps']
        self.beta_dist = torch.distributions.Beta(
            config.noise_scheduler['noise_beta_alpha'], 
            config.noise_scheduler['noise_beta_beta']
        )
        self.noise_s = config.noise_scheduler['noise_s']
        self.visual_encoder = VisualEncoder(**config.obs_encoder)

        action_key = config.action_key
        action_shape = config.shape_meta['action'][action_key]['shape']
        horizon = config.horizon
        hidden_size = config.hidden_size
        n_action_steps = config.n_action_steps
        n_obs_steps = config.n_obs_steps
        # parse shapes
        assert len(action_shape) == 1
        action_dim = action_shape[0]
        # get feature dim
        V, H, W, visual_feature_dim = self.visual_encoder.output_shape_meta() # V, H, W, D
        action_mask = torch.zeros(1, horizon).bool()
        action_mask[:,n_obs_steps:] = True

        # create diffusion model
        num_patches = H * W
        cond_len = config.n_obs_steps * V * num_patches
        cond_pos_embed_config = self.build_cond_pos_embed_config(
            mode=getattr(config, "cond_pos_embed_mode", "s_v_hw"),
            n_obs_steps=config.n_obs_steps,
            num_views=V,
            height=H,
            width=W,
        )

        # config.dtype is often the string "float32"; EntangledDiT defaults to bf16
        # which leaves DiT weights bf16 while the visual encoder stays fp32.
        _dtype = getattr(config, "dtype", torch.float32)
        if isinstance(_dtype, str):
            _dtype = getattr(torch, _dtype)
        self.model = EntangledDiT(
            horizon=horizon,
            hidden_size=hidden_size,
            depth=config.depth,
            num_heads=config.num_heads,
            cond_len=cond_len,
            cond_pos_embed_config=cond_pos_embed_config,
            dtype=_dtype,
            timestep_mode=getattr(config, 'timestep_mode', 'concat')
        )

        self.sigreg = SIGReg(**config.sigreg)

        # Initialize loss weights with defaults
        self.loss_weights = {
            'action': 1.0,
            'jepa': 1.0,
            'sigreg': 0.1
        }
        if getattr(config, 'loss_weights', None):
            self.loss_weights.update(config.loss_weights)

        self.visual_adaptor = self.build_condition_adapter(
            config.visual_adaptor, 
            in_features=visual_feature_dim, 
            out_features=hidden_size
        )

        self.state_encoder = MultiEmbodimentActionEncoder(
            action_dim=action_dim,
            hidden_size=hidden_size,
            output_dim=hidden_size,
            num_embodiments=1
        )

        self.action_decoder = MultiEmbodimentActionDecoder(
            input_dim=hidden_size,
            hidden_size=hidden_size,
            action_dim=action_dim,
            num_embodiments=1
        )

        self.normalizer = LinearNormalizer()
        # self.normalizer = None
        self.horizon = horizon
        self.visual_feature_dim = visual_feature_dim
        self.action_dim = action_dim
        self.n_action_steps = n_action_steps
        self.n_obs_steps = n_obs_steps
        self.action_mask = action_mask
        self.action_key = action_key
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
    
    def adapt_conditions(self, tokens):
        '''
        tokens: (batch_size, patch_num, img_token_dim)
        
        return: adpated (..., hidden_size) for all input tokens
        '''
        tokens = self.visual_adaptor(tokens)
        return tokens
    
    # ========= inference  ============
    def conditional_sample(
            self, visual_cond, norm_states, action_mask, return_flow:bool = False, **kwargs
        ) -> torch.Tensor:
        '''
        visual_cond: image conditional data, (batch_size, patch_num, hidden_size).
        norm_states: (batch_size, 1, action_dim), normalized state trajectory.
        
        return: (batch_size, horizon, action_dim)
        '''
        device = norm_states.device
        batch_size = norm_states.shape[0]

        noisy_actions = torch.randn_like(norm_states)
        flow = []
        if return_flow:
            flow.append(noisy_actions.detach().cpu().numpy())

        # Set step values
        num_steps = self.num_inference_timesteps
        dt = 1.0 / num_steps

        state_mask = ~action_mask
        for t in range(num_steps):
            # timesteps = t.unsqueeze(-1).to(device)
            t_discretized = int(t / float(num_steps) * self.num_timestep_buckets)
            timesteps = torch.full(
                size=(batch_size,), fill_value=t_discretized, device=device
            )
            # Prepare state-action trajectory
            noisy_actions[state_mask] = norm_states[state_mask]
            noisy_tokens = self.state_encoder(noisy_actions)
            
            # Predict the model output
            predict_action_latent, _ = self.model(noisy_tokens, timesteps, visual_cond)
            pred_velocity = self.action_decoder(predict_action_latent)
            
            # Compute previous actions: x_t -> x_t-1
            noisy_actions = noisy_actions + dt * pred_velocity
            if return_flow:
                flow.append(noisy_actions.detach().cpu().numpy())
        
        # Finally apply the action mask to mask invalid action dimensions
        noisy_actions[state_mask] = norm_states[state_mask]

        return noisy_actions, flow

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
        value = next(iter(obs_dict.values()))
        batch_size = value.shape[0]
        action_dim = self.action_dim
        n_obs_steps = self.n_obs_steps
        assert 'past_action' not in obs_dict # not implemented yet

        obs = {k: v.to(device=device, dtype=dtype) for k, v in obs_dict.items()}
        action_mask = self.action_mask.expand(batch_size, -1).to(device=device)
        states = torch.zeros(size=(*action_mask.shape, action_dim), device=device, dtype=dtype)
        states[:, :n_obs_steps] = obs[self.action_key][:, :n_obs_steps]
        norm_states = self.normalizer[self.action_key].normalize(states)

        # handle different ways of passing observation
        # reshape B, T, ... to B*T, matching compute_loss
        obs = dict_apply(obs, lambda x: x.reshape(-1, *x.shape[2:]))
        full_visual_features = self.adapt_conditions(self.visual_encoder(obs)) # BS, VP, D'
        num_obs_frames = full_visual_features.shape[0] // batch_size
        full_visual_features = full_visual_features.reshape(
            batch_size, num_obs_frames, *full_visual_features.shape[1:]
        )
        visual_features = full_visual_features[:, :n_obs_steps].flatten(1, 2) # (B, SVP, hidden_size)

        # run sampling
        predict_actions, flow = self.conditional_sample(visual_features, norm_states, action_mask, return_flow)
        
        # unnormalize prediction
        unnorm_actions = self.normalizer[self.action_key].unnormalize(predict_actions)

        # get action
        if gt is not None:
            gt_actions = gt[self.action_key]
            norm_actions = self.normalizer[self.action_key].normalize(gt_actions)
            action_loss = F.mse_loss(predict_actions[action_mask], norm_actions[action_mask], reduction="mean")
            mse_loss = F.mse_loss(unnorm_actions[action_mask], gt_actions[action_mask], reduction="mean")
            return {
                'loss': action_loss.detach(),
                'action loss': action_loss.detach(),
                'mse loss': mse_loss.detach()
            }
        
        actions = unnorm_actions[action_mask].view(batch_size, -1, action_dim)
        if 'delta' in self.action_key:
            key = self.action_key.replace('delta_', '')
            result = {
                'action': actions + obs_dict[key][:,0:1,...],
                'delta_action': actions,
                'predicted_raw': predict_actions,
            }
        else:
            result = {
                'action': actions,
                'predicted_raw': predict_actions,
            }

        if return_flow:
            flow = [smp[action_mask].view(batch_size, -1, action_dim) for smp in flow]
            result.update({
                'flow': flow,
                'gt': gt[self.action_key][action_mask].view(batch_size, -1, action_dim)
            })
        
        return result

    # ========= training  ============
    def set_normalizer(self, normalizer: LinearNormalizer):
        self.normalizer.load_state_dict(normalizer.state_dict())

    def sample_time(self, batch_size, device, dtype):
        sample = self.beta_dist.sample([batch_size]).to(device, dtype=dtype)
        return ((self.noise_s - sample) / self.noise_s).clamp(0, 1)

    def compute_loss(self, batch):
        # normalize input
        assert 'valid_mask' not in batch
        obs = {**batch['obs']}
        states = batch['obs'][self.action_key]
        actions = batch['action'][self.action_key]
        norm_states = self.normalizer[self.action_key].normalize(states)
        norm_actions = self.normalizer[self.action_key].normalize(actions)
        batch_size = norm_actions.shape[0]

        # handle different ways of passing observation
        # reshape B, T, ... to B*T
        obs = dict_apply(obs, lambda x: x.reshape(-1, *x.shape[2:]))
        full_visual_features = self.adapt_conditions(self.visual_encoder(obs))
        num_obs_frames = full_visual_features.shape[0] // batch_size
        # assert num_obs_frames == 2
        full_visual_features = full_visual_features.reshape(
            batch_size, num_obs_frames, *full_visual_features.shape[1:]
        )
        gt_visual_features = full_visual_features[:, -self.n_obs_steps:].flatten(1, 2)
        visual_features = full_visual_features[:, :self.n_obs_steps].flatten(1, 2)
        full_visual_features = full_visual_features.flatten(1, 2)

        action_mask = self.action_mask.expand(batch_size, -1).to(device=norm_actions.device)
        # Sample noise that we'll add to the images
        noise = torch.randn_like(norm_actions)
        # Sample a random timestep for each image
        t = self.sample_time(
            batch_size,
            device=noise.device,
            dtype=noise.dtype
        )
        t = t[:, None, None]  # shape (B,1,1) for broadcast
        # Convert (continuous) t -> discrete if needed
        timesteps = (t[:, 0, 0] * self.num_timestep_buckets).long() # shape (B,)
        
        # Add noise to the clean images according to the noise magnitude at each timestep
        # (this is the forward diffusion process)
        noisy_actions = (1 - t) * noise + t * norm_actions
        velocity = norm_actions - noise
        
        # compute loss mask
        cond_mask = ~action_mask
        # apply conditioning
        noisy_actions[cond_mask] = norm_states[cond_mask]
        noisy_tokens = self.state_encoder(noisy_actions)
        # Predict the noise residual
        predict_action_latent, predict_visual_features = self.model(noisy_tokens, timesteps, visual_features)
        predict_velocity = self.action_decoder(predict_action_latent)

        loss_action = F.mse_loss(
            predict_velocity[action_mask], velocity[action_mask], reduction="mean"
        )
        loss_sigreg = self.sigreg(full_visual_features.transpose(0, 1))
        loss_jepa = F.mse_loss(
            predict_visual_features, gt_visual_features, reduction="mean"
        )
        visual_diversity = visual_features.std(dim=0).mean()
        loss = (
            loss_action * self.loss_weights['action'] +
            loss_jepa * self.loss_weights['jepa'] +
            loss_sigreg * self.loss_weights['sigreg']
        )
        loss_dict = {
            'loss': loss, 
            'loss action': loss_action.detach(),
            'loss jepa': loss_jepa.detach(),
            'loss sigreg': loss_sigreg.detach(),
            'visual diversity': visual_diversity.detach(),
        }
        return loss_dict
