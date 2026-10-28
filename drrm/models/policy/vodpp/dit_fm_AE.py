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
from .module.action_codec_transformer import ActionTransformerEncoder, ActionTransformerDecoder
from .module.conditional_dit_head import DiT
from .module.blocks import SIGReg
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
class VODPPlusDitFlowMatchingAEConfig(PretrainedConfig):
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
    sigreg: dict = None
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


class VODPPlusDitFlowMatchingAE(BasePolicy, PreTrainedModel, ModuleAttrMixin):
    config_class = VODPPlusDitFlowMatchingAEConfig
    action_keys = ['state', 'delta_state', 'endpose', 'delta_endpose']

    def __init__(self, config: VODPPlusDitFlowMatchingAEConfig):
        super().__init__(config)
        self.num_timestep_buckets = config.noise_scheduler['num_train_timesteps']
        self.num_inference_timesteps = config.noise_scheduler['num_inference_timesteps']
        self.beta_dist = torch.distributions.Beta(
            config.noise_scheduler['noise_beta_alpha'], 
            config.noise_scheduler['noise_beta_beta']
        )
        self.noise_s = config.noise_scheduler['noise_s']
        self.obs_encoder = VisualEncoder(**config.obs_encoder)

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
        V, H, W, dim = self.obs_encoder.output_shape_meta() # V, H, W, D
        obs_feature_dim = dim
        action_mask = torch.zeros(1, horizon).bool()
        action_mask[:,n_obs_steps:] = True

        # create diffusion model
        num_patches = H * W
        scene_cond_len = (config.n_obs_steps * V * num_patches)
        scene_pos_embed_config = [("image", (config.n_obs_steps, V, num_patches)),]
        
        self.model = DiT(
            output_dim=action_dim,
            horizon=horizon,
            hidden_size=hidden_size,
            depth=config.depth,
            num_heads=config.num_heads,
            self_attn_first=config.self_attn_first,
            scene_cond_len=scene_cond_len,
            scene_pos_embed_config=scene_pos_embed_config,
            use_final_layer=False
        )

        self.scene_adaptor = self.build_condition_adapter(
            config.scene_adaptor, 
            in_features=obs_feature_dim, 
            out_features=hidden_size
        )

        self.state_encoder = ActionTransformerEncoder(
            action_dim = action_dim,
            hidden_size = hidden_size,
            num_heads = config.num_heads,
            horizon = horizon,
            depth = 1,
            is_causal = True
        )

        self.state_decoder = ActionTransformerDecoder(
            action_dim = action_dim,
            hidden_size = hidden_size,
            num_heads = config.num_heads,
            horizon = horizon,
            depth = 1,
            is_causal = False
        )

        self.prompt = nn.Embedding(horizon, hidden_size)

        self.sigreg = SIGReg(**config.sigreg)

        self.loss_weights = {
            'latent': 1.0,
            'reconstruct': 1.0,
            'consistency': 0.0,
            'predict': 1.0,
            'sigreg': 0.0,
            'direct_predict': 0.0
        }
        if getattr(config, 'loss_weights', None):
            self.loss_weights.update(config.loss_weights)

        self.normalizer = LinearNormalizer()
        # self.normalizer = None
        self.horizon = horizon
        self.obs_feature_dim = obs_feature_dim
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
    
    def adapt_conditions(self, scene_tokens):
        '''
        scene_cond: (batch_size, patch_num, img_token_dim)
        
        return: adpated (..., hidden_size) for all input tokens
        '''
        adpated_scene = self.scene_adaptor(scene_tokens)
        return adpated_scene
    
    # ========= inference  ============
    def conditional_sample(
            self, scene_cond, state_features, action_mask, return_flow:bool = False, **kwargs
        ) -> torch.Tensor:
        '''
        scene_cond: image conditional data, (batch_size, patch_num, hidden_size).
        state_features: (batch_size, 1, hidden_size), state features.
        
        return: (batch_size, horizon, action_dim)
        '''
        device = state_features.device
        batch_size = state_features.shape[0]

        noisy_tokens = torch.randn_like(state_features)
        noisy_tokens = noisy_tokens * kwargs.get('sigma', 0.0)
        noisy_tokens[:, self.n_obs_steps:, :] += self.prompt(
            torch.arange(self.n_obs_steps, self.horizon, device=state_features.device),
        ).to(dtype=state_features.dtype).unsqueeze(0)
        flow = []
        if return_flow:
            flow.append(noisy_tokens.detach().cpu().numpy())

        # Set step values
        num_steps = self.num_inference_timesteps
        dt = 1.0 / num_steps

        cond_mask = ~action_mask
        for t in range(num_steps):
            # timesteps = t.unsqueeze(-1).to(device)
            t_discretized = int(t / float(num_steps) * self.num_timestep_buckets)
            timesteps = torch.full(
                size=(batch_size,), fill_value=t_discretized, device=device
            )
            # Prepare state-action trajectory
            noisy_tokens[cond_mask] = state_features[cond_mask]
            
            # Predict the model output
            pred_velocity = self.model(noisy_tokens, timesteps, scene_cond)
            
            # Compute previous actions: x_t -> x_t-1
            noisy_tokens = noisy_tokens + dt * pred_velocity
            if return_flow:
                flow.append(noisy_tokens.detach().cpu().numpy())
        
        # Finally apply the action mask to mask invalid action dimensions
        noisy_tokens[cond_mask] = state_features[cond_mask]

        return noisy_tokens, flow

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

        # normalize input
        obs = {k: v.to(device=device, dtype=dtype) for k, v in obs_dict.items()}
        action_mask = self.action_mask.expand(batch_size, -1).to(device=device)
        cond_mask = ~action_mask
        norm_states = self.normalizer[self.action_key].normalize(obs[self.action_key])
        zeros = torch.zeros(size=(*action_mask.shape, action_dim), device=device, dtype=dtype)
        zeros[cond_mask] = norm_states[cond_mask]
        norm_states = zeros

        obs = dict_apply(obs, lambda x: x[:,:n_obs_steps,...].reshape(-1,*x.shape[2:]))
        obs_features = self.obs_encoder(obs) # (BS, VP, Do)
        scene_cond = self.adapt_conditions(obs_features) # (BS, VP, hidden_size)
        scene_cond = scene_cond.view(batch_size, -1, scene_cond.shape[-1]) # (B, SVP, hidden_size)


        state_features = self.state_encoder(norm_states)
        predict_features, flow = self.conditional_sample(scene_cond, state_features, action_mask, return_flow)
        predict_features[cond_mask] = state_features[cond_mask]
        predict_actions = self.state_decoder(predict_features)
        
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
        
        action = unnorm_actions[action_mask]
        if 'delta' in self.action_key:
            key = self.action_key.replace('delta_', '')
            result = {
                'action': action + obs_dict[key][:,0:1,...],
                'delta_action': action,
                'predicted_raw': predict_actions,
            }
        else:
            result = {
                'action': action,
                'predicted_raw': predict_actions,
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
        obs = {**batch['obs']}
        # norm_obs.update(self.normalizer.normalize(
        #     {key: norm_obs[key] for key in self.action_keys if key in norm_obs}
        # ))
        states = batch['obs'][self.action_key]
        norm_states = self.normalizer[self.action_key].normalize(states)
        actions = batch['action'][self.action_key]
        norm_actions = self.normalizer[self.action_key].normalize(actions)
        batch_size = norm_actions.shape[0]
        action_mask = self.action_mask.expand(batch_size, -1).to(device=norm_actions.device)

        # handle different ways of passing observation
        # reshape B, T, ... to B*T
        obs = dict_apply(obs, lambda x: x[:,:self.n_obs_steps,...].reshape(-1,*x.shape[2:]))
        obs_features = self.obs_encoder(obs) # (BS, VP, Do)
        scene_cond = self.adapt_conditions(obs_features) # (BS, VP, hidden_size)
        scene_cond = scene_cond.view(batch_size, -1, scene_cond.shape[-1]) # (B, SVP, hidden_size)

        state_features = self.state_encoder(norm_states)
        state_features_detached = state_features.detach()
        # Sample noise that we'll add to the images
        sigma = 0.05
        noise = torch.randn_like(state_features)
        noise = noise * sigma
        noise[:, self.n_obs_steps:, :] = self.prompt(
            torch.arange(self.n_obs_steps, self.horizon, device=state_features.device),
        ).to(dtype=state_features.dtype).unsqueeze(0)
        # Sample a random timestep for each image
        t = self.sample_time(
            noise.shape[0],
            device=noise.device,
            dtype=noise.dtype
        )
        t = t[:, None, None]  # shape (B,1,1) for broadcast
        # Convert (continuous) t -> discrete if needed
        timesteps = (t[:, 0, 0] * self.num_timestep_buckets).long() # shape (B,)
        
        # Add noise to the clean images according to the noise magnitude at each timestep
        # (this is the forward diffusion process)
        noisy_tokens = (1 - t) * noise + t * state_features_detached
        # velocity = state_features - noise
        
        # compute loss mask
        cond_mask = ~action_mask
        # apply conditioning
        noisy_tokens[cond_mask] = state_features_detached[cond_mask]
        
        # Predict the noise residual
        predict_velocity = self.model(noisy_tokens, timesteps, scene_cond)
        # loss_latent = F.mse_loss(predict_velocity[action_mask], velocity[action_mask], reduction="mean")
        predict_features = noisy_tokens + predict_velocity * (1 - t)


        # reconstruct loss
        reconstruct_actions = self.state_decoder(state_features)
        loss_reconstruct = F.mse_loss(
            reconstruct_actions[action_mask], norm_actions[action_mask], reduction="mean"
        )

        # direct loss
        loss_latent = F.mse_loss(
            predict_features[action_mask], state_features_detached[action_mask], reduction="mean"
        )

        predict_features[cond_mask] = state_features_detached[cond_mask]
        predict_actions = self.state_decoder(predict_features)
        loss_direct_predict = F.mse_loss(
            predict_actions[action_mask], norm_actions[action_mask], reduction="mean"
        )

        # sampled loss
        sampled_features, _ = self.conditional_sample(
            scene_cond=scene_cond,
            state_features=state_features_detached,
            action_mask=action_mask,
            return_flow=False,
            sigma = sigma
        )
        sampled_features[cond_mask] = state_features_detached[cond_mask]
        loss_consistency = F.mse_loss(
            sampled_features[action_mask], state_features_detached[action_mask], reduction="mean"
        )
        sampled_actions = self.state_decoder(sampled_features)
        loss_predict = F.l1_loss(
            sampled_actions[action_mask], norm_actions[action_mask], reduction="mean"
        )

        # sigreg loss
        loss_sigreg = self.sigreg(state_features.transpose(0, 1))
        state_latent_diversity = state_features.std(dim=0).mean()
        predict_latent_diversity = predict_features.std(dim=0).mean()

        loss = (
            loss_latent * self.loss_weights['latent'] + 
            loss_reconstruct * self.loss_weights['reconstruct'] + 
            loss_consistency * self.loss_weights['consistency'] +
            loss_predict * self.loss_weights['predict'] +
            loss_direct_predict * self.loss_weights['direct_predict'] +
            loss_sigreg * self.loss_weights['sigreg']
        )
        
        loss_dict = {
            'loss': loss,
            'loss latent': loss_latent.detach(),
            'loss reconstruct': loss_reconstruct.detach(),
            'loss consistency': loss_consistency.detach(),
            'loss predict': loss_predict.detach(),
            'loss sigreg': loss_sigreg.detach(),
            'state latent diversity': state_latent_diversity.detach(),
            'predict latent diversity': predict_latent_diversity.detach(),
        }
        return loss_dict