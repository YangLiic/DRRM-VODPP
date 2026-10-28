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
from .module.action_codec import MultiEmbodimentActionDecoder, MultiEmbodimentActionEncoder
from .module.action_codec_transformer import ActionTransformerDecoder
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
class VAR0FlowMatchingConfig(PretrainedConfig):
    shape_meta: dict
    noise_scheduler: dict
    vis_encoder: dict
    state_encoder: dict
    state_decoder: dict
    action_decoder: dict
    state_key: str
    state_tgt_key: str
    action_key: str
    hidden_size: int
    depth: int
    num_heads: int
    horizon: int
    n_action_steps: int
    n_obs_steps: int
    # obs_as_global_cond: bool = True
    diffusion_step_embed_dim: int = 256
    down_dims: tuple = (256,512,1024)
    kernel_size: int = 5
    n_groups: int = 8
    cond_predict_scale: bool = True
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

class VAR0FlowMatching(BasePolicy, PreTrainedModel, ModuleAttrMixin):
    config_class = VAR0FlowMatchingConfig

    def __init__(self, config: VAR0FlowMatchingConfig):
        super().__init__(config)
        # diffusion init
        self.num_timestep_buckets = config.noise_scheduler['num_train_timesteps']
        self.num_inference_timesteps = config.noise_scheduler['num_inference_timesteps']
        self.beta_dist = torch.distributions.Beta(
            config.noise_scheduler['noise_beta_alpha'], 
            config.noise_scheduler['noise_beta_beta']
        )
        self.noise_s = config.noise_scheduler['noise_s']

        # build observation encoder init
        # EncoderClass = hydra.utils.get_class(config.vis_encoder['_target_'])
        self.vis_encoder = VisualEncoder(**config.vis_encoder)
        V, H, W, vis_dim = self.vis_encoder.output_shape_meta() # V, H, W, D

        # build action encoder init
        action_shape = config.shape_meta['action']['shape']
        n_action_steps = config.n_action_steps
        n_obs_steps = config.n_obs_steps
        # parse shapes
        assert len(action_shape) == 1
        self.state_encoder = MultiEmbodimentActionEncoder(**config.state_encoder)
        
        state_decoder_type = config.state_decoder.get('type', None)
        StateDecoder = ActionTransformerDecoder \
            if state_decoder_type == 'transformer' \
                else MultiEmbodimentActionDecoder
        state_decoder_kwargs = {k: v for k, v in config.state_decoder.items() if k != 'type'}
        self.state_decoder = StateDecoder(**state_decoder_kwargs)
        
        action_dim = config.state_encoder['output_dim']
        if not config.pretraining:
            action_decoder_type = config.action_decoder.get('type', None)
            ActionDecoder = ActionTransformerDecoder \
                if action_decoder_type == 'transformer' \
                    else MultiEmbodimentActionDecoder
            decoder_kwargs = {k: v for k, v in config.action_decoder.items() if k != 'type'}
            self.action_decoder = ActionDecoder(**decoder_kwargs)

        # create world model
        horizon = config.horizon
        hidden_size = config.hidden_size
        num_patches = H * W
        cond_len = (config.n_obs_steps * V * num_patches)
        cond_pos_embed_config = [("image", (config.n_obs_steps, V, num_patches)),]
        self.model = EntangledDiT(
            horizon=horizon,
            hidden_size=hidden_size,
            depth=config.depth,
            num_heads=config.num_heads,
            cond_len=cond_len,
            cond_pos_embed_config=cond_pos_embed_config
        )

        self.sigreg = SIGReg(**config.sigreg)
        # self.sigreg_weight = config.sigreg.get('weight', 0.1)

        # Initialize loss weights with defaults
        self.loss_weights = {
            'emb_sta': 1.0,
            'emb_vis': 1.0,
            'sta': 1.0,
            'act': 1.0,
            'sigreg': 0.1
        }
        if getattr(config, 'loss_weights', None):
            self.loss_weights.update(config.loss_weights)

        # Condition adapters mapping between feature space and hidden dimension
        def create_adapter(config_type, in_dim, out_dim):
            return self.build_condition_adapter(config_type, in_features=in_dim, out_features=out_dim)
            
        self.vis_input_adaptor = create_adapter(config.vis_adaptor, vis_dim, hidden_size)
        self.vis_output_adaptor = create_adapter(config.vis_adaptor, hidden_size, vis_dim)
        self.action_input_adaptor = create_adapter(config.action_adaptor, action_dim, hidden_size)
        self.action_output_adaptor = create_adapter(config.action_adaptor, hidden_size, action_dim)

        self.normalizer = LinearNormalizer()
        # self.normalizer = None
        self.pretraining = config.pretraining
        self.action_key = config.action_key
        self.state_key = config.state_key
        self.state_tgt_key = config.state_tgt_key
        self.horizon = horizon
        self.vis_dim = vis_dim
        self.action_dim = action_dim
        self.n_action_steps = n_action_steps
        self.n_obs_steps = n_obs_steps
        self.action_mask = torch.zeros(1, horizon, action_dim).bool()
        self.action_mask[:,n_obs_steps:,:] = True
        self.obs_ind = torch.tensor(list(range(n_obs_steps))+[horizon], dtype=torch.long)
        # self.kwargs = kwargs
        self.kwargs = {} #
        
        # Apply freezing
        if getattr(config, 'freeze_keys', None):
            for key in config.freeze_keys:
                if key == 'vis_encoder':
                    for param in self.vis_encoder.parameters():
                        param.requires_grad_(False)
                elif key == 'state_encoder':
                    for param in self.state_encoder.parameters():
                        param.requires_grad_(False)
                elif key == 'state_decoder':
                    for param in self.state_decoder.parameters():
                        param.requires_grad_(False)
                elif key == 'predictor':
                    for block in self.model.blocks:
                        for param in block.block2.parameters():
                            param.requires_grad_(False)

    def build_condition_adapter(
        self, projector_type, in_features, out_features
    ):
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
  
    # ========= inference  ============
    def conditional_sample(
            self, 
            state_features: torch.Tensor, 
            vis_features: torch.Tensor, 
            action_mask: torch.Tensor, 
            return_flow: bool = False, 
            **kwargs
        ) -> torch.Tensor:
        '''
        vis_features: image conditional data, (batch_size, patch_num, hidden_size).
        state_features: (batch_size, 1, hidden_size), state trajectory.
        
        return: (batch_size, horizon, action_dim)
        '''
        device = state_features.device
        dtype = state_features.dtype
        batch_size = state_features.shape[0]
        noisy_features = torch.randn(
            size=(batch_size, self.horizon, self.action_dim), 
            dtype=dtype, device=device
        )
        flow = []
        if return_flow:
            flow.append(noisy_features.detach().cpu().numpy())

        vis_tokens = self.vis_input_adaptor(vis_features)
        vis_tokens = vis_tokens.view(batch_size, -1, vis_tokens.shape[-1])

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
            noisy_features[state_mask] = state_features[state_mask]
            
            # Predict the model output
            noisy_tokens = self.action_input_adaptor(noisy_features)
            pred_state_features, _ = self.model(noisy_tokens, timesteps, vis_tokens)
            pred_state_velocity = self.action_output_adaptor(pred_state_features)
            
            # Compute previous actions: x_t -> x_t-1
            noisy_features = noisy_features + dt * pred_state_velocity
            if return_flow:
                flow.append(noisy_features.detach().cpu().numpy())
            # noisy_features = noisy_features.to(state_features.dtype)
        
        # Finally apply the action mask to mask invalid action dimensions
        noisy_features[state_mask] = state_features[state_mask]

        return noisy_features, flow

    @torch.no_grad()
    def predict_action(self, obs: Dict[str, torch.Tensor], return_flow:bool = False, gt: Dict[str, torch.Tensor] = None) -> Dict[str, torch.Tensor]:
        """
        vis_dict: must include "obs" key
        result: must include "action" key
        """
        device = self.device
        dtype = self.dtype
        assert 'past_action' not in obs # not implemented yet
        # normalize input
        nobs = {k: v.to(device=device, dtype=dtype) for k, v in obs.items()}
        # filtered_obs_dict = {key: value for key, value in obs.items() 
        #         if key in self.normalizer.params_dict}
        filtered_obs_dict = {key: value for key, value in obs.items() 
                if key in ['endpose', 'agent_pos']}
        nobs.update(self.normalizer.normalize(filtered_obs_dict))
        value = next(iter(nobs.values()))
        B = value.shape[0]
        T = self.horizon
        Da = self.action_dim
        Do = self.vis_dim
        To = self.n_obs_steps
        Ts = value.shape[1]

        # handle different ways of passing observation
        # condition through global feature
        this_nobs = dict_apply(nobs, lambda x: x[:,:To,...].reshape(-1,*x.shape[2:]))
        vis_features = self.vis_encoder(this_nobs) # (BS, VP, Do)
        
        state_features = torch.zeros(size=(B, T, Da), device=device, dtype=dtype)
        state_features[:,:To,:] = self.state_encoder(this_nobs[self.state_key].view(B,To,-1))
        action_mask = self.action_mask.expand(B, -1, -1).to(device=device)

        # run sampling
        pred_state_features, flow = self.conditional_sample(state_features, vis_features, action_mask, return_flow)
        
        # unnormalize prediction
        pred_states = self.state_decoder(pred_state_features)
        pred_state_features[~action_mask] = state_features[~action_mask]
        pred_actions = self.action_decoder(pred_state_features)
        
        # We unnormalize separately to avoid key collisions if action_key == state_tgt_key
        pred_act_dict = self.normalizer.unnormalize({self.action_key: pred_actions})
        pred_sta_dict = self.normalizer.unnormalize({self.state_tgt_key: pred_states})

        # pad if needed
        pad_len = 1
        for d in [pred_act_dict, pred_sta_dict]:
            for k, v in d.items():
                pad_zeros = torch.zeros((v.shape[0], pad_len) + v.shape[2:], dtype=v.dtype, device=v.device)
                d[k] = torch.cat([v, pad_zeros], dim=1)

        # get action
        start = To
        end = start + self.n_action_steps
        if gt is not None:
            # 计算多样性 (沿 Batch 维度的标准差均值)，使用预测出来的去噪全时段轨迹特征计算
            state_diversity = state_features[:, :start].std(dim=0).mean()
            pred_state_diversity = pred_state_features[:, start:end].std(dim=0).mean()
            vis_diversity = vis_features.std(dim=0).mean()
            
            # 使用Normalized的原有空间，并限定在需要预测的时间窗口 [start:end] 中，以对齐训练 Loss 尺度
            action_loss = F.mse_loss(pred_actions[:,start:end], nobs[self.action_key][:,start:end], reduction="mean")
            state_loss = F.mse_loss(pred_states[:,start:end], nobs[self.state_tgt_key][:,start:end], reduction="mean")
            mse_loss = F.mse_loss(pred_act_dict[self.action_key][:,start:end], gt[self.action_key][:,start:end], reduction="mean")
            return {
                'loss': (action_loss + state_loss).detach(),
                'action loss': action_loss.detach(),
                'mse loss': mse_loss.detach(),
                'state loss': state_loss.detach(),
                'vision diversity': vis_diversity.detach(),
                'state diversity': state_diversity.detach(),
                'state predicted diversity': pred_state_diversity.detach(),
            }
        
        action = pred_act_dict[self.action_key][:,start:end]
        state = pred_sta_dict[self.state_tgt_key][:,start:end]
        if 'del' in self.action_key:
            key = self.action_key.replace('del_', '')
            action = action + obs[key][:,0:1,...]
        result = {
            'action': action,
            'action_pred': pred_act_dict[self.action_key],
            'state': state,
            'state_pred': pred_sta_dict[self.state_tgt_key],
        }
        
        if return_flow:
            flow = [smp[:,start:end] for smp in flow]
            result.update({
                'flow': flow,
                'gt': nobs['agent_pos'][:,start:end]
            })
        return result

    # ========= training  ============
    def set_normalizer(self, normalizer: LinearNormalizer):
        self.normalizer.load_state_dict(normalizer.state_dict())

    def sample_time(self, batch_size, device, dtype):
        sample = self.beta_dist.sample([batch_size]).to(device, dtype=dtype)
        return (self.noise_s - sample) / self.noise_s

    def compute_loss(self, obs, action, **kwargs):
        # normalize input
        assert 'valid_mask' not in kwargs
        nobs = {**obs}
        nobs.update(self.normalizer.normalize(
            {
                key: nobs[key] 
                for key in ['endpose', 'agent_pos', 'del_endpose', 'del_agent_pos']
                if key in nobs
            }
        ))
        # actions = self.normalizer['action'].normalize(action)
        if not isinstance(action, dict):
            action = {'action': action}
        action = self.normalizer.normalize(action)

        # set variables for easy access
        horizon = self.horizon
        value = next(iter(nobs.values()))
        batch_size = value.shape[0]
        data_length = value.shape[1]
        assert data_length >= horizon+1, "observation length should be horizon+1"

        # prepare action gt
        actions = action[self.action_key] if isinstance(action, dict) else action
        actions = actions[:,:horizon,...]

        # prepare state features
        gt_states = nobs[self.state_tgt_key][:,:horizon,...]
        states = nobs[self.state_key][:,:horizon,...]
        state_features = self.state_encoder(states) # (BS, horizon, hidden_size)

        # prepare vision features
        obs_ind = self.obs_ind.to(actions.device)
        this_nobs = dict_apply(nobs, lambda x: x[:,obs_ind,...].reshape(-1,*x.shape[2:]))
        sample_vis_features = self.vis_encoder(this_nobs) # (BS, VP, Do)
        sample_vis_features = sample_vis_features.reshape(batch_size, -1, *sample_vis_features.shape[1:]) # (BS, To, V, Do)
        gt_vis_features = sample_vis_features[:, -1, ...]
        vis_features = sample_vis_features[:, -2, ...]

        # Sample a random timestep
        t = self.sample_time(
            state_features.shape[0],
            device=state_features.device,
            dtype=state_features.dtype
        )
        t = t[:, None, None]  # shape (B,1,1) for broadcast
        # Convert (continuous) t -> discrete if needed
        timesteps = (t[:, 0, 0] * self.num_timestep_buckets).long() # shape (B,)
        
        # Sample noise
        action_mask = self.action_mask.expand(batch_size, -1, -1).to(device=state_features.device)
        noise = torch.randn(state_features.shape, device=state_features.device, dtype=state_features.dtype)
        velocity = state_features - noise
        noisy_features = (1 - t) * noise + t * state_features  
        # compute loss mask
        cond_mask = ~action_mask
        # apply conditioning
        noisy_features[cond_mask] = state_features[cond_mask]
        
        # Predict the noise residual
        noisy_tokens = self.action_input_adaptor(noisy_features)
        vis_tokens = self.vis_input_adaptor(vis_features)
        vis_tokens = vis_tokens.view(batch_size, -1, vis_tokens.shape[-1]) # (B, SVP, hidden_size)
        pred_state_features, pred_vis_features = self.model(noisy_tokens, timesteps, vis_tokens)
        pred_state_velocity = self.action_output_adaptor(pred_state_features)
        pred_vis_features = self.vis_output_adaptor(pred_vis_features)
        
        loss_emb_sta = F.mse_loss(pred_state_velocity[action_mask], velocity[action_mask], reduction="mean")
        loss_emb_vis = F.mse_loss(pred_vis_features, gt_vis_features, reduction="mean")

        pred_state_features = noisy_features + pred_state_velocity * (1-t)
        pred_state_features[cond_mask] = state_features[cond_mask]
        pred_states = self.state_decoder(pred_state_features)
        rec_states = self.state_decoder(state_features)

        # 计算多样性 (沿 Batch 维度的标准差均值)
        state_diversity = state_features.std(dim=0).mean()
        pred_state_diversity = pred_state_features.std(dim=0).mean()
        vis_diversity = vis_features.std(dim=0).mean()
        # gt_vis_diversity = gt_vis_features.std(dim=0).mean()
        # pred_vis_diversity = pred_vis_features.std(dim=0).mean()
        
        future_slice = slice(self.n_obs_steps, horizon)
        loss_sta = F.mse_loss(pred_states[:, future_slice], gt_states[:, future_slice], reduction="mean")
        loss_rec_sta = F.mse_loss(rec_states[:, future_slice], gt_states[:, future_slice], reduction="mean")

        loss_sigreg = self.sigreg(sample_vis_features.reshape(batch_size, -1, *sample_vis_features.shape[3:]).transpose(0, 1))
        # loss_sigreg += self.sigreg(pred_state_features.transpose(0, 1))
        # loss_sigreg += self.sigreg(state_features.transpose(0, 1))

        loss_act = 0.
        if not self.pretraining:
            # raise NotImplementedError("Only pretraining is implemented for now")
            # pred_state_features = noise + pred_action_velocity
            pred_actions = self.action_decoder(pred_state_features)
            loss_act = F.mse_loss(pred_actions[:, future_slice], actions[:, future_slice], reduction="mean")

        loss = (
            loss_emb_sta * self.loss_weights['emb_sta'] + 
            loss_emb_vis * self.loss_weights['emb_vis'] + 
            loss_sta * self.loss_weights['sta'] + 
            loss_rec_sta * self.loss_weights['rec_sta'] + 
            loss_act * self.loss_weights['act'] + 
            loss_sigreg * self.loss_weights['sigreg']
        )
        
        loss_dict = {
            'loss': loss, 
            'state emb loss': loss_emb_sta.detach(), 
            'vision emb loss': loss_emb_vis.detach(),
            'state loss': loss_sta.detach(),
            'state reconstruction loss': loss_rec_sta.detach(),
            'state diversity': state_diversity.detach(),
            'state predicted diversity': pred_state_diversity.detach(),
            'vision diversity': vis_diversity.detach(),
            # 'vision gt diversity': gt_vis_diversity.detach(),
            # 'vision predicted diversity': pred_vis_diversity.detach(),
            'signal regularization': loss_sigreg.detach(),
        }
        if not self.pretraining:
            loss_dict['action loss'] = loss_act.detach()
        return loss_dict