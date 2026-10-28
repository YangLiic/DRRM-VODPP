import copy
import logging
import math
import os
from pathlib import Path
import hydra
from omegaconf import OmegaConf

import diffusers
import torch
from torch.utils.data import RandomSampler, BatchSampler
import transformers
from transformers import AutoConfig, AutoModel
from transformers.models.deit.image_processing_deit import valid_images
from accelerate import Accelerator
from accelerate.utils import ProjectConfiguration, set_seed
from diffusers.optimization import get_scheduler
from diffusers.utils import is_wandb_available
from tqdm.auto import tqdm
from safetensors.torch import load_model

from drrm.common.ema_model import EMAModel


if is_wandb_available():
    import wandb

@torch.no_grad()
def log_sample_res(policy_model, args, dataloader, logger):
    logger.info(f"Running sampling for {args.num_val_batches} batches (bs={args.val_batch_size})...")

    policy_model.eval()
    
    loss_for_log = {}
    val_losses = {}
    for step, batch in enumerate(dataloader):
        if step >= args.num_val_batches:
            break
        loss = policy_model(**batch)
        if isinstance(loss, dict):
            for k, v in loss.items():
                if k not in val_losses:
                    val_losses[k] = []
                val_losses[k].append(v.item())
        else:
            if 'loss' not in val_losses:
                val_losses['loss'] = []
            val_losses['loss'].append(loss.item())
        
    for k, v_list in val_losses.items():
        loss_for_log[k] = torch.mean(torch.tensor(v_list)).item()

    policy_model.train()
    torch.cuda.empty_cache()

    return dict(loss_for_log)

def get_normalizer():
    raise NotImplementedError

def save_policy_custom(polciy, save_path):
    import inspect
    import shutil
    PolicyClass = polciy.__class__
    PolicyConfigClass = polciy.config_class
    code_path = inspect.getfile(PolicyClass)
    code_name = os.path.basename(code_path)
    if hasattr(polciy.config, 'obs_encoder') and 'emvis_config' in polciy.config.obs_encoder:
        emvis_config = polciy.config.obs_encoder['emvis_config']
        emvis_config['load_vggt_pretrain'] = False
        # emvis_config['load_vggt_heads'] = False
        emvis_config['visualize'] = False
    polciy.config.auto_map = {
        "AutoConfig": f"{os.path.splitext(code_name)[0]}.{PolicyConfigClass.__name__}",
        "AutoModel": f"{os.path.splitext(code_name)[0]}.{PolicyClass.__name__}"
    }
    shutil.copy(code_path, os.path.join(save_path, code_name))

    state_dict = polciy.state_dict()
    encoder_keys = [key for key in state_dict.keys() if 'vggt_encoder' in key]
    def is_shared(key):
        if not 'vggt_heads' in key:
            return False
        else:
            return 'vggt_encoder'.join(key.split('vggt_heads')) in encoder_keys
    filtered_state_dict = { 
        k:v
        for k,v in state_dict.items()
        if not is_shared(k)
    }
    polciy.save_pretrained(save_path, state_dict=filtered_state_dict, max_shard_size="10GB")

def resume_policy(ckp_path, ema_policy_model):
    load_model(ema_policy_model, os.path.join(ckp_path, "model.safetensors"), strict=False)
    return ema_policy_model

def train(args, logger):
    accelerator = Accelerator(
        gradient_accumulation_steps=args.gradient_accumulation_steps,
        mixed_precision=args.mixed_precision,
        log_with=args.report_to,
        project_dir=Path(args.output_dir, args.logging_dir),
        project_config=ProjectConfiguration(total_limit=args.checkpoints_total_limit),
    )

    if args.report_to == "wandb":
        if not is_wandb_available():
            raise ImportError("Make sure to install wandb if you want to use it for logging during training.")

    # Make one log on every process with the configuration for debugging.
    logging.basicConfig(
        format="%(asctime)s - %(levelname)s - %(name)s - %(message)s",
        datefmt="%m/%d/%Y %H:%M:%S",
        level=logging.INFO,
    )
    # Log the accelerator state information on all processes for debugging
    # This includes information about distributed training setup, device allocation,
    # mixed precision settings, and other accelerator configuration details
    logger.info(accelerator.state, main_process_only=False)
    if accelerator.is_local_main_process:
        transformers.utils.logging.set_verbosity_warning()
        diffusers.utils.logging.set_verbosity_info()
    else:
        transformers.utils.logging.set_verbosity_error()
        diffusers.utils.logging.set_verbosity_error()

    # If passed along, set the training seed now.
    if args.seed is not None:
        set_seed(args.seed)

    # For mixed precision training we cast the text_encoder and vae weights to half-precision
    # as these models are only used for inference, keeping weights in full precision is not required.
    weight_dtype = torch.float32
    if accelerator.mixed_precision == "fp16":
        weight_dtype = torch.float16
    elif accelerator.mixed_precision == "bf16":
        weight_dtype = torch.bfloat16

    # Policy Model creation
    model_args = OmegaConf.to_container(args.model)
    pkg_config = model_args.pop('target_config')
    pkg_policy = model_args.pop('target_policy')
    ConfigClass = hydra.utils.get_class(pkg_config)
    PolicyClass = hydra.utils.get_class(pkg_policy)
    model_args['pkg_map'] = {
        "AutoConfig": pkg_config,
        "AutoModel": pkg_policy
    }
    config = ConfigClass.from_dict(model_args)
    policy_model = PolicyClass(config)
    if hasattr(args, "checkpoint_dir") and args.checkpoint_dir:
        load_model(policy_model, os.path.join(args.checkpoint_dir, "model.safetensors"), strict=False)
    # policy_model = load_policy("checkpoints/dp_baseline/test", use_ckp_code = True)
    policy_model.to(accelerator.device)

    ema_policy_model = copy.deepcopy(policy_model)
    ema_model = EMAModel(
        ema_policy_model,
        update_after_step=args.ema.update_after_step,
        inv_gamma=args.ema.inv_gamma,
        power=args.ema.power,
        min_value=args.ema.min_value,
        max_value=args.ema.max_value,
    )

    # create custom saving & loading hooks so that `accelerator.save_state(...)` serializes in a nice format
    # which ensure saving model in huggingface format (config.json + pytorch_model.bin)
    def save_model_hook(models, weights, output_dir):
        if accelerator.is_main_process:
            for model in models:
                model_to_save = model.module if hasattr(model, "module") else model  # type: ignore
                if isinstance(model_to_save, type(accelerator.unwrap_model(policy_model))):
                    save_policy_custom(model_to_save, output_dir)

    accelerator.register_save_state_pre_hook(save_model_hook)

    if args.gradient_checkpointing:
        # TODO: 
        raise NotImplementedError("Gradient checkpointing is not yet implemented.")

    # Enable TF32 for faster training on Ampere GPUs,
    # cf https://pytorch.org/docs/stable/notes/cuda.html#tensorfloat-32-tf32-on-ampere-devices
    if args.allow_tf32:
        torch.backends.cuda.matmul.allow_tf32 = True

    # Optimizer creation
    params_to_optimize = policy_model.parameters()
    optimizer = hydra.utils.instantiate(args.optimizer, params=params_to_optimize)
    
    # Dataset and DataLoaders creation
    train_dataset = hydra.utils.instantiate(args.train_dataset)
    if hasattr(train_dataset, "get_normalizer"):
        normalizer = train_dataset.get_normalizer()
        policy_model.set_normalizer(normalizer)
        ema_policy_model.set_normalizer(normalizer)
    val_dataset = train_dataset.get_validation_dataset()
    sampler = RandomSampler(
        train_dataset,
        replacement=True,
        num_samples=len(train_dataset) * args.num_train_epochs, # total samples
    )
    batch_sampler = BatchSampler(
        sampler,
        batch_size=args.train_batch_size // accelerator.num_processes,  # batch size per device
        drop_last=True,
    )
    train_dataloader = torch.utils.data.DataLoader(
        train_dataset,
        batch_sampler=batch_sampler,
        # sampler=sampler,
        # batch_size=args.train_batch_size // accelerator.num_processes,
        # drop_last=True,
        num_workers=args.dataloader_num_workers // accelerator.num_processes,
        pin_memory=True,
        persistent_workers=True,
    )
    val_dataloader = torch.utils.data.DataLoader(
        val_dataset,
        batch_size=args.val_batch_size,
        shuffle=True,
        num_workers=math.ceil(args.val_batch_size / 16),
        pin_memory=True,
        persistent_workers=True,
    )

    lr_config = args.lr_scheduler
    lr_scheduler = get_scheduler(
        lr_config.name,
        optimizer=optimizer,
        num_warmup_steps=int(lr_config.warmup_ratio * len(train_dataloader)) 
            if hasattr(lr_config, "warmup_ratio") else 0,
        num_training_steps=len(train_dataloader),
        num_cycles=lr_config.num_cycles 
            if hasattr(lr_config, "num_cycles") else 1,
        power=lr_config.power 
            if hasattr(lr_config, "power") else 1.0,
    )

    # Prepare everything with `accelerator`.
    policy_model, optimizer, train_dataloader, val_dataloader, lr_scheduler = accelerator.prepare(
        policy_model, optimizer, train_dataloader, val_dataloader, lr_scheduler                   
    )
    ema_policy_model.to(accelerator.device, dtype=weight_dtype)

    # We need to initialize the trackers we use, and also store our configuration.
    # The trackers initializes automatically on the main process.
    if accelerator.is_main_process:
        # Convert omegaconf.DictConfig to primitive types so it doesn't get dumped as truncated JSON
        config_dict = OmegaConf.to_container(args, resolve=True)
        accelerator.init_trackers("DRRM", config=config_dict)

    # Train!
    max_iters = len(train_dataloader)
    logger.info("***** Running training *****")
    logger.info(f"  Num examples = {len(train_dataset)}")
    logger.info(f"  Num epochs = {args.num_train_epochs}")
    logger.info(f"  Total num examples = {len(train_dataset) * args.num_train_epochs}")
    logger.info(f"  Batch size per device = {args.train_batch_size // accelerator.num_processes}")
    logger.info(f"  Num processes = {accelerator.num_processes}")
    logger.info(f"  Total batch size = {args.train_batch_size}")
    logger.info(f"  Num iter per device = {len(train_dataloader)}")
    logger.info(f"  Num optimization steps per device = {len(train_dataloader) / args.gradient_accumulation_steps}")
    global_step = 0

    # Potentially load in the weights and states from a previous save
    if args.resume_from_checkpoint:
        if args.resume_from_checkpoint != "latest":
            path = os.path.basename(args.resume_from_checkpoint)
        else:
            # Get the mos recent checkpoint
            dirs = os.listdir(args.output_dir)
            dirs = [d for d in dirs if d.startswith("checkpoint")]
            dirs = sorted(dirs, key=lambda x: int(x.split("-")[1]))
            path = dirs[-1] if len(dirs) > 0 else None

        if path is None:
            accelerator.print(
                f"Checkpoint '{args.resume_from_checkpoint}' does not exist. Starting a new training run."
            )
            args.resume_from_checkpoint = None
        else:
            accelerator.print(f"Resuming from checkpoint {path}")
            # When resume_from_checkpoint is an absolute path, load directly from it;
            # output_dir is only used for saving new checkpoints.
            if os.path.isabs(args.resume_from_checkpoint) and args.resume_from_checkpoint != "latest":
                load_checkpoint_path = args.resume_from_checkpoint
            else:
                load_checkpoint_path = os.path.join(args.output_dir, path)
            try:
                accelerator.load_state(load_checkpoint_path, strict=False)
            except Exception as _ckpt_err:
                logger.info(f"Resuming training state failed ({_ckpt_err}). Attempting to only load from model checkpoint.")
                ds_ckpt = os.path.join(load_checkpoint_path, "pytorch_model", "mp_rank_00_model_states.pt")
                if os.path.exists(ds_ckpt):
                    # DeepSpeed-format checkpoint
                    checkpoint = torch.load(ds_ckpt)
                    policy_model.module.load_state_dict(checkpoint["module"])
                else:
                    # accelerator.load_state already loaded model weights before failing on optimizer;
                    # no further action needed for model weights.
                    logger.info("DeepSpeed checkpoint not found; model weights already loaded by accelerator.load_state.")
            
            load_model(ema_policy_model, os.path.join(load_checkpoint_path, "model.safetensors"), strict=False)
            global_step = int(path.split("-")[1])

            # normalizer is not load to device by default, so we need to do it manually
            if hasattr(policy_model, "normalizer"):
                policy_model.normalizer.to(accelerator.device)

    # Only show the progress bar once on each machine.
    progress_bar = tqdm(range(0, max_iters), disable=not accelerator.is_local_main_process)
    progress_bar.set_description("Steps")
    progress_bar.update(global_step)

    policy_model.train()
        
    # Forward and backward...
    for batch in train_dataloader:
        with accelerator.accumulate(policy_model):
            loss = policy_model(**batch)
            if isinstance(loss, dict):
                loss_for_log = loss
                loss = loss.pop("loss")
            else:
                loss_for_log = {"loss": loss.detach().item()}

            accelerator.backward(loss)
            if accelerator.sync_gradients:
                params_to_clip = policy_model.parameters()
                accelerator.clip_grad_norm_(params_to_clip, args.max_grad_norm)
            
            optimizer.step()
            lr_scheduler.step()
            optimizer.zero_grad(set_to_none=args.set_grads_to_none)


        ema_model.step(accelerator.unwrap_model(policy_model))

        # Checks if the accelerator has performed an optimization step behind the scenes
        logs = {}
        if accelerator.sync_gradients:
            progress_bar.update(1)

            if global_step % args.checkpointing_period == 0:
                save_path = os.path.join(args.output_dir, f"checkpoint-{global_step}")
                accelerator.save_state(save_path, False)
                ema_save_path = os.path.join(save_path, f"ema")
                accelerator.save_model(ema_policy_model, ema_save_path)
                logger.info(f"Saved state to {save_path}")

            if args.val_period > 0 and global_step % args.val_period == 0:
                logger.info(f"Sampling at step {global_step}")
                sample_loss_for_log = log_sample_res(
                    policy_model,    # We do not use EMA currently
                    args,
                    val_dataloader,
                    logger,
                )
                # logger.info(sample_loss_for_log)
                # accelerator.log(sample_loss_for_log, step=global_step)
                loss_for_log.update({f"val {k}": v for k, v in sample_loss_for_log.items()})
            global_step += 1
        
        logs.update({"loss": loss.detach().item(), "lr": lr_scheduler.get_last_lr()[0]})
        progress_bar.set_postfix(**logs)
        logs.update(loss_for_log)
        # logger.info(logs)
        accelerator.log(logs, step=global_step)

        if global_step >= max_iters:
            break

    # Create the pipeline using using the trained modules and save it.
    accelerator.wait_for_everyone()
    if accelerator.is_main_process:
        # accelerator.unwrap_model(policy_model).save_pretrained(args.output_dir)
        model_to_save = accelerator.unwrap_model(policy_model)
        save_policy_custom(model_to_save, args.output_dir)
        # TODO change
        # model_to_save.save_pretrained(args.output_dir, max_shard_size="10GB")

        ema_save_path = os.path.join(args.output_dir, f"ema")
        accelerator.save_model(ema_policy_model, ema_save_path)
        
        logger.info(f"Saved Model to {args.output_dir}")

    accelerator.end_training()