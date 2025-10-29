from __future__ import annotations

import argparse, json, os
from dataclasses import dataclass
from typing import Optional

import torch
from accelerate import Accelerator
from datasets import Dataset, concatenate_datasets
from peft import LoraConfig
from trl import SFTConfig, SFTTrainer, DPOConfig

from llms import ADAPTERS
from llms.base import BaseAdapter
from utils import set_seed, now_stamp, ensure_dir
from utils.vdpo import VDPOTrainer


@dataclass
class Args:
    stage: str
    model_family: str
    base_model: str
    output_dir: str
    image_prefix: str
    # data
    sft_train_jsonl: Optional[str]
    sft_val_jsonl: Optional[str]
    dpo_train_jsonl: Optional[str]
    dpo_val_jsonl: Optional[str]
    # sft hparams
    sft_epochs: int
    sft_lr: Optional[float]
    sft_per_device_train_batch_size: Optional[int]
    sft_per_device_eval_batch_size: Optional[int]
    sft_grad_accum: int
    # lora
    lora_r: int
    lora_alpha: int
    lora_dropout: float
    # ctl hparams
    lora_ckpt: Optional[str]
    beta: float
    rpo_alpha: float
    ctl_lr: float
    ctl_epochs: int
    keep_ratio_sft: float
    sft_sample_rate: float
    ctl_eval_steps: int
    ctl_save_steps: int
    # misc
    seed: int
    wandb_project: str
    no_wandb: bool
    resume_from: Optional[str]


def build_sft_trainer(adapter: BaseAdapter, args: Args, accelerator: Accelerator, out_dir: str):
    # data
    train_ds = adapter.make_sft_rows(args.sft_train_jsonl, args.image_prefix)
    val_ds = adapter.make_sft_rows(args.sft_val_jsonl, args.image_prefix)
    # model
    model, processor = adapter.load_model_and_processor_for_sft(args.base_model)
    model.config.use_cache = False
    model.gradient_checkpointing_enable()

    # family defaults
    if args.sft_lr is None:
        sft_lr = 2e-4 if adapter.family == "qwen2vl" else 1e-4
    else:
        sft_lr = args.sft_lr
    if args.sft_per_device_train_batch_size is None:
        bs = 3 if adapter.family == "qwen2vl" else 1
    else:
        bs = args.sft_per_device_train_batch_size
    bs_eval = args.sft_per_device_eval_batch_size or bs

    sft_cfg = SFTConfig(
        output_dir=out_dir,
        num_train_epochs=args.sft_epochs,
        per_device_train_batch_size=bs,
        per_device_eval_batch_size=bs_eval,
        gradient_accumulation_steps=args.sft_grad_accum,
        gradient_checkpointing=True,
        optim="adamw_bnb_8bit",
        learning_rate=sft_lr,
        lr_scheduler_type="cosine",
        logging_steps=1,
        eval_strategy="steps",
        eval_steps=50,
        save_strategy="steps",
        save_steps=50,
        save_total_limit=2,
        metric_for_best_model="eval_loss",
        greater_is_better=False,
        load_best_model_at_end=True,
        bf16=True,
        tf32=True,
        max_grad_norm=0.3,
        warmup_ratio=0.03,
        push_to_hub=False,
        report_to=([] if args.no_wandb else ["wandb"]),
        gradient_checkpointing_kwargs={"use_reentrant": False},
        dataset_text_field="",
        dataset_kwargs={"skip_prepare_dataset": True},
        remove_unused_columns=False,
    )

    if not accelerator.is_local_main_process:
        os.environ["WANDB_MODE"] = "disabled"

    lora_cfg = LoraConfig(
        r=args.lora_r,
        lora_alpha=args.lora_alpha,
        lora_dropout=args.lora_dropout,
        bias="none",
        task_type="CAUSAL_LM",
        target_modules=["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"],
    )

    return SFTTrainer(
        model=model,
        args=sft_cfg,
        train_dataset=train_ds,
        eval_dataset=val_ds,
        data_collator=lambda ex: adapter.collate_sft(ex, processor),
        peft_config=lora_cfg,
    )


def build_ctl_trainer(adapter: BaseAdapter, args: Args, accelerator: Accelerator, out_dir: str):
    sft_train_ds = adapter.make_sft_rows_pairs(args.sft_train_jsonl, args.image_prefix)
    sft_val_ds   = adapter.make_sft_rows_pairs(args.sft_val_jsonl,   args.image_prefix)
    dpo_train_ds = adapter.make_dpo_rows(args.dpo_train_jsonl, args.image_prefix)
    dpo_val_ds   = adapter.make_dpo_rows(args.dpo_val_jsonl,   args.image_prefix)

    train = concatenate_datasets([
        sft_train_ds.shuffle(seed=args.seed).select(range(int(len(sft_train_ds) * args.keep_ratio_sft))),
        dpo_train_ds,
    ]).shuffle(seed=args.seed)
    val = concatenate_datasets([
        sft_val_ds.shuffle(seed=args.seed).select(range(int(len(sft_val_ds) * args.keep_ratio_sft))),
        dpo_val_ds,
    ]).shuffle(seed=args.seed)

    weights = [1.0 if ex["is_dpo_data"] else args.sft_sample_rate for ex in train]
    sampler = torch.utils.data.WeightedRandomSampler(weights=weights, num_samples=len(weights), replacement=True)

    policy, processor = adapter.load_policy_for_ctl(args.base_model, args.lora_ckpt)
    for p in policy.get_input_embeddings().parameters():
        p.requires_grad_(True)
    policy.base_model.gradient_checkpointing_enable({"use_reentrant": False})

    dpo_cfg = DPOConfig(
        beta=args.beta,
        rpo_alpha=args.rpo_alpha,
        learning_rate=args.ctl_lr,
        per_device_train_batch_size=1,
        per_device_eval_batch_size=1,
        num_train_epochs=args.ctl_epochs,
        gradient_accumulation_steps=4,
        logging_steps=1,
        eval_strategy="steps",
        eval_steps=args.ctl_eval_steps,
        save_strategy="steps",
        save_steps=args.ctl_save_steps,
        save_total_limit=20,
        metric_for_best_model="eval_loss",
        greater_is_better=False,
        load_best_model_at_end=True,
        bf16=True,
        padding_value=processor.tokenizer.pad_token_id,
        remove_unused_columns=False,
        report_to=([] if args.no_wandb else ["wandb"]),
        output_dir=out_dir,
        precompute_ref_log_probs=False,
        max_grad_norm=0.3,
        warmup_ratio=0.03,
    )

    if not accelerator.is_local_main_process:
        os.environ["WANDB_MODE"] = "disabled"

    return VDPOTrainer(
        model=policy,
        ref_model=None,
        args=dpo_cfg,
        train_dataset=train,
        eval_dataset=val,
        processing_class=processor,
        train_sampler=sampler,
        data_collator=lambda ex: adapter.collate_ctl(ex, processor),
    )


def parse_cli() -> Args:
    p = argparse.ArgumentParser(description="Unified BEAT training entrypoint (SFT / CTL)")

    p.add_argument("--stage", choices=["sft", "ctl", "sft+ctl"], default="sft")
    p.add_argument("--model_family", choices=list(ADAPTERS.keys()), default="qwen2vl")

    p.add_argument("--base_model", type=str, required=True)
    p.add_argument("--output_dir", type=str, default="ckpts/beat")
    p.add_argument("--image_prefix", type=str, default="/workspace/backdoor-data")

    p.add_argument("--sft_train_jsonl", type=str)
    p.add_argument("--sft_val_jsonl", type=str)
    p.add_argument("--dpo_train_jsonl", type=str)
    p.add_argument("--dpo_val_jsonl", type=str)

    p.add_argument("--sft_epochs", type=int, default=3)
    p.add_argument("--sft_lr", type=float, default=None)
    p.add_argument("--sft_per_device_train_batch_size", type=int, default=None)
    p.add_argument("--sft_per_device_eval_batch_size", type=int, default=None)
    p.add_argument("--sft_grad_accum", type=int, default=3)

    p.add_argument("--lora_r", type=int, default=16)
    p.add_argument("--lora_alpha", type=int, default=32)
    p.add_argument("--lora_dropout", type=float, default=0.05)

    p.add_argument("--lora_ckpt", type=str)
    p.add_argument("--beta", type=float, default=0.05)
    p.add_argument("--rpo_alpha", type=float, default=0.4)
    p.add_argument("--ctl_lr", type=float, default=3e-5)
    p.add_argument("--ctl_epochs", type=int, default=2)
    p.add_argument("--keep_ratio_sft", type=float, default=1.0)
    p.add_argument("--sft_sample_rate", type=float, default=0.2)
    p.add_argument("--ctl_eval_steps", type=int, default=100)
    p.add_argument("--ctl_save_steps", type=int, default=100)

    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--wandb_project", type=str, default="BEAT")
    p.add_argument("--no_wandb", action="store_true")
    p.add_argument("--resume_from", type=str, default=None)

    a = p.parse_args()

    if a.stage in ("sft", "sft+ctl"):
        for k in ("sft_train_jsonl", "sft_val_jsonl"):
            if getattr(a, k) is None:
                p.error(f"--{k} is required for stage {a.stage}")
    if a.stage in ("ctl", "sft+ctl"):
        for k in ("sft_train_jsonl", "sft_val_jsonl", "dpo_train_jsonl", "dpo_val_jsonl"):
            if getattr(a, k) is None:
                p.error(f"--{k} is required for stage {a.stage}")

    return Args(**vars(a))


def main():
    args = parse_cli()
    set_seed(args.seed)

    AdapterCls = ADAPTERS[args.model_family]
    adapter: BaseAdapter = AdapterCls()

    root = ensure_dir(args.output_dir)

    if args.stage in ("sft", "sft+ctl"):
        sft_out = ensure_dir(os.path.join(root, f"sft-{adapter.family}-{now_stamp()}"))
        with open(os.path.join(sft_out, "args.json"), "w") as f:
            json.dump(vars(args), f, indent=2)
        accelerator = Accelerator()
        if not args.no_wandb and accelerator.is_local_main_process:
            import wandb
            wandb.init(project=args.wandb_project, name=os.path.basename(sft_out), config=vars(args))
        sft_tr = build_sft_trainer(adapter, args, accelerator, sft_out)
        sft_tr.train(resume_from_checkpoint=args.resume_from)
        sft_tr.save_model(os.path.join(sft_out, "best"))
        if not args.no_wandb and accelerator.is_local_main_process:
            import wandb
            wandb.finish()
        produced_lora = os.path.join(sft_out, "best")
    else:
        produced_lora = None

    if args.stage in ("ctl", "sft+ctl"):
        lora_for_ctl = args.lora_ckpt or produced_lora
        if lora_for_ctl is None:
            raise ValueError("CTL requires --lora_ckpt, or run SFT first (stage=sft+ctl)")
        ctl_out = ensure_dir(os.path.join(root, f"ctl-{adapter.family}-{now_stamp()}"))
        with open(os.path.join(ctl_out, "args.json"), "w") as f:
            d = vars(args).copy(); d["lora_ckpt"] = lora_for_ctl; json.dump(d, f, indent=2)
        args.lora_ckpt = lora_for_ctl
        accelerator = Accelerator()
        if not args.no_wandb and accelerator.is_local_main_process:
            import wandb
            wandb.init(project=args.wandb_project, name=os.path.basename(ctl_out), config=vars(args))
        ctl_tr = build_ctl_trainer(adapter, args, accelerator, ctl_out)
        ctl_tr.train(resume_from_checkpoint=args.resume_from)
        ctl_tr.save_model(ctl_out)
        if not args.no_wandb and accelerator.is_local_main_process:
            import wandb
            wandb.finish()

    print("Done.")


if __name__ == "__main__":
    main()