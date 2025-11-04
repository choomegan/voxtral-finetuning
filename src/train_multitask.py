"""
Multitask finetuning for ASR and ST
"""

import os
from datetime import datetime
import torch
import wandb
from omegaconf import OmegaConf
from peft import LoraConfig, get_peft_model
from transformers import (
    BitsAndBytesConfig,
    Trainer,
    TrainingArguments,
    VoxtralForConditionalGeneration,
    VoxtralProcessor,
)
from utils.dataset_utils import load_preprocessed_multitask_dataset
from utils.collators import FastASRCollator, FastSTCollator, FastMultiTaskCollator


def main():
    config = OmegaConf.load("config/train_multitask.yaml")

    # --- Setup WandB ---
    if config.exp_manager.logger == "wandb":
        wandb.init(
            project=config.exp_manager.wandb.project, name=config.exp_manager.name
        )

    # --- Experiment name ---
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    exp_name = f"{config.exp_manager.name}_{timestamp}"
    output_dir = os.path.join(config.exp_manager.exp_dir, exp_name)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    # --- Processor ---
    print("Loading processor...")
    processor = VoxtralProcessor.from_pretrained(config.model)

    # Load preprocessed datasets (this caches to disk)
    print("Loading or creating preprocessed datasets...")
    train_dataset, eval_dataset = load_preprocessed_multitask_dataset(
        train_manifest=config.data.train_manifest,
        eval_manifest=config.data.eval_manifest,
        processor=processor,
        model_id=config.model,
    )

    # --- Collators ---
    asr_collator = FastASRCollator(processor)
    st_collator = FastSTCollator(processor)
    multi_collator = FastMultiTaskCollator(asr_collator, st_collator)

    # --- Model ---
    print("Loading model...")
    if config.trainer.load_in_4bit:
        bnb_config = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_use_double_quant=True,
            bnb_4bit_quant_type="nf4",
            bnb_4bit_compute_dtype=torch.float16,
        )

        model = VoxtralForConditionalGeneration.from_pretrained(
            config.model,
            torch_dtype=torch.float16,
            quantization_config=bnb_config,
            device_map={"": int(config.device_id)},  # single GPU
        )
    else:
        model = VoxtralForConditionalGeneration.from_pretrained(
            config.model,
            torch_dtype=torch.bfloat16 if config.trainer.bf16 else torch.float16,
            device_map={"": int(config.device_id)},  # single GPU
        )

    # Freeze audio encoder
    for param in model.audio_tower.parameters():
        param.requires_grad = False

    # --- LoRA ---
    lora_config = LoraConfig(
        r=config.lora.r,
        lora_alpha=config.lora.alpha,
        lora_dropout=config.lora.dropout,
        bias="none",
        target_modules=list(config.lora.target_modules),
        task_type="SEQ_2_SEQ_LM",
    )
    model = get_peft_model(model, lora_config)
    model.enable_input_require_grads()
    model.print_trainable_parameters()

    # --- Training ---
    args = TrainingArguments(
        output_dir=output_dir,
        per_device_train_batch_size=config.trainer.train_batch_size,
        per_device_eval_batch_size=config.trainer.eval_batch_size,
        gradient_accumulation_steps=config.trainer.grad_accum,
        learning_rate=config.trainer.lr,
        num_train_epochs=config.trainer.epochs,
        warmup_steps=config.trainer.warmup_steps,
        bf16=config.trainer.bf16,
        logging_steps=config.trainer.logging_steps,
        eval_steps=config.trainer.eval_steps,
        save_steps=config.trainer.save_steps,
        save_strategy="steps",
        eval_strategy="steps",
        save_total_limit=config.trainer.save_total_limit,
        report_to=config.exp_manager.logger,
        remove_unused_columns=False,
        dataloader_num_workers=8,
        dataloader_prefetch_factor=2,  # Prefetch 2 batches per worker
        dataloader_pin_memory=True,
        gradient_checkpointing=False,
        lr_scheduler_type="cosine",
        seed=3407,
    )

    trainer = Trainer(
        model=model,
        args=args,
        train_dataset=train_dataset,
        eval_dataset=eval_dataset,
        data_collator=multi_collator,
    )

    print("Starting multi-task training...")
    trainer.train(resume_from_checkpoint=config.trainer.resume_from_checkpoint)
    print("Training complete!")

    trainer.save_model()
    processor.save_pretrained(output_dir)


if __name__ == "__main__":
    main()
