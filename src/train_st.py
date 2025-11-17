"""
Training script for speech translation task
"""

import logging
import os
from datetime import datetime

import torch
import wandb
from omegaconf import OmegaConf
from peft import LoraConfig, get_peft_model
from transformers import (
    BitsAndBytesConfig,
    TrainingArguments,
    VoxtralForConditionalGeneration,
    VoxtralProcessor,
)

from utils.dataset_utils import load_st_manifest_dataset
from utils.collators import StreamingSTCollator
from utils.train_utils import SafeTrainer

logging.basicConfig(
    level=logging.INFO,  # or DEBUG for more verbose logs
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)

logger = logging.getLogger(__name__)  # module-level logger


def main():
    # Load config
    config = OmegaConf.load("config/train_st.yaml")
    os.environ["CUDA_VISIBLE_DEVICES"] = str(config.device_id)

    # --- WandB setup ---
    if config.exp_manager.logger == "wandb":
        if config.trainer.resume_from_checkpoint and config.exp_manager.wandb.run_id:
            wandb.init(
                project=config.exp_manager.wandb.project,
                id=config.exp_manager.wandb.run_id,
                resume="must",
            )
        else:
            wandb.init(
                project=config.exp_manager.wandb.project,
                name=config.exp_manager.name,
            )

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    exp_prefix = (
        f"{config.exp_manager.name}_RESUME"
        if config.trainer.resume_from_checkpoint
        else config.exp_manager.name
    )
    exp_name = f"{exp_prefix}_{timestamp}"
    output_dir = os.path.join(config.exp_manager.exp_dir, exp_name)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    logger.info("Using device: %s", device)

    # --- Load processor ---
    logger.info("Loading processor...")
    processor = VoxtralProcessor.from_pretrained(config.model)

    # --- Load datasets ---
    logger.info("Loading datasets...")
    train_dataset, eval_dataset = load_st_manifest_dataset(
        train_manifest=config.data.train_manifest,
        eval_manifest=config.data.eval_manifest,
    )

    # --- Data collator ---
    data_collator = StreamingSTCollator(processor, model_id=config.model, incl_src_lang=config.tasks.s2tt.incl_src_lang)

    # --- Model & LoRA setup ---
    logger.info("Loading model...")

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

    for param in model.audio_tower.parameters():
        param.requires_grad = False

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
    model.gradient_checkpointing_enable()
    model.print_trainable_parameters()

    # --- Training args ---

    training_args = TrainingArguments(
        output_dir=output_dir,
        per_device_train_batch_size=config.trainer.train_batch_size,
        per_device_eval_batch_size=config.trainer.eval_batch_size,
        gradient_accumulation_steps=config.trainer.grad_accum,
        learning_rate=config.trainer.lr,
        num_train_epochs=config.trainer.epochs,
        warmup_steps=config.trainer.warmup_steps,
        bf16=config.trainer.bf16,
        logging_steps=config.trainer.logging_steps,
        eval_steps=config.trainer.eval_steps if eval_dataset else None,
        save_steps=config.trainer.save_steps,
        eval_strategy="steps" if eval_dataset else "no",
        save_strategy="steps",
        load_best_model_at_end=True,
        metric_for_best_model="eval_loss",
        greater_is_better=False,
        save_total_limit=config.trainer.save_total_limit,
        report_to=config.exp_manager.logger,
        remove_unused_columns=False,
        dataloader_num_workers=1,
        lr_scheduler_type="cosine",
        seed=3407,
    )

    # --- Trainer ---
    trainer = SafeTrainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=eval_dataset,
        data_collator=data_collator,
    )

    logger.info("Starting training...")
    trainer.train(resume_from_checkpoint=config.trainer.resume_from_checkpoint)
    logger.info("Training complete!")


if __name__ == "__main__":
    main()
