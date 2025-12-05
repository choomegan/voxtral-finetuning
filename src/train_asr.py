"""
Training script for ASR task
"""

import logging
import os
from datetime import datetime

import torch
from accelerate import Accelerator
from omegaconf import OmegaConf
from peft import LoraConfig, get_peft_model
from transformers import (
    BitsAndBytesConfig,
    TrainingArguments,
    VoxtralForConditionalGeneration,
    VoxtralProcessor,
)

import wandb
from utils.collators import StreamingASRCollator
from utils.dataset_utils import load_asr_manifest_dataset
from utils.train_utils import SafeTrainer

logging.basicConfig(
    level=logging.ERROR,  # or DEBUG for more verbose logs
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)

logger = logging.getLogger(__name__)  # module-level logger


def main():
    accelerator = Accelerator()
    # Load training config
    config = OmegaConf.load("config/train_asr.yaml")

    if config.exp_manager.logger == "wandb" and accelerator.is_main_process:
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

    # Generate timestamped experiment name
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    if config.trainer.resume_from_checkpoint:
        exp_prefix = f"{config.exp_manager.name}_RESUME"
    else:
        exp_prefix = config.exp_manager.name

    exp_name = f"{exp_prefix}_{timestamp}"

    # Configuration
    model_checkpoint = config.model
    output_dir = os.path.join(config.exp_manager.exp_dir, exp_name)

    # Set device
    torch_device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    logger.info("Using device: %s", torch_device)

    #################### Load datasets from manifest files #############################
    logger.info("Loading processor...")
    processor = VoxtralProcessor.from_pretrained(model_checkpoint)

    logger.info("Loading datasets...")
    train_dataset, eval_dataset = load_asr_manifest_dataset(
        train_manifest=config.data.train_manifest,
        eval_manifest=config.data.eval_manifest,
    )

    # Setup data collator
    data_collator = StreamingASRCollator(
        processor, model_id=config.model, lang=config.lang
    )

    ########################### Load processor and model ###############################
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
            # device_map={"": int(config.device_id)},  # single GPU
        )
    else:
        model = VoxtralForConditionalGeneration.from_pretrained(
            config.model,
            torch_dtype=torch.bfloat16 if config.trainer.bf16 else torch.float16,
            # device_map={"": int(config.device_id)},  # single GPU
        )

    # Load model with LoRA configuration
    lora_config = LoraConfig(
        r=config.lora.r,  # Rank of LoRA
        lora_alpha=config.lora.alpha,
        lora_dropout=config.lora.dropout,
        bias="none",
        target_modules=list(config.lora.target_modules),
        task_type="SEQ_2_SEQ_LM",
    )

    # Partial unFreeze the audio encoder model.audio_tower
    # for name, param in model.audio_tower.named_parameters():
    #     if any(f"block.{i}." in name for i in range(28,32)):  # top 4 layers
    #         param.requires_grad = True
    #     else:
    #         param.requires_grad = False
    # Optional - dont train audio encoder
    for param in model.audio_tower.parameters():
        param.requires_grad = False

    model = get_peft_model(model, lora_config)

    # Enable gradient checkpointing manually instead of via TrainingArguments.
    # When using LoRA/PEFT, enabling checkpointing through TrainerArgs alone causes
    # inputs to lose their requires_grad flag — leading to "tensor does not require grad" errors.
    # Calling these methods directly ensures input gradients are tracked correctly for LoRA layers.
    model.enable_input_require_grads()

    model.print_trainable_parameters()

    # Simple training arguments
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
        eval_on_start=config.trainer.eval_on_start,
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
        dataloader_num_workers=0,
        lr_scheduler_type="cosine",
        seed=3407,
    )

    # Setup trainer
    trainer = SafeTrainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=eval_dataset,
        data_collator=data_collator,
    )

    # Start training
    logger.info("Starting training...")
    if config.trainer.resume_from_checkpoint is not None:
        logger.info(
            "Resuming training from checkpoint: %s",
            config.trainer.resume_from_checkpoint,
        )
        trainer.train(resume_from_checkpoint=config.trainer.resume_from_checkpoint)
    else:
        trainer.train()

    # Save model and processor
    logger.info("Saving model to %s", output_dir)
    trainer.save_model()
    processor.save_pretrained(output_dir)

    # Final evaluation
    if eval_dataset:
        results = trainer.evaluate()
        logger.info("Final evaluation results: %s", results)

    logger.info("Training completed successfully!")


if __name__ == "__main__":
    main()
