"""
Multitask finetuning for ASR and ST
"""

import logging
import os
from collections import Counter
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
from utils.collators import (
    StreamingASRCollator,
    StreamingMultiTaskCollator,
    StreamingSTCollator,
)
from utils.dataset_utils import load_preprocessed_multitask_dataset
from utils.train_utils import SafeTrainer

logging.basicConfig(
    level=logging.INFO,  # or DEBUG for more verbose logs
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)

logger = logging.getLogger(__name__)  # module-level logger


def main():
    accelerator = Accelerator()
    config = OmegaConf.load("config/train_multitask.yaml")

    # --- Setup WandB ---
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
    # --- Experiment name ---
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    exp_name = f"{config.exp_manager.name}_{timestamp}"
    output_dir = os.path.join(config.exp_manager.exp_dir, exp_name)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    logger.info("Using device: %s", device)

    # --- Processor ---
    logger.info("Loading processor...")
    processor = VoxtralProcessor.from_pretrained(config.model)

    # Load preprocessed datasets (this caches to disk)
    logger.info("Loading or creating preprocessed datasets...")
    train_dataset, eval_dataset_asr, eval_dataset_st = (
        load_preprocessed_multitask_dataset(
            train_manifest=config.data.train_manifest,
            eval_manifest=config.data.eval_manifest,
        )
    )

    logger.info(
        "Eval dataset sizes — ASR: %s, ST: %s",
        len(eval_dataset_asr),
        len(eval_dataset_st),
    )

    # --- Compute language weights ---
    if config.trainer.lang_class_weighting:
        logger.info(
            "---------------- Adding language weighting to loss function -------------"
        )
        records = train_dataset.to_list()
        lang_counts = Counter([r["source_lang"] for r in records])
        logger.info("Language counts: %s", lang_counts)

        total = sum(lang_counts.values())
        lang_freq = {lang: count / total for lang, count in lang_counts.items()}

        # Inverse frequency
        lang_weights = {lang: 1.0 / freq for lang, freq in lang_freq.items()}

        # Optional: normalize weights to sum to 1
        norm_factor = sum(lang_weights.values())
        lang_weights = {lang: w / norm_factor for lang, w in lang_weights.items()}
        logger.info("Normalized language weights: %s", lang_weights)

    # --- Collators ---
    asr_collator = StreamingASRCollator(processor, model_id=config.model)
    st_collator = StreamingSTCollator(
        processor, model_id=config.model, incl_src_lang=config.tasks.s2tt.incl_src_lang
    )
    multi_collator = StreamingMultiTaskCollator(asr_collator, st_collator)

    # --- Model ---
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
        eval_on_start=config.trainer.eval_on_start,
        eval_steps=config.trainer.eval_steps,
        save_steps=config.trainer.save_steps,
        save_strategy="steps",
        eval_strategy="steps",
        save_total_limit=config.trainer.save_total_limit,
        load_best_model_at_end=True,
        metric_for_best_model="eval_combined_loss",
        greater_is_better=False,
        report_to=config.exp_manager.logger,
        remove_unused_columns=False,
        dataloader_num_workers=2,
        # dataloader_prefetch_factor=2,  # Prefetch 2 batches per worker
        dataloader_pin_memory=True,
        gradient_checkpointing=False,
        lr_scheduler_type="cosine",
        seed=3407,
        ddp_find_unused_parameters=False,
    )

    trainer = SafeTrainer(
        model=model,
        args=args,
        train_dataset=train_dataset,
        eval_dataset={
            "asr": eval_dataset_asr,
            "st": eval_dataset_st,
        },
        data_collator=multi_collator,
        lang_weight_map=lang_weights if config.trainer.lang_class_weighting else None,
        use_uncertainty=config.trainer.task_uncertainty_weighting,
        use_pcgrad=config.trainer.use_pcgrad,
        pcgrad_reduction=config.trainer.pcgrad_reduction,
    )

    logger.info("Starting multi-task training...")
    trainer.train(resume_from_checkpoint=config.trainer.resume_from_checkpoint)
    logger.info("Training complete!")

    trainer.save_model()
    processor.save_pretrained(output_dir)


if __name__ == "__main__":
    main()
