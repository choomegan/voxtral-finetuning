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
import traceback
from torch.utils.data import DataLoader
from utils.dataset_utils import load_preprocessed_multitask_dataset
from utils.collators import (
    StreamingASRCollator,
    StreamingMultiTaskCollator,
    StreamingSTCollator,
)


class SafeTrainer(Trainer):
    """
    Custom trainer that:
    1. Handles multiple eval datasets
    2. Properly handles None returns from collators (skipped batches)
    """

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.skipped_batches_train = 0
        self.skipped_batches_eval = 0
        self.total_train_batches = 0
        self.total_eval_batches = 0

    def evaluate(self, eval_dataset=None, **kwargs):
        """Handle dict of eval datasets"""
        if isinstance(self.eval_dataset, dict):
            results = {}
            for name, dataset in self.eval_dataset.items():
                print(f"\n🔍 Evaluating {name} dataset ({len(dataset)} samples)")
                self.skipped_batches_eval = 0  # Reset counter for this eval set
                self.total_eval_batches = 0

                try:
                    res = super().evaluate(
                        eval_dataset=dataset,
                        metric_key_prefix=f"eval_{name}",
                        **kwargs,
                    )
                    results[name] = res

                    if self.skipped_batches_eval > 0:
                        skip_pct = (
                            100
                            * self.skipped_batches_eval
                            / max(self.total_eval_batches, 1)
                        )
                        print(
                            f"ℹ️  Skipped {self.skipped_batches_eval}/{self.total_eval_batches} batches ({skip_pct:.1f}%) during {name} evaluation"
                        )

                except Exception as e:
                    print(f"❌ Error evaluating {name}: {e}")
                    traceback.print_exc()
                    results[name] = {"error": str(e)}

            return results

        # Default single dataset behavior
        return super().evaluate(eval_dataset=eval_dataset, **kwargs)

    def get_train_dataloader(self) -> DataLoader:
        """
        Override to add custom collate function that handles None returns
        """
        dataloader = super().get_train_dataloader()

        # Wrap the original collate function
        original_collate = dataloader.collate_fn

        def safe_collate_wrapper(batch):
            """Wrapper that filters out None returns and retries if needed"""
            result = original_collate(batch)

            # If collator returned None, skip this batch
            if result is None:
                self.skipped_batches_train += 1
                # Return an empty dict - trainer will skip it
                return None

            return result

        # Replace collate function
        dataloader.collate_fn = safe_collate_wrapper

        return dataloader

    def get_eval_dataloader(self, eval_dataset=None) -> DataLoader:
        """
        Override to add custom collate function that handles None returns
        """
        dataloader = super().get_eval_dataloader(eval_dataset)

        # Wrap the original collate function
        original_collate = dataloader.collate_fn

        def safe_collate_wrapper(batch):
            """Wrapper that filters out None returns"""
            result = original_collate(batch)

            # If collator returned None, skip this batch
            if result is None:
                self.skipped_batches_eval += 1
                return None

            return result

        # Replace collate function
        dataloader.collate_fn = safe_collate_wrapper

        return dataloader

    def training_step(self, model, inputs, num_items_in_batch=None):
        """
        Override training step to handle None inputs (skipped batches)
        """
        self.total_train_batches += 1

        # If inputs is None, skip this batch
        if inputs is None:
            self.skipped_batches_train += 1
            return torch.tensor(0.0, device=self.args.device, requires_grad=True)

        # Call parent with correct signature
        if num_items_in_batch is not None:
            return super().training_step(model, inputs, num_items_in_batch)
        else:
            return super().training_step(model, inputs)

    def prediction_step(self, model, inputs, prediction_loss_only, ignore_keys=None):
        """
        Override prediction step to handle None inputs (skipped batches)
        """
        self.total_eval_batches += 1

        # If inputs is None, skip this batch
        if inputs is None:
            self.skipped_batches_eval += 1
            # Return dummy values that will be filtered out later
            device = next(model.parameters()).device
            dummy_loss = torch.tensor(0.0, device=device)
            return (dummy_loss, None, None)

        return super().prediction_step(
            model, inputs, prediction_loss_only, ignore_keys=ignore_keys
        )


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
    train_dataset, eval_dataset_asr, eval_dataset_st = (
        load_preprocessed_multitask_dataset(
            train_manifest=config.data.train_manifest,
            eval_manifest=config.data.eval_manifest,
        )
    )

    print(
        f"Eval dataset sizes — ASR: {len(eval_dataset_asr)}, ST: {len(eval_dataset_st)}"
    )

    # --- Collators ---
    asr_collator = StreamingASRCollator(processor, model_id=config.model)
    st_collator = StreamingSTCollator(processor, model_id=config.model)
    multi_collator = StreamingMultiTaskCollator(asr_collator, st_collator)

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
        # dataloader_num_workers=8,
        # dataloader_prefetch_factor=2,  # Prefetch 2 batches per worker
        dataloader_pin_memory=True,
        gradient_checkpointing=False,
        lr_scheduler_type="cosine",
        seed=3407,
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
    )

    print("Starting multi-task training...")
    trainer.train(resume_from_checkpoint=config.trainer.resume_from_checkpoint)
    print("Training complete!")

    trainer.save_model()
    processor.save_pretrained(output_dir)


if __name__ == "__main__":
    main()
