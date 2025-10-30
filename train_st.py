import json
import os
from datetime import datetime
from typing import List, Dict, Any
import torch
from datasets import Audio, Dataset
from omegaconf import OmegaConf
from peft import LoraConfig, get_peft_model
from transformers import (
    BitsAndBytesConfig,
    Trainer,
    TrainingArguments,
    VoxtralForConditionalGeneration,
    VoxtralProcessor,
)
from torch.nn.utils.rnn import pad_sequence
import wandb
from utils.st_helper import load_st_manifest_dataset, build_convos

# ===============================
#  Voxtral Data Collator (Speech Translation)
# ===============================


class VoxtralSTDataCollator:
    """Data collator for Voxtral speech translation training."""

    def __init__(self, processor, model_id):
        self.processor = processor
        self.model_id = model_id
        self.tokenizer = processor.tokenizer
        self.pad_id = self.tokenizer.pad_token_id or self.tokenizer.eos_token_id

    def __call__(self, features: List[Dict[str, Any]]):
        # 1. Prepare two versions of conversations: Prompt Only and Full Conversation (Prompt + Target)
        prompts = []
        full_conversations = []

        for f in features:
            src_lang = f["source.lang"]
            tgt_lang = f["target.lang"]
            audio_path = f["source.audio_local_path"]

            # A. Prompt (User Message Only) - Used to calculate prompt length for masking
            prompt_messages = build_convos(src_lang, tgt_lang, audio_path)
            prompts.append(prompt_messages)

            # B. Full Conversation (User Message + Target/Assistant Response) - Used for final input_ids and labels
            full_messages = prompt_messages + [
                {
                    "role": "assistant",
                    "content": [{"type": "text", "text": f["target.text"]}],
                }
            ]
            full_conversations.append(full_messages)

        # 2. Tokenize the Prompt Only to determine the masking length
        # This ends with a USER message, which is usually fine without special flags.
        prompt_inputs = self.processor.apply_chat_template(
            prompts,
            return_tensors="pt",
            tokenize=True,
            padding="longest",
        )

        # Calculate the length of the prompt for each example in the batch
        prompt_lengths = torch.sum(
            prompt_inputs["input_ids"] != self.pad_id, dim=1
        ).tolist()

        # 3. Tokenize the Full Conversation to get the final input_ids and labels
        # We must add continue_final_message=True here because the conversation
        # ends with an 'assistant' role, which is typically not allowed for serving
        model_inputs = self.processor.apply_chat_template(
            full_conversations,
            return_tensors="pt",
            tokenize=True,
            padding="longest",
            continue_final_message=True,
        )

        input_ids = model_inputs["input_ids"]
        labels = input_ids.clone()  # Labels must have the exact same shape as input_ids

        # 4. Mask the Prompt tokens in the Labels tensor
        # Iterate over the batch and mask the prompt tokens with -100
        for i, prompt_len in enumerate(prompt_lengths):
            # Mask all tokens up to the calculated prompt length
            labels[i, :prompt_len] = -100

            # Mask padding tokens if they exist (though padding="longest" should handle this)
            padding_mask = input_ids[i] == self.pad_id
            labels[i][padding_mask] = -100

        batch = {
            **model_inputs,
            "labels": labels,
        }
        return batch


# ===============================
#  Main Training
# ===============================


def main():
    # Load config
    config = OmegaConf.load("config/train_ST.yaml")
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

    print(f"Using device: {device}")

    # --- Load datasets ---
    print("Loading datasets...")
    train_dataset = load_st_manifest_dataset(config.data.train_manifest)
    eval_dataset = load_st_manifest_dataset(config.data.eval_manifest)

    # --- Load processor ---
    print("Loading processor...")
    processor = VoxtralProcessor.from_pretrained(config.model)

    # --- Data collator ---
    data_collator = VoxtralSTDataCollator(processor, config.model)

    # Take a few samples from train_dataset to simulate a DataLoader batch
    sample_features = [train_dataset[i] for i in range(2)]  # 2 samples for testing

    batch = data_collator(sample_features)

    # --- Model & LoRA setup ---
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
    steps_per_epoch = (
        len(train_dataset)
        // config.trainer.train_batch_size
        // config.trainer.grad_accum
    )
    total_steps = steps_per_epoch * config.trainer.epochs
    warmup_steps = config.trainer.warmup_steps

    training_args = TrainingArguments(
        output_dir=output_dir,
        per_device_train_batch_size=config.trainer.train_batch_size,
        per_device_eval_batch_size=config.trainer.eval_batch_size,
        gradient_accumulation_steps=config.trainer.grad_accum,
        learning_rate=config.trainer.lr,
        num_train_epochs=config.trainer.epochs,
        warmup_steps=warmup_steps,
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
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=eval_dataset,
        data_collator=data_collator,
    )

    print("Starting training...")
    trainer.train(resume_from_checkpoint=config.trainer.resume_from_checkpoint)
    print("Training complete!")


if __name__ == "__main__":
    main()
