import os
from datetime import datetime

import torch
from omegaconf import OmegaConf
from peft import LoraConfig, get_peft_model
from transformers import (
    BitsAndBytesConfig,
    Trainer,
    TrainingArguments,
    VoxtralForConditionalGeneration,
    VoxtralProcessor,
)

import wandb

from utils.asr_helper import load_asr_manifest_dataset


class VoxtralDataCollator:
    """Data collator for Voxtral STT training - processes audio and text."""

    def __init__(self, processor, model_id):
        self.processor = processor
        self.model_id = model_id
        self.pad_id = processor.tokenizer.pad_token_id

    def __call__(self, features):
        """
        Each feature should have:
          - "audio": raw audio (whatever your processor expects)
          - "text":  transcription string
        """
        texts = [f["text"] for f in features]
        audios = [f["audio"]["array"] for f in features]

        # 1) Build the PROMPT part: [AUDIO]…[AUDIO] <transcribe>
        prompt = self.processor.apply_transcription_request(  # (same method you used)
            language="en",
            model_id=self.model_id if hasattr(self, "model_id") else None,
            audio=audios,
            format=["WAV"] * len(audios),
            return_tensors="pt",
        )
        # prompt["input_ids"]: shape [B, L_prompt]
        # keep any extra fields (e.g., audio features) to pass through to the model
        passthrough = {
            k: v for k, v in prompt.items() if k not in ("input_ids", "attention_mask")
        }

        prompt_ids = prompt["input_ids"]  # [B, Lp]
        prompt_attn = prompt["attention_mask"]  # [B, Lp]
        B = prompt_ids.size(0)

        tok = self.processor.tokenizer
        # 2) Tokenize transcriptions WITHOUT padding; we'll pad after concatenation
        text_tok = tok(
            texts,
            add_special_tokens=False,
            padding=False,
            truncation=True,
            max_length=256,
            return_tensors=None,
        )
        text_ids_list = text_tok["input_ids"]

        # 3) Concatenate: input_ids = [PROMPT] + [TEXT]
        input_ids, attention_mask, labels = [], [], []
        for i in range(B):
            p_ids = prompt_ids[i].tolist()
            p_att = prompt_attn[i].tolist()
            t_ids = text_ids_list[i]

            ids = p_ids + t_ids + [tok.eos_token_id]
            attn = p_att + [1] * (len(t_ids) + 1)
            # labels: mask prompt tokens, learn only on text tokens
            lab = [-100] * len(p_ids) + t_ids + [tok.eos_token_id]

            input_ids.append(ids)
            attention_mask.append(attn)
            labels.append(lab)

        # 4) Pad to max length in batch
        pad_id = tok.pad_token_id if tok.pad_token_id is not None else tok.eos_token_id
        max_len = max(len(x) for x in input_ids)

        def pad_to(seq, fill, L):
            return seq + [fill] * (L - len(seq))

        input_ids = [pad_to(x, pad_id, max_len) for x in input_ids]
        attention_mask = [pad_to(x, 0, max_len) for x in attention_mask]
        labels = [pad_to(x, -100, max_len) for x in labels]

        batch = {
            "input_ids": torch.tensor(input_ids, dtype=torch.long),
            "attention_mask": torch.tensor(attention_mask, dtype=torch.long),
            "labels": torch.tensor(labels, dtype=torch.long),
        }
        # 5) Include processor outputs needed by the model (e.g., audio features)
        for k, v in passthrough.items():
            batch[k] = v

        return batch


def main():
    # Load training config
    config = OmegaConf.load("config/train.yaml")

    if config.exp_manager.logger == "wandb":
        if config.trainer.resume_from_checkpoint and config.exp_manager.wandb.run_id:
            # resume existing run
            wandb.init(
                project=config.exp_manager.wandb.project,
                id=config.exp_manager.wandb.run_id,
                resume="must",
            )
        else:
            # start a new run
            wandb.init(
                project=config.exp_manager.wandb.project, name=config.exp_manager.name
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
    print(f"Using device: {torch_device}")

    print("Loading datasets...")
    #################### Load datasets from manifest files #############################
    train_dataset = load_asr_manifest_dataset(config.data.train_manifest)
    eval_dataset = load_asr_manifest_dataset(config.data.eval_manifest)

    print("Loading processor...")
    processor = VoxtralProcessor.from_pretrained(model_checkpoint)

    # Setup data collator
    data_collator = VoxtralDataCollator(processor, model_checkpoint)

    ########################### Load processor and model ###############################
    print("Loading model...")

    bnb_config = BitsAndBytesConfig(
        load_in_4bit=True,  # Quantize model weights to 4-bit
        bnb_4bit_use_double_quant=True,  # Secondary quantization for smaller memory footprint
        bnb_4bit_quant_type="nf4",  # NormalFloat4 – best balance of quality & compression
        bnb_4bit_compute_dtype=torch.float16,  # A6000 supports bf16 natively
    )

    # print number of parameters in model
    model = VoxtralForConditionalGeneration.from_pretrained(
        model_checkpoint,
        torch_dtype=torch.float16,
        quantization_config=bnb_config,
        device_map="auto",
    )

    print(type(model.audio_tower))
    print("audio encoder layers:", len(model.audio_tower.layers))

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
    model.gradient_checkpointing_enable()

    model.print_trainable_parameters()

    steps_per_epoch = (
        len(train_dataset)
        // config.trainer.train_batch_size
        // config.trainer.grad_accum
    )
    total_training_steps = steps_per_epoch * config.trainer.epochs

    # Compute warmup steps from ratio
    # warmup_ratio = config.trainer.warmup_ratio
    warmup_steps = (
        config.trainer.warmup_steps
    )  # int(total_training_steps * warmup_ratio)

    # Simple training arguments
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

    # Setup trainer
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=eval_dataset,
        data_collator=data_collator,
    )

    # Start training
    print("Starting training...")
    if config.trainer.resume_from_checkpoint is not None:
        print(
            f"Resuming training from checkpoint: {config.trainer.resume_from_checkpoint}"
        )
        trainer.train(resume_from_checkpoint=config.trainer.resume_from_checkpoint)
    else:
        trainer.train()

    # Save model and processor
    print(f"Saving model to {output_dir}")
    trainer.save_model()
    processor.save_pretrained(output_dir)

    # Final evaluation
    if eval_dataset:
        results = trainer.evaluate()
        print(f"Final evaluation results: {results}")

    print("Training completed successfully!")


if __name__ == "__main__":
    main()
