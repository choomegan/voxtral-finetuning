import json
import os
from datetime import datetime

import torch
from datasets import Audio, Dataset
from omegaconf import OmegaConf
from peft import LoraConfig, get_peft_model
from transformers import (BitsAndBytesConfig, Trainer, TrainingArguments,
                          VoxtralForConditionalGeneration, VoxtralProcessor)
from torch.nn.utils.rnn import pad_sequence
import wandb

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

    def __call__(self, features):
        """
        Each feature is expected to come from a JSONL manifest line:
        {
          "source": {"audio_local_path": str, "lang": "zsm", "sampling_rate": 16000},
          "target": {"text": str, "lang": "eng"}
        }
        """
        # print(features[0])
        # Load inputs
        audios = [f["source.audio"]["array"] for f in features]
        src_langs = [f["source.lang"] for f in features]
        tgt_langs = [f["target.lang"] for f in features]
        tgt_texts = [f["target.text"] for f in features]

        # --- Build chat prompt for each sample ---
        prompts = []
        for src, tgt in zip(src_langs, tgt_langs):
            messages = [
                {
                    "role": "system",
                    "content": "You are a multilingual speech translator.",
                },
                {
                    "role": "user",
                    "content": f"Translate this {src} audio into {tgt} text.",
                },
            ]
            prompts.append(self.processor.apply_chat_template(messages, tokenize=True))

        # --- Prepare multimodal input (prompt + audio) ---
        # Extract audio features with WhisperFeatureExtractor
        audio_features = self.processor.feature_extractor(
            raw_speech=audios,
            return_tensors="pt",
            sampling_rate=features[0]["source.sampling_rate"],
        )

        # Tokenize prompt with MistralCommonTokenizer
        # print("prompt: ", prompts[0])
        # Stack tokenized prompt tensors
        # Pad all input_ids and attention_masks to same length
        prompt_ids = pad_sequence(
            [p["input_ids"].squeeze(0) for p in prompts],  # ensure 1D tensors
            batch_first=True,
            padding_value=self.pad_id,  # use tokenizer’s pad token id
        )

        prompt_attn = pad_sequence(
            [p["attention_mask"].squeeze(0) for p in prompts],
            batch_first=True,
            padding_value=0,  # attention mask padding
        )

        B = prompt_ids.size(0)

        # keep any extra fields (e.g., audio features) to pass through to the model
        passthrough = {
            "input_features": audio_features[
                "input_features"
            ],  # main Whisper feature tensor
            "sampling_rate": features[0]["source.sampling_rate"],
        }

        # --- Tokenize target texts ---
        text_tokens = self.tokenizer(
            tgt_texts,
            add_special_tokens=False,
            padding=False,
            truncation=True,
            max_length=256,
            return_tensors=None,
        )

        # --- Concatenate prompt + target ---
        input_ids, attention_mask, labels = [], [], []
        for i in range(B):
            p_ids = prompt_ids[i].tolist()
            p_att = prompt_attn[i].tolist()
            t_ids = text_tokens["input_ids"][i]

            ids = p_ids + t_ids + [self.tokenizer.eos_token_id]
            attn = p_att + [1] * (len(t_ids) + 1)
            lab = [-100] * len(p_ids) + t_ids + [self.tokenizer.eos_token_id]

            input_ids.append(ids)
            attention_mask.append(attn)
            labels.append(lab)

        # --- Pad batch ---
        max_len = max(len(x) for x in input_ids)

        def pad_to(seq, fill, L):
            return seq + [fill] * (L - len(seq))

        input_ids = [pad_to(x, self.pad_id, max_len) for x in input_ids]
        attention_mask = [pad_to(x, 0, max_len) for x in attention_mask]
        labels = [pad_to(x, -100, max_len) for x in labels]

        batch = {
            "input_ids": torch.tensor(input_ids, dtype=torch.long),
            "attention_mask": torch.tensor(attention_mask, dtype=torch.long),
            "labels": torch.tensor(labels, dtype=torch.long),
        }
        batch.update(passthrough)
        return batch


# ===============================
#  Dataset Loader (Manifest)
# ===============================


def load_manifest_dataset(manifest_path, sample_rate=16000):
    print(f"Loading dataset from: {manifest_path}")
    root_dir = os.path.dirname(os.path.abspath(manifest_path))
    data = []

    with open(manifest_path, "r", encoding="utf-8") as f:
        for line in f:
            entry = json.loads(line.strip())

            # Normalize
            entry["target"]["text"] = entry["target"]["text"].strip()

            # Fix relative audio paths
            audio_path = entry["source"]["audio_local_path"]
            if not os.path.isabs(audio_path):
                audio_path = os.path.join(root_dir, audio_path)
            entry["source"]["audio_local_path"] = os.path.normpath(audio_path)
            data.append(entry)

    dataset = Dataset.from_list(data)

    # remove nested structure due to format of manifest file
    dataset = dataset.flatten()

    # Decode audio automatically
    dataset = dataset.cast_column(
        "source.audio_local_path", Audio(sampling_rate=sample_rate)
    )

    # Rename for collator compatibility
    dataset = dataset.rename_column("source.audio_local_path", "source.audio")
    print("Columns:", dataset.column_names)
    print(f"Loaded {len(dataset)} samples from {manifest_path}")
    return dataset


# ===============================
#  Main Training
# ===============================


def main():
    # Load config
    config = OmegaConf.load("config/train_ST.yaml")

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
    train_dataset = load_manifest_dataset(config.data.train_manifest)
    eval_dataset = load_manifest_dataset(config.data.eval_manifest)

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
            device_map="auto",
        )
    else:
        model = VoxtralForConditionalGeneration.from_pretrained(
            config.model,
            torch_dtype=torch.bfloat16 if config.trainer.bf16 else torch.float16,
            device_map="auto",
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
