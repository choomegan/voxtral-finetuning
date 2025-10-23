from datetime import datetime
import os
import json
import torch
from datasets import Audio, Dataset
from transformers import (
    VoxtralForConditionalGeneration,
    VoxtralProcessor,
    Trainer,
    TrainingArguments,
)
from peft import LoraConfig, get_peft_model
from omegaconf import OmegaConf


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


def load_manifest_dataset(manifest_path, sample_rate=16000):
    """
    Load dataset from a JSONL manifest file and make audio filepaths absolute.
    Each line should have:
    {
      "audio_filepath": "audio/audio_1.wav",
      "duration": 5.038,
      "start": 1166.599,
      "end": 1171.637,
      "text": "this is a transcript"
    }
    """
    print(f"Loading dataset from: {manifest_path}")
    root_dir = os.path.dirname(os.path.abspath(manifest_path))

    data = []
    with open(manifest_path, "r", encoding="utf-8") as f:
        for line in f:
            entry = json.loads(line.strip())
            # Prepend the manifest's directory if path is relative
            audio_path = entry["audio_filepath"]
            if not os.path.isabs(audio_path):
                audio_path = os.path.join(root_dir, audio_path)
            entry["audio_filepath"] = os.path.normpath(audio_path)
            data.append(entry)

    dataset = Dataset.from_list(data)

    # Decode audio on the fly
    dataset = dataset.cast_column("audio_filepath", Audio(sampling_rate=sample_rate))

    # Rename to match collator expectations
    dataset = dataset.rename_column("audio_filepath", "audio")

    print(f"Loaded {len(dataset)} samples from {manifest_path}")
    return dataset


def main():
    # Load training config
    config = OmegaConf.load("config/train.yaml")

    # Generate timestamped experiment name
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    exp_name = f"{config.exp_manager.name}_{timestamp}"

    # Configuration
    model_checkpoint = config.model
    output_dir = os.path.join(config.exp_manager.exp_dir, exp_name)

    # Set device
    torch_device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {torch_device}")

    # Load processor and model
    print("Loading processor and model...")
    processor = VoxtralProcessor.from_pretrained(model_checkpoint)
    # Load model with LoRA configuration
    lora_config = LoraConfig(
        r=config.lora.r,  # Rank of LoRA
        lora_alpha=config.lora.alpha,
        lora_dropout=config.lora.dropout,
        bias="none",
        target_modules=list(config.lora.target_modules),
        task_type="SEQ_2_SEQ_LM",
    )
    # print number of parameters in model
    model = VoxtralForConditionalGeneration.from_pretrained(
        model_checkpoint, torch_dtype=torch.bfloat16, device_map="auto"
    )
    # Freeze the audio encoder model.audio_tower
    for param in model.audio_tower.parameters():
        param.requires_grad = False

    model = get_peft_model(model, lora_config)
    model.print_trainable_parameters()

    # Load datasets from manifest files
    train_dataset = load_manifest_dataset(config.data.train_manifest)
    eval_dataset = load_manifest_dataset(config.data.eval_manifest)

    # Setup data collator
    data_collator = VoxtralDataCollator(processor, model_checkpoint)

    # Simple training arguments
    training_args = TrainingArguments(
        output_dir=output_dir,
        per_device_train_batch_size=config.trainer.train_batch_size,
        per_device_eval_batch_size=config.trainer.eval_batch_size,
        gradient_accumulation_steps=config.trainer.grad_accum,
        learning_rate=config.trainer.lr,
        num_train_epochs=config.trainer.epochs,
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
