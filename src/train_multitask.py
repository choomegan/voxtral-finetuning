"""
Multitask finetuning for ASR and ST
"""

"""
Multi-task finetuning for ASR + Speech Translation
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
from utils.dataset_utils import load_multitask_manifest_dataset
from train_asr import VoxtralDataCollator
from train_st import VoxtralSTDataCollator
import random
from typing import List, Dict, Any


# --- Combine dataset and collator as above ---
class MultiTaskCollator:
    """
    Collator for multi-task training from a shared dataset.
    Each entry produces:
      - one ASR example (input: audio, label: source.text)
      - one ST example  (input: audio, label: target.text)
    """

    def __init__(self, asr_collator, st_collator):
        self.asr_collator = asr_collator
        self.st_collator = st_collator

    def __call__(self, features: List[Dict[str, Any]]) -> Dict[str, Any]:
        """
        Merges ASR and ST batches, handling variable lengths and the required
        multimodal input_features tensor.
        """
        # Create duplicated entries: one for ASR, one for ST
        asr_features, st_features = [], []

        for f in features:

            # --- ASR example ---
            asr_features.append(
                {
                    "audio": f["audio"],
                    "text": f["text"],  # transcription target
                }
            )

            # --- ST example ---
            st_features.append(
                {
                    "source.lang": f["source.lang"],
                    "source.audio_local_path": f.get("audio_local_path", None),
                    "target.text": f["target.text"],
                }
            )

        # Mix both tasks
        all_features = [{"task": "asr", "data": f} for f in asr_features] + [
            {"task": "st", "data": f} for f in st_features
        ]

        random.shuffle(all_features)
        # 1. Split features by task (assuming a 'task_name' key is present)
        asr_batch_data = [f for f in all_features if f["task"] == "asr"]
        st_batch_data = [f for f in all_features if f["task"] == "st"]

        # 2. Apply task-specific collators
        asr_out = (
            self.asr_collator([f["data"] for f in asr_batch_data])
            if asr_batch_data
            else {}
        )
        st_out = (
            self.st_collator([f["data"] for f in st_batch_data])
            if st_batch_data
            else {}
        )

        # Determine overall batch sizes
        B_asr = (
            asr_out["input_ids"].size(0) if asr_out.get("input_ids") is not None else 0
        )
        B_st = st_out["input_ids"].size(0) if st_out.get("input_ids") is not None else 0

        if B_asr + B_st == 0:
            return {}

        # 3. Determine the overall max sequence length for tokens (input_ids, labels)
        max_seq_len = 0
        if B_asr > 0:
            # Token sequence length for ASR is typically at asr_out["input_ids"].size(1)
            max_seq_len = max(max_seq_len, asr_out["input_ids"].size(1))
        if B_st > 0:
            max_seq_len = max(max_seq_len, st_out["input_ids"].size(1))

        # print(f"\n--- Batch Info ---")
        # print(
        #     f"ASR Batch Size: {B_asr}, ST Batch Size: {B_st}, Max Token Seq Len: {max_seq_len}"
        # )

        # 4. Helper for padding and merging token tensors
        def _pad_and_merge_token_tensor(key, fill_val):
            """
            Find the global max token length.
            Pad ASR tensor and ST tensors to this length and merges tensors together.
            """
            tensors = []

            # --- ASR Tensor Processing ---
            if B_asr > 0:
                t_asr = asr_out[key]
                # Calculate padding required
                pad_shape_asr = (B_asr, max_seq_len - t_asr.size(1))
                # Create padding tensor
                padding_asr = torch.full(
                    pad_shape_asr,
                    fill_val,
                    dtype=torch.long,
                    device=t_asr.device,
                )
                # Concatenate the token tensor with padding
                tensors.append(torch.cat([t_asr, padding_asr], dim=1))

            # --- ST Tensor Processing ---
            if B_st > 0:
                t_st = st_out[key]
                # Calculate padding required
                pad_shape_st = (B_st, max_seq_len - t_st.size(1))
                # Create padding tensor
                padding_st = torch.full(
                    pad_shape_st, fill_val, dtype=torch.long, device=t_st.device
                )
                # Concatenate the token tensor with padding
                tensors.append(torch.cat([t_st, padding_st], dim=1))

            # Concatenate all parts (ASR then ST)
            return torch.cat(tensors, dim=0)

        # 5. Merge core tensors
        final_batch = {}

        # input_ids: Tokens (use tokenizer.pad_token_id)
        final_batch["input_ids"] = _pad_and_merge_token_tensor(
            "input_ids", self.asr_collator.pad_id
        )

        # attention_mask: Mask (use 0 for padded tokens)
        final_batch["attention_mask"] = _pad_and_merge_token_tensor("attention_mask", 0)

        # labels: Target sequence (use -100 for ignored loss indices)
        final_batch["labels"] = _pad_and_merge_token_tensor("labels", -100)

        # 6. Handle Multimodal/Audio Features ('input_features') - THE CRITICAL FIX
        AUDIO_KEY = "input_features"
        asr_audio_features = asr_out.get(AUDIO_KEY)

        if asr_audio_features is not None:
            # The ASR batch is present and has created audio features

            if B_st > 0:
                # Need to create a zero-tensor placeholder for the ST (text-only) examples

                # The shape must match the ASR audio feature shape: (B_st, Feature_Dim_1, Feature_Dim_2, ...)
                # Example ASR shape: [1, 80, 3000]. We need [B_st, 80, 3000]
                audio_padding_shape = (B_st, *asr_audio_features.shape[1:])

                # Use the same device and dtype as the ASR features (typically float)
                audio_padding = torch.zeros(
                    audio_padding_shape,
                    dtype=asr_audio_features.dtype,
                    device=asr_audio_features.device,
                )

                # Concatenate ASR features and the zero-padded ST features
                final_batch[AUDIO_KEY] = torch.cat(
                    [asr_audio_features, audio_padding], dim=0
                )
            else:
                # Only ASR present, just use its features
                final_batch[AUDIO_KEY] = asr_audio_features
        # If only ST is present (B_asr == 0), we omit AUDIO_KEY, relying on the model
        # to accept text-only input when 'input_features' is missing.

        # 7. Include task names (useful for custom loss functions)
        final_batch["task"] = ["asr"] * B_asr + ["st"] * B_st

        # print(f"Final Batch Size: {B_asr + B_st}")
        # print(f"input_ids shape: {final_batch['input_ids'].shape}")
        # if "input_features" in final_batch:
        #     print(f"input_features shape: {final_batch['input_features'].shape}")

        return final_batch


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

    # --- Load datasets ---
    print("Loading datasets...")
    train = load_multitask_manifest_dataset(config.data.train_manifest)
    eval = load_multitask_manifest_dataset(config.data.eval_manifest)

    # --- Processor ---
    print("Loading processor...")
    processor = VoxtralProcessor.from_pretrained(config.model)

    # --- Collators ---
    asr_collator = VoxtralDataCollator(processor, config.model)
    st_collator = VoxtralSTDataCollator(processor, config.model)
    multi_collator = MultiTaskCollator(asr_collator, st_collator)

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
    model.gradient_checkpointing_enable()
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
        dataloader_num_workers=1,
        lr_scheduler_type="cosine",
        seed=3407,
    )

    trainer = Trainer(
        model=model,
        args=args,
        train_dataset=train,
        eval_dataset=eval,
        data_collator=multi_collator,
    )

    print("Starting multi-task training...")
    trainer.train(resume_from_checkpoint=config.trainer.resume_from_checkpoint)
    print("Training complete!")

    trainer.save_model()
    processor.save_pretrained(output_dir)


if __name__ == "__main__":
    main()
