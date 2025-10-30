#!/usr/bin/env python3
import json
import os
import sys

import torch

# datasets is imported in st_helper, no need to import Audio here
from omegaconf import OmegaConf
from peft import PeftModel
from tqdm import tqdm
from transformers import VoxtralForConditionalGeneration, VoxtralProcessor
import evaluate

# Assuming st_helper.py is available in a utils/ directory
# If using a mock environment, ensure this path is correct or copy the functions directly
from utils.st_helper import load_st_manifest_dataset, build_convos


def translate_sample(model, processor, audio_path, src_lang, tgt_lang, device):
    """
    Run speech translation inference on one audio sample using the path-based
    multimodal chat template. The processor handles feature extraction internally.

    Args:
        model: The Voxtral model (potentially wrapped with PeftModel).
        processor: The VoxtralProcessor.
        audio_path (str): Path to the audio file.
        src_lang (str): Source language code.
        tgt_lang (str): Target language code.
        device (torch.device): CUDA or CPU device.

    Returns:
        str: The predicted translated text.
    """
    with torch.no_grad():
        # 1. Build the full multimodal prompt structure, including the audio file path
        # This now uses the audio_path argument per your request.
        messages = build_convos(src_lang, tgt_lang, audio_path)

        # 2. Tokenize and process the full multimodal input
        # The processor automatically loads the audio from the 'audio_path'
        # specified in the messages and extracts features.
        model_inputs = processor.apply_chat_template(
            messages,
            return_tensors="pt",
            tokenize=True,
            padding="longest",
        )

        # Move inputs to device
        input_ids = model_inputs["input_ids"].to(device)
        attention_mask = model_inputs["attention_mask"].to(device)
        input_features = model_inputs["input_features"].to(device)

        # NOTE: model_inputs contains input_ids, attention_mask, and input_features.
        generate_inputs = {
            "input_ids": input_ids,
            "attention_mask": attention_mask,
            "input_features": input_features,
        }

        # 3. Generate the translation tokens
        generated_tokens = model.generate(
            **generate_inputs,
            max_new_tokens=256,
            do_sample=False,
            num_beams=1,
        )

        # 4. Decode the results
        # CRITICAL STEP: Slice off the prompt tokens (input_ids.shape[1])
        # from the generated sequence before decoding.
        decoded = processor.batch_decode(
            generated_tokens[:, input_ids.shape[1] :], skip_special_tokens=True
        )

    return decoded[0].strip()


def main():
    """
    Main function to load the model, run inference over a dataset,
    and calculate corpus-level evaluation metrics (BLEU).
    """
    # Load configuration from YAML file
    config = OmegaConf.load("config/eval_ST.yaml")

    # Device Setup
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    # Load Model & Processor
    print(f"Loading processor and base model: {config.model}")
    processor = VoxtralProcessor.from_pretrained(config.model)

    # Use bfloat16 precision if CUDA is available for memory efficiency
    base_model = VoxtralForConditionalGeneration.from_pretrained(
        config.model,
        torch_dtype=torch.bfloat16 if torch.cuda.is_available() else torch.float32,
    ).to(device)

    # Load LoRA adapter if available (common for finetuning)
    adapter_config = os.path.join(config.checkpoint_path, "adapter_config.json")
    if os.path.exists(adapter_config):
        print(f"Detected LoRA adapter at {config.checkpoint_path} — loading it...")
        # Use PeftModel to wrap the base model with the adapter weights
        model = PeftModel.from_pretrained(base_model, config.checkpoint_path)
    else:
        print(f"No adapter_config.json found — assuming full model checkpoint.")
        model = base_model

    # Set model to evaluation mode (disables dropout, etc.)
    model.eval()

    # Load dataset. Note: The dataset now contains the 'audio_path' column.
    dataset = load_st_manifest_dataset(config.manifest, sample_rate=16000)

    # Prepare output files
    os.makedirs(os.path.dirname(config.output_path), exist_ok=True)
    if os.path.exists(config.output_path):
        print(f"Overwriting existing results file: {config.output_path}")
        os.remove(config.output_path)

    # Initialize metrics
    bleu = evaluate.load("bleu")
    all_predictions, all_references = [], []

    print(f"Running speech translation inference on {len(dataset)} samples...")
    with open(config.output_path, "a", encoding="utf-8") as f_out:
        for i, sample in enumerate(tqdm(dataset)):
            src_lang = sample["source.lang"]
            tgt_lang = sample["target.lang"]
            reference = sample["target.text"].strip()
            # New: Get the audio path instead of the loaded audio array
            audio_path = sample["source.audio_local_path"]

            # Run translation
            prediction = translate_sample(
                model, processor, audio_path, src_lang, tgt_lang, device  # Pass path
            )

            # Collect results for corpus-level metrics
            all_predictions.append(prediction)
            all_references.append([reference])

            # Write per-sample result
            meta = {k: v for k, v in sample.items()}
            result = {
                "prediction": prediction,
                "reference": reference,
                **meta,
            }
            json.dump(result, f_out, ensure_ascii=False)
            f_out.write("\n")

    # --- Corpus-level metrics ---
    corpus_bleu = bleu.compute(predictions=all_predictions, references=all_references)[
        "bleu"
    ]

    print(f"\n✅ Corpus-level BLEU: {corpus_bleu:.4f}")

    # Save summary
    summary_file = config.output_path + ".summary.json"
    with open(summary_file, "w", encoding="utf-8") as f_sum:
        json.dump(
            {
                "corpus_bleu": corpus_bleu,
                "model": config.model,
                "checkpoint": config.checkpoint_path,
                "manifest": config.manifest,
            },
            f_sum,
            indent=2,
            ensure_ascii=False,
        )

    print(f"📁 Saved per-sample results to: {config.output_path}")
    print(f"📊 Saved summary to: {summary_file}")


if __name__ == "__main__":
    main()
