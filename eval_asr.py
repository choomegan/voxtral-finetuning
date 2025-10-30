#!/usr/bin/env python3
import json
import os

import jiwer
import torch
from datasets import Audio, Dataset
from omegaconf import OmegaConf
from peft import PeftModel
from tqdm import tqdm
from transformers import VoxtralForConditionalGeneration, VoxtralProcessor


def load_manifest_dataset(manifest_path, sample_rate=16000):
    """
    Load dataset from a JSONL manifest file and automatically decode audio.
    """
    print(f"Loading dataset from: {manifest_path}")

    root_dir = os.path.dirname(os.path.abspath(manifest_path))

    # Read manifest lines and fix relative paths
    with open(manifest_path, "r", encoding="utf-8") as f:
        data = []
        for line in f:
            entry = json.loads(line.strip())
            if not os.path.isabs(entry["audio_filepath"]):
                entry["audio_filepath"] = os.path.join(
                    root_dir, entry["audio_filepath"]
                )
            data.append(entry)

    # Create Hugging Face dataset and cast audio column
    dataset = Dataset.from_list(data)
    dataset = dataset.add_column(
        "original_audio_filepath",
        [x["audio_filepath"].replace(root_dir, "") for x in data],
    )

    # Cast audio column and rename for processor compatibility
    dataset = dataset.cast_column("audio_filepath", Audio(sampling_rate=sample_rate))
    dataset = dataset.rename_column("audio_filepath", "audio")

    print(f"Loaded {len(dataset)} samples.")
    return dataset


def transcribe_batch(model, base_model_name, processor, audio_batch, device):
    """
    Run inference on a batch of audio clips.
    """
    with torch.no_grad():
        inputs = processor.apply_transcription_request(
            language="en",
            audio=audio_batch["array"],
            format=["WAV"],
            model_id=base_model_name,
        ).to(device)

        generated_tokens = model.generate(
            **inputs,
            max_new_tokens=256,
            do_sample=False,
            num_beams=1,
        )

        # slices off the input prompt tokens
        decoded_outputs = processor.batch_decode(
            generated_tokens[:, inputs.input_ids.shape[1] :], skip_special_tokens=True
        )
    return decoded_outputs


def main():

    config = OmegaConf.load("config/eval.yaml")

    # Load config
    manifest_path = config.manifest
    sample_rate = 16000

    # Device
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    # Load model & processor

    print(f"Loading base model: {config.model}")
    processor = VoxtralProcessor.from_pretrained(config.model)
    base_model = VoxtralForConditionalGeneration.from_pretrained(
        config.model,
        torch_dtype=torch.bfloat16 if torch.cuda.is_available() else torch.float32,
    ).to(device)

    # Check if checkpoint contains LoRA adapter
    adapter_config = os.path.join(config.checkpoint_path, "adapter_config.json")
    if os.path.exists(adapter_config):
        print(f"🔗 Detected LoRA adapter at {config.checkpoint_path} — loading it...")
        model = PeftModel.from_pretrained(base_model, config.checkpoint_path)
    else:
        print(f"⚠️ No adapter_config.json found — assuming full model checkpoint.")
        model = base_model

    model.eval()

    # Load dataset
    dataset = load_manifest_dataset(manifest_path, sample_rate=sample_rate)

    # Prepare output file
    os.makedirs(os.path.dirname(config.output_path), exist_ok=True)
    if os.path.exists(config.output_path):
        os.remove(config.output_path)

    all_predictions, all_references = [], []

    print("Running inference...")
    with open(config.output_path, "a", encoding="utf-8") as f_out:
        for i, sample in enumerate(tqdm(dataset)):
            reference = sample["text"].strip()
            audio = sample["audio"]

            # Run transcription
            prediction = transcribe_batch(
                model, config.model, processor, audio, device
            )[0].strip()

            # Compute WER and CER for this example
            sample_wer = jiwer.wer(reference, prediction)
            sample_cer = jiwer.cer(reference, prediction)

            # Collect for corpus metrics
            all_predictions.append(prediction)
            all_references.append(reference)

            # Merge metadata except heavy fields
            meta = {k: v for k, v in sample.items() if k != "audio"}

            # Write result line
            result = {
                "prediction": prediction,
                "wer": sample_wer,
                "cer": sample_cer,
                **meta,
            }
            json.dump(result, f_out, ensure_ascii=False)
            f_out.write("\n")

    # Compute corpus-level metrics
    corpus_wer = jiwer.wer(all_references, all_predictions)
    corpus_cer = jiwer.cer(all_references, all_predictions)

    print(f"\n✅ Corpus-level WER: {corpus_wer:.4f}")
    print(f"✅ Corpus-level CER: {corpus_cer:.4f}")

    # Save summary at the end
    summary_file = config.output_path + ".summary.json"
    with open(summary_file, "w", encoding="utf-8") as f_sum:
        json.dump(
            {"corpus_wer": corpus_wer, "corpus_cer": corpus_cer},
            f_sum,
            indent=2,
            ensure_ascii=False,
        )
    print(f"📁 Saved per-sample results to: {config.output_path}")
    print(f"📊 Saved summary to: {summary_file}")


if __name__ == "__main__":
    main()
