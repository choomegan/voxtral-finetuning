"""
Evaluation script for ASR task
"""

import json
import logging
import os

import jiwer
import torch
from omegaconf import OmegaConf
from tqdm import tqdm
from transformers import VoxtralProcessor

from utils.constants import LANGCODE_MAP
from utils.dataset_utils import load_eval_asr_manifest_dataset
from utils.eval_helper import load_model_for_evaluation

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)

logger = logging.getLogger(__name__)


def transcribe_sample(
    model,
    base_model_name: str,
    processor: VoxtralProcessor,
    audio_batch,
    lang: str,
    device: torch.device,
    is_task_routing: bool = False,
):
    """
    Run inference on a batch of audio clips.
    """
    with torch.no_grad():
        inputs = processor.apply_transcription_request(
            language=lang,
            audio=audio_batch["array"],
            format=["WAV"],
            model_id=base_model_name,
        ).to(device)

        if is_task_routing:
            generated_tokens = model.base_model.generate(
                **inputs,
                max_new_tokens=256,
                do_sample=False,
                num_beams=1,
            )
        else:
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

    config = OmegaConf.load("config/eval_asr.yaml")

    # Load config
    manifest_path = config.manifest
    sample_rate = 16000

    # Device Setup
    device = torch.device(
        f"cuda:{int(config.device)}" if torch.cuda.is_available() else "cpu"
    )
    logger.info("Using device: %s", device)

    # Load Processor
    logger.info("Loading processor: %s", config.model)
    processor = VoxtralProcessor.from_pretrained(config.model)

    # Load Model (handles all checkpoint types)
    model, is_task_routing = load_model_for_evaluation(config, device, logger)

    # Load dataset
    dataset = load_eval_asr_manifest_dataset(manifest_path, sample_rate=sample_rate)

    # Prepare output file
    os.makedirs(os.path.dirname(config.output_path), exist_ok=True)
    if os.path.exists(config.output_path):
        os.remove(config.output_path)

    all_predictions, all_references = [], []

    print("Running inference...")
    with open(config.output_path, "a", encoding="utf-8") as f_out:
        for sample in tqdm(dataset):
            reference = sample["text"].strip()
            audio = sample["audio"]
            if config.lang:
                lang = config.lang
            else:
                lang = sample["source"]["lang"]

            # Run transcription
            prediction = transcribe_sample(
                model,
                config.model,
                processor,
                audio,
                LANGCODE_MAP[lang],
                device,
                is_task_routing,
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
