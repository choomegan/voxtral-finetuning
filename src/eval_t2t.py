"""
Evaluation script for text-to-text translation task
"""

import json
import logging
import os

import torch
from omegaconf import OmegaConf
from sacrebleu.metrics import BLEU
from tqdm import tqdm
from transformers import VoxtralProcessor

from utils.chat_template_utils import build_t2t_prompt_no_src_lang
from utils.dataset_utils import load_eval_st_manifest_dataset
from utils.eval_helper import load_model_for_evaluation

logging.basicConfig(
    level=logging.INFO,  # or DEBUG for more verbose logs
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)

logger = logging.getLogger(__name__)  # module-level logger


def translate_sample(
    model,
    processor: VoxtralProcessor,
    src_text: str,
    device: torch.device,
    src_lang: str = None,
    is_task_routing: bool = False,
):
    """
    Run speech translation inference on one audio sample using the path-based
    multimodal chat template. The processor handles feature extraction internally.

    Args:
        model: The Voxtral model (potentially wrapped with PeftModel).
        processor: The VoxtralProcessor.
        audio_path (str): Path to the audio file.
        src_lang (str): Source language code.
        device (torch.device): CUDA or CPU device.

    Returns:
        str: The predicted translated text.
    """
    with torch.no_grad():
        # 1. Build chat prompt
        messages = build_t2t_prompt_no_src_lang(src_text)

        # 2. Tokenize and process the full multimodal input
        model_inputs = processor.apply_chat_template(
            messages,
            return_tensors="pt",
            tokenize=True,
            padding="longest",
        )

        # Move inputs to device
        input_ids = model_inputs["input_ids"].to(device)
        attention_mask = model_inputs["attention_mask"].to(device)

        # 4. Construct Generation Inputs Dynamically
        # CRITICAL FIX: Only add input_features if they exist.
        # For T2T, the processor likely returns None for input_features.
        generate_inputs = {
            "input_ids": input_ids,
            "attention_mask": attention_mask,
        }

        if (
            "input_features" in model_inputs
            and model_inputs["input_features"] is not None
        ):
            print("input features is PRESENT!!!!!!")
            generate_inputs["input_features"] = model_inputs["input_features"].to(
                device
            )

        # 5. Generate
        if is_task_routing:
            generated_tokens = model.base_model.generate(
                **generate_inputs,
                max_new_tokens=256,
                do_sample=False,
                num_beams=1,
            )
        else:
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
    config = OmegaConf.load("config/eval_t2t.yaml")

    # Device Setup
    device = torch.device(
        f"cuda:{int(config.device)}" if torch.cuda.is_available() else "cpu"
    )
    logger.info("Using device: %s", device)

    # Load Model & Processor
    logger.info("Loading processor and base model: %s", config.model)
    processor = VoxtralProcessor.from_pretrained(config.model)

    # Load Model (handles all checkpoint types)
    model, is_task_routing = load_model_for_evaluation(config, device, logger)

    # Load dataset. Note: The dataset now contains the 'audio_path' column.
    dataset = load_eval_st_manifest_dataset(config.manifest)

    # Prepare output files
    os.makedirs(os.path.dirname(config.output_path), exist_ok=True)
    if os.path.exists(config.output_path):
        logger.info("Overwriting existing results file: %s", config.output_path)
        os.remove(config.output_path)

    # Initialize metrics
    bleu = BLEU(tokenize="spm", effective_order=True)
    all_predictions, all_references = [], []

    logger.info(
        "Running text to text translation inference on %s samples...", len(dataset)
    )

    with open(config.output_path, "a", encoding="utf-8") as f_out:
        for sample in tqdm(dataset):
            reference = sample["target.text"].strip()
            # New: Get the audio path instead of the loaded audio array
            src_text = sample["source.text"]

            # Run translation
            prediction = translate_sample(
                model=model,
                processor=processor,
                src_text=src_text,
                device=device,
                src_lang=None,
                is_task_routing=is_task_routing,
            )

            # Collect results for corpus-level metrics
            all_predictions.append(prediction)
            all_references.append(reference)

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
    corpus_bleu = bleu.corpus_score(all_predictions, [all_references]).score

    logger.info("\n✅ Corpus-level BLEU: %s", round(corpus_bleu, 4))

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

    logger.info("📁 Saved per-sample results to: %s", config.output_path)
    logger.info("📊 Saved summary to: %s", summary_file)


if __name__ == "__main__":
    main()
