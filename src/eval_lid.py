"""
Evaluation script for Language Identification (LID)

- No task tokens
- Routing is controlled purely by task_type
- Uses VoxtralWithTaskTokenRouting
"""

import json
import logging
import os
from collections import defaultdict

import torch
from omegaconf import OmegaConf
from tqdm import tqdm
from transformers import VoxtralProcessor
from sklearn.metrics import (
    confusion_matrix,
    precision_recall_fscore_support,
    classification_report,
)

from utils.constants import TASKTYPE2ID, ID2SRCLANG
from utils.dataset_utils import load_eval_lid_manifest_dataset
from utils.eval_helper import load_model_for_evaluation

# -----------------------------------------------------------------------------
# Logging
# -----------------------------------------------------------------------------

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)

logger = logging.getLogger(__name__)
LID_TASK_ID = TASKTYPE2ID["lid"]

# -----------------------------------------------------------------------------
# Inference
# -----------------------------------------------------------------------------


@torch.no_grad()
def run_lid_inference(
    model,
    processor: VoxtralProcessor,
    audio_path: str,
    device: torch.device,
):
    """
    Runs LID inference on a single audio sample.

    Returns:
        int: predicted language ID
    """

    messages = [
        {
            "role": "user",
            "content": [
                {"type": "audio", "path": audio_path},
            ],
        }
    ]

    model_inputs = processor.apply_chat_template(
        messages,
        return_tensors="pt",
        tokenize=True,
        padding="longest",
        continue_final_message=True,
    )

    input_ids = model_inputs["input_ids"].to(device)
    attention_mask = model_inputs["attention_mask"].to(device)
    input_features = model_inputs["input_features"].to(device)

    task_type = torch.full(
        (input_ids.size(0),),
        LID_TASK_ID,
        dtype=torch.long,
        device=device,
    )

    preds = model.generate(
        input_ids=input_ids,
        input_features=input_features,
        attention_mask=attention_mask,
        task_type=task_type,
    )

    return preds.item()


# -----------------------------------------------------------------------------
# Main
# -----------------------------------------------------------------------------


def main():
    config = OmegaConf.load("config/eval_lid.yaml")

    device = torch.device(
        f"cuda:{int(config.device)}" if torch.cuda.is_available() else "cpu"
    )
    logger.info("Using device: %s", device)

    processor = VoxtralProcessor.from_pretrained(config.model)

    model, is_task_routing = load_model_for_evaluation(config, device, logger)
    assert is_task_routing, "LID eval requires VoxtralWithTaskTokenRouting"
    model.eval()

    dataset = load_eval_lid_manifest_dataset(config.manifest)

    os.makedirs(os.path.dirname(config.output_path), exist_ok=True)
    if os.path.exists(config.output_path):
        os.remove(config.output_path)

    # -------------------------------------------------------------------------
    # Metrics storage
    # -------------------------------------------------------------------------

    total = 0
    correct = 0
    per_lang_total = defaultdict(int)
    per_lang_correct = defaultdict(int)

    y_true = []
    y_pred = []

    logger.info("Running LID evaluation on %d samples...", len(dataset))

    with open(config.output_path, "a", encoding="utf-8") as f_out:
        for sample in tqdm(dataset):
            audio_path = sample["source.audio_local_path"]
            gold_lang = sample["source.lang_id"]

            pred_lang = run_lid_inference(
                model=model,
                processor=processor,
                audio_path=audio_path,
                device=device,
            )

            total += 1
            per_lang_total[gold_lang] += 1

            y_true.append(gold_lang)
            y_pred.append(pred_lang)

            is_correct = pred_lang == gold_lang
            if is_correct:
                correct += 1
                per_lang_correct[gold_lang] += 1

            json.dump(
                {
                    "audio": audio_path,
                    "gold_lang": ID2SRCLANG.get(gold_lang),
                    "pred_lang": ID2SRCLANG.get(pred_lang),
                    "correct": is_correct,
                },
                f_out,
                ensure_ascii=False,
            )
            f_out.write("\n")

    # -------------------------------------------------------------------------
    # Metrics
    # -------------------------------------------------------------------------

    accuracy = correct / total if total > 0 else 0.0

    per_lang_acc = {
        lang: per_lang_correct[lang] / per_lang_total[lang] for lang in per_lang_total
    }

    precision, recall, f1, support = precision_recall_fscore_support(
        y_true,
        y_pred,
        average=None,
        zero_division=0,
    )

    precision_macro, recall_macro, f1_macro, _ = precision_recall_fscore_support(
        y_true, y_pred, average="macro", zero_division=0
    )

    precision_weighted, recall_weighted, f1_weighted, _ = (
        precision_recall_fscore_support(
            y_true, y_pred, average="weighted", zero_division=0
        )
    )

    cm = confusion_matrix(y_true, y_pred)

    logger.info("✅ LID Accuracy: %.4f", accuracy)

    logger.info(
        "\n%s",
        classification_report(
            y_true,
            y_pred,
            target_names=[ID2SRCLANG.get(i, str(i)) for i in sorted(set(y_true))],
            digits=4,
            zero_division=0,
        ),
    )

    # -------------------------------------------------------------------------
    # Save summary
    # -------------------------------------------------------------------------

    summary = {
        "accuracy": accuracy,
        "precision_macro": precision_macro,
        "recall_macro": recall_macro,
        "f1_macro": f1_macro,
        "precision_weighted": precision_weighted,
        "recall_weighted": recall_weighted,
        "f1_weighted": f1_weighted,
        "per_language_accuracy": per_lang_acc,
        "confusion_matrix": cm.tolist(),
        "num_samples": total,
        "model": config.model,
        "checkpoint": config.checkpoint_path,
        "manifest": config.manifest,
        "routing": "task_type_only",
    }

    summary_path = config.output_path + ".summary.json"
    with open(summary_path, "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2, ensure_ascii=False)

    logger.info("📁 Saved results to: %s", config.output_path)
    logger.info("📊 Saved summary to: %s", summary_path)


# -----------------------------------------------------------------------------

if __name__ == "__main__":
    main()
