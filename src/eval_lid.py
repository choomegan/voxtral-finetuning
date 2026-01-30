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

import librosa
import soundfile as sf
import torch
from omegaconf import OmegaConf
from tqdm import tqdm
from transformers import VoxtralProcessor
from sklearn.metrics import (
    confusion_matrix,
    precision_recall_fscore_support,
    classification_report,
)

from utils.constants import TASKTYPE2ID, ID2SRCLANG, SRCLANG2ID
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
    Directly replicates StreamingLIDCollator logic to ensure
    mel input features are exactly length 3000.
    """
    # 1. Match Collator Resampling
    audio, sr = sf.read(audio_path)

    if sr != 16000:
        audio = librosa.resample(audio, orig_sr=sr, target_sr=16000)

    # 2. Extract Features
    # Note: We do manual padding/truncation to be 100% sure it matches training
    EXPECTED_MEL_LENGTH = 3000
    audio_features = processor.feature_extractor(
        audio,
        sampling_rate=16000,
        return_tensors="pt",
    )
    # Capture the ACTUAL length T before padding
    T_actual = audio_features.input_features.shape[-1]
    audio_lengths = torch.tensor([T_actual], dtype=torch.long, device=device)

    ## Apply manual padding/ truncation to match collator behavior
    input_features = audio_features.input_features.to(device)  # [1, 128, T]
    B, mel_dim, T = input_features.shape

    if T < EXPECTED_MEL_LENGTH:
        pad_length = EXPECTED_MEL_LENGTH - T
        input_features = torch.nn.functional.pad(
            input_features, (0, pad_length), value=0.0
        )
    elif T > EXPECTED_MEL_LENGTH:
        input_features = input_features[:, :, :EXPECTED_MEL_LENGTH]

    # 4. Dummy Text Inputs (Identical to your Collator)
    pad_id = processor.tokenizer.pad_token_id
    input_ids = torch.full((1, 1), pad_id, dtype=torch.long, device=device)
    attention_mask = torch.zeros((1, 1), dtype=torch.long, device=device)

    # 5. Task Routing
    task_type = torch.tensor([LID_TASK_ID], dtype=torch.long, device=device)

    # 6. Forward pass through custom model
    # Diagnostic
    non_zero_frames = (input_features.abs() > 1e-5).sum().item() / 128
    logger.info(f"Audio Signal: {non_zero_frames} active frames out of 3000")
    logger.info(
        f"Feature Max: {input_features.max().item():.4f}, Mean: {input_features.mean().item():.4f}"
    )

    lid_logits = model.generate(
        input_ids=input_ids,
        input_features=input_features,
        audio_lengths=audio_lengths,
        attention_mask=attention_mask,
        task_type=task_type,
        return_lid_logits=True,
    )
    print("lid logits: ,", lid_logits)
    # Compute softmax probabilities
    probs = torch.softmax(lid_logits, dim=-1)  # [1, num_languages]
    pred_lang = lid_logits.argmax(dim=-1).item()

    return (
        pred_lang,
        probs.squeeze(0).float().cpu().numpy(),
    )  # Return both pred_lang and probs


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
    y_probs = []  # Store all probabilities

    logger.info("Running LID evaluation on %d samples...", len(dataset))

    with open(config.output_path, "a", encoding="utf-8") as f_out:
        for sample in tqdm(dataset):
            audio_path = sample["source.audio_local_path"]
            gold_lang = sample["source.lang_id"]

            pred_lang, probs = run_lid_inference(
                model=model,
                processor=processor,
                audio_path=audio_path,
                device=device,
            )

            total += 1
            per_lang_total[gold_lang] += 1

            y_true.append(gold_lang)
            y_pred.append(pred_lang)
            y_probs.append(probs)

            is_correct = pred_lang == gold_lang
            if is_correct:
                correct += 1
                per_lang_correct[gold_lang] += 1

            # Create probability dict with language names as keys
            prob_dict = {
                ID2SRCLANG.get(lang_id, f"lang_{lang_id}"): float(probs[lang_id])
                for lang_id in range(len(probs))
            }

            json.dump(
                {
                    "audio": audio_path,
                    "gold_lang": ID2SRCLANG.get(gold_lang),
                    "pred_lang": ID2SRCLANG.get(pred_lang),
                    "correct": is_correct,
                    "confidence": float(probs[pred_lang]),  # Confidence in prediction
                    "probabilities": prob_dict,  # Full probability distribution
                },
                f_out,
                ensure_ascii=False,
            )
            f_out.write("\n")

    # -------------------------------------------------------------------------
    # Compute Per-Language Metrics
    # -------------------------------------------------------------------------

    num_langs = len(SRCLANG2ID)
    id2lang = {v: k for k, v in SRCLANG2ID.items()}

    # Get per-class precision, recall, f1
    precision_per_class, recall_per_class, f1_per_class, support = (
        precision_recall_fscore_support(
            y_true,
            y_pred,
            labels=list(range(num_langs)),
            average=None,  # Return per-class metrics
            zero_division=0,
        )
    )

    # Get macro averages
    precision_macro, recall_macro, f1_macro, _ = precision_recall_fscore_support(
        y_true, y_pred, average="macro", zero_division=0
    )

    # Get weighted averages
    precision_weighted, recall_weighted, f1_weighted, _ = (
        precision_recall_fscore_support(
            y_true, y_pred, average="weighted", zero_division=0
        )
    )

    cm = confusion_matrix(y_true, y_pred)

    # -------------------------------------------------------------------------
    # Log Results
    # -------------------------------------------------------------------------

    logger.info("=" * 80)
    logger.info("LID EVALUATION RESULTS")
    logger.info("=" * 80)

    logger.info("\n📊 MACRO AVERAGES:")
    logger.info(f"  Precision: {precision_macro:.4f}")
    logger.info(f"  Recall:    {recall_macro:.4f}")
    logger.info(f"  F1-Score:  {f1_macro:.4f}")

    logger.info("\n📊 WEIGHTED AVERAGES:")
    logger.info(f"  Precision: {precision_weighted:.4f}")
    logger.info(f"  Recall:    {recall_weighted:.4f}")
    logger.info(f"  F1-Score:  {f1_weighted:.4f}")

    logger.info("\n📋 PER-LANGUAGE METRICS:")
    logger.info(
        "\n%s",
        classification_report(
            y_true,
            y_pred,
            target_names=[ID2SRCLANG.get(i, str(i)) for i in range(num_langs)],
            digits=4,
            zero_division=0,
        ),
    )

    # -------------------------------------------------------------------------
    # Save Summary
    # -------------------------------------------------------------------------

    # Build per-language metrics dict
    per_language_metrics = {}
    for lang_id in range(num_langs):
        lang_name = id2lang.get(lang_id, f"lang_{lang_id}")
        per_language_metrics[lang_name] = {
            "precision": float(precision_per_class[lang_id]),
            "recall": float(recall_per_class[lang_id]),
            "f1": float(f1_per_class[lang_id]),
            "support": int(support[lang_id]),  # Number of samples
        }

    summary = {
        # Macro averages
        "precision_macro": float(precision_macro),
        "recall_macro": float(recall_macro),
        "f1_macro": float(f1_macro),
        # Weighted averages
        "precision_weighted": float(precision_weighted),
        "recall_weighted": float(recall_weighted),
        "f1_weighted": float(f1_weighted),
        # Per-language breakdown
        "per_language_metrics": per_language_metrics,
        # Additional info
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

    logger.info("\n📁 Saved detailed results to: %s", config.output_path)
    logger.info("📊 Saved summary to: %s", summary_path)
    logger.info("=" * 80)


# -----------------------------------------------------------------------------

if __name__ == "__main__":
    main()
