import argparse
import json
import numpy as np
from collections import defaultdict
from sacrebleu.metrics import BLEU, CHRF, TER
from comet import download_model, load_from_checkpoint

# --- Metric initialization ---
bleu = BLEU(tokenize="spm", effective_order=True)
# chrf = CHRF()
# ter = TER()

# print("Loading XCOMET-XL model...")
# xcomet_model_path = download_model("Unbabel/XCOMET-XL")
# xcomet_model = load_from_checkpoint(xcomet_model_path)


def load_predictions_and_references(manifest_path):
    """
    Loads predictions, references, sources, and language tags from a JSONL manifest.
    Expected keys: 'prediction', 'reference' or 'target.text',
                   'source.text', and 'source.lang'
    """
    data = []

    with open(manifest_path, "r", encoding="utf-8") as f:
        for line in f:
            if not line.strip():
                continue
            entry = json.loads(line)

            pred = entry.get("prediction")
            ref = entry.get("reference", entry.get("target", {}).get("text"))
            src = entry.get("source.text", entry.get("source", {}).get("text"))
            lang = entry.get("source.lang", entry.get("source", {}).get("lang"))

            if pred is None or ref is None:
                continue  # skip incomplete entries

            data.append(
                {
                    "src": src,
                    "pred": pred,
                    "ref": ref,
                    "lang": lang,
                }
            )

    print(f"Loaded {len(data)} samples from {manifest_path}")
    return data


def compute_metrics_for_group(group):
    preds = [x["pred"] for x in group]
    refs = [x["ref"] for x in group]
    sources = [x["src"] for x in group]  # required for XCOMET if used later

    # --- Compute SACREBLEU / CHRF / TER ---
    bleu_score = bleu.corpus_score(preds, [refs])
    # chrf_score = chrf.corpus_score(preds, [refs])
    # ter_score = ter.corpus_score(preds, [refs])

    # --- Compute XCOMET if enabled ---
    # xcomet_scores = xcomet_model.predict(
    #     [{"src": s, "mt": p, "ref": r} for s, p, r in zip(sources, preds, refs)]
    # ).scores
    # xcomet_mean = float(np.mean(xcomet_scores))

    result = {
        "BLEU": bleu_score.score,
        # "CHRF": chrf_score.score,
        # "TER": ter_score.score,
        # "XCOMET": xcomet_mean,
    }

    return result


def compute_metrics(manifest_path):
    # --- Load data ---
    data = load_predictions_and_references(manifest_path)

    # --- Group by language ---
    lang_groups = defaultdict(list)
    for item in data:
        lang = item["lang"] or "unknown"
        lang_groups[lang].append(item)

    print("\n===== Overall Metrics =====")
    overall = compute_metrics_for_group(data)
    for k, v in overall.items():
        print(f"{k}: {v:.4f}")

    print("\n===== Metrics by source.lang =====")
    results_by_lang = {}

    for lang, group in lang_groups.items():
        print(f"\n--- Language: {lang} ({len(group)} samples) ---")
        metrics = compute_metrics_for_group(group)
        results_by_lang[lang] = metrics

        for k, v in metrics.items():
            print(f"{k}: {v:.4f}")

    return {
        "overall": overall,
        "by_lang": results_by_lang,
    }


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Normalize and compute WER for fine-tuned Voxtral model."
    )
    parser.add_argument(
        "--manifest",
        type=str,
        help="Path to a manifest file with 'prediction' and 'text' fields.",
    )
    args = parser.parse_args()

    compute_metrics(args.manifest)
