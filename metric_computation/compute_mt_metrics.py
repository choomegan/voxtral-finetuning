import argparse
import json
import numpy as np
from sacrebleu.metrics import BLEU, CHRF, TER
from comet import download_model, load_from_checkpoint

# --- Metric initialization ---
bleu = BLEU(tokenize="spm", effective_order=True)
chrf = CHRF()
ter = TER()

print("Loading XCOMET-XL model...")
xcomet_model_path = download_model("Unbabel/XCOMET-XL")
xcomet_model = load_from_checkpoint(xcomet_model_path)


def load_predictions_and_references(manifest_path):
    """
    Loads predictions, references, and sources from a JSONL manifest.
    Expected keys: 'prediction', 'reference' or 'target.text', and optionally 'source.text'.
    """
    predictions, references, sources = [], [], []

    with open(manifest_path, "r", encoding="utf-8") as f:
        for line in f:
            if not line.strip():
                continue
            entry = json.loads(line)

            pred = entry.get("prediction")
            ref = entry.get("reference", entry.get("target", {}).get("text"))
            src = entry.get("source.text", entry.get("source", {}).get("text"))

            if pred is None or ref is None:
                continue  # skip incomplete entries

            predictions.append(pred)
            references.append(ref)
            sources.append(src)

    print(f"Loaded {len(predictions)} samples from {manifest_path}")
    return sources, predictions, references


def compute_metrics(manifest_path):
    # --- Load data ---
    sources, preds, refs = load_predictions_and_references(manifest_path)

    # --- Compute metrics ---
    print("Computing SacreBLEU / CHRF / TER...")
    bleu_score = bleu.corpus_score(preds, [refs])
    chrf_score = chrf.corpus_score(preds, [refs])
    ter_score = ter.corpus_score(preds, [refs])

    print("Computing XCOMET scores...")
    xcomet_scores = xcomet_model.predict(
        [{"src": s, "mt": p, "ref": r} for s, p, r in zip(sources, preds, refs)]
    ).scores
    xcomet_mean = float(np.mean(xcomet_scores))

    # --- Combine results ---
    result = {
        "BLEU": bleu_score.score,
        "CHRF": chrf_score.score,
        "TER": ter_score.score,
        "XCOMET": xcomet_mean,
    }

    print("\n Translation Quality Metrics:")
    for k, v in result.items():
        print(f"{k}: {v:.3f}")

    return result


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
