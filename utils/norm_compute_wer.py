"""
Script to take in model evaluation from eval_lora.py and compute corpus-level WER after normalization.
"""
import argparse
import pandas as pd
from jiwer import wer
from normalization import normalize_text

def normalize_and_compute_wer(manifest_path: str, remove_unk: bool = True) -> None:
    """
    Load eval results, normalize texts, compute corpus-level WER
    """

    df = pd.read_json(manifest_path, lines=True)

    print("Percentage of samples with <UNK> in reference text: ", round((df['text'].apply(lambda x: "<UNK>" in x).sum()/len(df) * 100), 2), "%")

    if remove_unk:
        print(len(df), "samples before removing <UNK> references.")
        df = df[df['text'].apply(lambda x: "<UNK>" not in x)].reset_index(drop=True)
        print(len(df), "samples remaining after removing <UNK> references.")


    def _remove_unk(text):
        """
        remove unk in text
        """
        return text.replace("<unk>", "").strip()

    predictions = df['prediction'].apply(normalize_text).tolist()
    references = df['text'].str.lower().apply(_remove_unk).tolist()

    print("------------------------- Sample predictions -------------------------")
    print(predictions[:10])
    print("------------------------- Sample references -------------------------")
    print(references[:10])


    corpus_wer = wer(references, predictions)
    print(f"\n Corpus-level WER: {corpus_wer:.4f}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Normalize and compute WER for fine-tuned Voxtral model.")
    parser.add_argument(
        "--manifest", type=str, help="Path to a manifest file with 'prediction' and 'text' fields."
    )
    parser.add_argument(
        "--remove_unk", action="store_true", help="Whether to remove samples with <UNK> from references before computing WER."
    )

    args = parser.parse_args()

    normalize_and_compute_wer(args.manifest, args.remove_unk)