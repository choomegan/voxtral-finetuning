"""
Calculate WER, CER, MER overall and per-language for each row in a manifest
"""

import json
import argparse
import logging
import re
from typing import Tuple, List, Dict
from collections import defaultdict
import tqdm
import jiwer
from num2words import num2words

logging.basicConfig(
    format="%(levelname)s | %(asctime)s | %(message)s", level=logging.INFO
)


def save_manifest(items: List[Dict[str, str]], save_path: str) -> None:
    with open(save_path, "w") as fw:
        for item in items:
            fw.write(json.dumps(item, ensure_ascii=False) + "\n")
    logging.info(f"Saved manifest file to: {save_path}")


def compute_asr_metrics(reference: str, hypothesis: str) -> Tuple[float]:
    wer = round(jiwer.wer(reference, hypothesis), 5)
    cer = round(jiwer.cer(reference, hypothesis), 5)
    mer = round(jiwer.mer(reference, hypothesis), 5)
    return wer, cer, mer


def convert_numbers_to_words(text: str) -> str:
    number_pattern = re.compile(r"\d+(?:\.\d+)?")

    def replacer(match):
        number_str = match.group(0)
        try:
            if "." in number_str:
                return num2words(float(number_str))
            else:
                return num2words(int(number_str))
        except ValueError:
            return number_str

    return number_pattern.sub(replacer, text)


def normalise(text: str) -> str:
    text = text.lower()
    if "<unk>" in text:
        text = re.sub(r"\s*<unk>\s*", " ", text)

    cleaned = re.sub(r"[^A-Za-z0-9#\' ]+", " ", text)
    cleaned = re.sub(r"\s+", " ", cleaned)
    return cleaned.strip()


def main():
    parser = argparse.ArgumentParser(description="Calculate WER, CER and MER")
    parser.add_argument("-i", "--manifest_dir", help="input manifest dir")
    parser.add_argument("-o", "--output_dir", help="output dir to save manifest")
    parser.add_argument("-n", "--normalise_text", help="boolean to normalise text")

    args = parser.parse_args()

    logging.info("Text normalisation: %s", args.normalise_text)

    # -------------------------------
    # Load manifest
    # -------------------------------
    data = []
    with open(args.manifest_dir, "r") as f:
        for line in f:
            data.append(json.loads(line))

    references = []
    hypotheses = []

    # For per-language metrics
    refs_by_lang = defaultdict(list)
    hyps_by_lang = defaultdict(list)

    for row in tqdm.tqdm(data):
        lang = (
            row.get("source", {}).get("lang")
            or row.get("source.lang")
            or row.get("lang")
            or "unknown"
        )

        # ---------------------------
        # Normalisation (optional)
        # ---------------------------
        if args.normalise_text:
            pred_text = convert_numbers_to_words(row["prediction"])
            text = normalise(row["source"]["text"])
            pred_text = normalise(pred_text)
            row["text_normalised"] = text
            row["pred_text_normalised"] = pred_text
        else:
            text = row["text"]
            pred_text = row["pred_text"]

        references.append(text)
        hypotheses.append(pred_text)

        # Track by language
        refs_by_lang[lang].append(text)
        hyps_by_lang[lang].append(pred_text)

        # Compute per-row metrics
        wer, cer, mer = compute_asr_metrics(text, pred_text)
        row["wer"] = wer
        row["cer"] = cer

    # --------------------------------
    # Compute overall metrics
    # --------------------------------
    overall_ref = " ".join(references)
    overall_hyp = " ".join(hypotheses)

    overall_wer, overall_cer, overall_mer = compute_asr_metrics(
        overall_ref, overall_hyp
    )

    logging.info("===== OVERALL METRICS =====")
    logging.info(f"Overall WER: {overall_wer}")
    logging.info(f"Overall CER: {overall_cer}")
    logging.info(f"Overall MER: {overall_mer}")

    # --------------------------------
    # Compute per-language metrics
    # --------------------------------
    logging.info("\n===== PER-LANGUAGE METRICS =====")

    for lang in sorted(refs_by_lang.keys()):
        lang_ref = " ".join(refs_by_lang[lang])
        lang_hyp = " ".join(hyps_by_lang[lang])

        lang_wer, lang_cer, lang_mer = compute_asr_metrics(lang_ref, lang_hyp)

        logging.info(f"\n--- language: {lang} ---")
        logging.info(f"WER: {lang_wer}")
        logging.info(f"CER: {lang_cer}")
        logging.info(f"MER: {lang_mer}")

    # Save manifest with row-level WER & CER
    save_manifest(items=data, save_path=args.output_dir)


if __name__ == "__main__":
    main()
