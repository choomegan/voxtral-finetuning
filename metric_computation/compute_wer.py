"""
Calculate WER, CER, MER overall and per-language for each row in a manifest
Optionally compute metrics only on verified entries (is_verified == "true").
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
    parser.add_argument(
        "-v",
        "--verified_only",
        help="compute overall metrics only on is_verified=='true'",
        action="store_true",
    )

    args = parser.parse_args()

    logging.info("Text normalisation: %s", args.normalise_text)
    logging.info("Verified only mode: %s", args.verified_only)

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

    # For verified-only logic
    verified_refs = []
    verified_hyps = []
    verified_refs_by_lang = defaultdict(list)
    verified_hyps_by_lang = defaultdict(list)

    for row in tqdm.tqdm(data):
        lang = (
            row.get("source", {}).get("lang")
            or row.get("source.lang")
            or row.get("lang")
            or "unknown"
        )

        # Check verification
        is_verified = str(row.get("is_verified", "")).lower() == "true"

        # Normalisation (optional)
        if args.normalise_text:
            pred_text = convert_numbers_to_words(row["prediction"])
            text = normalise(row["source"]["text"])
            pred_text = normalise(pred_text)
            row["text_normalised"] = text
            row["pred_text_normalised"] = pred_text
        else:
            text = row["text"]
            pred_text = row["pred_text"]

        # Always compute row-level metrics
        wer, cer, mer = compute_asr_metrics(text, pred_text)
        row["wer"] = wer
        row["cer"] = cer

        # Track all rows
        references.append(text)
        hypotheses.append(pred_text)
        refs_by_lang[lang].append(text)
        hyps_by_lang[lang].append(pred_text)

        # Track verified rows
        if is_verified:
            verified_refs.append(text)
            verified_hyps.append(pred_text)
            verified_refs_by_lang[lang].append(text)
            verified_hyps_by_lang[lang].append(pred_text)

    # --------------------------------
    # Compute overall metrics
    # --------------------------------
    if args.verified_only:
        logging.info("===== OVERALL METRICS (VERIFIED ONLY) =====")
        overall_wer, overall_cer, overall_mer = compute_asr_metrics(
            verified_refs, verified_hyps
        )
    else:
        logging.info("===== OVERALL METRICS (ALL DATA) =====")
        overall_wer, overall_cer, overall_mer = compute_asr_metrics(
            references, hypotheses
        )

    logging.info(f"Overall WER: {overall_wer}")
    logging.info(f"Overall CER: {overall_cer}")
    logging.info(f"Overall MER: {overall_mer}")

    # --------------------------------
    # Compute per-language metrics
    # --------------------------------
    logging.info("\n===== PER-LANGUAGE METRICS =====")

    langs = sorted(refs_by_lang.keys())

    for lang in langs:
        if args.verified_only:
            logging.info(
                "Number of rows for lang %s: %d", lang, len(verified_refs_by_lang[lang])
            )
            if len(verified_refs_by_lang[lang]) == 0:
                logging.info(f"\n--- language: {lang} (no verified rows) ---")
                continue
            lang_refs = verified_refs_by_lang[lang]
            lang_hyps = verified_hyps_by_lang[lang]
        else:
            lang_refs = refs_by_lang[lang]
            lang_hyps = hyps_by_lang[lang]

        lang_wer, lang_cer, lang_mer = compute_asr_metrics(lang_refs, lang_hyps)

        logging.info(f"\n--- language: {lang} ---")
        logging.info(f"WER: {lang_wer}")
        logging.info(f"CER: {lang_cer}")
        logging.info(f"MER: {lang_mer}")

    # Save manifest with row-level WER & CER
    save_manifest(items=data, save_path=args.output_dir)


if __name__ == "__main__":
    main()
