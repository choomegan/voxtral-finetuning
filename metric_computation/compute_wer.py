"""
Calculate WER, CER, MER for each row in a manifest
"""

import json
import argparse
import logging
import re
from typing import Tuple, List, Dict

import jiwer
from num2words import num2words

logging.basicConfig(
    format="%(levelname)s | %(asctime)s | %(message)s", level=logging.INFO
)
def save_manifest(items: List[Dict[str, str]], save_path: str) -> None:
    """
    Save manifest file
    """
    with open(save_path, "w") as fw:
        for item in items:
            fw.write(json.dumps(item, ensure_ascii=False) + "\n")
    logging.info(f"Saved manifest file to: {save_path}")


def compute_asr_metrics(reference: str, hypothesis: str) -> Tuple[float]:
    """
    Use Jiwer to calculate asr metrics
    """
    wer = round(jiwer.wer(reference, hypothesis), 5)
    cer = round(jiwer.cer(reference, hypothesis), 5)
    mer = round(jiwer.mer(reference, hypothesis), 5)

    return wer, cer, mer


def convert_numbers_to_words(text: str) -> str:
    """
    Convert all integer or decimal numbers in a string to their word form.

    Args:
        text (str): Input string containing numbers.

    Returns:
        str: String with numbers replaced by words.
    """
    # This regex finds integers and decimals (e.g., 123, 45.67)
    number_pattern = re.compile(r"\d+(?:\.\d+)?")

    def replacer(match):
        number_str = match.group(0)
        try:
            if "." in number_str:
                return num2words(float(number_str))
            else:
                return num2words(int(number_str))
        except ValueError:
            return number_str  # fallback in case of unexpected format

    return number_pattern.sub(replacer, text)


def normalise(text: str) -> str:
    text = text.lower()
    if "<unk>" in text:
        text = re.sub(r"\s*<unk>\s*", " ", text)

    # remove unwanted chars
    cleaned = re.sub(r"[^A-Za-z0-9#\' ]+", " ", text)
    # remove multiple spaces
    cleaned = re.sub(r"\s+", " ", cleaned)
    return cleaned.strip()


def main():
    parser = argparse.ArgumentParser(description="Calculate WER, CER and Word Accuracy")
    parser.add_argument(
        "-i", "--manifest_dir", help="input manifest dir with 'text' and 'pred_text'"
    )
    parser.add_argument("-o", "--output_dir", help="output dir to save manifest")
    parser.add_argument(
        "-n", "--normalise_text", help="boolean on whether to normalise text"
    )

    args = parser.parse_args()

    logging.info("Text normalisation: %s", args.normalise_text)

    data = []
    with open(args.manifest_dir, "r") as f:
        for line in f:
            data.append(json.loads(line))

    references = []
    hypotheses = []

    for row in data:
        if args.normalise_text:
            pred_text = convert_numbers_to_words(row["prediction"])  # num normalisation
            text = normalise(row["source"]["text"])
            pred_text = normalise(pred_text)
            row["text_normalised"] = text
            row["pred_text_normalised"] = pred_text

        else:
            text = row["text"]
            pred_text = row["pred_text"]

        references.append(text)
        hypotheses.append(pred_text)
        wer, cer, mer = compute_asr_metrics(text, pred_text)
        row["wer"] = wer
        row["cer"] = cer

    overall_wer, overall_cer, overall_mer = compute_asr_metrics(references, hypotheses)
    logging.info(f"WER: {overall_wer}")
    logging.info(f"CER: {overall_cer}")

    save_manifest(items=data, save_path=args.output_dir)


if __name__ == "__main__":
    main()
