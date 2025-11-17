"""
Calculate WER, CER, MER for each row in a manifest
"""

import json
import argparse
import logging
import re
from typing import Tuple, List, Dict

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


import re


def normalise(text: str) -> str:
    """
    ASR-friendly normalizer for Indonesian/Malay.

    Key points:
    - Convert Unicode dashes to ASCII hyphen BEFORE removing non-allowed chars
    - Preserve hyphens between letters (reduplication: 'pewaris-pewaris')
    - Preserve hyphens between digits (optional; keeps '2-3' ranges)
    - Remove hyphens used as bullets or standalone (e.g. '-halo' -> 'halo')
    - Keep decimals like '75.000' and attach '%' to numbers ('90%')
    - Remove sentence punctuation ('.', '?', '!') unless part of a number
    - Return None when normalized result is empty / punctuation-only
    """

    if not text or text.strip() == "":
        return None

    text = text.lower()

    # 1) Normalize unicode dashes to ASCII hyphen first (important)
    text = re.sub(r"[–—]", "-", text)

    # 2) Remove <unk> tokens
    text = re.sub(r"\s*<unk>\s*", " ", text)

    # 3) Keep only meaningful chars (allow hyphen because we've normalized it)
    text = re.sub(r"[^a-z0-9\-\.\%\' ]+", " ", text)

    # 4) Preserve valid hyphens by marking them with a temporary placeholder
    #    - letters-letter (reduplication: pewaris-pewaris)
    #    - digits-digit (numeric ranges: 2-3)  <-- keep if you want ranges preserved
    text = re.sub(r"([a-z])\s*-\s*([a-z])", r"\1<<<H>>>\2", text)
    text = re.sub(r"(\d)\s*-\s*(\d)", r"\1<<<H>>>\2", text)

    # 5) Number fixes
    #    Collapse spaced decimals: "75 . 000" -> "75.000"
    text = re.sub(r"(\d)\s*\.\s*(\d)", r"\1.\2", text)
    #    Attach percent to number: "90 %" -> "90%"
    text = re.sub(r"(\d)\s*%", r"\1%", text)

    # 6) Remove dots that are NOT part of numbers (must be after decimal fix)
    text = re.sub(r"(?<!\d)\.(?!\d)", " ", text)

    # 7) Remove hyphens that are NOT between alnum (the placeholder protects valid ones)
    #    This removes bullets and leading/trailing hyphens like "-halo" or "halo-"
    text = re.sub(r"(?<![a-z0-9])-", " ", text)   # hyphen with no alnum before
    text = re.sub(r"-(?![a-z0-9])", " ", text)    # hyphen with no alnum after

    # 8) Collapse multiple hyphens (safety) — placeholders are safe
    text = re.sub(r"-{2,}", "-", text)

    # 9) Restore preserved hyphens
    text = text.replace("<<<H>>>", "-")

    # 10) Remove apostrophes not inside words (standalone quotes)
    text = re.sub(r"(?<![a-z])'(?![a-z])", " ", text)

    # 11) Cleanup whitespace
    text = re.sub(r"\s+", " ", text).strip()

    # 12) Final drop conditions: empty or punctuation-only results -> return None
    if text == "":
        return None
    if re.fullmatch(r"[\.\-%']+", text):
        return None

    return text






def is_empty_or_punctuation(text: str) -> bool:
    """
    Returns True if text is None, empty, whitespace-only, or contains only punctuation characters.
    Safe to call with text==None.
    """
    # handle None early
    if text is None:
        return True

    # strip whitespace
    s = text.strip()
    if s == "":
        return True

    # if the string is only made of punctuation characters we consider it empty
    # include common punctuation we might keep earlier: . , ! ? - ' % and spaces (already stripped)
    if re.fullmatch(r"^[\.\,\!\?\-\'%]+$", s):
        return True

    return False


def main():
    parser = argparse.ArgumentParser(description="Calculate WER, CER, MER")
    parser.add_argument(
        "-i", "--manifest_dir", help="input manifest with 'text' and 'pred_text'"
    )
    parser.add_argument("-o", "--output_dir", help="output dir to save manifest")

    args = parser.parse_args()

    data = []
    with open(args.manifest_dir, "r") as f:
        for line in f:
            data.append(json.loads(line))

    cleaned = []
    total = len(data)
    skipped = 0

    for row in data:
        raw = row.get("text", "")
        norm = normalise(raw)

        # skip None or punctuation-only normalized outputs
        if is_empty_or_punctuation(norm):
            skipped += 1
            continue

        row["text_raw"] = raw
        row["text"] = norm
        cleaned.append(row)


    # ---- PRINT STATS ----
    if total > 0:
        pct = (skipped / total) * 100
    else:
        pct = 0

    print(f"\nSkipped {skipped}/{total} entries ({pct:.2f}%).")

    # Save if output_dir provided
    if args.output_dir:
        save_manifest(cleaned, args.output_dir)



if __name__ == "__main__":
    main()
