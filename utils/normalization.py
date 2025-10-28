"""
Text normalization for mms domain data
"""

import re
import string
from num2words import num2words

def normalize_text(text: str) -> str:
    """
    Normalize text:
      - Expand ALL-CAPS acronyms into spaced letters (done from original text)
      - Expand decimals (9.081 -> 'nine decimal zero eight one')
      - Remove all punctuation except apostrophes (replace with spaces so tokens don't merge)
      - Expand numbers: short numbers -> num2words, long digit sequences -> digit-by-digit
      - Lowercase and normalize whitespace
    """
    if not text:
        return text

    original = text

    # 1) Expand ALL-CAPS acronyms from original text into spaced lowercase letters.
    #    Example: "BTI SOS" -> "b t i s o s"
    def split_acro(m):
        w = m.group(0)
        return " ".join(list(w.lower()))
    # Replace acronyms in the original string so we preserve casing info
    working = re.sub(r"\b[A-Z]{2,}\b", split_acro, original)

    # 2) Expand decimals first (so the decimal point isn't removed).
    #    We'll convert the integer part with num2words and spell fractional digits individually.
    def expand_decimal(m):
        num = m.group(0)
        left, right = num.split(".", 1)
        left_word = num2words(int(left)) if left != "" else ""
        # fractional digits spelled individually
        right_words = " ".join(num2words(int(d)) for d in right)
        if left_word:
            return f"{left_word} decimal {right_words}"
        else:
            return f"decimal {right_words}"
    working = re.sub(r"\b\d+\.\d+\b", expand_decimal, working)

    # 3) Remove punctuation except apostrophes.
    #    Replace punctuation with a single space so tokens don't get fused.
    punctuation = string.punctuation.replace("'", "")   # keep apostrophes
    working = re.sub(rf"[{re.escape(punctuation)}]+", " ", working)

    # 4) Expand remaining integer digit tokens.
    #    - If token length > 2 (e.g. "01011") treat as digit-by-digit: "zero one zero one one"
    #    - Else use num2words for the integer (e.g. "10" -> "ten")
    def expand_int_token(m):
        s = m.group(0)
        # preserve leading zeros by treating as digit sequence if any leading zero or length>2
        if len(s) > 2 or (len(s) > 1 and s.startswith("0")):
            return " ".join(num2words(int(d)) for d in s)
        else:
            return num2words(int(s))
    working = re.sub(r"\b\d+\b", expand_int_token, working)

    # 5) Lowercase and normalize whitespace
    working = working.lower()
    working = re.sub(r"\s+", " ", working).strip()

    return working