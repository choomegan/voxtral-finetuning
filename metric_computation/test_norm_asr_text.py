import pytest
from clean_asr_texts import normalise  # adjust if your module name differs

TEST_CASES = [

    # -------------------------
    # Hyphen-related behaviour
    # -------------------------

    # Hyphens inside Malay/Indo words MUST be preserved
    ("pewaris-pewaris kita", "pewaris-pewaris kita"),
    ("akhir-akhir ini", "akhir-akhir ini"),

    # Hyphens between words reconstructed
    ("akhir - akhir ini", "akhir-akhir ini"),

    # Bullet hyphens removed
    ("- Disclaimer, Mas.", "disclaimer mas"),
    ("-- tidak penting --", "tidak penting"),

    # Hyphens not between text removed
    ("halo -", "halo"),
    ("-halo", "halo"),
    (" - test - ", "test"),

    # Unicode hyphens normalized
    ("pewaris–pewaris", "pewaris-pewaris"),

    # -------------------------
    # Fullstop handling
    # -------------------------

    # Remove trailing sentence punctuation
    ("itu pasti.", "itu pasti"),

    # Remove mid-sentence punctuation not part of numbers
    ("Iya. Kita melihatnya", "iya kita melihatnya"),
    ("iya.", "iya"),
    ("iya..", "iya"),

    # Decimal numbers remain valid
    ("75.000 tahun yang lalu.", "75.000 tahun yang lalu"),

    # -------------------------
    # Number normalization
    # -------------------------

    # Collapse spaced thousands
    ("75 . 000 tahun", "75.000 tahun"),

    # Fix percent spacing
    ("90 %", "90%"),
    ("90% di antaranya", "90% di antaranya"),

    # -------------------------
    # Punctuation removal
    # -------------------------

    ("masa jabatan Presiden Jokowi yang tinggal 2 - 3 tahun, kan sangat cepat.",
     "masa jabatan presiden jokowi yang tinggal 2-3 tahun kan sangat cepat"),

    ("\"Sains, Teknologi, Teknik dan Matematika\" itu pasti.",
     "sains teknologi teknik dan matematika itu pasti"),

    ("((Nah, akhir-akhir ini kita melihat))",
     "nah akhir-akhir ini kita melihat"),

    # -------------------------
    # Lines that must be dropped
    # -------------------------

    (".", None),
    (",", None),
    ("-", None),
    ("!!!", None),
    ("", None),
    ("   ", None),

    # Edge cases
    ("...'...", None),
    ("--", None),
]


@pytest.mark.parametrize("raw, expected", TEST_CASES)
def test_normalise(raw, expected):
    result = normalise(raw)

    if expected is None:
        # For drop cases: either empty string or None
        assert result is None or result.strip() == ""
    else:
        assert result == expected
