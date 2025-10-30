import os
import json
from datasets import Audio, Dataset
from typing import List, Dict


def load_st_manifest_dataset(manifest_path, sample_rate=16000):
    """
    Loading manifest file for S2TT
    """
    print(f"Loading dataset from: {manifest_path}")
    root_dir = os.path.dirname(os.path.abspath(manifest_path))
    data = []

    with open(manifest_path, "r", encoding="utf-8") as f:
        for line in f:
            entry = json.loads(line.strip())

            # Normalize
            entry["target"]["text"] = entry["target"]["text"].strip()

            # Fix relative audio paths
            audio_path = entry["source"]["audio_local_path"]
            if not os.path.isabs(audio_path):
                audio_path = os.path.join(root_dir, audio_path)
            entry["source"]["audio_local_path"] = os.path.normpath(audio_path)
            data.append(entry)

    dataset = Dataset.from_list(data)

    # remove nested structure due to format of manifest file
    dataset = dataset.flatten()

    # Rename for collator compatibility
    print("Columns:", dataset.column_names)
    print(f"Loaded {len(dataset)} samples from {manifest_path}")
    return dataset


def build_convos(src_lang: str, tgt_lang: str, audio_path: str) -> List[Dict]:
    """
    Build chat prompt
    """
    source_lang = "Indonesian" if src_lang == "ind" else "Malay"

    return [
        {
            "role": "user",
            "content": [
                {"type": "audio", "path": audio_path},
                {
                    "type": "text",
                    "text": f"Translate this {source_lang} audio into English text.",
                },
            ],
        },
    ]
