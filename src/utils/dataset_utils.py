"""
Helper functions for dataset loading
"""

import os
import json
from datasets import Audio, Dataset


def load_asr_manifest_dataset(manifest_path: str, sample_rate: int = 16000) -> Dataset:
    """
    Data loader for ASR

    Load dataset from a JSON manifest file and make audio filepaths absolute.
    Each line should have:
    {
      "audio_filepath": "audio/audio_1.wav",
      "duration": 5.038,
      "start": 1166.599,
      "end": 1171.637,
      "text": "this is a transcript"
    }
    """
    print(f"Loading dataset from: {manifest_path}")
    root_dir = os.path.dirname(os.path.abspath(manifest_path))

    data = []
    with open(manifest_path, "r", encoding="utf-8") as f:
        for line in f:
            entry = json.loads(line.strip())

            # Normalize text
            entry["text"] = entry["text"].lower().strip()

            # Prepend the manifest's directory if path is relative
            audio_path = entry["audio_filepath"]
            if not os.path.isabs(audio_path):
                audio_path = os.path.join(root_dir, audio_path)
            entry["audio_filepath"] = os.path.normpath(audio_path)
            data.append(entry)

    dataset = Dataset.from_list(data)

    # add column to retain original audio_filepath
    dataset = dataset.add_column(
        "original_audio_filepath",
        [x["audio_filepath"].replace(root_dir, "") for x in data],
    )
    # Decode audio on the fly
    dataset = dataset.cast_column("audio_filepath", Audio(sampling_rate=sample_rate))

    # Rename to match collator expectations
    dataset = dataset.rename_column("audio_filepath", "audio")

    print(f"Loaded {len(dataset)} samples from {manifest_path}")
    return dataset


def load_st_manifest_dataset(manifest_path: str) -> Dataset:
    """
    Data loader for speech translation

    Load dataset from a JSON manifest file and make audio filepaths absolute.
    Each line should have:
    {
        "source":
            {
                "text": "CEO LTAT",
                "lang": "zsm",
                "audio_local_path": "audio/TeF4KD586kk-254.wav",
                "sampling_rate": 16000
            },
        "target":
            {
                "text": "the CEO of LTAT.",
                "lang": "eng"}
            }
    }
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
