"""
Helper functions for dataset loading
"""

import logging
import json
import os
from typing import Tuple
from datasets import Audio, Dataset, interleave_datasets
from utils.preprocess_utils import (
    preprocess_asr_dataset,
    preprocess_st_dataset,
    preprocess_t2t_dataset,
    preprocess_lid_dataset,
)

logger = logging.getLogger(__name__)  # module-level logger


def load_asr_manifest_dataset(
    train_manifest: str,
    eval_manifest: str,
) -> Tuple[Dataset, Dataset]:
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
    eval_dataset = preprocess_asr_dataset(eval_manifest)
    train_dataset = preprocess_asr_dataset(train_manifest)
    logger.info(f"✅ Train dataset size: {len(train_dataset)}")
    logger.info(f"✅ Eval dataset size: {len(eval_dataset)}")

    return train_dataset, eval_dataset


def load_st_manifest_dataset(
    train_manifest: str, eval_manifest: str
) -> Tuple[Dataset, Dataset]:
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
    train_dataset = preprocess_st_dataset(train_manifest)
    eval_dataset = preprocess_st_dataset(eval_manifest)
    logger.info(f"✅ Train dataset size: {len(train_dataset)}")
    logger.info(f"✅ Eval dataset size: {len(eval_dataset)}")
    return train_dataset, eval_dataset


def load_t2t_manifest_dataset(
    train_manifest: str, eval_manifest: str
) -> Tuple[Dataset, Dataset]:
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
    train_dataset = preprocess_t2t_dataset(train_manifest)
    eval_dataset = preprocess_t2t_dataset(eval_manifest)
    logger.info(f"✅ Train dataset size: {len(train_dataset)}")
    logger.info(f"✅ Eval dataset size: {len(eval_dataset)}")
    return train_dataset, eval_dataset


def load_lid_manifest_dataset(
    train_manifest: str, eval_manifest: str
) -> Tuple[Dataset, Dataset]:
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
    train_dataset = preprocess_lid_dataset(train_manifest)
    eval_dataset = preprocess_lid_dataset(eval_manifest)
    logger.info(f"✅ Train dataset size: {len(train_dataset)}")
    logger.info(f"✅ Eval dataset size: {len(eval_dataset)}")
    return train_dataset, eval_dataset


def load_preprocessed_multitask_dataset(
    train_manifest: str,
    eval_manifest: str,
    incl_asr: bool = False,
    incl_s2tt: bool = False,
    incl_t2t: bool = False,
    incl_lid: bool = False,
) -> Tuple[Dataset, Dataset, Dataset, Dataset, Dataset]:
    """
    Load or create preprocessed datasets for both tasks.
    Uses datasets' automatic caching - memory mapped and cached to disk automatically.
    """
    train_datasets = []
    task_names = []
    eval_datasets = {}

    logger.info("Loading/preprocessing training datasets...")

    # ========== ASR ==========
    if incl_asr:
        logger.info("Loading/preprocessing ASR dataset...")

        train_asr = preprocess_asr_dataset(
            train_manifest,
        )
        eval_asr = preprocess_asr_dataset(eval_manifest)

        train_datasets.append(train_asr)
        task_names.append("asr")
        eval_datasets["asr"] = eval_asr
    else:
        eval_datasets["asr"] = None
        logger.info("⏭️  ASR task disabled")

    # ========== ST (Speech Translation) ==========
    if incl_s2tt:
        train_st = preprocess_st_dataset(
            train_manifest,
        )
        eval_st = preprocess_st_dataset(eval_manifest)

        train_datasets.append(train_st)
        task_names.append("st")
        eval_datasets["st"] = eval_st
    else:
        eval_datasets["st"] = None
        logger.info("⏭️  ST task disabled")

    # ========== T2T (Text-to-Text Translation) ==========
    if incl_t2t:
        train_t2t = preprocess_t2t_dataset(
            train_manifest,
        )
        eval_t2t = preprocess_t2t_dataset(eval_manifest)

        train_datasets.append(train_t2t)
        task_names.append("t2t")
        eval_datasets["t2t"] = eval_t2t
    else:
        eval_datasets["t2t"] = None
        logger.info("⏭️  T2T task disabled")

    # ========== LID (Language Identification) ==========
    if incl_lid:
        train_lid = preprocess_lid_dataset(
            train_manifest,
        )
        eval_lid = preprocess_lid_dataset(eval_manifest)

        train_datasets.append(train_lid)
        task_names.append("lid")
        eval_datasets["lid"] = eval_lid
    else:
        eval_datasets["lid"] = None
        logger.info("⏭️  LID task disabled")

    probabilities = [1.0 / len(train_datasets)] * len(train_datasets)

    # Interleave dataset with equal probability
    train_dataset = interleave_datasets(
        train_datasets,
        probabilities=probabilities,
        seed=42,
        stopping_strategy="all_exhausted",
    )
    logger.info(f"Multi-task mode: {', '.join(task_names)}")

    logger.info(f"Total training samples: {len(train_dataset)}")

    return (
        train_dataset,
        eval_datasets["asr"],
        eval_datasets["st"],
        eval_datasets["t2t"],
        eval_datasets["lid"],
    )


def load_eval_asr_manifest_dataset(
    manifest_path: str, sample_rate: int = 16000
) -> Dataset:
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
    logger.info("Loading dataset from: %s", manifest_path)
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

    logger.info(
        "Loaded %s samples from %s",
        len(dataset),
        manifest_path,
    )
    return dataset


def load_eval_st_manifest_dataset(manifest_path: str) -> Dataset:
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
    logger.info("Loading dataset from: %s", manifest_path)
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
    logger.info("Columns: %s", dataset.column_names)
    logger.info(
        "Loaded %s samples from %s",
        len(dataset),
        manifest_path,
    )
    return dataset
