"""
Helper functions for dataset loading
"""

import logging
import json
import os
import torch
from collections import Counter
from typing import Tuple
from datasets import Audio, Dataset, interleave_datasets
from utils.preprocess_utils import (
    preprocess_asr_dataset,
    preprocess_st_dataset,
    preprocess_t2t_dataset,
    preprocess_lid_dataset,
)
from utils.constants import SRCLANG2ID

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


def load_eval_lid_manifest_dataset(manifest_path: str) -> Dataset:
    logger.info("Loading LID eval dataset from: %s", manifest_path)
    root_dir = os.path.dirname(os.path.abspath(manifest_path))
    data = []

    with open(manifest_path, "r", encoding="utf-8") as f:
        for line in f:
            entry = json.loads(line.strip())

            # ---- Fix audio path ----
            audio_path = entry["source"]["audio_local_path"]
            if not os.path.isabs(audio_path):
                audio_path = os.path.join(root_dir, audio_path)
            entry["source"]["audio_local_path"] = os.path.normpath(audio_path)

            # ---- Create numeric LID label ----
            src_lang = entry["source"]["lang"]
            entry["source"]["lang_id"] = SRCLANG2ID[src_lang]

            data.append(entry)

    dataset = Dataset.from_list(data)

    # Flatten nested fields
    dataset = dataset.flatten()

    logger.info("Columns: %s", dataset.column_names)
    logger.info("Loaded %d samples from %s", len(dataset), manifest_path)

    return dataset


def compute_lid_class_weights(
    train_dataset: Dataset,
    method: str = "inverse_freq",
    beta: float = 0.9999,
    normalize: bool = True,
) -> torch.Tensor:
    """
    Compute class weights for LID task from multitask training dataset.

    Args:
        train_dataset: Interleaved multitask dataset containing LID samples
        method: Weighting method
            - 'inverse_freq': Weight = Total / (num_classes * count_per_class)
            - 'effective_samples': Weight = (1-beta) / (1-beta^n) [better for extreme imbalance]
            - 'sqrt_inverse_freq': Weight = sqrt(Total / count_per_class) [softer balancing]
        beta: Hyperparameter for effective_samples method (0.9999 recommended)
        normalize: Whether to normalize weights to sum to num_classes

    Returns:
        torch.Tensor: Class weights [num_classes], where index corresponds to SRCLANG2ID values
    """
    logger.info(f"Computing LID class weights using method: {method}")

    # ================================================
    # Step 1: Extract LID samples from multitask dataset
    # ================================================
    # Filter only LID task samples
    lid_samples = train_dataset.filter(
        lambda x: x.get("task") == "lid", desc="Filtering LID samples"
    )

    num_lid_samples = len(lid_samples)
    if num_lid_samples == 0:
        logger.warning("⚠️  No LID samples found in training dataset!")
        return torch.ones(len(SRCLANG2ID))

    logger.info(f"Found {num_lid_samples} LID samples in training set")

    # ================================================
    # Step 2: Count samples per language
    # ================================================
    # Get all source_lang values (these are language strings like 'zsm', 'ind')
    lang_counts = Counter(lid_samples["source_lang"])

    logger.info(f"Language distribution in LID samples:")
    for lang, count in sorted(lang_counts.items()):
        percentage = 100 * count / num_lid_samples
        logger.info(f"  {lang}: {count:,} samples ({percentage:.2f}%)")

    # ================================================
    # Step 3: Convert to tensor indexed by SRCLANG2ID
    # ================================================
    num_classes = len(SRCLANG2ID)
    counts = torch.zeros(num_classes, dtype=torch.float32)

    for lang, count in lang_counts.items():
        if lang not in SRCLANG2ID:
            logger.warning(f"⚠️  Unknown language '{lang}' found in dataset, skipping")
            continue
        lang_id = SRCLANG2ID[lang]
        counts[lang_id] = count

    # Check for zero counts (languages in SRCLANG2ID but not in data)
    zero_count_langs = [
        lang for lang, lang_id in SRCLANG2ID.items() if counts[lang_id] == 0
    ]
    if zero_count_langs:
        logger.warning(
            f"⚠️  Languages in SRCLANG2ID but not in training data: {zero_count_langs}"
        )

    # ================================================
    # Step 4: Compute weights based on method
    # ================================================
    total_samples = counts.sum()

    if method == "inverse_freq":
        # Standard inverse frequency: w_i = N / (K * n_i)
        # N = total samples, K = num classes, n_i = samples in class i
        weights = total_samples / (num_classes * counts.clamp(min=1))

        if normalize:
            weights = weights / weights.mean()

    elif method == "effective_samples":
        # Effective number of samples (Cui et al., 2019)
        # "Class-Balanced Loss Based on Effective Number of Samples"
        # w_i = (1 - beta) / (1 - beta^n_i)
        # Better for extreme imbalance (e.g., 1000:1 ratio)
        effective_num = 1.0 - torch.pow(beta, counts)
        weights = (1.0 - beta) / effective_num.clamp(min=1e-7)

        if normalize:
            weights = weights / weights.mean()

    elif method == "sqrt_inverse_freq":
        # Softer balancing: w_i = sqrt(N / n_i)
        # Less aggressive than full inverse frequency
        weights = torch.sqrt(total_samples / counts.clamp(min=1))

        if normalize:
            weights = weights / weights.mean()

    else:
        raise ValueError(
            f"Unknown method '{method}'. "
            f"Choose from: 'inverse_freq', 'effective_samples', 'sqrt_inverse_freq'"
        )

    # ================================================
    # Step 5: Log final weights
    # ================================================
    logger.info(f"Computed class weights ({method}):")
    id2lang = {v: k for k, v in SRCLANG2ID.items()}

    for lang_id, weight in enumerate(weights):
        lang = id2lang.get(lang_id, f"UNKNOWN_{lang_id}")
        count = int(counts[lang_id].item())

        if count > 0:
            logger.info(
                f"  {lang} (id={lang_id}): weight={weight:.4f}, count={count:,}"
            )
        else:
            logger.info(
                f"  {lang} (id={lang_id}): weight={weight:.4f}, count=0 (NOT IN DATA)"
            )

    # Log weight ratio for reference
    if weights.numel() == 2 and (weights > 0).all():
        ratio = weights.max() / weights.min()
        logger.info(f"Weight ratio (max/min): {ratio:.2f}x")

    return weights
