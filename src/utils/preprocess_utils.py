import logging
import os
import json
from datasets import Dataset

logger = logging.getLogger(__name__)


# --- 1. Define the Unified Schema ---
def get_base_entry(task_name):
    """
    Returns a template dictionary with ALL possible columns for ALL tasks.
    This ensures interleave_datasets never drops data due to schema mismatch.
    """
    return {
        "task": task_name,
        "id": None,  # Useful for debugging
        "audio_path": None,  # ASR, ST, LID
        "source_text": None,  # T2T
        "target_text": None,  # ST, T2T
        "text": None,  # ASR (transcript)
        "source_lang": None,  # All
    }


def preprocess_asr_dataset(manifest_path: str) -> Dataset:
    logger.info("Preparing unified ASR dataset from: %s", manifest_path)
    root_dir = os.path.dirname(os.path.abspath(manifest_path))
    data = []

    with open(manifest_path, "r", encoding="utf-8") as f:
        for line in f:
            entry = json.loads(line.strip())
            src = entry.get("source", {})

            # Use base entry to ensure all keys exist
            row = get_base_entry("asr")

            # Fill relevant fields
            audio_path = src.get("audio_local_path")
            if audio_path:
                if not os.path.isabs(audio_path):
                    audio_path = os.path.join(root_dir, audio_path)
                row["audio_path"] = os.path.normpath(audio_path)

            row["text"] = src.get("text", "").strip()
            row["source_lang"] = src.get("lang")
            row["id"] = src.get("id")

            data.append(row)

    return Dataset.from_list(data)


def preprocess_st_dataset(manifest_path: str) -> Dataset:
    logger.info("Preparing unified ST dataset from: %s", manifest_path)
    root_dir = os.path.dirname(os.path.abspath(manifest_path))
    data = []

    with open(manifest_path, "r", encoding="utf-8") as f:
        for line in f:
            entry = json.loads(line.strip())
            src = entry.get("source", {})
            tgt = entry.get("target", {})

            row = get_base_entry("st")

            audio_path = src.get("audio_local_path")
            if audio_path:
                if not os.path.isabs(audio_path):
                    audio_path = os.path.join(root_dir, audio_path)
                row["audio_path"] = os.path.normpath(audio_path)

            row["target_text"] = tgt.get("text", "").strip()
            row["source_lang"] = src.get("lang")
            row["id"] = src.get("id")

            data.append(row)

    return Dataset.from_list(data)


def preprocess_t2t_dataset(manifest_path: str) -> Dataset:
    logger.info("Preparing unified T2T dataset from: %s", manifest_path)
    data = []

    with open(manifest_path, "r", encoding="utf-8") as f:
        for line in f:
            entry = json.loads(line.strip())
            src = entry.get("source", {})
            tgt = entry.get("target", {})

            row = get_base_entry("t2t")

            # Crucial: T2T fills these two, leaving audio_path as None
            row["source_text"] = src.get("text", "").strip()
            row["target_text"] = tgt.get("text", "").strip()
            row["source_lang"] = src.get("lang")
            row["id"] = src.get("id")

            # Validate to prevent "None" strings
            if row["source_text"] and row["target_text"]:
                data.append(row)

    return Dataset.from_list(data)


def preprocess_lid_dataset(manifest_path: str) -> Dataset:
    logger.info("Preparing unified LID dataset from: %s", manifest_path)
    root_dir = os.path.dirname(os.path.abspath(manifest_path))
    data = []

    with open(manifest_path, "r", encoding="utf-8") as f:
        for line in f:
            entry = json.loads(line.strip())
            src = entry.get("source", {})

            row = get_base_entry("lid")

            audio_path = src.get("audio_local_path")
            if audio_path:
                if not os.path.isabs(audio_path):
                    audio_path = os.path.join(root_dir, audio_path)
                row["audio_path"] = os.path.normpath(audio_path)

            row["source_lang"] = src.get("lang")
            row["id"] = src.get("id")

            data.append(row)

    return Dataset.from_list(data)
