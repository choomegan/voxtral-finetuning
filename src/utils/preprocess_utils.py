import logging
import os
import json
from datasets import Dataset

logger = logging.getLogger(__name__)  # module-level logger


def preprocess_asr_dataset(manifest_path: str) -> Dataset:
    """
    Lightweight preprocessing for ASR.
    Only normalizes paths and text. No feature extraction or tokenization.
    """
    logger.info("Preparing lightweight ASR dataset from: %s", manifest_path)
    root_dir = os.path.dirname(os.path.abspath(manifest_path))

    data = []
    with open(manifest_path, "r", encoding="utf-8") as f:
        for line in f:
            entry = json.loads(line.strip())
            audio_path = entry["source"]["audio_local_path"]
            if not os.path.isabs(audio_path):
                audio_path = os.path.join(root_dir, audio_path)
            audio_path = os.path.normpath(audio_path)

            data.append(
                {
                    "task": "asr",
                    "audio_path": audio_path,
                    "text": entry["source"]["text"].strip(),
                    "source_lang": entry["source"]["lang"],
                }
            )

    return Dataset.from_list(data)


def preprocess_st_dataset(manifest_path: str) -> Dataset:
    """
    Lightweight preprocessing for ST.
    Only stores text, language, and audio paths.
    """
    logger.info("Preparing lightweight ST dataset from: %s", manifest_path)
    root_dir = os.path.dirname(os.path.abspath(manifest_path))

    data = []
    with open(manifest_path, "r", encoding="utf-8") as f:
        for line in f:
            entry = json.loads(line.strip())
            audio_path = entry["source"]["audio_local_path"]
            if not os.path.isabs(audio_path):
                audio_path = os.path.join(root_dir, audio_path)
            audio_path = os.path.normpath(audio_path)

            data.append(
                {
                    "task": "st",
                    "audio_path": audio_path,
                    "source_lang": entry["source"]["lang"],
                    "target_text": entry["target"]["text"].strip(),
                }
            )

    return Dataset.from_list(data)

def preprocess_t2t_dataset(manifest_path: str) -> Dataset:
    """
    Lightweight preprocessing for text-to-text translation.
    Stores source text, target text, and language codes.
    
    Expected manifest format (one JSON per line):
    {
        "source": {
            "text": "CEO LTAT",
            "lang": "zsm"
        },
        "target": {
            "text": "the CEO of LTAT.",
            "lang": "eng"
        }
    }
    """
    logger.info("Preparing lightweight T2T translation dataset from: %s", manifest_path)
    
    data = []
    with open(manifest_path, "r", encoding="utf-8") as f:
        for line in f:
            entry = json.loads(line.strip())
            
            data.append(
                {
                    "task": "t2t", 
                    "source_lang": entry["source"]["lang"],
                    "source_text": entry["source"]["text"].strip(),
                    "target_text": entry["target"]["text"].strip(),
                }
            )
    
    return Dataset.from_list(data)
