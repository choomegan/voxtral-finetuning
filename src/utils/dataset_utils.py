"""
Helper functions for dataset loading
"""

from datasets import Dataset
from utils.preprocess_utils import preprocess_asr_dataset, preprocess_st_dataset


def load_asr_manifest_dataset(
    train_manifest: str, eval_manifest: str, processor, model_id: str
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
    train_dataset = preprocess_asr_dataset(train_manifest, processor, model_id)
    eval_dataset = preprocess_asr_dataset(eval_manifest, processor, model_id)
    return train_dataset, eval_dataset


def load_st_manifest_dataset(
    train_manifest: str, eval_manifest: str, processor, model_id: str
) -> Dataset:
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
    train_dataset = preprocess_st_dataset(train_manifest, processor, model_id)
    eval_dataset = preprocess_st_dataset(eval_manifest, processor, model_id)
    return train_dataset, eval_dataset


def load_preprocessed_multitask_dataset(
    train_manifest: str,
    eval_manifest: str,
    processor,
    model_id: str,
):
    """
    Load or create preprocessed datasets for both tasks.

    This caches the preprocessed data to disk so you only pay the cost once.
    """
    # Load or create train datasets
    train_asr = preprocess_asr_dataset(train_manifest, processor, model_id)
    train_st = preprocess_st_dataset(train_manifest, processor, model_id)

    # Load or create eval datasets
    eval_asr = preprocess_asr_dataset(eval_manifest, processor, model_id)
    eval_st = preprocess_st_dataset(eval_manifest, processor, model_id)

    # Combine and shuffle
    from datasets import concatenate_datasets

    train_dataset = concatenate_datasets([train_asr, train_st]).shuffle(seed=42)
    eval_dataset = concatenate_datasets([eval_asr, eval_st]).shuffle(seed=42)

    return train_dataset, eval_dataset
