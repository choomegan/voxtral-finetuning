"""
Script for dataset preprocessing utilities for ASR and ST tasks.
"""

import os
import json
import torch
from datasets import Dataset, Sequence, Value
import numpy as np
from tqdm import tqdm

from utils.chat_template_utils import build_st_prompt

MAX_AUDIO_DURATION_SEC = 30.0


def preprocess_asr_dataset(
    manifest_path: str, processor, model_id: str, sample_rate: int = 16000
):
    """
    Preprocess ASR dataset - tokenize text and process audio features offline.

    This moves all expensive operations OUT of the collator:
    - Audio feature extraction (apply_transcription_request)
    - Text tokenization

    The collator then only needs to pad and stack.
    """
    print(f"Preprocessing ASR dataset from: {manifest_path}")
    root_dir = os.path.dirname(os.path.abspath(manifest_path))

    data = []
    with open(manifest_path, "r", encoding="utf-8") as f:
        lines = [json.loads(line.strip()) for line in f]

    # Process in batches for efficiency
    batch_size = 32
    tokenizer = processor.tokenizer

    max_samples = int(MAX_AUDIO_DURATION_SEC * sample_rate)  # Add this line

    for i in tqdm(range(0, len(lines), batch_size), desc="Preprocessing ASR"):
        batch = lines[i : i + batch_size]

        # Collect batch data
        texts = []
        audio_paths = []
        audios = []

        for entry in batch:
            audio_path = entry["source"]["audio_local_path"]
            if not os.path.isabs(audio_path):
                audio_path = os.path.join(root_dir, audio_path)
            audio_path = os.path.normpath(audio_path)

            texts.append(entry["source"]["text"])
            audio_paths.append(audio_path)

            # Load audio
            import soundfile as sf

            audio, sr = sf.read(audio_path)
            if sr != sample_rate:
                import librosa

                audio = librosa.resample(audio, orig_sr=sr, target_sr=sample_rate)

            # Check duration based on the number of samples
            if audio.shape[0] > max_samples:
                print(
                    f"Skipping sample {audio_path}: duration ({audio.shape[0] / sr:.2f}s) exceeds {MAX_AUDIO_DURATION_SEC}s."
                )
                continue  # Skip this sample

            audios.append(audio)

        # Process audio features in batch
        prompt = processor.apply_transcription_request(
            language="en",
            model_id=model_id,
            audio=audios,
            format=["WAV"] * len(audios),
            return_tensors="pt",
        )

        # Tokenize text in batch
        text_tokens = tokenizer(
            texts,
            add_special_tokens=False,
            padding=False,
            truncation=True,
            max_length=256,
        )

        # Store preprocessed data
        for j, entry in enumerate(batch):
            # Extract per-sample data
            prompt_ids = prompt["input_ids"][j]
            prompt_attn = prompt["attention_mask"][j]

            # Get actual prompt length (non-padded)
            prompt_len = prompt_attn.sum().item()

            # Get text tokens
            text_ids = text_tokens["input_ids"][j]

            # Extract audio features if present and cast to float32
            input_features = None
            if "input_features" in prompt:
                input_features = np.array(prompt["input_features"][j], dtype=np.float32)

            # Store preprocessed sample
            data.append(
                {
                    "task": "asr",
                    # Preprocessed features
                    "prompt_input_ids": prompt_ids[
                        :prompt_len
                    ].tolist(),  # Remove padding
                    "text_input_ids": text_ids,
                    # Audio features (if present in prompt output)
                    "input_features": input_features,
                    # Metadata
                    "source_lang": entry["source"]["lang"],
                }
            )

    dataset = Dataset.from_list(data)
    print(f"Preprocessed {len(dataset)} ASR samples")

    if "input_features" in dataset.column_names:
        dataset = dataset.cast_column(
            "input_features", Sequence(Sequence(Value("float32")))
        )

    return dataset


def preprocess_st_dataset(manifest_path: str, processor, model_id: str):
    """
    Preprocess ST dataset - pre-tokenize prompts and full conversations.

    This moves tokenization OUT of the collator.
    """
    print(f"Preprocessing ST dataset from: {manifest_path}")
    root_dir = os.path.dirname(os.path.abspath(manifest_path))

    data = []
    with open(manifest_path, "r", encoding="utf-8") as f:
        lines = [json.loads(line.strip()) for line in f]

    # Process samples
    for entry in tqdm(lines, desc="Preprocessing ST"):
        audio_path = entry["source"]["audio_local_path"]
        if not os.path.isabs(audio_path):
            audio_path = os.path.join(root_dir, audio_path)
        audio_path = os.path.normpath(audio_path)

        src_lang = entry["source"]["lang"]
        target_text = entry["target"]["text"]

        # Build prompt
        prompt_messages = build_st_prompt(src_lang, audio_path)

        # Build full conversation
        full_messages = prompt_messages + [
            {
                "role": "assistant",
                "content": [{"type": "text", "text": target_text}],
            }
        ]

        # Tokenize prompt only
        prompt_tokens = processor.apply_chat_template(
            [prompt_messages],
            return_tensors="pt",
            tokenize=True,
            padding=False,
        )

        # Tokenize full conversation
        # We must add continue_final_message=True here because the conversation
        # ends with an 'assistant' role, which is typically not allowed for serving
        full_tokens = processor.apply_chat_template(
            [full_messages],
            return_tensors="pt",
            tokenize=True,
            padding=False,
            continue_final_message=True,
        )

        # Calculate prompt length for label masking
        prompt_len = (
            (prompt_tokens["input_ids"][0] != processor.tokenizer.pad_token_id)
            .sum()
            .item()
        )

        # Store preprocessed sample
        data.append(
            {
                "task": "st",
                "input_ids": full_tokens["input_ids"][0].tolist(),
                "attention_mask": full_tokens["attention_mask"][0].tolist(),
                "prompt_length": prompt_len,
                "input_features": np.zeros(
                    (1, 1), dtype=np.float32
                ).tolist(),  # for schema to match asr
                # Store other features if present
                **{
                    k: v[0].tolist() if torch.is_tensor(v[0]) else v[0]
                    for k, v in full_tokens.items()
                    if k not in ["input_ids", "attention_mask"]
                },
            }
        )

    dataset = Dataset.from_list(data)
    print(f"Preprocessed {len(dataset)} ST samples")
    dataset = dataset.cast_column(
        "input_features", Sequence(Sequence(Value("float32")))
    )
    return dataset
