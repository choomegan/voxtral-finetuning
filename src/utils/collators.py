"""
Script for dataset collators for different tasks.
These collators assume that the datasets have already been preprocessed
"""

import logging
import torch
from typing import List, Dict, Any
import numpy as np
import soundfile as sf
import librosa

from utils.chat_template_utils import build_st_prompt, build_st_prompt_no_src_lang
from concurrent.futures import ThreadPoolExecutor
from utils.constants import SRCLANG2ID, LANGCODE_MAP

logger = logging.getLogger(__name__)  # module-level logger


def validate_batch_alignment(batch_dict, expected_batch_size, context=""):
    """
    Validate that all tensors in a batch have matching batch dimensions.
    Returns True if valid, False if invalid.
    """
    for key, value in batch_dict.items():
        if not isinstance(value, torch.Tensor):
            continue

        actual_batch = value.size(0)
        if actual_batch != expected_batch_size:
            logger.debug("\n❌ Batch alignment error in %s:", context)
            logger.debug(
                "   %s: expected %s, got %s (shape: %s)",
                key,
                expected_batch_size,
                actual_batch,
                value.shape,
            )
            return False

    return True


class StreamingASRCollator:
    """
    Fixed ASR collator that creates labels aligned with input_ids.
    Skips entire batch on any alignment issues.
    """

    def __init__(
        self, processor, model_id, sample_rate=16000, lang=None, num_workers=4
    ):
        self.processor = processor
        self.tokenizer = self.processor.tokenizer
        self.model_id = model_id
        self.sample_rate = sample_rate
        self.lang = lang
        self.pad_id = self.tokenizer.pad_token_id
        self.eos_id = self.tokenizer.eos_token_id
        self.num_workers = num_workers

    def _load_and_resample(self, audio_path):
        """Load and resample a single audio file."""
        audio, sr = sf.read(audio_path)
        if sr != self.sample_rate:
            audio = librosa.resample(audio, orig_sr=sr, target_sr=self.sample_rate)
        return audio

    def __call__(self, features):
        if not features:
            return None  # Return None instead of {} to signal skip

        # Collect paths and texts
        audio_paths = [f["audio_path"] for f in features]
        texts = [f["text"] for f in features]
        if not self.lang:  # assume that language is provided in the per-row manifest
            source_langs = [LANGCODE_MAP[f["source_lang"]] for f in features]
        else:
            source_langs = None

        expected_batch_size = len(features)

        try:
            audios = []
            for audio_path in audio_paths:
                audios.append(self._load_and_resample(audio_path))

            # Get prompt tokens + audio features
            prompt = self.processor.apply_transcription_request(
                language=self.lang if self.lang else source_langs,
                model_id=self.model_id,
                audio=audios,
                format=["WAV"] * len(audios),
                return_tensors="pt",
            )

        except Exception as e:
            logger.error("\n❌ Error in ASR collator during audio processing: %s", e)
            logger.error(
                "⏭️  Skipping entire ASR batch (%s samples)", expected_batch_size
            )
            return None

        # Validate batch alignment
        if not validate_batch_alignment(
            prompt, expected_batch_size, context="ASR features"
        ):
            logger.debug(
                f"⏭️  Skipping entire ASR batch (%s samples)", expected_batch_size
            )
            return None

        prompt_ids = prompt["input_ids"]  # [B, prompt_len]
        prompt_attn = prompt["attention_mask"]  # [B, prompt_len]

        # Tokenize target text (no padding yet)
        text_tokens = self.tokenizer(
            texts,
            add_special_tokens=False,
            padding=False,
            truncation=True,
            max_length=256,
            return_tensors=None,  # Return lists
        )
        text_ids_list = text_tokens["input_ids"]

        # Build input_ids and labels WITH SAME LENGTH
        B = len(texts)

        # Calculate actual prompt lengths (exclude padding)
        prompt_lens = prompt_attn.sum(dim=1).tolist()
        text_lens = [len(t) for t in text_ids_list]

        # Max length: prompt + text + EOS
        max_len = max(pl + tl + 1 for pl, tl in zip(prompt_lens, text_lens))

        # Pre-allocate tensors
        input_ids = torch.full((B, max_len), self.pad_id, dtype=torch.long)
        attention_mask = torch.zeros((B, max_len), dtype=torch.long)
        labels = torch.full((B, max_len), -100, dtype=torch.long)

        # Fill tensors
        for i in range(B):
            pl = prompt_lens[i]
            tl = text_lens[i]

            # Copy prompt
            input_ids[i, :pl] = prompt_ids[i, :pl]
            attention_mask[i, :pl] = 1
            # labels[i, :pl] stays -100 (ignore prompt in loss)

            # Copy text
            text_tensor = torch.tensor(text_ids_list[i], dtype=torch.long)
            input_ids[i, pl : pl + tl] = text_tensor
            attention_mask[i, pl : pl + tl] = 1
            labels[i, pl : pl + tl] = text_tensor  # Learn text tokens

            # Add EOS
            input_ids[i, pl + tl] = self.eos_id
            attention_mask[i, pl + tl] = 1
            labels[i, pl + tl] = self.eos_id

        batch = {
            "input_ids": input_ids,
            "attention_mask": attention_mask,
            "labels": labels,
            "input_features": prompt.get("input_features", None),
        }

        # Convert input_features to tensor if needed
        if batch["input_features"] is not None:
            if isinstance(batch["input_features"], np.ndarray):
                batch["input_features"] = torch.tensor(
                    batch["input_features"], dtype=torch.float32
                )
            elif isinstance(batch["input_features"], list):
                batch["input_features"] = torch.tensor(
                    np.stack(batch["input_features"]), dtype=torch.float32
                )

        batch["source_lang"] = torch.tensor(
            [SRCLANG2ID[f["source_lang"]] for f in features], dtype=torch.long
        )
        return batch


class StreamingSTCollator:
    """
    ST collator - skips entire batch on alignment issues.
    """

    def __init__(
        self,
        processor,
        model_id: str,
        incl_src_lang: bool = True,
        max_length: int = 512,
    ):
        self.processor = processor
        self.model_id = model_id
        self.tokenizer = processor.tokenizer
        self.incl_src_lang = incl_src_lang

        self.max_length = max_length
        self.pad_id = (
            processor.tokenizer.pad_token_id or processor.tokenizer.eos_token_id
        )
        #  Find the [/INST] token ID (marks end of instruction, start of response)
        self.inst_end_token_id = 4
        logger.info("Using [/INST] token ID: %s", self.inst_end_token_id)

        # Verify it decodes correctly
        decoded = self.tokenizer.decode([self.inst_end_token_id])
        logger.info(
            "Token %s decodes to: '%s'",
            self.inst_end_token_id,
            decoded,
        )

    def __call__(self, batch):
        if not batch:
            return None  # Return None instead of {}

        expected_batch_size = len(batch)

        try:
            # Collect all prompts and conversations
            full_conversations = []

            for entry in batch:
                audio_path = entry["audio_path"]
                src_lang = entry["source_lang"]
                tgt_text = entry["target_text"]

                # Build prompt
                prompt_messages = (
                    build_st_prompt(src_lang, audio_path)
                    if self.incl_src_lang
                    else build_st_prompt_no_src_lang(audio_path)
                )

                # Build full conversation
                full_messages = prompt_messages + [
                    {
                        "role": "assistant",
                        "content": [{"type": "text", "text": tgt_text}],
                    }
                ]
                full_conversations.append(full_messages)

            # Process only ONCE
            model_inputs = self.processor.apply_chat_template(
                full_conversations,
                return_tensors="pt",
                tokenize=True,
                padding="longest",
                continue_final_message=True,
            )

        except Exception as e:
            logger.debug("\n❌ Error in ST collator during processing: %s", e)
            logger.debug(
                "⏭️  Skipping entire ST batch (%s samples)", expected_batch_size
            )
            return None

        # Validate batch alignment
        if not validate_batch_alignment(
            model_inputs, expected_batch_size, context="ST features"
        ):
            logger.debug(
                "⏭️  Skipping entire ST batch (%s samples)", expected_batch_size
            )
            return None

        input_ids = model_inputs["input_ids"]
        labels = input_ids.clone()

        # Find [/INST] token to determine where assistant response starts
        for i in range(input_ids.size(0)):
            # Find position of [/INST] token
            inst_end_positions = (input_ids[i] == self.inst_end_token_id).nonzero(
                as_tuple=True
            )[0]

            if len(inst_end_positions) > 0:
                # Mask everything up to and including [/INST]
                prompt_len = inst_end_positions[0].item() + 1
                labels[i, :prompt_len] = -100
            else:
                # Fallback: if [/INST] not found, mask everything (shouldn't happen)
                logger.error("Warning: [/INST] token not found in sample %s", i)
                labels[i, :] = -100

            # Mask padding
            labels[i][input_ids[i] == self.pad_id] = -100

        return {
            "input_ids": input_ids,
            "attention_mask": model_inputs["attention_mask"],
            "labels": labels,
            "input_features": model_inputs.get("input_features"),
            "source_lang": torch.tensor(
                [SRCLANG2ID[f["source_lang"]] for f in batch], dtype=torch.long
            ),
        }


class StreamingMultiTaskCollator:
    """
    Multi-task collator that merges ASR and ST batches.
    Properly handles None returns from sub-collators.
    """

    def __init__(self, asr_collator, st_collator):
        self.asr_collator = asr_collator
        self.st_collator = st_collator
        self.pad_id = asr_collator.pad_id

    def __call__(self, features: List[Dict[str, Any]]) -> Dict[str, Any]:
        # Split by task
        asr_data = [f for f in features if f["task"] == "asr"]
        st_data = [f for f in features if f["task"] == "st"]

        # Process each task
        asr_batch = self.asr_collator(asr_data) if asr_data else None
        st_batch = self.st_collator(st_data) if st_data else None

        # Check for None (skipped batches)
        asr_is_valid = (
            asr_batch is not None
            and isinstance(asr_batch, dict)
            and "input_ids" in asr_batch
            and asr_batch["input_ids"] is not None
            and asr_batch["input_ids"].numel() > 0
        )

        st_is_valid = (
            st_batch is not None
            and isinstance(st_batch, dict)
            and "input_ids" in st_batch
            and st_batch["input_ids"] is not None
            and st_batch["input_ids"].numel() > 0
        )

        # Handle cases where one or both batches are invalid
        if not asr_is_valid and not st_is_valid:
            logger.info("⏭️  Both ASR and ST batches invalid, skipping entire batch")
            return None

        if not asr_is_valid:
            return st_batch

        if not st_is_valid:
            return asr_batch

        # Both batches valid - merge them
        try:
            merged = self._merge_batches(asr_batch, st_batch)

            # Final validation of merged batch
            total_samples = len(asr_data) + len(st_data)
            if not validate_batch_alignment(
                merged, total_samples, context="Multi-task merged batch"
            ):
                logger.info("⏭️  Merged batch validation failed, skipping")
                return None

            return merged

        except Exception as e:
            logger.info("\n❌ Error merging batches: %s", e)
            logger.info("⏭️  Skipping merged batch")
            return None

    def _merge_batches(self, asr_batch: Dict, st_batch: Dict) -> Dict:
        max_seq_len = max(asr_batch["input_ids"].size(1), st_batch["input_ids"].size(1))

        def pad_and_cat(key, fill_value):
            asr_tensor = asr_batch[key]
            st_tensor = st_batch[key]

            if asr_tensor.size(1) < max_seq_len:
                asr_tensor = torch.nn.functional.pad(
                    asr_tensor, (0, max_seq_len - asr_tensor.size(1)), value=fill_value
                )

            if st_tensor.size(1) < max_seq_len:
                st_tensor = torch.nn.functional.pad(
                    st_tensor, (0, max_seq_len - st_tensor.size(1)), value=fill_value
                )

            return torch.cat([asr_tensor, st_tensor], dim=0)

        merged_batch = {
            "input_ids": pad_and_cat("input_ids", self.pad_id),
            "attention_mask": pad_and_cat("attention_mask", 0),
            "labels": pad_and_cat("labels", -100),
        }

        # Both batches have input_features -> just concat
        if "input_features" in asr_batch and "input_features" in st_batch:
            asr_audio = asr_batch["input_features"]
            st_audio = st_batch["input_features"]

            # Verify shapes match (they should, since same model processes both)
            if asr_audio.shape[1:] != st_audio.shape[1:]:
                raise ValueError(
                    f"Audio feature shape mismatch! "
                    f"ASR: {asr_audio.shape}, ST: {st_audio.shape}"
                )

            # Simply concatenate real audio features
            merged_batch["input_features"] = torch.cat([asr_audio, st_audio], dim=0)

        elif "input_features" in asr_batch:
            merged_batch["input_features"] = asr_batch["input_features"]
        elif "input_features" in st_batch:
            merged_batch["input_features"] = st_batch["input_features"]

        # Merge source_lang as simple concatenation
        merged_batch["source_lang"] = torch.cat(
            [asr_batch["source_lang"], st_batch["source_lang"]], dim=0
        )
        return merged_batch
