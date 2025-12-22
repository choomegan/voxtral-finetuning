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

from utils.chat_template_utils import (
    build_st_prompt,
    build_st_prompt_no_src_lang,
    build_t2t_prompt_no_src_lang,
)
from utils.constants import SRCLANG2ID, LANGCODE_MAP, TASKTYPE2ID

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
            logger.error("\n❌ Batch alignment error in %s:", context)
            logger.error(
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
        batch["task_type"] = torch.tensor(
            [TASKTYPE2ID["asr"]] * len(features), dtype=torch.long
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
            logger.error("\n❌ Error in ST collator during processing: %s", e)
            logger.error(
                "⏭️  Skipping entire ST batch (%s samples)", expected_batch_size
            )
            return None

        # Validate batch alignment
        if not validate_batch_alignment(
            model_inputs, expected_batch_size, context="ST features"
        ):
            logger.error(
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
            "task_type": torch.tensor(
                [TASKTYPE2ID["s2tt"]] * len(batch), dtype=torch.long
            ),
        }


class StreamingT2TCollator:
    """
    Text-to-text translation collator.
    Handles pure text translation without audio processing.
    """

    def __init__(
        self,
        processor,
        model_id: str = None,
        max_length: int = 512,
    ):
        self.processor = processor
        self.tokenizer = processor.tokenizer
        self.max_length = max_length
        self.pad_id = self.tokenizer.pad_token_id or self.tokenizer.eos_token_id

        # Find [/INST] token ID
        self.inst_end_token_id = 4
        logger.info("Using [/INST] token ID: %s", self.inst_end_token_id)

    def __call__(self, batch):
        if not batch:
            return None

        expected_batch_size = len(batch)

        try:
            # Build conversations for each sample
            conversations = []
            for entry in batch:
                src_lang = entry["source_lang"]
                src_text = entry["source_text"]
                tgt_text = entry["target_text"]

                # Build translation prompt
                prompt_message = build_t2t_prompt_no_src_lang(src_text)

                # Build full conversation
                full_messages = prompt_message + [
                    {
                        "role": "assistant",
                        "content": [{"type": "text", "text": tgt_text}],
                    }
                ]
                conversations.append(full_messages)

            # Process with chat template
            model_inputs = self.processor.apply_chat_template(
                conversations,
                return_tensors="pt",
                tokenize=True,
                padding="longest",
                continue_final_message=True,
            )

        except Exception as e:
            logger.error("\n❌ Error in T2T collator during processing: %s", e)
            logger.error(
                "⏭️  Skipping entire T2T batch (%s samples)", expected_batch_size
            )
            return None

        # Validate batch alignment
        if not validate_batch_alignment(
            model_inputs, expected_batch_size, context="T2T features"
        ):
            logger.error(
                "⏭️  Skipping entire T2T batch (%s samples)", expected_batch_size
            )
            return None

        input_ids = model_inputs["input_ids"]
        labels = input_ids.clone()

        # Mask prompt tokens (everything before [/INST])
        for i in range(input_ids.size(0)):
            inst_end_positions = (input_ids[i] == self.inst_end_token_id).nonzero(
                as_tuple=True
            )[0]

            if len(inst_end_positions) > 0:
                prompt_len = inst_end_positions[0].item() + 1
                labels[i, :prompt_len] = -100
            else:
                logger.error("Warning: [/INST] token not found in sample %s", i)
                labels[i, :] = -100

            # Mask padding
            labels[i][input_ids[i] == self.pad_id] = -100

        return {
            "input_ids": input_ids,
            "attention_mask": model_inputs["attention_mask"],
            "labels": labels,
            "input_features": None,  # No audio features for T2T
            "source_lang": torch.tensor(
                [SRCLANG2ID[f["source_lang"]] for f in batch], dtype=torch.long
            ),
            "task_type": torch.tensor(
                [TASKTYPE2ID["t2t"]] * len(batch), dtype=torch.long
            ),
        }


class StreamingMultiTaskCollator:
    """
    Multi-task collator that merges ASR and ST batches.
    Properly handles None returns from sub-collators.
    """

    def __init__(self, asr_collator, st_collator, t2t_collator):
        self.asr_collator = asr_collator
        self.st_collator = st_collator
        self.t2t_collator = t2t_collator
        self.pad_id = asr_collator.pad_id

    def __call__(self, features: List[Dict[str, Any]]) -> Dict[str, Any]:
        # Split by task
        asr_data = [f for f in features if f["task"] == "asr"]
        st_data = [f for f in features if f["task"] == "st"]
        t2t_data = [f for f in features if f["task"] == "t2t"]

        # Process each task
        asr_batch = self.asr_collator(asr_data) if asr_data else None
        st_batch = self.st_collator(st_data) if st_data else None
        t2t_batch = self.t2t_collator(t2t_data) if t2t_data else None

        # Collect valid batches (non-None, non-empty)
        valid_batches = []

        if self._is_valid_batch(asr_batch):
            valid_batches.append(("ASR", asr_batch))
        elif asr_data:
            logger.debug(f"⚠️  ASR collator returned None for {len(asr_data)} samples")

        if self._is_valid_batch(st_batch):
            valid_batches.append(("ST", st_batch))
        elif st_data:
            logger.debug(f"⚠️  ST collator returned None for {len(st_data)} samples")

        if self._is_valid_batch(t2t_batch):
            valid_batches.append(("T2T", t2t_batch))
        elif t2t_data:
            logger.debug(f"⚠️  T2T collator returned None for {len(t2t_data)} samples")

        # Handle empty case
        if len(valid_batches) == 0:
            logger.debug(
                f"⏭️  All sub-batches invalid. Input: ASR={len(asr_data)}, "
                f"ST={len(st_data)}, T2T={len(t2t_data)}"
            )
            return None

        # Single task - return directly
        if len(valid_batches) == 1:
            return valid_batches[0][1]

        # Multiple tasks - merge them
        try:
            task_names = [name for name, _ in valid_batches]
            batches = [batch for _, batch in valid_batches]

            merged = self._merge_batches(batches)

            # Validate merged batch
            expected_size = sum(b["input_ids"].size(0) for b in batches)
            actual_size = merged["input_ids"].size(0)

            if actual_size != expected_size:
                logger.warning(
                    f"⚠️  Merge size mismatch! Expected {expected_size}, got {actual_size}. "
                    f"Tasks: {task_names}"
                )
                return None

            return merged

        except Exception as e:
            logger.error(f"❌ Error merging batches: {e}")
            import traceback

            logger.debug(traceback.format_exc())
            return None

    def _is_valid_batch(self, batch):
        """Check if a batch is valid."""
        return (
            batch is not None
            and isinstance(batch, dict)
            and "input_ids" in batch
            and batch["input_ids"] is not None
            and batch["input_ids"].numel() > 0
        )

    def _merge_batches(self, batches: List[Dict]) -> Dict:
        """
        Merge multiple task batches into one.
        Handles audio/non-audio mixing correctly.
        """
        # Find max sequence length
        max_seq_len = max(b["input_ids"].size(1) for b in batches)

        # Helper to pad and concatenate tensors
        def pad_and_concat(key, fill_value):
            tensors = []
            for batch in batches:
                tensor = batch[key]
                if tensor.size(1) < max_seq_len:
                    tensor = torch.nn.functional.pad(
                        tensor, (0, max_seq_len - tensor.size(1)), value=fill_value
                    )
                tensors.append(tensor)
            return torch.cat(tensors, dim=0)

        # Merge text tensors
        merged = {
            "input_ids": pad_and_concat("input_ids", self.pad_id),
            "attention_mask": pad_and_concat("attention_mask", 0),
            "labels": pad_and_concat("labels", -100),
        }

        # Merge audio features - this is the tricky part
        merged["input_features"] = self._merge_audio_features(batches)

        # Merge metadata
        merged["source_lang"] = torch.cat([b["source_lang"] for b in batches], dim=0)
        merged["task_type"] = torch.cat([b["task_type"] for b in batches], dim=0)

        return merged

    def _merge_audio_features(self, batches: List[Dict]):
        """
        Merge audio features, handling cases where some batches have None.

        Cases:
        1. All batches have audio → concatenate
        2. Some have audio, some None → concatenate audio + create dummy for None
        3. All None → return None
        """
        # Separate batches with and without audio
        audio_batches = []
        non_audio_batches = []

        for batch in batches:
            if batch.get("input_features") is not None:
                audio_batches.append(batch)
            else:
                non_audio_batches.append(batch)

        # Case 3: No audio at all (all T2T)
        if not audio_batches:
            return None

        # Extract and validate audio shapes
        audio_features = [b["input_features"] for b in audio_batches]
        audio_shape = audio_features[0].shape[1:]  # (time, features) or similar

        # Verify all audio has same shape
        for i, af in enumerate(audio_features):
            if af.shape[1:] != audio_shape:
                raise ValueError(
                    f"Audio shape mismatch at index {i}! "
                    f"Expected {audio_shape}, got {af.shape[1:]}"
                )

        # Case 1: All batches have audio → simple concat
        if not non_audio_batches:
            return torch.cat(audio_features, dim=0)

        # Concatenate: real audio + dummy audio
        # IMPORTANT: Order must match the order in merged["input_ids"]
        # We need to interleave them correctly

        all_features = []
        for batch in batches:
            if batch.get("input_features") is not None:
                all_features.append(batch["input_features"])
            else:
                # Create dummy for this batch
                batch_size = batch["input_ids"].size(0)
                batch_dummy = torch.zeros(
                    batch_size,
                    *audio_shape,
                    dtype=audio_features[0].dtype,
                    device=audio_features[0].device,
                )
                all_features.append(batch_dummy)

        return torch.cat(all_features, dim=0)
