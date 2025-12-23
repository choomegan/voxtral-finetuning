"""
Script for dataset collators for different tasks.
These collators assume that the datasets have already been preprocessed
"""

from collections import defaultdict
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


class BaseChatCollator:
    """
    Parent class for ST and T2T tasks.
    Handles Chat Template application, Tokenization, and Label Masking.
    """

    def __init__(self, processor, model_id, inst_token_id=4):
        self.processor = processor
        self.tokenizer = processor.tokenizer
        self.model_id = model_id
        self.pad_id = self.tokenizer.pad_token_id or self.tokenizer.eos_token_id
        self.inst_end_token_id = inst_token_id

        # Validation
        decoded = self.tokenizer.decode([self.inst_end_token_id])
        logger.info(
            f"Initialized {self.__class__.__name__}. Token {self.inst_end_token_id} = '{decoded}'"
        )

    def build_conversations(self, batch):
        """Child classes must implement this to return a list of message lists."""
        raise NotImplementedError

    def __call__(self, batch):
        if not batch:
            return None

        try:
            # 1. Build Conversations (Child logic)
            conversations = self.build_conversations(batch)

            # 2. Apply Template (Shared logic)
            model_inputs = self.processor.apply_chat_template(
                conversations,
                return_tensors="pt",
                tokenize=True,
                padding="longest",
                continue_final_message=True,
            )

            # 3. Validate
            if not validate_batch_alignment(
                model_inputs, len(batch), context=self.__class__.__name__
            ):
                logger.error("⏭️  Skipping entire batch (%s samples)")
                return None

            # 4. Create Labels (Shared logic)
            input_ids = model_inputs["input_ids"]
            labels = self._mask_chat_labels(
                input_ids, self.pad_id, self.inst_end_token_id
            )

            # 5. Return Standardized Dict
            return {
                "input_ids": input_ids,
                "attention_mask": model_inputs["attention_mask"],
                "labels": labels,
                # ST will have input_features, T2T will have None. Safe .get() handles both.
                "input_features": model_inputs.get("input_features"),
                "source_lang": torch.tensor(
                    [SRCLANG2ID[f["source_lang"]] for f in batch], dtype=torch.long
                ),
                "task_type": torch.tensor(
                    [self.TASK_ID] * len(batch), dtype=torch.long
                ),
            }

        except Exception as e:
            logger.error(f"❌ Error in {self.__class__.__name__}: {e}")
            return None

    def _mask_chat_labels(self, input_ids, pad_id, inst_end_token_id):
        """
        Creates labels for chat-based tasks.
        Masks everything up to [/INST] and masks padding.
        """
        labels = input_ids.clone()

        # 1. Mask Padding
        labels[input_ids == pad_id] = -100

        # 2. Mask Prompt (Instruction)
        # Find position of [/INST] token
        for i in range(input_ids.size(0)):
            inst_end_positions = (input_ids[i] == inst_end_token_id).nonzero(
                as_tuple=True
            )[0]

            if len(inst_end_positions) > 0:
                # Mask everything up to and including [/INST]
                prompt_len = inst_end_positions[0].item() + 1
                labels[i, :prompt_len] = -100
            else:
                logger.warning(
                    f"Warning: Instruction end token {inst_end_token_id} not found in sample {i}"
                )
                # Fallback: Mask everything to avoid training on garbage
                labels[i, :] = -100

        return labels


class StreamingT2TCollator(BaseChatCollator):
    TASK_ID = TASKTYPE2ID["t2t"]

    def build_conversations(self, batch):
        conversations = []
        for entry in batch:
            prompt_msg = build_t2t_prompt_no_src_lang(entry["source_text"])
            conversations.append(
                prompt_msg
                + [
                    {
                        "role": "assistant",
                        "content": [{"type": "text", "text": entry["target_text"]}],
                    }
                ],
            )
        return conversations


class StreamingSTCollator(BaseChatCollator):
    """
    Speech Translation Collator that inherits from BaseChatCollator.
    """

    TASK_ID = TASKTYPE2ID["s2tt"]

    def __init__(self, processor, model_id, incl_src_lang=True, **kwargs):
        super().__init__(processor, model_id, **kwargs)

        self.incl_src_lang = incl_src_lang

    def build_conversations(self, batch):
        """
        Build speech translation conversation with or without source language
        """
        conversations = []
        for entry in batch:
            # Logic specific to ST prompt building
            prompt_msg = (
                build_st_prompt(entry["source_lang"], entry["audio_path"])
                if self.incl_src_lang
                else build_st_prompt_no_src_lang(entry["audio_path"])
            )

            conversations.append(
                prompt_msg
                + [
                    {
                        "role": "assistant",
                        "content": [{"type": "text", "text": entry["target_text"]}],
                    }
                ]
            )
        return conversations


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

        try:
            # Collect paths and texts
            audio_paths = [f["audio_path"] for f in features]
            texts = [f["text"] for f in features]
            if (
                not self.lang
            ):  # assume that language is provided in the per-row manifest
                source_langs = [LANGCODE_MAP[f["source_lang"]] for f in features]
            else:
                source_langs = None

            expected_batch_size = len(features)

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

            # We explicitly cast to tensors here to clean up the main logic flow
            feat = prompt.get("input_features")
            if feat is not None and not isinstance(feat, torch.Tensor):
                feat = torch.tensor(
                    np.stack(feat) if isinstance(feat, list) else feat,
                    dtype=torch.float32,
                )

            return {
                "input_ids": input_ids,
                "attention_mask": attention_mask,
                "labels": labels,
                "input_features": feat,
                "source_lang": torch.tensor(
                    [SRCLANG2ID[f["source_lang"]] for f in features], dtype=torch.long
                ),
                "task_type": torch.tensor(
                    [TASKTYPE2ID["asr"]] * len(features), dtype=torch.long
                ),
            }

        except Exception as e:
            logger.error(f"❌ Error in ASR Collator: {e}")
            return None


class StreamingMultiTaskCollator:
    """
    Multi-task collator that merges ASR and ST batches.
    Properly handles None returns from sub-collators.
    """

    def __init__(self, asr_collator, st_collator, t2t_collator):
        self.task_collators = {
            "asr": asr_collator,
            "st": st_collator,
            "t2t": t2t_collator,
        }
        self.pad_id = asr_collator.pad_id

    def __call__(self, features: List[Dict[str, Any]]) -> Dict[str, Any]:
        # 2. Group inputs by task (Single pass grouping)
        task_inputs = defaultdict(list)
        for f in features:
            task_inputs[f["task"]].append(f)

        # 3. Process sub-batches dynamically
        valid_batches = []
        for task_name, collator in self.task_collators.items():
            if task_name not in task_inputs:
                continue

            # Run the specific collator
            batch = collator(task_inputs[task_name])

            if self._is_valid_batch(batch):
                valid_batches.append(batch)
            else:
                logger.debug(
                    f"⚠️  {task_name.upper()} collator skipped (invalid/empty result)."
                )

        # 4. Handle results
        if not valid_batches:
            logger.debug("⏭️  All sub-batches invalid.")
            return None

        if len(valid_batches) == 1:
            return valid_batches[0]

        try:
            return self._merge_batches(valid_batches)
        except Exception as e:
            logger.error(f"❌ Error merging batches: {e}", exc_info=True)
            return None

    @staticmethod
    def _is_valid_batch(batch):
        """Check if a batch is valid."""
        return (
            batch is not None
            and isinstance(batch, dict)
            and "input_ids" in batch
            and batch["input_ids"] is not None
            and batch["input_ids"].numel() > 0
        )

    @staticmethod
    def _pad_and_concat(batches, key, fill_value, max_seq_len):
        """
        Helper to pad and concatenate tensors
        """

        tensors = []
        for batch in batches:
            tensor = batch[key]
            if tensor.size(1) < max_seq_len:
                tensor = torch.nn.functional.pad(
                    tensor, (0, max_seq_len - tensor.size(1)), value=fill_value
                )
            tensors.append(tensor)
        return torch.cat(tensors, dim=0)

    def _merge_batches(self, batches: List[Dict]) -> Dict:
        """
        Merge multiple task batches into one.
        Handles audio/non-audio mixing correctly.
        """
        # Find max sequence length
        max_seq_len = max(b["input_ids"].size(1) for b in batches)

        merged = {
            "input_ids": self._pad_and_concat(
                batches, "input_ids", self.pad_id, max_seq_len
            ),
            "attention_mask": self._pad_and_concat(
                batches, "attention_mask", 0, max_seq_len
            ),
            "labels": self._pad_and_concat(batches, "labels", -100, max_seq_len),
            # Simplified audio merging call
            "input_features": self._merge_audio_features(batches),
            "source_lang": torch.cat([b["source_lang"] for b in batches], dim=0),
            "task_type": torch.cat([b["task_type"] for b in batches], dim=0),
        }
        return merged

    def _merge_audio_features(self, batches: List[Dict]):
        """
        Streamlined audio merger. Finds a reference tensor and fills gaps with zeros.
        """
        # 1. Find a reference audio tensor to get shape/dtype/device
        ref_batch = next(
            (b for b in batches if b.get("input_features") is not None), None
        )

        # Case: No audio in ANY batch (Pure Text tasks)
        if ref_batch is None:
            return None

        ref_features = ref_batch["input_features"]
        audio_shape = ref_features.shape[1:]

        # 2. Build the list, creating dummy zeros where needed
        all_features = []
        for batch in batches:
            feat = batch.get("input_features")

            if feat is not None:
                # Shape validation
                if feat.shape[1:] != audio_shape:
                    raise ValueError(
                        f"Audio shape mismatch! Got {feat.shape[1:]}, expected {audio_shape}"
                    )
                all_features.append(feat)
            else:
                # Create Dummy (zeros)
                dummy = torch.zeros(
                    (batch["input_ids"].size(0), *audio_shape),
                    dtype=ref_features.dtype,
                    device=ref_features.device,
                )
                all_features.append(dummy)

        return torch.cat(all_features, dim=0)
