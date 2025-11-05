"""
Script for dataset collators for different tasks.
These collators assume that the datasets have already been preprocessed
"""

import torch
from typing import List, Dict, Any
import numpy as np
import soundfile as sf
import librosa

from utils.chat_template_utils import build_st_prompt

LANGCODE_MAP = {  # from 3-letter ISO code to 2-letter code
    "en": "en",
    "zsm": "ms",
    "ind": "id",
}


class StreamingASRCollator:
    """
    Fixed ASR collator that creates labels aligned with input_ids.
    """

    def __init__(self, processor, model_id, sample_rate=16000, lang=None):
        self.processor = processor
        self.tokenizer = self.processor.tokenizer
        self.model_id = model_id
        self.sample_rate = sample_rate
        self.lang = lang
        self.pad_id = self.tokenizer.pad_token_id
        self.eos_id = self.tokenizer.eos_token_id

    def __call__(self, features):
        if not features:
            return {}

        # Collect paths and texts
        audio_paths = [f["audio_path"] for f in features]
        texts = [f["text"] for f in features]
        if not self.lang:  # assume that language is provied in the per-row manifest
            source_langs = [LANGCODE_MAP[f["source_lang"]] for f in features]
        else:
            source_langs = None

        # Load and resample audio
        audios = []
        for p in audio_paths:
            audio, sr = sf.read(p)
            if sr != self.sample_rate:
                audio = librosa.resample(audio, orig_sr=sr, target_sr=self.sample_rate)
            audios.append(audio)

        # Get prompt tokens + audio features
        prompt = self.processor.apply_transcription_request(
            language=self.lang if self.lang else source_langs,
            model_id=self.model_id,
            audio=audios,
            format=["WAV"] * len(audios),
            return_tensors="pt",
        )

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

        return batch


class StreamingSTCollator:
    """
    ST collator - already correct, but cleaned up for consistency.
    """

    def __init__(self, processor, model_id: str, max_length: int = 512):
        self.processor = processor
        self.model_id = model_id
        self.max_length = max_length
        self.pad_id = (
            processor.tokenizer.pad_token_id or processor.tokenizer.eos_token_id
        )

    def __call__(self, batch):
        if not batch:
            return {}

        data = []

        for entry in batch:
            audio_path = entry["audio_path"]
            src_lang = entry["source_lang"]
            tgt_text = entry["target_text"]

            # Build prompt
            prompt_messages = build_st_prompt(src_lang, audio_path)

            # Build full conversation
            full_messages = prompt_messages + [
                {
                    "role": "assistant",
                    "content": [{"type": "text", "text": tgt_text}],
                }
            ]

            # Tokenize prompt
            prompt_tokens = self.processor.apply_chat_template(
                [prompt_messages],
                return_tensors="pt",
                tokenize=True,
                padding=False,
            )

            # Tokenize full conversation
            full_tokens = self.processor.apply_chat_template(
                [full_messages],
                return_tensors="pt",
                tokenize=True,
                padding=False,
                continue_final_message=True,
            )

            # Get 1D tensors
            prompt_input_ids = prompt_tokens["input_ids"].squeeze(0)
            prompt_attn = prompt_tokens["attention_mask"].squeeze(0)
            full_input_ids = full_tokens["input_ids"].squeeze(0)
            full_attn = full_tokens["attention_mask"].squeeze(0)

            # Calculate prompt length
            prompt_len = (prompt_attn != 0).sum().item()

            # Create labels (same length as input_ids)
            labels = full_input_ids.clone()
            labels[:prompt_len] = -100  # Mask prompt

            data.append(
                {
                    "input_ids": full_input_ids,
                    "attention_mask": full_attn,
                    "labels": labels,
                }
            )

        # Pad all sequences to same length
        input_ids = torch.nn.utils.rnn.pad_sequence(
            [d["input_ids"] for d in data], batch_first=True, padding_value=self.pad_id
        )
        attention_mask = torch.nn.utils.rnn.pad_sequence(
            [d["attention_mask"] for d in data], batch_first=True, padding_value=0
        )
        labels = torch.nn.utils.rnn.pad_sequence(
            [d["labels"] for d in data], batch_first=True, padding_value=-100
        )

        return {
            "input_ids": input_ids,
            "attention_mask": attention_mask,
            "labels": labels,
        }


class StreamingMultiTaskCollator:
    """
    Multi-task collator that merges ASR and ST batches.
    Now both tasks have properly aligned labels.
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

        # Handle edge cases
        if asr_batch is None and st_batch is None:
            return {}
        if asr_batch is None:
            return st_batch
        if st_batch is None:
            return asr_batch

        # Merge batches
        return self._merge_batches(asr_batch, st_batch)

    def _merge_batches(self, asr_batch: Dict, st_batch: Dict) -> Dict:
        """
        Merge ASR + ST batches.
        Now both have input_ids.shape == labels.shape, so this works correctly.
        """
        B_asr = asr_batch["input_ids"].size(0)
        B_st = st_batch["input_ids"].size(0)

        # Find max sequence length
        max_seq_len = max(asr_batch["input_ids"].size(1), st_batch["input_ids"].size(1))

        def pad_and_cat(key, fill_value):
            """Pad both tensors to max_seq_len and concatenate."""
            asr_tensor = asr_batch[key]
            st_tensor = st_batch[key]

            # Pad ASR
            if asr_tensor.size(1) < max_seq_len:
                asr_tensor = torch.nn.functional.pad(
                    asr_tensor, (0, max_seq_len - asr_tensor.size(1)), value=fill_value
                )

            # Pad ST
            if st_tensor.size(1) < max_seq_len:
                st_tensor = torch.nn.functional.pad(
                    st_tensor, (0, max_seq_len - st_tensor.size(1)), value=fill_value
                )

            return torch.cat([asr_tensor, st_tensor], dim=0)

        # Merge token tensors
        merged_batch = {
            "input_ids": pad_and_cat("input_ids", self.pad_id),
            "attention_mask": pad_and_cat("attention_mask", 0),
            "labels": pad_and_cat("labels", -100),
        }

        # Handle audio features
        if "input_features" in asr_batch and asr_batch["input_features"] is not None:
            asr_audio = asr_batch["input_features"]

            # Create zero padding for ST samples
            st_audio_padding = torch.zeros(
                (B_st, *asr_audio.shape[1:]),
                dtype=asr_audio.dtype,
                device=asr_audio.device,
            )

            merged_batch["input_features"] = torch.cat(
                [asr_audio, st_audio_padding], dim=0
            )

        # Debug: Verify shapes match
        assert (
            merged_batch["input_ids"].shape == merged_batch["labels"].shape
        ), f"Shape mismatch: input_ids {merged_batch['input_ids'].shape} vs labels {merged_batch['labels'].shape}"

        return merged_batch
