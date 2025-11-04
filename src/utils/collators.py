"""
Script for dataset collators for different tasks.
These collators assume that the datasets have already been preprocessed
"""

import torch
from typing import List, Dict, Any


class FastASRCollator:
    """
    Lightning-fast ASR collator - data is already preprocessed.
    Just needs to pad and stack tensors.
    """

    def __init__(self, processor):
        self.pad_id = processor.tokenizer.pad_token_id
        self.eos_id = processor.tokenizer.eos_token_id

    def __call__(self, features: List[Dict[str, Any]]) -> Dict[str, Any]:
        if not features:
            return {}

        B = len(features)

        # Calculate max lengths
        max_prompt_len = max(len(f["prompt_input_ids"]) for f in features)
        max_text_len = max(len(f["text_input_ids"]) for f in features)
        max_total_len = max_prompt_len + max_text_len + 1  # +1 for EOS

        # Pre-allocate tensors
        input_ids = torch.full((B, max_total_len), self.pad_id, dtype=torch.long)
        attention_mask = torch.zeros((B, max_total_len), dtype=torch.long)
        labels = torch.full((B, max_total_len), -100, dtype=torch.long)

        # Stack audio features if present
        input_features = []

        for i, f in enumerate(features):
            prompt_ids = f["prompt_input_ids"]
            text_ids = f["text_input_ids"]
            pl = len(prompt_ids)
            tl = len(text_ids)

            # Fill tensors
            input_ids[i, :pl] = torch.tensor(prompt_ids, dtype=torch.long)
            input_ids[i, pl : pl + tl] = torch.tensor(text_ids, dtype=torch.long)
            input_ids[i, pl + tl] = self.eos_id

            attention_mask[i, : pl + tl + 1] = 1

            labels[i, pl : pl + tl] = torch.tensor(text_ids, dtype=torch.long)
            labels[i, pl + tl] = self.eos_id

            # Collect audio features
            if f.get("input_features") is not None:
                input_features.append(f["input_features"])

        batch = {
            "input_ids": input_ids,
            "attention_mask": attention_mask,
            "labels": labels,
        }

        # Handle input_features efficiently
        if input_features:
            input_features = [
                torch.as_tensor(f["input_features"], dtype=torch.float32)
                for f in features
            ]
            batch["input_features"] = torch.stack(input_features)

        return batch


class FastSTCollator:
    """
    Lightning-fast ST collator - data is already preprocessed.
    Just needs to pad and create labels.
    """

    def __init__(self, processor):
        self.pad_id = (
            processor.tokenizer.pad_token_id or processor.tokenizer.eos_token_id
        )

    def __call__(self, features: List[Dict[str, Any]]) -> Dict[str, Any]:
        if not features:
            return {}

        # Find max length
        max_len = max(len(f["input_ids"]) for f in features)
        B = len(features)

        # Pre-allocate tensors
        input_ids = torch.full((B, max_len), self.pad_id, dtype=torch.long)
        attention_mask = torch.zeros((B, max_len), dtype=torch.long)
        labels = torch.full((B, max_len), -100, dtype=torch.long)

        for i, f in enumerate(features):
            seq_len = len(f["input_ids"])
            prompt_len = f["prompt_length"]

            # Fill tensors
            input_ids[i, :seq_len] = torch.tensor(f["input_ids"], dtype=torch.long)
            attention_mask[i, :seq_len] = torch.tensor(
                f["attention_mask"], dtype=torch.long
            )

            # Labels: mask prompt, learn on response
            labels[i, prompt_len:seq_len] = torch.tensor(
                f["input_ids"][prompt_len:seq_len], dtype=torch.long
            )

        batch = {
            "input_ids": input_ids,
            "attention_mask": attention_mask,
            "labels": labels,
        }

        return batch


class FastMultiTaskCollator:
    """
    Ultra-fast multitask collator using preprocessed data.
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
        if asr_batch is None:
            return st_batch
        if st_batch is None:
            return asr_batch

        # Merge batches (same logic as before)
        return self._merge_batches(asr_batch, st_batch)

    def _merge_batches(self, asr_batch: Dict, st_batch: Dict) -> Dict:
        """Merge ASR + ST batches with vectorized padding + concat."""
        B_asr, B_st = asr_batch["input_ids"].size(0), st_batch["input_ids"].size(0)
        max_seq_len = max(asr_batch["input_ids"].size(1), st_batch["input_ids"].size(1))

        def pad_and_cat(a, b, fill_value):
            if a.size(1) < max_seq_len:
                a = torch.nn.functional.pad(
                    a, (0, max_seq_len - a.size(1)), value=fill_value
                )
            if b.size(1) < max_seq_len:
                b = torch.nn.functional.pad(
                    b, (0, max_seq_len - b.size(1)), value=fill_value
                )
            return torch.cat([a, b], dim=0)

        merged_batch = {
            "input_ids": pad_and_cat(
                asr_batch["input_ids"], st_batch["input_ids"], self.pad_id
            ),
            "attention_mask": pad_and_cat(
                asr_batch["attention_mask"], st_batch["attention_mask"], 0
            ),
            "labels": pad_and_cat(asr_batch["labels"], st_batch["labels"], -100),
        }

        if "input_features" in asr_batch:
            asr_audio = asr_batch["input_features"]
            st_audio_padding = torch.zeros(
                (B_st, *asr_audio.shape[1:]),
                dtype=asr_audio.dtype,
                device=asr_audio.device,
            )
            merged_batch["input_features"] = torch.cat(
                [asr_audio, st_audio_padding], dim=0
            )

        return merged_batch
