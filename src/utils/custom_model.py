"""
Custom model for Voxtral with task token-based routing.
Main changes is to implement classification head for LID.
"""

import logging
import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import VoxtralForConditionalGeneration
from utils.constants import TASKTYPE2ID
from utils.loss import FocalLoss

logging.basicConfig(
    level=logging.INFO,  # or DEBUG for more verbose logs
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)

logger = logging.getLogger(__name__)  # module-level logger


class VoxtralForConditionalGenerationWithLID(VoxtralForConditionalGeneration):
    """Extends Voxtral to optionally include LID classification head."""

    def __init__(self, config, num_languages=None):
        super().__init__(config)

        if num_languages is not None:
            hidden_size = self.audio_tower.config.hidden_size
            self.lid_head = nn.Sequential(
                nn.Dropout(0.1),
                nn.Linear(hidden_size, hidden_size),
                nn.GELU(),
                nn.Dropout(0.1),
                nn.Linear(hidden_size, num_languages),
            )
            logger.info(f"✅ Initialized LID head with {num_languages} classes")
        else:
            self.lid_head = None

    def save_pretrained(self, save_directory, **kwargs):
        """Override to ensure LID head is saved."""
        super().save_pretrained(save_directory, **kwargs)
        logger.info("✅ Saved model with LID head")


class VoxtralWithTaskTokenRouting(nn.Module):
    """
    Voxtral with task routing.

    - Detects task token in input_ids
    - Routes LID to classification head
    - Routes others to full generative model
    """

    def __init__(
        self,
        base_model,
        lid_class_weights=None,
        use_focal_loss=False,
        focal_gamma=2.0,
    ):
        super().__init__()
        self.base_model = base_model
        self.lid_loss_weight = 1.0  # FIXME: to be customised next time
        self.gen_loss_weight = 1.0

        # Use LID head from base model
        assert (
            hasattr(base_model, "lid_head") and base_model.lid_head is not None
        ), "Base model must have lid_head for task routing"

        logger.info("✅ Using LID head from VoxtralForConditionalGenerationWithLID")

        # Create LID loss function
        if use_focal_loss:
            self.lid_loss_fn = FocalLoss(alpha=lid_class_weights, gamma=focal_gamma)
        else:
            # Store weights for cross entropy
            if lid_class_weights is not None:
                self.register_buffer("lid_class_weights", lid_class_weights)
            else:
                self.lid_class_weights = None

    def set_task_token_ids(self, task_token_ids):
        """Set task token IDs after tokenizer initialization."""
        self.task_token_ids = task_token_ids
        logger.info(f"Task token IDs set: {task_token_ids}")

    def _pool_audio(self, audio_features, audio_lengths, mel_lengths):
        """
        Pool audio features over time dimension using scaled lengths. Scaled lengths
        are required as the audio tower upsamples the time dimension.
        """
        T_enc = audio_features.size(1)
        dtype = audio_features.dtype  # Capture the model's dtype (BFloat16)
        scale = float(T_enc) / float(mel_lengths)
        logger.debug(f"LID scale factor: {scale:.3f}")

        enc_lengths = (audio_lengths.float() * scale).round().clamp(min=1).long()
        mask = (
            torch.arange(T_enc, device=audio_features.device)[None, :]
            < enc_lengths[:, None]
        ).to(dtype)

        pooled = (audio_features * mask.unsqueeze(-1)).sum(dim=1)
        return pooled / enc_lengths.unsqueeze(-1).to(dtype)

    def forward(
        self,
        input_features=None,
        input_ids=None,
        attention_mask=None,
        labels=None,
        source_lang=None,  # Ground truth for LID
        task_type=None,
        audio_lengths=None,
        **kwargs,
    ):
        """
        Forward pass with task token-based routing.
        """

        # Detect task from input_ids
        lid_mask = task_type == TASKTYPE2ID["lid"]
        gen_mask = (
            (task_type == TASKTYPE2ID["asr"])
            | (task_type == TASKTYPE2ID["s2tt"])
            | (task_type == TASKTYPE2ID["t2tt"])
        )

        has_lid = lid_mask.any().item()
        has_gen = gen_mask.any().item()

        # ✅ CRITICAL FIX: Initialize ALL possible keys upfront
        # This ensures DataParallel can gather outputs from all GPUs
        outputs = {
            "loss": None,
            "lid_loss": None,
            "lid_logits": None,
            "gen_loss": None,
            "logits": None,
        }

        # ====================================
        # Route 1: LID Classification
        # ====================================
        if has_lid:
            lid_indices = lid_mask.nonzero(as_tuple=True)[0]

            # Extract audio for LID samples
            lid_audio = input_features[lid_indices]

            # Forward through audio encoder
            audio_outputs = self.base_model.audio_tower(lid_audio)
            audio_features = (
                audio_outputs.last_hidden_state
            )  # [B_lid, Time frames, Hidden size]

            # Pool audio features
            pooled_features = self._pool_audio(
                audio_features,
                audio_lengths[lid_indices],
                mel_lengths=lid_audio.size(-1),
            )  # [B_lid, Hidden size]

            # Classification
            lid_logits = self.base_model.lid_head(
                pooled_features
            )  # [B_lid, num_languages]
            outputs["lid_logits"] = lid_logits

            # Compute loss if labels provided
            if source_lang is not None:
                lid_labels = source_lang[lid_indices]
                # Use configured loss function

                # ┌─────────────────────────────────────────────┐
                # │ TIER 1: Check for Focal Loss (Priority 1)   │
                # └─────────────────────────────────────────────┘
                if hasattr(self, "lid_loss_fn"):
                    lid_loss = self.lid_loss_fn(lid_logits, lid_labels)

                # ┌─────────────────────────────────────────────┐
                # │ TIER 2: Check for Class Weights (Priority 2)│
                # └─────────────────────────────────────────────┘
                elif self.lid_class_weights is not None:
                    lid_loss = F.cross_entropy(
                        lid_logits, lid_labels, weight=self.lid_class_weights
                    )
                # ┌─────────────────────────────────────────────┐
                # │ TIER 3: Fallback to Standard CE (Priority 3)│
                # └─────────────────────────────────────────────┘
                else:
                    lid_loss = F.cross_entropy(lid_logits, lid_labels)

                outputs["lid_loss"] = lid_loss

        # ====================================
        # Route 2: Generative Tasks
        # ====================================
        if has_gen:
            gen_indices = gen_mask.nonzero(as_tuple=True)[0]

            # Filter inputs for generative tasks
            gen_input_features = (
                input_features[gen_indices] if input_features is not None else None
            )
            gen_input_ids = input_ids[gen_indices]
            gen_attention_mask = attention_mask[gen_indices]
            gen_labels = labels[gen_indices] if labels is not None else None

            # Full generative forward pass
            gen_outputs = self.base_model(
                input_features=gen_input_features,
                input_ids=gen_input_ids,
                attention_mask=gen_attention_mask,
                labels=gen_labels,
                **kwargs,
            )

            outputs["gen_loss"] = gen_outputs.loss
            outputs["logits"] = gen_outputs.logits

        # ====================================
        # Combine losses
        # ====================================
        if has_lid and has_gen:
            outputs["loss"] = (
                self.lid_loss_weight * outputs["lid_loss"]
                + self.gen_loss_weight * outputs["gen_loss"]
            )
        elif has_lid:
            outputs["loss"] = outputs["lid_loss"]
        else:
            outputs["loss"] = outputs["gen_loss"]

        return outputs

    def _get_base_voxtral(self):
        """Recursively unwraps PeftModel to get the core Voxtral model."""
        model = self.base_model
        # PeftModel wraps the core model in .base_model.model
        if hasattr(model, "base_model") and hasattr(model.base_model, "model"):
            return model.base_model.model
        return model

    def _get_lid_head(self):
        return self._get_base_voxtral().lid_head

    def _get_audio_tower(self):
        return self._get_base_voxtral().audio_tower

    @torch.no_grad()
    def generate(
        self,
        input_ids=None,
        input_features=None,
        attention_mask=None,
        task_type=None,
        audio_lengths=None,
        return_lid_logits=False,
        **generation_kwargs,
    ):
        """
        Override generate to handle LID classification vs text generation.
        """
        lid_mask = task_type == TASKTYPE2ID["lid"]

        # If all samples are LID, do classification instead of generation
        if lid_mask.all():
            # All LID - return classification results
            audio_tower = self._get_audio_tower()
            audio_outputs = audio_tower(input_features)
            audio_features = audio_outputs.last_hidden_state

            # Pool audio features
            pooled_features = self._pool_audio(
                audio_features,
                audio_lengths,
                mel_lengths=input_features.size(-1),  # original mel length
            )  # [B_lid, Hidden size]

            lid_logits = self._get_lid_head()(pooled_features)
            if return_lid_logits:
                return lid_logits  # [B, num_languages]

            # Return predicted class IDs (mimicking generated token IDs)
            pred_classes = lid_logits.argmax(dim=-1, keepdim=True)  # [B, 1]

            return pred_classes

        elif lid_mask.any():
            # Mixed batch - not supported in generation, should batch separately
            raise ValueError(
                "Cannot mix LID and generative tasks in generate(). "
                "Please batch LID samples separately."
            )
        else:
            # All generative - use standard generation
            return self.base_model.generate(
                input_ids=input_ids,
                input_features=input_features,
                attention_mask=attention_mask,
                **generation_kwargs,
            )
