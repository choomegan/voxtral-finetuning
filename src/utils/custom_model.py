"""
Custom model for Voxtral with task token-based routing.
Main changes is to implement classification head for LID.
"""

import logging
import torch
import torch.nn as nn
import torch.nn.functional as F

logging.basicConfig(
    level=logging.INFO,  # or DEBUG for more verbose logs
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)

logger = logging.getLogger(__name__)  # module-level logger


class VoxtralWithTaskTokenRouting(nn.Module):
    """
    Voxtral with task token-based routing.

    - Detects task token in input_ids
    - Routes LID to classification head
    - Routes others to full generative model
    """

    def __init__(self, base_model, num_languages, hidden_size):
        super().__init__()
        self.base_model = base_model

        # LID classification head
        self.lid_head = nn.Sequential(
            nn.Dropout(0.1),
            nn.Linear(hidden_size, hidden_size),
            nn.GELU(),
            nn.Dropout(0.1),
            nn.Linear(hidden_size, num_languages),
        )

        # Store task token IDs for routing
        self.task_token_ids = None  # Will be set after tokenizer is ready

    def set_task_token_ids(self, task_token_ids):
        """Set task token IDs after tokenizer initialization."""
        self.task_token_ids = task_token_ids
        logger.info(f"Task token IDs set: {task_token_ids}")

    def detect_task_from_tokens(self, input_ids):
        """
        Detect task type from input_ids by looking for task tokens.

        Returns:
            task_mask: Boolean tensor [batch_size] indicating which samples are each task
        """
        batch_size = input_ids.size(0)

        # Create masks for each task
        lid_mask = torch.zeros(batch_size, dtype=torch.bool, device=input_ids.device)
        asr_mask = torch.zeros(batch_size, dtype=torch.bool, device=input_ids.device)
        st_mask = torch.zeros(batch_size, dtype=torch.bool, device=input_ids.device)
        t2t_mask = torch.zeros(batch_size, dtype=torch.bool, device=input_ids.device)

        for i in range(batch_size):
            # Check if any task token appears in this sample
            sample_tokens = input_ids[i]

            if self.task_token_ids["lid"] in sample_tokens:
                lid_mask[i] = True
            elif self.task_token_ids["asr"] in sample_tokens:
                asr_mask[i] = True
            elif self.task_token_ids["s2tt"] in sample_tokens:
                st_mask[i] = True
            elif self.task_token_ids["t2tt"] in sample_tokens:
                t2t_mask[i] = True
            else:
                # Fallback: if no task token found, assume generative task
                # You could also raise an error here
                logger.warning(f"No task token found in sample {i}, defaulting to ASR")
                asr_mask[i] = True

        return {
            "lid": lid_mask,
            "asr": asr_mask,
            "st": st_mask,
            "t2t": t2t_mask,
        }

    def forward(
        self,
        input_features=None,
        input_ids=None,
        attention_mask=None,
        labels=None,
        source_lang=None,  # Ground truth for LID
        **kwargs,
    ):
        """
        Forward pass with task token-based routing.
        """

        # Detect task from input_ids
        task_masks = self.detect_task_from_tokens(input_ids)

        lid_mask = task_masks["lid"]
        gen_mask = task_masks["asr"] | task_masks["st"] | task_masks["t2t"]

        has_lid = lid_mask.any().item()
        has_gen = gen_mask.any().item()

        outputs = {}

        # ====================================
        # Route 1: LID Classification
        # ====================================
        if has_lid:
            lid_indices = lid_mask.nonzero(as_tuple=True)[0]

            # Extract audio for LID samples
            lid_audio = input_features[lid_indices]

            # Forward through audio encoder
            audio_outputs = self.base_model.audio_tower(lid_audio)
            audio_features = audio_outputs.last_hidden_state  # [B_lid, T, H]

            # Pool over time dimension
            pooled_features = audio_features.mean(dim=1)  # [B_lid, H]

            # Classification
            lid_logits = self.lid_head(pooled_features)  # [B_lid, num_languages]

            # Compute loss if labels provided
            if source_lang is not None:
                lid_labels = source_lang[lid_indices]
                lid_loss = F.cross_entropy(lid_logits, lid_labels)
                outputs["lid_loss"] = lid_loss
                outputs["lid_logits"] = lid_logits
                outputs["lid_indices"] = lid_indices

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
            outputs["gen_indices"] = gen_indices

        # ====================================
        # Combine losses
        # ====================================
        if has_lid and has_gen:
            outputs["loss"] = outputs["lid_loss"] + outputs["gen_loss"]
        elif has_lid:
            outputs["loss"] = outputs["lid_loss"]
        else:
            outputs["loss"] = outputs["gen_loss"]

        return outputs

    @torch.no_grad()
    def generate(
        self,
        input_ids=None,
        input_features=None,
        attention_mask=None,
        **generation_kwargs,
    ):
        """
        Override generate to handle LID classification vs text generation.
        """
        # Detect task from input_ids
        task_masks = self.detect_task_from_tokens(input_ids)

        lid_mask = task_masks["lid"]

        # If all samples are LID, do classification instead of generation
        if lid_mask.all():
            # All LID - return classification results
            audio_outputs = self.base_model.audio_tower(input_features)
            audio_features = audio_outputs.last_hidden_state
            pooled_features = audio_features.mean(dim=1)
            lid_logits = self.lid_head(pooled_features)

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
