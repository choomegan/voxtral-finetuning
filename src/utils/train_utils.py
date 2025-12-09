"""
Trainer utils
"""

import logging
import traceback

import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from transformers import Trainer

from utils.constants import SRCLANG2ID, TASKTYPE2ID

logger = logging.getLogger(__name__)  # recommended at module level


class _FilteredDataLoader:
    """
    Wrapper that filters out None batches returned by collator.
    This prevents None from reaching the trainer's training loop.
    """

    def __init__(self, dataloader, skip_counter_callback=None):
        self.dataloader = dataloader
        self.skip_counter_callback = skip_counter_callback

    def __iter__(self):
        for batch in self.dataloader:
            if batch is None:
                # Increment skip counter if callback provided
                if self.skip_counter_callback:
                    self.skip_counter_callback()
                # Skip this batch by not yielding it
                continue
            yield batch

    def __len__(self):
        return len(self.dataloader)

    def __getattr__(self, name):
        # Forward all other attribute access to the underlying dataloader
        return getattr(self.dataloader, name)


class UncertaintyModule(nn.Module):
    """
    Separate module for uncertainty parameters.
    This will be wrapped by DDP independently.
    """

    def __init__(self):
        super().__init__()
        self.log_var_asr = nn.Parameter(torch.zeros(1))
        self.log_var_st = nn.Parameter(torch.zeros(1))


class SafeTrainer(Trainer):
    """
    Custom trainer that:
    1. Handles multiple eval datasets
    2. Properly handles None returns from collators (skipped batches)
    """

    def __init__(self, *args, lang_weight_map=None, use_uncertainty=False, **kwargs):
        # Store flags
        self.use_uncertainty = use_uncertainty
        self.use_lang_weighting = lang_weight_map is not None

        # Create uncertainty module BEFORE super().__init__
        if self.use_uncertainty:
            self.uncertainty_module = UncertaintyModule()
            logger.info("✓ Created uncertainty module")
        else:
            self.uncertainty_module = None

        # Call parent init
        super().__init__(*args, **kwargs)

        # Initialize counters
        self.skipped_batches_train = 0
        self.skipped_batches_eval = 0
        self.total_train_batches = 0
        self.total_eval_batches = 0

        # Setup language weighting
        if self.use_lang_weighting:

            def _convert_lang_weight_map(lang_weight_map):
                num_langs = len(SRCLANG2ID)
                vec = torch.zeros(num_langs, dtype=torch.float32)
                for lang, weight in lang_weight_map.items():
                    lang_id = SRCLANG2ID[lang]
                    vec[lang_id] = weight
                return vec

            self.lang_weight_map = _convert_lang_weight_map(lang_weight_map)
            logger.info("✓ Language class weighting enabled")
        else:
            self.lang_weight_map = None

        # Move uncertainty module to device and wrap with Accelerator
        if self.use_uncertainty:
            device = next(self.model.parameters()).device
            self.uncertainty_module = self.uncertainty_module.to(device)

            # Wrap with Accelerator if available
            if hasattr(self, "accelerator") and self.accelerator is not None:
                self.uncertainty_module = self.accelerator.prepare(
                    self.uncertainty_module
                )
                logger.info("✓ Uncertainty module wrapped by Accelerator")

            logger.info("✓ Uncertainty weighting enabled")

        # Log configuration
        if self.use_uncertainty and self.use_lang_weighting:
            logger.warning("⚠️  Both uncertainty AND language weighting enabled")
        elif not self.use_uncertainty and not self.use_lang_weighting:
            logger.info("Using standard cross-entropy loss")

    def create_optimizer(self):
        """
        Override to add uncertainty parameters to optimizer.
        """
        if not self.use_uncertainty:
            return super().create_optimizer()

        # Get model (might be wrapped by DDP/Accelerator)
        opt_model = self.model_wrapped if hasattr(self, "model_wrapped") else self.model

        # Get optimizer class and kwargs from parent
        optimizer_cls, optimizer_kwargs = Trainer.get_optimizer_cls_and_kwargs(
            self.args, opt_model
        )

        # Get uncertainty parameters (unwrap if needed)
        if isinstance(self.uncertainty_module, nn.parallel.DistributedDataParallel):
            uncertainty_params = list(self.uncertainty_module.module.parameters())
        else:
            uncertainty_params = list(self.uncertainty_module.parameters())

        # Combine model and uncertainty parameters
        all_params = [
            {"params": opt_model.parameters()},
            {"params": uncertainty_params, "weight_decay": 0.0},
        ]

        optimizer = optimizer_cls(all_params, **optimizer_kwargs)

        self.optimizer = optimizer

        logger.info("✓ Created optimizer with model + uncertainty parameters")
        return optimizer

    def compute_loss(
        self, model, inputs, return_outputs=False, num_items_in_batch=None
    ):
        """
        Compute loss with three possible routes:
        1. Normal loss (no weighting)
        2. Language class weighting (per-sample)
        3. Uncertainty task-based weighting (per-task)

        Routes 2 and 3 are mutually exclusive in practice, but can be combined if needed.
        """
        # Extract and remove source_lang from model inputs
        source_lang = inputs.pop("source_lang")
        task_type = inputs.pop("task_type")

        # Get model outputs
        outputs = model(**inputs)

        # ============================================
        # ROUTE 1: Default loss computation (no weighting)
        # ============================================
        if not self.use_lang_weighting and not self.use_uncertainty:
            loss = outputs["loss"]
            return (loss, outputs) if return_outputs else loss

        # -------------------------------
        # Compute per-sample loss manually for weighting
        # -------------------------------
        logits = outputs.logits  # [batch_size, seq_len, vocab_size]
        labels = inputs["labels"]  # [batch_size, seq_len]
        batch_size = labels.size(0)

        # Shift for next-token prediction
        shift_logits = logits[..., :-1, :].contiguous()
        shift_labels = labels[..., 1:].contiguous()

        # Compute per-token loss (no reduction)
        loss_fct = torch.nn.CrossEntropyLoss(reduction="none", ignore_index=-100)
        per_token_loss = loss_fct(
            shift_logits.view(-1, shift_logits.size(-1)), shift_labels.view(-1)
        )

        # Reshape to [batch_size, seq_len-1]
        per_token_loss = per_token_loss.view(batch_size, -1)

        # Average per sample (only over non-ignored tokens)
        valid_mask = (shift_labels != -100).float()
        per_sample_loss = per_token_loss.sum(dim=1) / valid_mask.sum(dim=1).clamp(min=1)

        # ============================================
        # ROUTE 2: Language class weighting
        # ============================================
        if self.use_lang_weighting and not self.use_uncertainty:
            lang_weight_map_device = self.lang_weight_map.to(source_lang.device)
            lang_weights = lang_weight_map_device[source_lang]  # [batch_size]

            # Apply language weights and normalize
            weighted_loss = (per_sample_loss * lang_weights).sum() / lang_weights.sum()

            # Restore metadata
            inputs["source_lang"] = source_lang
            inputs["task_type"] = task_type

            return (weighted_loss, outputs) if return_outputs else weighted_loss

        # ============================================
        # ROUTE 3: Uncertainty-based task weighting
        # ============================================
        if self.use_uncertainty:
            # Separate losses by task type
            asr_mask = (task_type == TASKTYPE2ID["asr"]).float()  # 0 = ASR
            st_mask = (task_type == TASKTYPE2ID["s2tt"]).float()  # 1 = ST

            # Optionally apply language weighting first (if both enabled)
            if self.use_lang_weighting:
                lang_weight_map_device = self.lang_weight_map.to(source_lang.device)
                lang_weights = lang_weight_map_device[source_lang]  # [batch_size]
                per_sample_loss = per_sample_loss * lang_weights

            # Compute task-specific losses
            asr_loss = (per_sample_loss * asr_mask).sum()
            st_loss = (per_sample_loss * st_mask).sum()

            num_asr = asr_mask.sum().clamp(min=1)
            num_st = st_mask.sum().clamp(min=1)

            # Average per task
            asr_loss = asr_loss / num_asr
            st_loss = st_loss / num_st

            # Apply uncertainty weighting: L = (1/(2σ²)) * L_task + log(σ)
            # Using precision (inverse variance) for numerical stability
            if isinstance(self.uncertainty_module, nn.parallel.DistributedDataParallel):
                log_var_asr = self.uncertainty_module.module.log_var_asr
                log_var_st = self.uncertainty_module.module.log_var_st
            else:
                log_var_asr = self.uncertainty_module.log_var_asr
                log_var_st = self.uncertainty_module.log_var_st

            precision_asr = torch.exp(-log_var_asr)
            precision_st = torch.exp(-log_var_st)

            weighted_loss = (
                precision_asr * asr_loss
                + log_var_asr
                + precision_st * st_loss
                + log_var_st
            )

            # Log uncertainty values and losses periodically
            if (
                self.model.training
                and self.state.global_step % self.args.logging_steps == 0
            ):
                sigma_asr = torch.exp(0.5 * log_var_asr).item()
                sigma_st = torch.exp(0.5 * log_var_st).item()

                # Log to wandb/tensorboard
                self.log(
                    {
                        "uncertainty/sigma_asr": sigma_asr,
                        "uncertainty/sigma_st": sigma_st,
                        "uncertainty/weight_asr": 1.0 / (2 * sigma_asr**2),
                        "uncertainty/weight_st": 1.0 / (2 * sigma_st**2),
                        "loss/asr_unweighted": asr_loss.item(),
                        "loss/st_unweighted": st_loss.item(),
                        "batch/num_asr_samples": num_asr.item(),
                        "batch/num_st_samples": num_st.item(),
                    }
                )

            return (weighted_loss, outputs) if return_outputs else weighted_loss

    def evaluate(self, eval_dataset=None, **kwargs):
        """Handle dict of eval datasets"""
        if isinstance(self.eval_dataset, dict):
            results = {}
            for name, dataset in self.eval_dataset.items():
                logger.info(
                    "🔍 Evaluating %s dataset (%s samples)",
                    name,
                    len(dataset),
                )
                self.skipped_batches_eval = 0  # Reset counter for this eval set
                self.total_eval_batches = 0

                try:
                    res = super().evaluate(
                        eval_dataset=dataset,
                        metric_key_prefix=f"eval_{name}",
                        **kwargs,
                    )
                    results[name] = res

                    if self.skipped_batches_eval > 0:
                        skip_pct = (
                            100
                            * self.skipped_batches_eval
                            / max(self.total_eval_batches, 1)
                        )
                        logger.info(
                            "ℹ️  Skipped %s/%s batches (%.2f%%) during %s evaluation",
                            self.skipped_batches_eval,
                            self.total_eval_batches,
                            skip_pct,
                            name,
                        )

                except Exception as e:
                    logger.error("Error evaluating %s: %s", name, e)
                    traceback.print_exc()
                    results[name] = {"error": str(e)}

            return results

        # Default single dataset behavior
        return super().evaluate(eval_dataset=eval_dataset, **kwargs)

    def get_train_dataloader(self) -> DataLoader:
        """
        Override to wrap dataloader with None-filtering wrapper
        """
        dataloader = super().get_train_dataloader()

        # Wrap with filtered dataloader that skips None batches
        def increment_skip():
            self.skipped_batches_train += 1
            self.total_train_batches += 1

        return _FilteredDataLoader(dataloader, skip_counter_callback=increment_skip)

    def get_eval_dataloader(self, eval_dataset=None) -> DataLoader:
        """
        Override to wrap dataloader with None-filtering wrapper
        """
        dataloader = super().get_eval_dataloader(eval_dataset)

        # Wrap with filtered dataloader that skips None batches
        def increment_skip():
            self.skipped_batches_eval += 1
            self.total_eval_batches += 1

        return _FilteredDataLoader(dataloader, skip_counter_callback=increment_skip)

    def training_step(self, model, inputs, num_items_in_batch=None):
        """
        Override training step to count batches
        """
        self.total_train_batches += 1

        # Call parent with correct signature
        if num_items_in_batch is not None:
            return super().training_step(model, inputs, num_items_in_batch)
        else:
            return super().training_step(model, inputs)

    def prediction_step(self, model, inputs, prediction_loss_only, ignore_keys=None):
        """
        Override prediction step to count batches
        """
        self.total_eval_batches += 1

        return super().prediction_step(
            model, inputs, prediction_loss_only, ignore_keys=ignore_keys
        )
