"""
Clean Trainer implementation with proper separation of concerns:
- compute_loss(): Handles ALL loss computation (normal, language weighting, uncertainty)
- training_step(): Only handles PCGrad gradient manipulation
- No code duplication
"""

import logging
import traceback

import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from transformers import Trainer

from utils.constants import SRCLANG2ID, TASKTYPE2ID
from utils.pcgrad import PCGrad

logger = logging.getLogger(__name__)


class _FilteredDataLoader:
    """Wrapper that filters out None batches returned by collator."""

    def __init__(self, dataloader, skip_counter_callback=None):
        self.dataloader = dataloader
        self.skip_counter_callback = skip_counter_callback

    def __iter__(self):
        for batch in self.dataloader:
            if batch is None:
                if self.skip_counter_callback:
                    self.skip_counter_callback()
                continue
            yield batch

    def __len__(self):
        return len(self.dataloader)

    def __getattr__(self, name):
        return getattr(self.dataloader, name)


class UncertaintyModule(nn.Module):
    """Separate module for uncertainty parameters."""

    def __init__(self):
        super().__init__()
        self.log_var_asr = nn.Parameter(torch.zeros(1))
        self.log_var_st = nn.Parameter(torch.zeros(1))


class SafeTrainer(Trainer):
    """
    Custom trainer supporting 4 independent strategies:

    1. Normal - baseline
    2. Language weighting - per-sample weighting by language frequency
    3. Uncertainty weighting - learnable task weights
    4. PCGrad - gradient-level conflict resolution

    PCGrad is orthogonal to loss strategies:
    - PCGrad operates on gradients AFTER loss computation
    - Can be combined with language weighting or used alone
    - Should NOT be combined with uncertainty weighting
    """

    def __init__(
        self,
        *args,
        lang_weight_map=None,
        use_uncertainty=False,
        use_pcgrad=False,
        pcgrad_reduction="mean",
        **kwargs,
    ):
        self.use_uncertainty = use_uncertainty
        self.use_lang_weighting = lang_weight_map is not None
        self.use_pcgrad = use_pcgrad
        self.pcgrad_reduction = pcgrad_reduction

        # Create uncertainty module if needed
        if self.use_uncertainty:
            self.uncertainty_module = UncertaintyModule()
            logger.info("✓ Created uncertainty module")
        else:
            self.uncertainty_module = None

        # Verify PCGrad is available
        if self.use_pcgrad:
            try:
                self.pcgrad_cls = PCGrad
                logger.info(f"✓ PCGrad enabled (reduction={pcgrad_reduction})")
            except Exception as e:
                raise ImportError(f"PCGrad not available: {e}")
        else:
            self.pcgrad_cls = None

        # Call parent init
        super().__init__(*args, **kwargs)

        # Initialize counters
        self.skipped_batches_train = 0
        self.skipped_batches_eval = 0
        self.total_train_batches = 0
        self.total_eval_batches = 0

        # Storage for task losses (used by PCGrad)
        self.current_task_losses = None

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

        # Move uncertainty module to device
        if self.use_uncertainty:
            device = next(self.model.parameters()).device
            self.uncertainty_module = self.uncertainty_module.to(device)

            if hasattr(self, "accelerator") and self.accelerator is not None:
                self.uncertainty_module = self.accelerator.prepare(
                    self.uncertainty_module
                )
                logger.info("✓ Uncertainty module wrapped by Accelerator")

            logger.info("✓ Uncertainty weighting enabled")

        # Log configuration warnings
        if self.use_pcgrad and self.use_uncertainty:
            logger.warning(
                "⚠️  PCGrad + Uncertainty weighting: Not recommended, use one or the other"
            )
        if self.use_pcgrad and self.use_lang_weighting:
            logger.info("ℹ️  PCGrad + Language weighting: Compatible combination")

    def create_optimizer(self):
        """
        Create optimizer and optionally wrap with PCGrad.

        IMPORTANT: For PCGrad, we return the wrapper but store the base optimizer
        so that the learning rate scheduler can access it properly.
        """
        opt_model = self.model_wrapped if hasattr(self, "model_wrapped") else self.model
        optimizer_cls, optimizer_kwargs = Trainer.get_optimizer_cls_and_kwargs(
            self.args, opt_model
        )

        # Build parameter groups
        param_groups = [{"params": opt_model.parameters()}]

        # Add uncertainty parameters if needed
        if self.use_uncertainty:
            if isinstance(self.uncertainty_module, nn.parallel.DistributedDataParallel):
                uncertainty_params = list(self.uncertainty_module.module.parameters())
            else:
                uncertainty_params = list(self.uncertainty_module.parameters())
            param_groups.append({"params": uncertainty_params, "weight_decay": 0.0})
            logger.info("✓ Added uncertainty parameters to optimizer")

        # Create base optimizer
        base_optimizer = optimizer_cls(param_groups, **optimizer_kwargs)

        # Wrap with PCGrad if enabled
        if self.use_pcgrad:
            pcgrad_wrapper = self.pcgrad_cls(
                base_optimizer, reduction=self.pcgrad_reduction
            )
            logger.info(
                f"✓ Wrapped optimizer with PCGrad (reduction={self.pcgrad_reduction})"
            )
            # Store the wrapper for training_step to use
            self.optimizer = pcgrad_wrapper
            # IMPORTANT: Return the wrapper so it gets used
            return pcgrad_wrapper
        else:
            self.optimizer = base_optimizer
            return base_optimizer

    def create_scheduler(self, num_training_steps: int, optimizer=None):
        """
        Override to handle PCGrad wrapper - pass base optimizer to scheduler.
        """
        # If PCGrad is enabled, unwrap to get base optimizer for scheduler
        if self.use_pcgrad and optimizer is not None:
            # Pass the base optimizer to the scheduler
            base_optimizer = optimizer._optim
            logger.info("✓ Using base optimizer for LR scheduler")
            return super().create_scheduler(
                num_training_steps, optimizer=base_optimizer
            )
        else:
            return super().create_scheduler(num_training_steps, optimizer=optimizer)

    def compute_loss(
        self, model, inputs, return_outputs=False, num_items_in_batch=None
    ):
        """
        Compute loss using one of 3 strategies:

        1. Normal - use model's built-in loss
        2. Language weighting - weight samples by language frequency
        3. Uncertainty weighting - learnable task weights

        For PCGrad: Also stores individual task losses in self.current_task_losses
        so training_step can use them for gradient projection.

        NOTE: PCGrad is applied at the GRADIENT level in training_step,
        not at the LOSS level here. This maintains clean separation.
        """
        # Extract metadata
        source_lang = inputs.pop("source_lang")
        task_type = inputs.pop("task_type")

        # Get model outputs
        outputs = model(**inputs)

        # ============================================
        # ROUTE 1: Normal loss (no weighting)
        # ============================================
        if not self.use_lang_weighting and not self.use_uncertainty:
            loss = outputs["loss"]

            # For PCGrad: still need to compute task-specific losses
            if self.use_pcgrad and model.training:
                # Recompute per-sample losses to separate by task
                logits = outputs.logits
                labels = inputs["labels"]
                batch_size = labels.size(0)

                shift_logits = logits[..., :-1, :].contiguous()
                shift_labels = labels[..., 1:].contiguous()

                loss_fct = nn.CrossEntropyLoss(reduction="none", ignore_index=-100)
                per_token_loss = loss_fct(
                    shift_logits.view(-1, shift_logits.size(-1)),
                    shift_labels.view(-1),
                ).view(batch_size, -1)

                valid_mask = (shift_labels != -100).float()
                per_sample_loss = per_token_loss.sum(dim=1) / valid_mask.sum(
                    dim=1
                ).clamp(min=1)

                # Separate by task
                asr_mask = (task_type == TASKTYPE2ID["asr"]).float()
                st_mask = (task_type == TASKTYPE2ID["s2tt"]).float()

                num_asr = asr_mask.sum().clamp(min=1)
                num_st = st_mask.sum().clamp(min=1)

                asr_loss = (per_sample_loss * asr_mask).sum() / num_asr
                st_loss = (per_sample_loss * st_mask).sum() / num_st

                # Store for training_step
                self.current_task_losses = [asr_loss, st_loss]
            else:
                self.current_task_losses = None

            # Restore metadata
            inputs["source_lang"] = source_lang
            inputs["task_type"] = task_type
            return (loss, outputs) if return_outputs else loss

        # ============================================
        # For weighted routes: Compute per-sample losses
        # ============================================
        logits = outputs.logits
        labels = inputs["labels"]
        batch_size = labels.size(0)

        shift_logits = logits[..., :-1, :].contiguous()
        shift_labels = labels[..., 1:].contiguous()

        loss_fct = nn.CrossEntropyLoss(reduction="none", ignore_index=-100)
        per_token_loss = loss_fct(
            shift_logits.view(-1, shift_logits.size(-1)),
            shift_labels.view(-1),
        ).view(batch_size, -1)

        valid_mask = (shift_labels != -100).float()
        per_sample_loss = per_token_loss.sum(dim=1) / valid_mask.sum(dim=1).clamp(min=1)

        # Separate by task
        asr_mask = (task_type == TASKTYPE2ID["asr"]).float()
        st_mask = (task_type == TASKTYPE2ID["s2tt"]).float()

        num_asr = asr_mask.sum().clamp(min=1)
        num_st = st_mask.sum().clamp(min=1)

        # Apply language weighting if enabled (before task-specific computation)
        if self.use_lang_weighting:
            lang_weight_map_device = self.lang_weight_map.to(source_lang.device)
            lang_weights = lang_weight_map_device[source_lang]
            per_sample_loss = per_sample_loss * lang_weights

        # Compute task-specific losses
        asr_loss = (per_sample_loss * asr_mask).sum() / num_asr
        st_loss = (per_sample_loss * st_mask).sum() / num_st

        # Store task losses for PCGrad (if enabled and training)
        if self.use_pcgrad and model.training:
            self.current_task_losses = [asr_loss, st_loss]
        else:
            self.current_task_losses = None

        # ============================================
        # ROUTE 2: Language weighting only
        # ============================================
        if self.use_lang_weighting and not self.use_uncertainty:
            final_loss = asr_loss + st_loss

            inputs["source_lang"] = source_lang
            inputs["task_type"] = task_type
            return (final_loss, outputs) if return_outputs else final_loss

        # ============================================
        # ROUTE 3: Uncertainty weighting (with optional language weighting)
        # ============================================
        if self.use_uncertainty:
            # Get uncertainty parameters
            if isinstance(self.uncertainty_module, nn.parallel.DistributedDataParallel):
                log_var_asr = self.uncertainty_module.module.log_var_asr
                log_var_st = self.uncertainty_module.module.log_var_st
            else:
                log_var_asr = self.uncertainty_module.log_var_asr
                log_var_st = self.uncertainty_module.log_var_st

            precision_asr = torch.exp(-log_var_asr)
            precision_st = torch.exp(-log_var_st)

            # Apply uncertainty weighting: L = (1/(2σ²)) * L_task + log(σ)
            weighted_loss = (
                precision_asr * asr_loss
                + log_var_asr
                + precision_st * st_loss
                + log_var_st
            )

            # Log uncertainty values periodically
            if model.training and self.state.global_step % self.args.logging_steps == 0:
                sigma_asr = torch.exp(0.5 * log_var_asr).item()
                sigma_st = torch.exp(0.5 * log_var_st).item()

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

            inputs["source_lang"] = source_lang
            inputs["task_type"] = task_type
            return (weighted_loss, outputs) if return_outputs else weighted_loss

        # Should never reach here
        raise ValueError("Invalid trainer configuration")

    def training_step(self, model, inputs, num_items_in_batch=None):
        """
        Override training_step to use PCGrad when enabled.

        PCGrad flow:
        1. Call compute_loss() to get combined loss (and store task losses)
        2. Retrieve task losses from self.current_task_losses
        3. Use pcgrad.pc_backward() instead of loss.backward()
        4. Trainer handles optimizer.step() automatically

        Non-PCGrad flow:
        - Use default Trainer behavior (calls loss.backward())
        """
        self.total_train_batches += 1

        # ========================================
        # Non-PCGrad: Use default Trainer behavior
        # ========================================
        if not self.use_pcgrad:
            if num_items_in_batch is not None:
                return super().training_step(model, inputs, num_items_in_batch)
            else:
                return super().training_step(model, inputs)

        # ========================================
        # PCGrad: Custom backward pass
        # ========================================
        model.train()
        inputs = self._prepare_inputs(inputs)

        # Compute loss (this also stores task losses in self.current_task_losses)
        with self.compute_loss_context_manager():
            loss = self.compute_loss(
                model, inputs, num_items_in_batch=num_items_in_batch
            )

        # Retrieve task losses
        task_losses = self.current_task_losses

        if task_losses is None:
            # Fallback: if no task losses available, use standard backward
            logger.warning(
                "PCGrad enabled but task losses not available, using standard backward"
            )
            if self.args.gradient_accumulation_steps > 1:
                loss = loss / self.args.gradient_accumulation_steps

            if self.use_amp:
                self.scaler.scale(loss).backward()
            else:
                loss.backward()

            return loss.detach()

        # Scale task losses for gradient accumulation
        if self.args.gradient_accumulation_steps > 1:
            task_losses = [
                l / self.args.gradient_accumulation_steps for l in task_losses
            ]

        # Use PCGrad backward (sets gradients, doesn't call step)
        self.optimizer.pc_backward(task_losses)

        # Log periodically (unscale for display)
        # Note: With DDP, losses might have multiple elements, so use mean
        if self.state.global_step % self.args.logging_steps == 0:
            scale = self.args.gradient_accumulation_steps

            # Helper to safely convert to scalar (handles DDP gathered tensors)
            def to_scalar(tensor):
                """Convert tensor to scalar, handling DDP multi-element tensors."""
                if isinstance(tensor, torch.Tensor):
                    if tensor.numel() > 1:
                        return tensor.mean().item()
                    return tensor.item()
                return float(tensor)

            self.log(
                {
                    "train/asr_loss": to_scalar(task_losses[0]) * scale,
                    "train/st_loss": to_scalar(task_losses[1]) * scale,
                    "train/total_loss": to_scalar(loss),
                }
            )

        # Ensure scalar loss (Accelerate does this implicitly, python does not)
        if isinstance(loss, torch.Tensor) and loss.numel() > 1:
            loss = loss.mean()

        # Return scaled combined loss for logging
        if self.args.gradient_accumulation_steps > 1:
            return loss.detach() / self.args.gradient_accumulation_steps
        else:
            return loss.detach()

    def evaluate(self, eval_dataset=None, **kwargs):
        if isinstance(self.eval_dataset, dict):
            results = {}
            losses = {}

            for name, dataset in self.eval_dataset.items():
                logger.info("🔍 Evaluating %s dataset (%s samples)", name, len(dataset))
                self.skipped_batches_eval = 0
                self.total_eval_batches = 0

                res = super().evaluate(
                    eval_dataset=dataset,
                    metric_key_prefix=f"eval_{name}",
                    **kwargs,
                )

                results[name] = res

                # Extract loss if present
                loss_key = f"eval_{name}_loss"
                if loss_key in res:
                    losses[name] = res[loss_key]

            # ===============================
            # ADD COMBINED LOSS
            # ===============================
            if "asr" in losses and "st" in losses:
                combined_loss = losses["asr"] + losses["st"]

                # Log it so Trainer & TensorBoard see it
                self.log({"eval_combined_loss": combined_loss})

                # Also return it (important!)
                results["combined"] = {"eval_combined_loss": combined_loss}

                logger.info(
                    "📊 Combined eval loss: %.4f (ASR %.4f + ST %.4f)",
                    combined_loss,
                    losses["asr"],
                    losses["st"],
                )

            return results

        return super().evaluate(eval_dataset=eval_dataset, **kwargs)

    def get_train_dataloader(self) -> DataLoader:
        """Wrap dataloader to filter None batches."""
        dataloader = super().get_train_dataloader()

        def increment_skip():
            self.skipped_batches_train += 1
            self.total_train_batches += 1

        return _FilteredDataLoader(dataloader, skip_counter_callback=increment_skip)

    def get_eval_dataloader(self, eval_dataset=None) -> DataLoader:
        """Wrap dataloader to filter None batches."""
        dataloader = super().get_eval_dataloader(eval_dataset)

        def increment_skip():
            self.skipped_batches_eval += 1
            self.total_eval_batches += 1

        return _FilteredDataLoader(dataloader, skip_counter_callback=increment_skip)

    def prediction_step(self, model, inputs, prediction_loss_only, ignore_keys=None):
        """Override to count eval batches."""
        self.total_eval_batches += 1
        return super().prediction_step(
            model, inputs, prediction_loss_only, ignore_keys=ignore_keys
        )
