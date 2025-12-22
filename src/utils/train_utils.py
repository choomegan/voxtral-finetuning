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
        Compute loss using one of 4 strategies:

        Route 0: Normal loss - use model's built-in loss (no weighting, no PCGrad)
        Route 1: PCGrad - gradient projection for multi-task learning
        Route 2: Language weighting - weight samples by language frequency
        Route 3: Uncertainty weighting - learnable task weights

        For PCGrad (Route 1): Stores individual task losses in self.current_task_losses
        so training_step can use them for gradient projection.

        NOTE: PCGrad is applied at the GRADIENT level in training_step,
        not at the LOSS level here. This maintains clean separation.
        """
        # Extract metadata
        source_lang = inputs["source_lang"]
        task_type = inputs["task_type"]

        # Get model outputs
        outputs = model(**inputs)

        # ============================================
        # ROUTE 0: Normal loss (no weighting, no PCGrad)
        # ============================================
        if (
            not self.use_pcgrad
            and not self.use_lang_weighting
            and not self.use_uncertainty
        ):
            loss = outputs["loss"]
            self.current_task_losses = None
            return (loss, outputs) if return_outputs else loss

        # ============================================
        # Common computation for Routes 1-3: Per-sample losses
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

        has_asr = asr_mask.any().item()
        has_st = st_mask.any().item()

        num_asr = asr_mask.sum().clamp(min=1)
        num_st = st_mask.sum().clamp(min=1)

        # ============================================
        # ROUTE 1: PCGrad only (no language weighting, no uncertainty)
        # ============================================
        if self.use_pcgrad and not self.use_lang_weighting and not self.use_uncertainty:
            # Compute task-specific losses
            asr_loss = (per_sample_loss * asr_mask).sum() / num_asr
            st_loss = (per_sample_loss * st_mask).sum() / num_st

            # Check if we can apply PCGrad (requires both tasks in batch)
            if has_asr and has_st and model.training:
                # Both tasks present - enable PCGrad
                self.current_task_losses = [asr_loss, st_loss]
                loss = asr_loss + st_loss
            else:
                # Single task or not training - disable PCGrad
                self.current_task_losses = None
                loss = asr_loss + st_loss

                if (
                    model.training
                    and self.state.global_step % self.args.logging_steps == 0
                ):
                    task_status = (
                        "ASR only"
                        if has_asr and not has_st
                        else (
                            "ST only"
                            if has_st and not has_asr
                            else "both tasks (eval mode)"
                        )
                    )
                    logger.info(
                        f"⚠️  Single-task batch ({task_status}), skipping PCGrad"
                    )

            return (loss, outputs) if return_outputs else loss

        # ============================================
        # Apply language weighting if enabled (for Routes 2 & 3)
        # ============================================
        if self.use_lang_weighting:
            lang_weight_map_device = self.lang_weight_map.to(source_lang.device)
            lang_weights = lang_weight_map_device[source_lang]
            per_sample_loss = per_sample_loss * lang_weights

        # Compute task losses (after optional language weighting)
        asr_loss = (per_sample_loss * asr_mask).sum() / num_asr
        st_loss = (per_sample_loss * st_mask).sum() / num_st

        # Handle PCGrad for Routes 2 & 3 (if enabled)
        if self.use_pcgrad and model.training:
            if has_asr and has_st:
                # Both tasks present - enable PCGrad
                self.current_task_losses = [asr_loss, st_loss]
            else:
                # Single task - disable PCGrad
                self.current_task_losses = None
                if self.state.global_step % self.args.logging_steps == 0:
                    task_status = "ASR only" if has_asr and not has_st else "ST only"
                    logger.info(
                        f"⚠️  Single-task batch ({task_status}), skipping PCGrad"
                    )
        else:
            self.current_task_losses = None

        # ============================================
        # ROUTE 2: Language weighting only (no uncertainty)
        # ============================================
        if self.use_lang_weighting and not self.use_uncertainty:
            final_loss = asr_loss + st_loss

            # Preserve metadata
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

            # Preserve metadata
            inputs["source_lang"] = source_lang
            inputs["task_type"] = task_type

            return (weighted_loss, outputs) if return_outputs else weighted_loss

        # Should never reach here
        raise ValueError(
            f"Invalid trainer configuration: "
            f"use_pcgrad={self.use_pcgrad}, "
            f"use_lang_weighting={self.use_lang_weighting}, "
            f"use_uncertainty={self.use_uncertainty}"
        )

    def training_step(self, model, inputs, num_items_in_batch=None):
        """
        Override training_step to use PCGrad with correct Mixed Precision and Accumulation.
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

        # 1. Compute Loss (Get individual task losses)
        with self.compute_loss_context_manager():
            loss = self.compute_loss(
                model, inputs, num_items_in_batch=num_items_in_batch
            )

        task_losses = self.current_task_losses

        # 2. Fallback: Standard Backward (Single Task or Validation)
        if task_losses is None:
            if self.state.global_step % 100 == 0:
                logger.info(
                    f"⚠️ Step {self.state.global_step}: Standard backward (single-task batch)"
                )

            if self.args.gradient_accumulation_steps > 1:
                loss = loss / self.args.gradient_accumulation_steps

            # Accelerator handles scaling and accumulation automatically here
            self.accelerator.backward(loss)
            return loss.detach()

        # 3. PCGrad Logic
        # Scale losses for accumulation
        if self.args.gradient_accumulation_steps > 1:
            task_losses = [
                l / self.args.gradient_accumulation_steps for l in task_losses
            ]

        # --- CRITICAL FIX: REMOVED ZERO_GRAD BLOCK HERE ---
        # Do not zero grads here. The Trainer zeros them after optimizer.step().
        # If we zero here, we lose the accumulated gradients from previous micro-batches.

        # 4. Handle Mixed Precision Scaling Manually
        # PCGrad needs scaled losses to compute gradients large enough for FP16
        if self.accelerator.mixed_precision == "fp16" and getattr(
            self.accelerator, "scaler", None
        ):
            scaled_losses = [self.accelerator.scaler.scale(l) for l in task_losses]
            self.optimizer.pc_backward(scaled_losses)
        else:
            self.optimizer.pc_backward(task_losses)

        # 5. Logging
        if self.state.global_step % self.args.logging_steps == 0:

            def to_scalar(tensor):
                if isinstance(tensor, torch.Tensor):
                    return tensor.mean().item() if tensor.numel() > 1 else tensor.item()
                return float(tensor)

            # Calculate grad norm (computationally expensive, use sparingly)
            grad_norm = 0.0
            for param in model.parameters():
                if param.grad is not None:
                    grad_norm += param.grad.norm().item() ** 2
            grad_norm = grad_norm**0.5

            scale = self.args.gradient_accumulation_steps
            self.log(
                {
                    "asr_loss": to_scalar(task_losses[0]) * scale,
                    "st_loss": to_scalar(task_losses[1]) * scale,
                    "total_loss": to_scalar(loss),
                    "pcgrad_grad_norm": grad_norm,
                }
            )

        # Return loss for Trainer statistics
        if self.args.gradient_accumulation_steps > 1:
            return loss.detach() / self.args.gradient_accumulation_steps
        return loss.detach()

    def evaluate(self, eval_dataset=None, **kwargs):
        """Handle dict of eval datasets."""
        if isinstance(self.eval_dataset, dict):
            flat_metrics = {}
            losses = {}

            # ensure eval datasets are not None
            active_datasets = {
                k: v for k, v in self.eval_dataset.items() if v is not None
            }

            for name, dataset in active_datasets.items():
                logger.info("🔍 Evaluating %s dataset (%s samples)", name, len(dataset))
                self.skipped_batches_eval = 0
                self.total_eval_batches = 0

                try:
                    res = super().evaluate(
                        eval_dataset=dataset,
                        metric_key_prefix=f"eval_{name}",
                        **kwargs,
                    )

                    # 🔑 Flatten metrics so Trainer can see them
                    flat_metrics.update(res)

                    # Extract per-task loss
                    loss_key = f"eval_{name}_loss"
                    if loss_key in res:
                        losses[name] = res[loss_key]

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
                    # Still record something so Trainer doesn’t crash
                    flat_metrics[f"eval_{name}_error"] = 1.0

            # ===============================
            # DYNAMIC COMBINED LOSS
            # ===============================
            if losses:
                combined_loss = sum(losses.values())
                flat_metrics["eval_combined_loss"] = combined_loss

                # Create a dynamic log string for the breakdown
                loss_breakdown = " + ".join(
                    [f"{k.upper()} {v:.4f}" for k, v in losses.items()]
                )

                logger.info(
                    "📊 Combined eval loss: %.4f (%s)",
                    combined_loss,
                    loss_breakdown,
                )
            else:
                # Fallback if no losses were found (e.g. all evals failed or no datasets)
                # We return infinity so this checkpoint is not saved as "best"
                flat_metrics["eval_combined_loss"] = float("inf")
                logger.warning("⚠️ No losses collected during evaluation.")

            return flat_metrics
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
