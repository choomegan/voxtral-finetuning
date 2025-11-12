"""
Trainer utils
"""

import logging
from transformers import Trainer
import traceback
from torch.utils.data import DataLoader

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


class SafeTrainer(Trainer):
    """
    Custom trainer that:
    1. Handles multiple eval datasets
    2. Properly handles None returns from collators (skipped batches)
    """

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.skipped_batches_train = 0
        self.skipped_batches_eval = 0
        self.total_train_batches = 0
        self.total_eval_batches = 0

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
