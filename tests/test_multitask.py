"""
Script to unit test FilteredDataLoader in the case where there are None batches returned
from Collator
"""

import sys, os
import torch
from torch.utils.data import Dataset, DataLoader

# Ensure src/ is on the path so we can import
PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
sys.path.insert(0, os.path.join(PROJECT_ROOT, "src"))

from utils.train_utils import _FilteredDataLoader


class MockDataset(Dataset):
    """
    Mock dataset that sometimes returns None
    """

    def __init__(self):
        # Mix of valid and None samples
        self.samples = [torch.tensor([1.0]), torch.tensor([2.0]), None, None, None]

    def __getitem__(self, idx):
        return self.samples[idx]

    def __len__(self):
        return len(self.samples)


# --- mock collator that simulates StreamingMultiTaskCollator returning None ---
def mock_collator(batch):
    """
    Mock collator to simulate StreamingMultiTaskCollator returning None
    """
    # Filter out None items inside batch
    filtered = [b for b in batch if b is not None]
    if len(filtered) == 0:
        # Simulate collator failure (returns None)
        print("Mock collator returning None (simulated ASR batch misalignment)")
        return None
    return torch.stack(filtered)


def test_filtered_dataloader_removes_none():
    dataset = MockDataset()

    # Standard DataLoader first
    base_dataloader = DataLoader(
        dataset, batch_size=2, shuffle=False, drop_last=False, collate_fn=mock_collator
    )

    skipped_batches = 0

    # Callback for when batches are skipped
    def increment_skip():
        nonlocal skipped_batches
        skipped_batches += 1

    # Wrap with FilteredDataLoader
    filtered_loader = _FilteredDataLoader(
        base_dataloader, skip_counter_callback=increment_skip
    )

    # Collect all yielded batches
    batches = list(filtered_loader)

    # --- Assertions ---
    assert len(batches) > 0, "No batches were returned by FilteredDataLoader"
    for batch in batches:
        assert batch is not None, "FilteredDataLoader yielded a None batch"
        assert isinstance(batch, torch.Tensor), "Each batch should be a tensor"

    # Check that skip counter was triggered at least once
    assert skipped_batches > 0, "FilteredDataLoader did not skip any None batches"

    print(
        f"✅ Passed test: {len(batches)} valid batches, {skipped_batches} skipped batches"
    )
