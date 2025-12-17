"""
Clean PCGrad implementation - fully decoupled from loss weighting strategies
"""

import torch
import random


class PCGrad:
    """
    Project Conflicting Gradients optimizer wrapper.

    Reference: Yu et al. "Gradient Surgery for Multi-Task Learning" (NeurIPS 2020)

    Note: This is a wrapper, not a subclass of torch.optim.Optimizer.
    For compatibility with LR schedulers, access the base optimizer via ._optim
    """

    def __init__(self, optimizer, reduction="mean"):
        """
        Args:
            optimizer: Base PyTorch optimizer
            reduction: How to combine projected gradients ('mean' or 'sum')
        """
        self._optim = optimizer
        self.reduction = reduction

    def zero_grad(self):
        """Zero gradients of underlying optimizer."""
        self._optim.zero_grad()

    def step(self, closure=None):
        """Perform optimization step (called by Trainer)."""
        return self._optim.step(closure)

    @property
    def param_groups(self):
        """Expose param_groups for Trainer/scheduler compatibility."""
        return self._optim.param_groups

    @property
    def defaults(self):
        """Expose defaults for scheduler compatibility."""
        return self._optim.defaults

    def state_dict(self):
        """Save optimizer state."""
        return self._optim.state_dict()

    def load_state_dict(self, state_dict):
        """Load optimizer state."""
        self._optim.load_state_dict(state_dict)

    def pc_backward(self, losses):
        """
        Compute per-task gradients, project conflicting ones, set combined gradient.

        Does NOT call optimizer.step() - that's handled by the Trainer.

        Args:
            losses: List of scalar loss tensors, one per task
        """
        if len(losses) == 1:
            # Single task - just do normal backward
            self.zero_grad()
            losses[0].backward()
            return

        # Compute per-task gradients
        grads = []
        for i, loss in enumerate(losses):
            self.zero_grad()
            # Retain graph for all but last task
            retain = i < len(losses) - 1
            loss.backward(retain_graph=retain)
            grads.append(self._get_flat_grads())

        # Apply PCGrad projection
        proj_grads = self._project_conflicting(grads)

        # Combine projected gradients
        if self.reduction == "mean":
            combined_grad = torch.stack(proj_grads).mean(dim=0)
        elif self.reduction == "sum":
            combined_grad = torch.stack(proj_grads).sum(dim=0)
        else:
            raise ValueError(f"Unknown reduction: {self.reduction}")

        # Set combined gradient on parameters
        self.zero_grad()
        self._set_flat_grads(combined_grad)

    def _get_flat_grads(self):
        """Flatten all parameter gradients into a single vector."""
        grads = []
        for group in self._optim.param_groups:
            for p in group["params"]:
                if p.grad is not None:
                    grads.append(p.grad.view(-1))
        return torch.cat(grads) if grads else torch.tensor([])

    def _set_flat_grads(self, flat_grad):
        """Set parameter gradients from a flattened vector."""
        offset = 0
        for group in self._optim.param_groups:
            for p in group["params"]:
                if p.grad is None:
                    continue
                numel = p.grad.numel()
                p.grad.copy_(flat_grad[offset : offset + numel].view_as(p.grad))
                offset += numel

    def _project_conflicting(self, grads):
        """
        Apply PCGrad projection to resolve gradient conflicts.

        For each task i:
        1. Start with its original gradient
        2. Project against all other tasks' ORIGINAL gradients where conflict exists
        3. Return projected gradient

        Args:
            grads: List of flattened gradient tensors, one per task

        Returns:
            List of projected gradients in same order as input
        """
        num_tasks = len(grads)

        # Random shuffle task processing order (recommended in paper)
        task_indices = list(range(num_tasks))
        random.shuffle(task_indices)

        # Project each task's gradient
        proj_grads = []
        for i in task_indices:
            # Start with original gradient (clone to avoid modifying original)
            grad_i = grads[i].clone()

            # Project against all other tasks' ORIGINAL gradients
            for j in range(num_tasks):
                if i == j:
                    continue

                grad_j = grads[j]  # Use original, not projected
                dot_product = torch.dot(grad_i, grad_j)

                # Only project if gradients conflict (negative dot product)
                if dot_product < 0:
                    # Project grad_i onto normal plane of grad_j
                    # Formula: g_i' = g_i - ((g_i · g_j) / ||g_j||²) * g_j
                    grad_i = (
                        grad_i
                        - (dot_product / (torch.dot(grad_j, grad_j) + 1e-12)) * grad_j
                    )

            proj_grads.append(grad_i)

        # Return in original order (undo shuffle)
        result = [None] * num_tasks
        for idx, task_idx in enumerate(task_indices):
            result[task_idx] = proj_grads[idx]

        return result
