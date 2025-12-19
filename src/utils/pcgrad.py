import torch
import torch.nn as nn
import random


class PCGrad:
    """
    Reference: Yu et al. "Gradient Surgery for Multi-Task Learning" (NeurIPS 2020)
    """

    def __init__(self, optimizer, reduction="mean"):
        self._optim = optimizer
        self.reduction = reduction

    def zero_grad(self, *args, **kwargs):
        self._optim.zero_grad(*args, **kwargs)

    def step(self, closure=None):
        return self._optim.step(closure)

    @property
    def param_groups(self):
        return self._optim.param_groups

    @property
    def defaults(self):
        return self._optim.defaults

    def state_dict(self):
        return self._optim.state_dict()

    def load_state_dict(self, state_dict):
        self._optim.load_state_dict(state_dict)

    def pc_backward(self, losses):
        """
        Compute per-task gradients, project conflicting ones, apply to model.
        Args:
            losses: List of scalar loss tensors, one per task
        """
        # 1. Prepare parameters that require gradients
        # We need a flat list for autograd.grad
        params = []
        for group in self._optim.param_groups:
            for p in group["params"]:
                if p.requires_grad:
                    params.append(p)

        # 2. Compute per-task gradients independently using autograd.grad
        # This keeps them isolated from each other AND from accumulated grads
        task_grads = []
        for i, loss in enumerate(losses):
            # retain_graph=True is needed because we backprop multiple times
            # through the same shared encoder
            retain_graph = True

            # Compute gradients pure for this task
            grads = torch.autograd.grad(
                loss, params, retain_graph=retain_graph, allow_unused=True
            )

            # autograd.grad returns None for unused params, we need to handle that
            # but our logic expects shapes, so we keep the structure
            grads_list = []
            for g in grads:
                if g is not None:
                    grads_list.append(g.detach())  # Detach to save memory
                else:
                    grads_list.append(None)

            task_grads.append(grads_list)

        # 3. Apply PCGrad projection
        proj_grads = self._project_conflicting(task_grads)

        # 4. Apply the projected gradients to the model
        # Crucial: We must ADD to existing .grad to support Gradient Accumulation
        self._set_grads(proj_grads, params)

    def _set_grads(self, proj_grads, params):
        """
        Set gradients from projected gradients.
        Args:
            proj_grads: List of lists of gradients (one per task)
            params: List of parameters matching the flat structure
        """
        # Combine projected gradients
        if self.reduction == "mean":
            combined = []
            for task_grads in zip(*proj_grads):
                valid_grads = [g for g in task_grads if g is not None]
                if valid_grads:
                    combined.append(sum(valid_grads) / len(valid_grads))
                else:
                    combined.append(None)
        elif self.reduction == "sum":
            combined = []
            for task_grads in zip(*proj_grads):
                valid_grads = [g for g in task_grads if g is not None]
                if valid_grads:
                    combined.append(sum(valid_grads))
                else:
                    combined.append(None)
        else:
            raise ValueError(f"Unknown reduction: {self.reduction}")

        # Apply to parameters (Iterate over the flat params list we created)
        for i, p in enumerate(params):
            if combined[i] is not None:
                if p.grad is None:
                    p.grad = combined[i]
                else:
                    # FIX: Add to existing grad for accumulation
                    p.grad += combined[i]

    def _project_conflicting(self, task_grads):
        """
        Core PCGrad projection algorithm.
        Same logic as before, but handles the input format from autograd.grad
        """
        num_tasks = len(task_grads)
        task_indices = list(range(num_tasks))
        random.shuffle(task_indices)

        # Flatten gradients
        flat_grads = []
        shapes = []

        # We assume all tasks have same param structure, so we capture shapes from first task
        # But we need to handle None carefully per task

        for task_grad_list in task_grads:
            flat = []
            for g in task_grad_list:
                if g is not None:
                    flat.append(g.view(-1))

            # Check if this task had any gradients
            if flat:
                flat_grads.append(torch.cat(flat))
            else:
                flat_grads.append(None)

        # Capture shapes for unflattening (using the first valid gradient list found)
        # Note: We assume all tasks share the same model structure
        shapes = []
        for g in task_grads[0]:
            if g is not None:
                shapes.append(g.shape)
            else:
                shapes.append(None)

        # Project in shuffled order
        proj_flat_grads = [None] * num_tasks
        for i, task_idx in enumerate(task_indices):
            if flat_grads[task_idx] is None:
                continue

            grad = flat_grads[task_idx].clone()

            for j in range(i):
                prev_task_idx = task_indices[j]
                prev_grad = proj_flat_grads[prev_task_idx]
                if prev_grad is None:
                    continue

                dot = torch.dot(grad, prev_grad)
                if dot < 0:
                    grad = (
                        grad
                        - dot / (torch.dot(prev_grad, prev_grad) + 1e-12) * prev_grad
                    )

            proj_flat_grads[task_idx] = grad

        # Unflatten
        proj_task_grads = []
        for task_idx in range(num_tasks):
            flat = proj_flat_grads[task_idx]
            if flat is None:
                # Return list of Nones matching shape
                proj_task_grads.append([None] * len(shapes))
                continue

            unflat = []
            offset = 0
            for shape in shapes:
                if shape is not None:
                    numel = torch.prod(torch.tensor(shape)).item()
                    unflat.append(flat[offset : offset + numel].view(shape))
                    offset += numel
                else:
                    unflat.append(None)

            proj_task_grads.append(unflat)

        return proj_task_grads
