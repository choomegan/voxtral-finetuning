"""
PCGrad implementation. Doesn't work with DDP.
"""

import torch
import torch.nn as nn
import random


class PCGrad:
    """    
    Reference: Yu et al. "Gradient Surgery for Multi-Task Learning" (NeurIPS 2020)
    """

    def __init__(self, optimizer, reduction='mean'):
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
        """Perform optimization step (called by Trainer) using modified gradients."""
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
        Compute per-task gradients, project conflicting ones, apply to model.
        
        Args:
            losses: List of scalar loss tensors, one per task
        """
        if len(losses) == 1:
            # Single task - just do normal backward
            self.zero_grad()
            losses[0].backward()
            return
        
        # Compute per-task gradients
        # Single-GPU only: multiple backward passes are safe here
        task_grads = []
        for i, loss in enumerate(losses):
            self.zero_grad() # Clear previous gradients
            retain = (i < len(losses) - 1)
            loss.backward(retain_graph=retain)
            
            # # DEBUG: Check if gradients were computed
            # grad_count = 0
            # total_norm = 0.0
            # for group in self._optim.param_groups:
            #     for p in group["params"]:
            #         if p.grad is not None:
            #             grad_count += 1
            #             total_norm += p.grad.norm().item() ** 2
            # total_norm = total_norm ** 0.5
            # print(f"  Task {i}: {grad_count} params with gradients, norm = {total_norm:.4f}")
            
            # Collect gradients
            task_grads.append(self._get_grads())

        # Apply PCGrad projection
        proj_grads = self._project_conflicting(task_grads)

        # Now apply the projected gradients
        self.zero_grad()
        self._set_grads(proj_grads)
        
        # # DEBUG: Check final gradient norm
        # final_norm = 0.0
        # final_count = 0
        # for group in self._optim.param_groups:
        #     for p in group["params"]:
        #         if p.grad is not None:
        #             final_count += 1
        #             final_norm += p.grad.norm().item() ** 2
        # final_norm = final_norm ** 0.5
        # print(f"  After projection: {final_count} params with gradients, norm = {final_norm:.4f}")

    def _get_grads(self):
        """
        Get gradients as a list of tensors (not flattened).
        Returns a list of (param, grad) tuples for parameters with gradients.
        """
        grads = []
        for group in self._optim.param_groups:
            for p in group["params"]:
                if p.grad is not None:
                    grads.append(p.grad.clone())  # Clone to avoid modification
                else:
                    grads.append(None)
        return grads

    def _set_grads(self, proj_grads):
        """
        Set gradients from projected gradients.
        
        Args:
            proj_grads: List of lists of gradients (one per task)
        """
        # Combine projected gradients
        if self.reduction == 'mean':
            # Average across tasks
            combined = []
            for task_grads in zip(*proj_grads):
                # Filter out None values
                valid_grads = [g for g in task_grads if g is not None]
                if valid_grads:
                    combined.append(sum(valid_grads) / len(valid_grads))
                else:
                    combined.append(None)
        elif self.reduction == 'sum':
            # Sum across tasks
            combined = []
            for task_grads in zip(*proj_grads):
                valid_grads = [g for g in task_grads if g is not None]
                if valid_grads:
                    combined.append(sum(valid_grads))
                else:
                    combined.append(None)
        else:
            raise ValueError(f"Unknown reduction: {self.reduction}")

        # Set gradients on parameters
        idx = 0
        for group in self._optim.param_groups:
            for p in group["params"]:
                if combined[idx] is not None:
                    p.grad = combined[idx]
                idx += 1

    def _project_conflicting(self, task_grads):
        """
        Core PCGrad projection algorithm to resolve conflicting gradients between tasks.
        
        Args:
            task_grads: List of lists of gradients (one per task) from _get_grads()
        """
        num_tasks = len(task_grads)
        task_indices = list(range(num_tasks))
        random.shuffle(task_indices)

        # Flatten gradients
        flat_grads = []
        shapes = []
        for task_grad_list in task_grads:
            flat = []
            task_shapes = []
            for g in task_grad_list:
                if g is not None:
                    task_shapes.append(g.shape)
                    flat.append(g.view(-1))
                else:
                    task_shapes.append(None)
            shapes = task_shapes
            flat_grads.append(torch.cat(flat) if flat else None)

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
                    grad = grad - dot / (torch.dot(prev_grad, prev_grad) + 1e-12) * prev_grad

            proj_flat_grads[task_idx] = grad

        # 🔑 UNFLATTEN
        proj_task_grads = []
        for task_idx in range(num_tasks):
            flat = proj_flat_grads[task_idx]
            if flat is None:
                proj_task_grads.append([None] * len(shapes))
                continue

            unflat = []
            offset = 0
            for shape in shapes:
                if shape is not None:
                    numel = torch.prod(torch.tensor(shape)).item()
                    unflat.append(flat[offset:offset + numel].view(shape))
                    offset += numel
                else:
                    unflat.append(None)

            proj_task_grads.append(unflat)

        return proj_task_grads
