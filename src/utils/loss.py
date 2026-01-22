"""
Custom loss functions
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


class FocalLoss(nn.Module):
    """
    Focal Loss is a modified Cross-Entropy Loss that tackles class imbalance in deep
    learning, especially object detection, by down-weighting easily classified examples
    and focusing training on hard-to-classify examples

    Focal Loss: FL(p_t) = -alpha_t * (1 - p_t)^gamma * log(p_t)
    """

    def __init__(self, alpha=None, gamma=2.0, reduction="mean"):
        super().__init__()

        if alpha is not None:
            if isinstance(alpha, (list, tuple)):
                alpha = torch.tensor(alpha, dtype=torch.float32)
            self.register_buffer("alpha", alpha)
        else:
            self.alpha = None

        self.gamma = gamma
        self.reduction = reduction

    def forward(self, logits, targets):
        """
        Args:
            logits: [batch_size, num_classes] - raw logits
            targets: [batch_size] - class indices
        """
        # Compute log probabilities directly (more numerically stable)
        log_probs = F.log_softmax(logits, dim=-1)  # [B, C]

        # Get log_prob of correct class for each sample
        log_p_t = log_probs.gather(1, targets.unsqueeze(1)).squeeze(1)  # [B]

        # Compute p_t from log_p_t (one exp instead of full softmax)
        p_t = log_p_t.exp()  # [B]

        # Focal weight
        focal_weight = (1 - p_t) ** self.gamma  # [B]

        # Focal loss = -focal_weight * log(p_t)
        focal_loss = -focal_weight * log_p_t  # [B]

        # Apply class weights if provided
        if self.alpha is not None:
            alpha_t = self.alpha.gather(0, targets)  # [B]
            focal_loss = alpha_t * focal_loss

        # Reduction
        if self.reduction == "mean":
            return focal_loss.mean()
        elif self.reduction == "sum":
            return focal_loss.sum()
        return focal_loss
