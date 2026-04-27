"""
Multipositive InfoNCE loss for seasonal geo-localization.

Generated drone views serve as additional positive anchors:
- Original drone embeddings are the primary queries
- Generated drone embeddings are extra rows in the similarity matrix
- All drone embeddings (original + generated) at location A should match satellite A
- The target distributes probability across matching satellites

Batch structure:
    features_query:   [N, D]    — original drone embeddings
    features_ref:     [N, D]    — satellite embeddings
    features_gen:     [G, D]    — generated drone embeddings (variable per batch)
    gen_loc_ids:      [G]       — location index for each generated view

Loss matrix: [N+G, N] where row i is positive for column j
if they share the same location.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


class MultipositiveInfoNCE(nn.Module):
    """InfoNCE with generated drone views as additional positive anchors.

    The loss extends the standard [N, N] similarity matrix to [N+G, N]
    where G generated views provide extra positive signal without being
    used as primary queries.
    """

    def __init__(self, label_smoothing=0.1, device='cuda'):
        super().__init__()
        self.label_smoothing = label_smoothing
        self.device = device

    def forward(self, features_query, features_ref, logit_scale,
                loc_ids=None, features_gen=None, gen_loc_ids=None):
        """
        Args:
            features_query: Original drone features [N, D].
            features_ref: Satellite features [N, D].
            logit_scale: Learnable temperature (scalar).
            loc_ids: Location indices for primary pairs [N].
            features_gen: Generated drone features [G, D] (optional).
            gen_loc_ids: Location indices for generated views [G] (optional).
        """
        features_query = F.normalize(features_query, dim=-1)
        features_ref = F.normalize(features_ref, dim=-1)

        N = len(features_query)

        if features_gen is not None and len(features_gen) > 0:
            features_gen = F.normalize(features_gen, dim=-1)
            # Concatenate: [original; generated] → [N+G, D]
            all_query = torch.cat([features_query, features_gen], dim=0)
            all_loc_ids = torch.cat([loc_ids, gen_loc_ids], dim=0)
        else:
            all_query = features_query
            all_loc_ids = loc_ids

        # Similarity: [N+G, N]
        logits_q2r = logit_scale * all_query @ features_ref.T

        # Standard direction: satellite → original drone only [N, N]
        logits_r2q = logit_scale * features_ref @ features_query.T

        # Build targets
        # For q→r: each row (drone, original or gen) matches columns where
        # loc_ids match. Since each column j corresponds to loc_ids[j],
        # row i is positive for column j if all_loc_ids[i] == loc_ids[j]
        targets_q2r = self._build_targets(all_loc_ids, loc_ids)

        # For r→q: standard diagonal (satellite → original drone only)
        targets_r2q = self._build_targets(loc_ids, loc_ids)

        loss = (
            self._soft_cross_entropy(logits_q2r, targets_q2r)
            + self._soft_cross_entropy(logits_r2q, targets_r2q)
        ) / 2

        return loss

    def _build_targets(self, row_ids, col_ids):
        """Build soft target matrix where match[i,j] = 1 if same location."""
        row_ids = row_ids.unsqueeze(1)  # [R, 1]
        col_ids = col_ids.unsqueeze(0)  # [1, C]
        match = (row_ids == col_ids).float()  # [R, C]

        n_positives = match.sum(dim=1, keepdim=True).clamp(min=1)
        targets = match / n_positives

        B = targets.shape[1]
        if self.label_smoothing > 0:
            targets = (1 - self.label_smoothing) * targets + self.label_smoothing / B

        return targets.to(self.device)

    @staticmethod
    def _soft_cross_entropy(logits, targets):
        """Cross entropy with soft targets."""
        log_probs = F.log_softmax(logits, dim=1)
        loss = -(targets * log_probs).sum(dim=1).mean()
        return loss
