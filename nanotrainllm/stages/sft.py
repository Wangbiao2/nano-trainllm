"""pt and sft: one trainer, two label masks.

There is no `pt.py`, deliberately.  Continued pretraining and supervised
fine-tuning differ in *exactly one* place -- which positions carry a label -- and
that place is `template.py`: `encode_pt` skips the chat template and sets
`labels = input_ids`, `encode_sft` supervises only the assistant spans.  By the
time a batch reaches this file the arithmetic is identical, so writing it twice
would only create two places for the same denominator bug to hide.

That denominator is the one subtlety here: `denom` counts `labels[:, 1:]`, not all
of `labels`, because the causal shift in `shift_for_causal_lm` drops column 0.
Counting the unshifted labels over-counts by one supervised token per row and
makes the reported loss quietly too small.
"""

from __future__ import annotations

import torch

from ..logprobs import ce_loss, shift_for_causal_lm
from ..template import IGNORE
from . import Stage, all_sum


class CrossEntropy(Stage):
    def denom(self, batches: list) -> float:
        """Supervised tokens in this optimiser step, summed over every rank."""
        n = sum(int((b.labels[:, 1:] != IGNORE).sum()) for b in batches)
        # A step in which nothing at all is supervised would divide by zero; the
        # loss is then exactly 0 anyway, so any positive denominator will do.
        return max(all_sum(n, device=self.policy.device), 1.0)

    def loss(self, batch, denom: float) -> tuple[torch.Tensor, dict]:
        hidden = self.policy.hidden(batch)
        h, y = shift_for_causal_lm(hidden, batch.labels)
        loss, n = ce_loss(h, self.policy.lm_head, y, denom=denom, chunk=self.cfg.logit_chunk)
        return loss, {"tokens": int(n)}
