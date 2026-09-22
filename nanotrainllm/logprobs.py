"""Memory-bounded log-prob extraction.  This is the load-bearing file.

The vocabulary is ~250k tokens.  One 8192-token sequence's logits are
8192 x 250k x 2 bytes = **~4 GiB** in bf16, log_softmax wants fp32, and autograd keeps
it alive until backward -- ~8 GiB for a single sequence.

So nothing here ever holds a `[batch, seq, vocab]` tensor.  Every consumer of logits
goes through one of the four functions below, all of which run the LM head in chunks of
`chunk` rows and wrap each chunk in `torch.utils.checkpoint` when gradients are needed.
Peak logits memory is then O(chunk x vocab) rather than O(N x vocab): 1.0 GiB fp32 at
chunk=1024, whatever the sequence length.

Two numerical conventions used throughout:

  * `logp = logits.gather(target) - logits.logsumexp(-1)`, not
    `log_softmax(-1).gather(target)` -- identical, but log_softmax materialises a
    second `[chunk, vocab]` fp32 tensor.
  * logits are cast to fp32 *before* any reduction.  A bf16 logsumexp over 250k terms
    loses ~2 decimal digits, enough to make an importance ratio of a truly-unchanged
    policy read as 1.02.
"""

from __future__ import annotations

import torch
import torch.nn as nn
from torch.utils.checkpoint import checkpoint

NEG_INF = float("-inf")


def _ranges(n: int, chunk: int):
    for start in range(0, n, chunk):
        yield start, min(start + chunk, n)


def _head(lm_head: nn.Module, hidden: torch.Tensor, temperature: float = 1.0) -> torch.Tensor:
    logits = lm_head(hidden).float()
    if temperature != 1.0:
        logits = logits / temperature
    return logits


def _run_chunked(fn, *tensors, chunk: int, grad: bool):
    """Apply `fn` to row-slices of `tensors` and concatenate along dim 0.

    `fn` must return one row per input row, each small (a scalar, or `topk` columns):
    that is what makes checkpointing profitable -- save the tiny output, throw away the
    enormous intermediate.
    """
    n = tensors[0].shape[0]
    out = []
    for start, end in _ranges(n, chunk):
        args = tuple(t[start:end] for t in tensors)
        if grad:
            out.append(checkpoint(fn, *args, use_reentrant=False))
        else:
            out.append(fn(*args))
    return torch.cat(out, dim=0)


# ---------------------------------------------------------------------------
# 1. log p(target_t | prefix) for one given token per position
# ---------------------------------------------------------------------------


def token_logprobs(
    hidden: torch.Tensor,
    lm_head: nn.Module,
    target_ids: torch.Tensor,
    temperature: float = 1.0,
    chunk: int = 1024,
) -> torch.Tensor:
    """`hidden [N, H]`, `target_ids [N]` -> `logp [N]`.

    The caller is responsible for the shift: `hidden[i]` must be the state that predicts
    `target_ids[i]`.  Deliberately not hidden inside this function -- see
    `shift_for_causal_lm` below.
    """
    assert hidden.ndim == 2 and target_ids.ndim == 1
    assert hidden.shape[0] == target_ids.shape[0]

    def step(h, t):
        logits = _head(lm_head, h, temperature)
        return logits.gather(-1, t[:, None].long()).squeeze(-1) - logits.logsumexp(-1)

    return _run_chunked(step, hidden, target_ids, chunk=chunk, grad=hidden.requires_grad)


# ---------------------------------------------------------------------------
# 2. cross-entropy for pt / sft
# ---------------------------------------------------------------------------


def ce_loss(
    hidden: torch.Tensor,
    lm_head: nn.Module,
    labels: torch.Tensor,
    denom: torch.Tensor | float | None = None,
    chunk: int = 1024,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Token-level cross entropy.  Returns `(loss, num_tokens)`.

    `denom` is the *global* token count for the whole optimiser step, summed over every
    micro-batch and every rank.  Dividing each micro-batch by its own count is a real and
    silent bug: under accumulation it weights short micro-batches more heavily, so
    `micro_batch=1 x accum=4` and `micro_batch=4 x accum=1` stop agreeing.  `denom=None`
    gives the plain per-micro-batch mean, correct only when `grad_accum == 1`.
    """
    assert hidden.ndim == 2 and labels.ndim == 1
    keep = labels != -100
    n_tokens = keep.sum()
    if n_tokens == 0:
        return hidden.sum() * 0.0, n_tokens
    logp = token_logprobs(hidden[keep], lm_head, labels[keep], chunk=chunk)
    total = -logp.sum()
    if denom is None:
        return total / n_tokens, n_tokens
    return total / denom, n_tokens


# ---------------------------------------------------------------------------
# 3. entropy, for monitoring only
# ---------------------------------------------------------------------------


def entropy(
    hidden: torch.Tensor,
    lm_head: nn.Module,
    temperature: float = 1.0,
    chunk: int = 1024,
) -> torch.Tensor:
    """`H = logsumexp(z) - sum(softmax(z) * z)`, per row.  Always no-grad.

    Falling entropy with flat reward is the classic GRPO collapse signature, so
    this is worth logging even though it costs an extra LM-head pass.
    """
    with torch.no_grad():

        def step(h):
            logits = _head(lm_head, h, temperature)
            lse = logits.logsumexp(-1)
            return lse - (logits.softmax(-1) * logits).sum(-1)

        return _run_chunked(step, hidden, chunk=chunk, grad=False)


# ---------------------------------------------------------------------------
# 4. union top-k log-probs, for on-policy distillation
# ---------------------------------------------------------------------------


def _union_indices(
    logits_s: torch.Tensor, logits_t: torch.Tensor, k: int
) -> tuple[torch.Tensor, torch.Tensor]:
    """Row-wise union of the two top-k index sets, kept dense.

    A true per-row union is ragged (between k and 2k entries).  Instead we keep `[C, 2k]`
    and return a `dup` mask marking the repeats; callers set those columns to -inf in
    *both* distributions, where `exp(-inf) = 0` makes them exactly equivalent to absent.
    """
    idx = torch.cat([logits_s.topk(k, -1).indices, logits_t.topk(k, -1).indices], dim=-1)
    idx, _ = idx.sort(dim=-1)
    dup = torch.zeros_like(idx, dtype=torch.bool)
    dup[:, 1:] = idx[:, 1:] == idx[:, :-1]
    return idx, dup


def _logprobs_at(
    logits: torch.Tensor, idx: torch.Tensor, dup: torch.Tensor, with_tail: bool
) -> torch.Tensor:
    """Gather `logits` at `idx` and normalise, optionally with a tail bucket.

    Without a tail bucket we renormalise over the selected support only -- the usual
    "top-k distillation" approximation, which is not a bound on anything.  With one, the
    appended column holds `logsumexp(everything not selected)`, so the result sums to
    exactly 1 and the KL from it is a real KL between distributions over a *partition* of
    the vocabulary: a lower bound on the full-vocabulary KL, tightening with k.
    """
    total = logits.logsumexp(-1, keepdim=True)
    kept = logits.gather(-1, idx)
    kept = kept.masked_fill(dup, NEG_INF)
    if not with_tail:
        return kept - kept.logsumexp(-1, keepdim=True)
    kept_lse = kept.logsumexp(-1, keepdim=True)
    # tail = log(exp(total) - exp(kept_lse)); clamp keeps log1p out of log(0).
    gap = (kept_lse - total).clamp(max=-1e-6)
    tail = total + torch.log1p(-gap.exp())
    return torch.cat([kept, tail], dim=-1) - total


def union_topk_logprobs(
    hidden_s: torch.Tensor,
    lm_head_s: nn.Module,
    hidden_t: torch.Tensor,
    lm_head_t: nn.Module,
    k: int = 64,
    union: bool = True,
    with_tail: bool = True,
    chunk: int = 1024,
) -> tuple[torch.Tensor, torch.Tensor]:
    """-> `(student_logp, teacher_logp)`, both `[N, M]` on a shared support.

    `M = 2k (+1)` when `union`, else `k (+1)`.  Gradients flow through the student only.

    Why the union, rather than gathering the student at the *teacher's* top-k?  Reverse KL
    `sum_v p_S(v) log(p_S(v)/p_T(v))` is driven by tokens where the **student** puts mass --
    exactly the ones a teacher-only support throws away.  There a student confidently
    emitting a token the teacher dislikes gets renormalised out of existence and pays no
    penalty, which is the mode-seeking pressure reverse KL is supposed to supply.  Because
    the teacher is in-process, both sides are gathered *exactly* at the union indices.

    The extra student LM-head pass in stage 1 is deliberate: deriving the union inside the
    gradient-tracked loop would force the teacher's logits to be recomputed during backward.
    """
    assert hidden_s.ndim == 2 and hidden_t.ndim == 2
    assert hidden_s.shape[0] == hidden_t.shape[0]

    # -- stage 1: pick the shared support and score the teacher on it (no grad)
    with torch.no_grad():

        def teacher_step(hs, ht):
            logits_t = _head(lm_head_t, ht)
            if union:
                logits_s = _head(lm_head_s, hs)
                idx, dup = _union_indices(logits_s, logits_t, k)
            else:
                idx = logits_t.topk(k, -1).indices
                dup = torch.zeros_like(idx, dtype=torch.bool)
            return torch.cat(
                [idx.float(), dup.float(), _logprobs_at(logits_t, idx, dup, with_tail)], dim=-1
            )

        packed = _run_chunked(teacher_step, hidden_s, hidden_t, chunk=chunk, grad=False)

    m = k * 2 if union else k
    idx = packed[:, :m].long()
    dup = packed[:, m : 2 * m].bool()
    teacher_logp = packed[:, 2 * m :]

    # -- stage 2: score the student on that same support (grad flows here)
    def student_step(h, i, d):
        return _logprobs_at(_head(lm_head_s, h), i, d, with_tail)

    student_logp = _run_chunked(
        student_step, hidden_s, idx, dup, chunk=chunk, grad=hidden_s.requires_grad
    )
    return student_logp, teacher_logp


# ---------------------------------------------------------------------------
# the shift, written once
# ---------------------------------------------------------------------------


def shift_for_causal_lm(
    hidden: torch.Tensor, labels: torch.Tensor
) -> tuple[torch.Tensor, torch.Tensor]:
    """`hidden [B, S, H]`, `labels [B, S]` -> `hidden [B*(S-1), H]`, `labels [B*(S-1)]`.

    Position `i` of the hidden state predicts token `i+1`, so drop the last hidden state
    and the first label.  Doing this in exactly one place is the only defence against the
    off-by-one that otherwise appears independently in four paths.  Under packing the
    shift leaks one position across every sub-sequence boundary; `collate.py` sets the
    boundary label to -100 so the leaked pair is masked out rather than corrected here.
    """
    assert hidden.ndim == 3 and labels.ndim == 2
    h = hidden[:, :-1, :].reshape(-1, hidden.shape[-1])
    y = labels[:, 1:].reshape(-1)
    return h, y
