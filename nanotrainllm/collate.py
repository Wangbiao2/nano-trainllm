"""Encoded samples -> the exact tensors `Qwen3_5ForConditionalGeneration` wants.

Three mutually exclusive layouts, one exit:

  * **padded** -- `[B, S]` with right padding and an `attention_mask`.  The
    baseline; every other layout is checked against it.
  * **padding-free** -- `[1, total]`, no pad tokens, no `attention_mask`, with
    `cu_seq_lens_q/k` marking the boundaries.  Viable on this model because the
    GatedDeltaNet layers accept `cu_seqlens` too, not just full-attention ones.
  * **packing** -- `dataset.py` best-fits samples into `max_length` bins first,
    then the bin goes down the padding-free path.

Two things here are load-bearing and neither fails loudly:

1. **We always build `position_ids` ourselves.**  The model only computes them
   when `position_ids is None`, so passing them means its `self.rope_deltas` cache
   is never read *or* written during training.  That cache is mutable state left
   over from `generate()`, and a GRPO loop alternates the two.

2. **Sub-sequence boundaries must be masked in the labels.**  The causal shift
   pairs the last token of sub-sequence j with the first token of j+1, so
   `labels[start] = -100` for every sub-sequence: one supervised token lost per
   sample, in exchange for no cross-sample leakage.

Positions are not token counts: under M-RoPE a text token advances all three axes
by 1, but an image advances them by `max(h, w) // merge_size` in total, because the
h/w axes are spatial indices.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import torch

from .template import IGNORE, Encoded


@dataclass(slots=True)
class Batch:
    input_ids: torch.Tensor  # [B, S]
    labels: torch.Tensor  # [B, S]
    position_ids: torch.Tensor  # [3, B, S]
    mm_token_type_ids: torch.Tensor  # [B, S]
    attention_mask: torch.Tensor | None = None  # None in the padding-free layout
    pixel_values: torch.Tensor | None = None
    image_grid_thw: torch.Tensor | None = None
    cu_seq_lens_q: torch.Tensor | None = None
    cu_seq_lens_k: torch.Tensor | None = None
    max_length_q: int = 0
    max_length_k: int = 0
    seq_lens: list[int] = field(default_factory=list)  # per sample, in order
    prompt_lens: list[int] = field(default_factory=list)
    meta: list[dict] = field(default_factory=list)
    # `[B, S-1]` log p_old on the completion frame, cached at rollout time for the
    # off-policy (num_iterations>1 or global<rollout) PPO update; None => on-policy,
    # in which case `rl/loss.py` uses logp.detach() and the ratio is identically 1.
    old_logp: torch.Tensor | None = None

    @property
    def padding_free(self) -> bool:
        return self.attention_mask is None

    def to(self, device, dtype=None) -> "Batch":
        for name in (
            "input_ids",
            "labels",
            "position_ids",
            "mm_token_type_ids",
            "attention_mask",
            "pixel_values",
            "image_grid_thw",
            "cu_seq_lens_q",
            "cu_seq_lens_k",
        ):
            t = getattr(self, name)
            if t is not None:
                # only pixel_values is floating point; casting the rest would break them
                setattr(self, name, t.to(device, dtype) if t.is_floating_point() else t.to(device))
        # old_logp is floating point but must keep its own dtype: casting it to bf16
        # would quantise log p_old and bias the importance ratio, so move device only.
        if self.old_logp is not None:
            self.old_logp = self.old_logp.to(device)
        return self

    def model_kwargs(self) -> dict:
        """Exactly the keyword arguments to splat into the model call.

        `labels` is deliberately absent: this framework never lets transformers
        compute the loss, because that path materialises `[B, S, ~250k]` logits.
        """
        kw = {
            "input_ids": self.input_ids,
            "position_ids": self.position_ids,
            "mm_token_type_ids": self.mm_token_type_ids,
            "use_cache": False,
        }
        if self.attention_mask is not None:
            kw["attention_mask"] = self.attention_mask
        if self.pixel_values is not None:
            kw["pixel_values"] = self.pixel_values
            kw["image_grid_thw"] = self.image_grid_thw
        if self.cu_seq_lens_q is not None:
            kw["cu_seq_lens_q"] = self.cu_seq_lens_q
            kw["cu_seq_lens_k"] = self.cu_seq_lens_k
            kw["max_length_q"] = self.max_length_q
            kw["max_length_k"] = self.max_length_k
        return kw


def runs(values: list[int]) -> list[tuple[int, int, int]]:
    """`[(value, start, end)]` for each maximal run of equal values."""
    out = []
    i = 0
    while i < len(values):
        j = i
        while j < len(values) and values[j] == values[i]:
            j += 1
        out.append((values[i], i, j))
        i = j
    return out


def build_position_ids(
    mm_token_type_ids: list[int], grid: torch.Tensor | None, merge: int
) -> torch.Tensor:
    """M-RoPE position ids `[3, L]` for **one** sequence, starting at 0.

    Text runs get `arange(n) + cur` on all three axes; an image run gets the
    `(t, h, w)` mesh offset by `cur`, after which `cur` advances by only
    `max(llm_h, llm_w)` -- the h/w axes are spatial indices, not token counts.
    """
    n = len(mm_token_type_ids)
    pos = torch.zeros(3, n, dtype=torch.long)
    cur = 0
    seen = 0
    for kind, start, end in runs(mm_token_type_ids):
        if kind == 0:
            length = end - start
            pos[:, start:end] = torch.arange(length).view(1, -1) + cur
            cur += length
            continue
        assert kind == 1, f"only images are supported, got mm_token_type_id {kind}"
        assert grid is not None and seen < len(grid), "vision run without a matching grid"
        t, h, w = (int(x) for x in grid[seen])
        seen += 1
        gt, gh, gw = t, h // merge, w // merge
        assert gt * gh * gw == end - start, (
            f"grid {(t, h, w)} implies {gt * gh * gw} tokens but the run is {end - start}"
        )
        axis_t = (torch.arange(gt) + cur).view(-1, 1, 1).expand(gt, gh, gw)
        axis_h = (torch.arange(gh) + cur).view(1, -1, 1).expand(gt, gh, gw)
        axis_w = (torch.arange(gw) + cur).view(1, 1, -1).expand(gt, gh, gw)
        pos[:, start:end] = torch.stack([axis_t, axis_h, axis_w]).reshape(3, -1)
        cur += max(gh, gw)
    assert seen == (0 if grid is None else len(grid)), "unused image grids"
    return pos


def _vision(encs: list[Encoded]) -> tuple[torch.Tensor | None, torch.Tensor | None]:
    """Concatenate the vision tensors in sample order.

    Order matters and the model does not check it: the pieces are scattered into the
    placeholder positions in row-major order, so the n-th grid must describe the n-th
    placeholder run of the flattened batch.
    """
    px = [e.pixel_values for e in encs if e.pixel_values is not None]
    grids = [e.image_grid_thw for e in encs if e.image_grid_thw is not None]
    if not px:
        return None, None
    return torch.cat(px, dim=0), torch.cat(grids, dim=0)


def collate(encs: list[Encoded], cfg) -> Batch:
    assert encs
    return _padding_free(encs, cfg) if cfg.padding_free else _padded(encs, cfg)


def _padded(encs: list[Encoded], cfg) -> Batch:
    merge = cfg.merge_size
    b, s = len(encs), max(len(e) for e in encs)
    input_ids = torch.full((b, s), cfg.pad_token_id, dtype=torch.long)
    labels = torch.full((b, s), IGNORE, dtype=torch.long)
    mm = torch.zeros((b, s), dtype=torch.long)
    mask = torch.zeros((b, s), dtype=torch.long)
    pos = torch.zeros((3, b, s), dtype=torch.long)
    for i, e in enumerate(encs):
        n = len(e)
        input_ids[i, :n] = torch.tensor(e.input_ids)
        labels[i, :n] = torch.tensor(e.labels)
        mm[i, :n] = torch.tensor(e.mm_token_type_ids)
        mask[i, :n] = 1
        # pad slots keep position 0, matching what the model would compute itself
        pos[:, i, :n] = build_position_ids(e.mm_token_type_ids, e.image_grid_thw, merge)
    px, grid = _vision(encs)
    return Batch(
        input_ids=input_ids,
        labels=labels,
        position_ids=pos,
        mm_token_type_ids=mm,
        attention_mask=mask,
        pixel_values=px,
        image_grid_thw=grid,
        seq_lens=[len(e) for e in encs],
        prompt_lens=[e.prompt_len for e in encs],
        meta=[e.meta for e in encs],
    )


def _padding_free(encs: list[Encoded], cfg) -> Batch:
    merge = cfg.merge_size
    ids: list[int] = []
    lab: list[int] = []
    mmt: list[int] = []
    pos: list[torch.Tensor] = []
    cu = [0]
    for e in encs:
        ids += e.input_ids
        # the first label of every sub-sequence is the one the previous
        # sub-sequence's last hidden state would "predict"; drop it
        lab += [IGNORE] + e.labels[1:]
        mmt += e.mm_token_type_ids
        pos.append(build_position_ids(e.mm_token_type_ids, e.image_grid_thw, merge))
        cu.append(cu[-1] + len(e))
    px, grid = _vision(encs)
    longest = max(len(e) for e in encs)
    return Batch(
        input_ids=torch.tensor(ids).unsqueeze(0),
        labels=torch.tensor(lab).unsqueeze(0),
        position_ids=torch.cat(pos, dim=1).unsqueeze(1),
        mm_token_type_ids=torch.tensor(mmt).unsqueeze(0),
        attention_mask=None,
        pixel_values=px,
        image_grid_thw=grid,
        cu_seq_lens_q=torch.tensor(cu, dtype=torch.int32),
        cu_seq_lens_k=torch.tensor(cu, dtype=torch.int32),
        max_length_q=longest,
        max_length_k=longest,
        seq_lens=[len(e) for e in encs],
        prompt_lens=[e.prompt_len for e in encs],
        meta=[e.meta for e in encs],
    )


def sequence_index(batch: Batch) -> torch.Tensor:
    """`[B, S]` telling you which sample each position belongs to; -1 for padding.

    Nothing in the forward pass needs this -- it is for the loss aggregators and
    `tools/inspect_batch.py`, which group tokens back into samples.
    """
    out = torch.full_like(batch.input_ids, -1)
    if batch.padding_free:
        for i, (a, b) in enumerate(zip(batch.cu_seq_lens_q[:-1], batch.cu_seq_lens_q[1:])):
            out[0, a:b] = i
    else:
        for i, n in enumerate(batch.seq_lens):
            out[i, :n] = i
    return out
