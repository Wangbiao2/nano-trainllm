"""jsonl -> encoded samples, length statistics, and best-fit packing bins.

1. **Encoding is eager.**  Every row is run through `Template` in `__init__`, so
   the whole dataset (including `pixel_values`) sits in memory.  A real cost, paid
   on purpose: packing needs every length up front, and the alternative is an
   offline caching pass -- more machinery than a readable framework should carry.
2. **The atomic unit is a *group*, not a sample.**  `Template.encode` returns a
   list: one entry for most stages, two adjacent ones for DPO (chosen, rejected).
   Shuffling, sharding and batching all move *groups*, which is what keeps a DPO
   pair inside one forward pass.  Split it across an accumulation boundary and
   `logp_chosen - logp_rejected` quietly spans two different micro-batches.
3. **Packing is best-fit-decreasing, not first-fit.**  Sort by length descending,
   place each sample in the *fullest* bin that still fits: a few percent of tokens
   wasted where first-fit wastes tens of percent, in six lines.
"""

from __future__ import annotations

import json
import random

from .template import Encoded, Template


def read_jsonl(path: str) -> list[dict]:
    rows = []
    with open(path, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                rows.append(json.loads(line))
    assert rows, f"no usable rows in {path!r}"
    return rows


def pack_bins(lengths: list[int], max_length: int) -> list[list[int]]:
    """Best-fit-decreasing bin packing.  Returns bins of indices into `lengths`.

    `room[b]` is the free space left in bin `b`; "best fit" is the smallest such room
    that still admits the item, which is what keeps the bins tight.  The scan is
    O(n_bins) per item -- fine for one step's batch, and a heap would only obscure it.
    """
    order = sorted(range(len(lengths)), key=lambda i: -lengths[i])
    bins: list[list[int]] = []
    room: list[int] = []
    for i in order:
        need = lengths[i]
        assert need <= max_length, f"sample {i} has {need} tokens, bin size is {max_length}"
        best = -1
        for b, free in enumerate(room):
            if free >= need and (best < 0 or free < room[best]):
                best = b
        if best < 0:
            bins.append([i])
            room.append(max_length - need)
        else:
            bins[best].append(i)
            room[best] -= need
    return bins


class Dataset:
    """Encoded groups plus the batching policy.  No torch tensors are built here.

    `batches(epoch)` is the only thing `loop.py` calls.  It returns a list of
    micro-batches, each a flat `list[Encoded]`, already sharded for this rank.
    """

    def __init__(self, path: str, template: Template, cfg):
        self.cfg = cfg
        self.groups: list[list[Encoded]] = []
        self.dropped = 0
        for row in read_jsonl(path):
            group = [template.truncate(e, cfg.max_length) for e in template.encode(row)]
            if any(e is None for e in group):
                # A group is all-or-nothing: half a DPO pair is not a sample.
                self.dropped += 1
                continue
            self.groups.append(group)
        assert self.groups, f"every one of the rows in {path!r} was dropped by truncation"

    def __len__(self) -> int:
        return len(self.groups)

    def __getitem__(self, i: int) -> list[Encoded]:
        return self.groups[i]

    @property
    def group_lengths(self) -> list[int]:
        return [sum(len(e) for e in g) for g in self.groups]

    # -- batching ----------------------------------------------------------
    def batches(self, epoch: int = 0) -> list[list[Encoded]]:
        """Shuffle -> shard by rank -> group into micro-batches.

        The shuffle seed folds in the epoch so successive epochs differ, and *not* the
        rank, so every rank agrees on the permutation before slicing its own stride out
        of it.  Ranks then always see the same number of micro-batches, which is what
        keeps `all_reduce` from deadlocking on a ragged tail.
        """
        idx = list(range(len(self.groups)))
        random.Random(self.cfg.seed + epoch).shuffle(idx)

        if self.cfg.packing:
            lengths = self.group_lengths
            bins = pack_bins([lengths[i] for i in idx], self.cfg.max_length)
            units = [[idx[i] for i in b] for b in bins]
        else:
            # On-policy stages draw *prompts*, one per unit: `micro_batch_size` counts
            # generated completions, not prompts, and `grpo.micro_batches` slices those
            # into forward micro-batches after generation.
            per = 1 if self.cfg.is_on_policy else self.cfg.micro_batch_size
            units = [idx[s : s + per] for s in range(0, len(idx), per)]

        # Truncate to a rank-independent count *before* sharding: with a stride shard,
        # rank 0 would otherwise get one micro-batch more than the last rank and the
        # extra step's all_reduce would hang waiting for everyone else.
        keep = len(units) // self.cfg.world_size * self.cfg.world_size
        units = units[:keep][self.cfg.rank :: self.cfg.world_size]
        return [[e for i in u for e in self.groups[i]] for u in units]

    # -- dry-run reporting -------------------------------------------------
    def stats(self) -> dict:
        lengths = self.group_lengths
        n_sup = sum(sum(1 for y in e.labels if y != -100) for g in self.groups for e in g)
        n_tok = sum(lengths)
        n_img = sum(e.num_image_tokens for g in self.groups for e in g)
        srt = sorted(lengths)
        return {
            "groups": len(self.groups),
            "samples": sum(len(g) for g in self.groups),
            "dropped": self.dropped,
            "tokens": n_tok,
            "supervised": n_sup,
            "supervised_frac": n_sup / max(n_tok, 1),
            "image_tokens": n_img,
            "len_min": srt[0],
            "len_p50": srt[len(srt) // 2],
            "len_p99": srt[min(len(srt) - 1, int(len(srt) * 0.99))],
            "len_max": srt[-1],
        }
