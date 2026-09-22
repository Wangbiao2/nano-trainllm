"""Print exactly what one micro-batch looks like, without loading any weights.

The single most useful debugging tool in this repo.  Every silent training bug
in this space is a batch bug -- a label off by one, a placeholder run that
doesn't match its grid, a position id that restarts in the wrong place -- and
all of them are visible here.

    python tools/inspect_batch.py --stage sft --data examples/data/sft.jsonl --packing

Only the processor and `config.json` are read, so this runs on a CPU-only box in
a couple of seconds.  Supervised tokens are printed inside `[[...]]`; the marker
is placed by walking the labels, not by re-deriving the spans, so what you see
is what the loss will use.
"""

from __future__ import annotations

import argparse
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from nanotrainllm.collate import collate, sequence_index  # noqa: E402
from nanotrainllm.config import Config  # noqa: E402
from nanotrainllm.dataset import Dataset  # noqa: E402
from nanotrainllm.template import IGNORE, Template  # noqa: E402


def marked(tok, ids: list[int], labels: list[int], mm: list[int]) -> str:
    """Decode `ids`, bracketing the maximal runs whose label is supervised.

    Vision runs are collapsed to `<N image tokens>` -- printing 64 identical
    placeholders teaches nobody anything, and a 4-megapixel image would be
    thousands of them.
    """
    out, i = [], 0
    while i < len(ids):
        j, sup, vis = i, labels[i] != IGNORE, mm[i] != 0
        while j < len(ids) and (labels[j] != IGNORE) == sup and (mm[j] != 0) == vis:
            j += 1
        text = f"<{j - i} image tokens>" if vis else tok.decode(ids[i:j])
        out.append(f"[[{text}]]" if sup else text)
        i = j
    return "".join(out)


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--model", default="Qwen/Qwen3.5-4B")
    p.add_argument("--data", required=True)
    p.add_argument("--stage", default="sft")
    p.add_argument("--teacher", default=None)
    p.add_argument("--max-length", type=int, default=4096)
    p.add_argument("--micro-batch-size", type=int, default=2)
    p.add_argument("--num-generations", type=int, default=2)
    p.add_argument("--packing", action="store_true")
    p.add_argument("--padding-free", action="store_true")
    p.add_argument("--truncation", default="right")
    p.add_argument("--epoch", type=int, default=0)
    p.add_argument("--batch", type=int, default=0, help="which micro-batch of the epoch")
    p.add_argument("--tokens", type=int, default=0, help="also dump the first N token rows")
    args = p.parse_args()
    if args.packing:
        args.micro_batch_size = 1  # a packed micro-batch is one bin; Config asserts this

    from transformers import AutoConfig, AutoProcessor

    processor = AutoProcessor.from_pretrained(args.model)
    cfg = Config.from_kwargs(**{k.replace("-", "_"): v for k, v in vars(args).items()})
    cfg.bind_model_meta(AutoConfig.from_pretrained(args.model), processor.tokenizer)
    tpl = Template(processor, cfg)

    ds = Dataset(args.data, tpl, cfg)
    stats = {k: round(v, 4) if isinstance(v, float) else v for k, v in ds.stats().items()}
    print("dataset:", " ".join(f"{k}={v}" for k, v in stats.items()))
    batches = ds.batches(args.epoch)
    print(f"micro-batches this epoch: {len(batches)}  (showing #{args.batch})\n")
    batch = collate(batches[args.batch], cfg)

    layout = "padding-free" if batch.padding_free else "padded"
    print(f"layout      : {layout}")
    print(f"input_ids   : {tuple(batch.input_ids.shape)}")
    print(f"position_ids: {tuple(batch.position_ids.shape)}  max={int(batch.position_ids.max())}")
    if batch.cu_seq_lens_q is not None:
        print(f"cu_seq_lens : {batch.cu_seq_lens_q.tolist()}  max_length_q={batch.max_length_q}")
    if batch.pixel_values is not None:
        print(f"pixel_values: {tuple(batch.pixel_values.shape)}")
        print(f"image_grid  : {batch.image_grid_thw.tolist()}")
    n_sup = int((batch.labels != IGNORE).sum())
    n_tok = int(batch.input_ids.numel() if batch.padding_free else batch.attention_mask.sum())
    print(f"tokens      : {n_tok} real, {n_sup} supervised ({n_sup / n_tok:.1%})")
    print(f"model kwargs: {sorted(batch.model_kwargs())}\n")

    seq = sequence_index(batch)
    tok = processor.tokenizer
    for i, n in enumerate(batch.seq_lens):
        where = seq == i
        ids = batch.input_ids[where].tolist()
        labels = batch.labels[where].tolist()
        mm = batch.mm_token_type_ids[where].tolist()
        pos = batch.position_ids[:, where]
        print(f"--- sample {i}  len={n}  prompt_len={batch.prompt_lens[i]}  meta={batch.meta[i]}")
        print(f"    positions {pos[:, 0].tolist()} .. {pos[:, -1].tolist()}")
        print("    " + marked(tok, ids, labels, mm).replace("\n", "\\n"))
        for k in range(min(args.tokens, n)):
            lab = labels[k]
            print(
                f"    {k:>5} id={ids[k]:>6} pos={pos[:, k].tolist()} mm={mm[k]} "
                f"label={'-' if lab == IGNORE else lab:>6} {tok.decode([ids[k]])!r}"
            )
        print()


if __name__ == "__main__":
    main()
