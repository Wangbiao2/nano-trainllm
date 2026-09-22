"""messages -> (input_ids, labels, vision tensors).

Everything model-specific lives here, including the hard-coded chat-structure
token ids below: this framework targets one model family, and looking the ids up
by string would turn a loud failure into a silent one.

The invariant that catches almost every vision bug, asserted on every encode:
    `(input_ids == image_token_id).sum() == sum(grid.prod(-1) // merge_size**2)`
"""

from __future__ import annotations

import os
from dataclasses import dataclass, field

import torch

# Qwen3.5 chat-structure token ids, verified against the checkpoint's tokenizer.
IM_START = 248045
IM_END = 248046
ROLE_ASSISTANT = 74455
NEWLINE = 198
# `<think>` `\n\n` `</think>` `\n\n` -- what the last assistant turn renders to
# when it has no reasoning of its own.  Four tokens, because `\n\n` is one (271).
EMPTY_THINK = (248068, 271, 248069, 271)
# `tokenizer.eos_token_id` is <|im_end|>, but text_config.eos_token_id is
# <|endoftext|>.  Generation must stop on either.
STOP_TOKEN_IDS = (248046, 248044)

IGNORE = -100


@dataclass(slots=True)
class Encoded:
    input_ids: list[int]
    labels: list[int]
    mm_token_type_ids: list[int]
    pixel_values: torch.Tensor | None = None
    image_grid_thw: torch.Tensor | None = None
    prompt_len: int = 0  # tokens before the part we generate / score
    meta: dict = field(default_factory=dict)
    # The *undecoded* images, for backends that preprocess themselves: vLLM owns its
    # own image processor and wants PIL objects, not our `pixel_values`.  A typed
    # field rather than a `meta` key because `to_encoded` copies `meta` into every
    # trainable row, and PIL objects have no business travelling that far.
    images: list = field(default_factory=list)

    def __len__(self) -> int:
        return len(self.input_ids)

    @property
    def num_image_tokens(self) -> int:
        return sum(1 for t in self.mm_token_type_ids if t != 0)


def assistant_spans(ids: list[int]) -> list[tuple[int, int]]:
    """Half-open `[start, end)` spans to supervise, one per assistant turn.

    `start` is just after `<|im_start|>assistant\\n`, `end` just after the turn's
    `<|im_end|>` -- included on purpose, since a model that never learns to emit it
    never stops generating.  Found by walking token ids rather than re-encoding
    `messages[:i]`, which would re-run the image preprocessor once per turn.
    """
    spans: list[tuple[int, int]] = []
    i, n = 0, len(ids)
    while i < n:
        if ids[i : i + 3] == [IM_START, ROLE_ASSISTANT, NEWLINE]:
            j = i + 3
            while j < n and ids[j] != IM_END:
                j += 1
            assert j < n, "unterminated assistant turn; the chat template changed?"
            spans.append((i + 3, j + 1))
            i = j + 1
        else:
            i += 1
    return spans


class Template:
    def __init__(self, processor, cfg):
        self.processor = processor
        self.cfg = cfg
        self.tokenizer = processor.tokenizer
        self.image_token_id = cfg.image_token_id
        self.merge_size = processor.image_processor.merge_size

    # -- message normalisation --------------------------------------------
    def _load_image(self, ref):
        from PIL import Image

        if hasattr(ref, "convert"):
            return ref.convert("RGB")
        assert isinstance(ref, str) and os.path.isfile(ref), f"image not found: {ref!r}"
        return Image.open(ref).convert("RGB")

    def normalize(self, row: dict) -> tuple[list[dict], int]:
        """Accept both common row shapes.

        Marker style: `{"messages": [...content: "<image>what?"], "images": ["a.jpg"]}`,
        where each `<image>` consumes the next entry of `images`.  HF style: content is
        already a list of typed parts.  Returns the messages and the number of images
        consumed, so callers can assert the pool was fully used.
        """
        pool = list(row.get("images") or [])
        used = 0
        inline = False  # did any content part carry its own image object?
        out = []
        for msg in row["messages"]:
            content = msg["content"]
            if isinstance(content, list):
                parts = []
                for p in content:
                    if p.get("type") == "image":
                        src = p.get("image", pool[used] if used < len(pool) else None)
                        if p.get("image") is None:
                            used += 1
                        else:
                            inline = True
                        parts.append({"type": "image", "image": self._load_image(src)})
                    else:
                        parts.append(p)
                out.append({"role": msg["role"], "content": parts})
                continue
            parts = []
            # `enumerate`, not `if parts`: a leading "<image>..." produces an empty
            # first chunk, so "parts is still empty" is a different question from "is
            # this the first chunk" -- using it loses every leading marker.
            for j, chunk in enumerate(str(content).split("<image>")):
                if j:
                    assert used < len(pool), "more <image> markers than images"
                    parts.append({"type": "image", "image": self._load_image(pool[used])})
                    used += 1
                if chunk:
                    parts.append({"type": "text", "text": chunk})
            out.append({"role": msg["role"], "content": parts or [{"type": "text", "text": ""}]})
        # Images listed but never referenced are prepended to the first user turn,
        # which is what VL datasets in the wild mean by it.  Doing that when the
        # content already carries its own image objects would silently double the
        # image, so it is an error instead -- that assert is why this is idempotent.
        if used < len(pool):
            assert not inline, "row mixes inline images with an unconsumed `images` pool"
            extra = [{"type": "image", "image": self._load_image(p)} for p in pool[used:]]
            for m in out:
                if m["role"] == "user":
                    m["content"] = extra + m["content"]
                    break
            used = len(pool)
        return out, used

    # -- the single call into the HF processor -----------------------------
    def _apply(self, messages: list[dict], add_generation_prompt: bool) -> Encoded:
        kwargs = {}
        if add_generation_prompt and not self.cfg.enable_thinking:
            kwargs["enable_thinking"] = False
        out = self.processor.apply_chat_template(
            messages,
            tokenize=True,
            add_generation_prompt=add_generation_prompt,
            return_dict=True,
            return_tensors="pt",
            **kwargs,
        )
        ids = out["input_ids"][0].tolist()
        # Mandatory, not optional: transformers raises if `image_grid_thw` arrives
        # without it, because M-RoPE needs to know which positions are vision.  Being
        # per-token, it is padded and packed exactly like `input_ids`.
        mm = out["mm_token_type_ids"][0].tolist()
        grid = out.get("image_grid_thw")
        px = out.get("pixel_values")
        enc = Encoded(
            input_ids=ids,
            labels=[IGNORE] * len(ids),
            mm_token_type_ids=mm,
            pixel_values=px,
            image_grid_thw=grid,
            images=[p["image"] for m in messages for p in m["content"] if p.get("type") == "image"],
        )
        self.check_vision(enc)
        return enc

    def check_vision(self, enc: Encoded) -> None:
        n_ph = sum(1 for t in enc.input_ids if t == self.image_token_id)
        if enc.image_grid_thw is None:
            assert n_ph == 0, f"{n_ph} image placeholders but no image_grid_thw"
            return
        expect = int((enc.image_grid_thw.prod(-1) // self.merge_size**2).sum())
        assert n_ph == expect, (
            f"image placeholder count {n_ph} != grid-derived count {expect}; "
            "the processor and the template disagree"
        )
        assert enc.pixel_values is not None

    # -- per-stage encoders ------------------------------------------------
    def encode_pt(self, row: dict) -> Encoded:
        """Continued pretraining: no chat template, every token is supervised."""
        text = row.get("text")
        if text is None:  # tolerate a messages row by concatenating the contents
            text = "".join(str(m["content"]) for m in row["messages"])
        ids = self.tokenizer(text, add_special_tokens=False)["input_ids"]
        ids = ids + [STOP_TOKEN_IDS[1]]
        return Encoded(input_ids=ids, labels=list(ids), mm_token_type_ids=[0] * len(ids))

    def encode_sft(self, row: dict) -> Encoded:
        """Supervise every assistant turn -- except an *empty* `<think>` block.

        The template renders `<think>\\n{reasoning}\\n</think>\\n\\n` even with no
        reasoning, collapsing to `EMPTY_THINK`.  Those four tokens stay in the context
        but leave the loss: inference always prefills *through* `<think>`, so they are
        never sampled, and supervising them puts probability on "close the block and
        answer" exactly where thinking data teaches the opposite.
        """
        messages, _ = self.normalize(row)
        enc = self._apply(messages, add_generation_prompt=False)
        spans = assistant_spans(enc.input_ids)
        assert spans, "no assistant turn to supervise"
        k = len(EMPTY_THINK)
        for a, b in spans:
            enc.labels[a:b] = enc.input_ids[a:b]
            if tuple(enc.input_ids[a : a + k]) == EMPTY_THINK:
                enc.labels[a : a + k] = [IGNORE] * k
        enc.prompt_len = spans[0][0]
        return enc

    def encode_prompt(self, row: dict) -> Encoded:
        """Prompt only, ready for `generate`.  Nothing is supervised yet."""
        messages, _ = self.normalize(row)
        messages = [m for m in messages if m["role"] != "assistant"]
        enc = self._apply(messages, add_generation_prompt=True)
        enc.prompt_len = len(enc.input_ids)
        enc.meta = {k: row[k] for k in ("solution", "answer", "label") if k in row}
        # The raw question, for `--log-samples-every` to print beside the completion
        # and its reward.  Image parts are dropped; only the text asked.
        enc.meta["question"] = " ".join(
            p["text"]
            for m in messages
            for p in m["content"]
            if p.get("type") == "text" and p.get("text", "").strip()
        ).strip()
        return enc

    def encode_pair(self, row: dict) -> tuple[Encoded, Encoded]:
        """DPO: one shared prompt, two completions, encoded as two rows.

        Kept adjacent in the batch so `stages/dpo.py` can recover pairs with `[0::2]` /
        `[1::2]` after one concatenated forward -- both sides then see identical
        numerics.
        """
        messages, _ = self.normalize(row)
        base = [m for m in messages if m["role"] != "assistant"]
        out = []
        for key in ("chosen", "rejected"):
            assert key in row, f"dpo row needs a {key!r} field"
            full = base + [{"role": "assistant", "content": [{"type": "text", "text": row[key]}]}]
            # `base` already holds loaded PIL objects, so the images pool must not
            # be handed over a second time.
            enc = self.encode_sft({"messages": full})
            enc.meta = {"side": key}
            out.append(enc)
        assert out[0].prompt_len == out[1].prompt_len, "chosen/rejected prompts differ"
        return out[0], out[1]

    def encode(self, row: dict) -> list[Encoded]:
        stage = self.cfg.stage
        if stage == "pt":
            return [self.encode_pt(row)]
        if stage == "dpo":
            return list(self.encode_pair(row))
        if stage in ("grpo", "opd-rl"):
            return [self.encode_prompt(row)]
        return [self.encode_sft(row)]

    # -- truncation --------------------------------------------------------
    def truncate(self, enc: Encoded, max_length: int) -> Encoded | None:
        """Return a shortened copy, or None if the sample must be dropped.

        Vision tokens are never cut: slicing away half an image's placeholders breaks
        the placeholder-count invariant and the model raises deep inside the vision
        merge.  A cut that would land inside a vision run drops the sample instead.
        """
        n = len(enc)
        if n <= max_length:
            return enc
        if self.cfg.truncation == "drop":
            return None
        if self.cfg.truncation == "right":
            lo, hi = 0, max_length
        else:
            lo, hi = n - max_length, n
        if any(t != 0 for t in enc.mm_token_type_ids[:lo] + enc.mm_token_type_ids[hi:]):
            return None  # would orphan a pixel_values block
        return Encoded(
            input_ids=enc.input_ids[lo:hi],
            labels=enc.labels[lo:hi],
            mm_token_type_ids=enc.mm_token_type_ids[lo:hi],
            pixel_values=enc.pixel_values,
            image_grid_thw=enc.image_grid_thw,
            prompt_len=max(0, min(enc.prompt_len - lo, max_length)),
            meta=enc.meta,
            images=enc.images,
        )
