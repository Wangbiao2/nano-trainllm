"""Rule-based rewards.  No reward model, on purpose.

A learned reward model doubles the memory footprint and hides the part of GRPO
worth studying, which is what happens *between* the reward and the gradient.
Verifiable rewards are one scalar per completion, computable on CPU, and
recomputable by hand when a run looks wrong.

    fn(text, ref, cfg, n_tokens) -> float

`text` is the completion only (never the prompt), `ref` is whatever the row put in
`"solution"`, `n_tokens` is its generated length.  The signature is uniform so
`compute_rewards` can dispatch by name from `--reward-funcs accuracy,format`; add
a rule by adding one entry to `REWARDS`.

Two traps, both handled below:

  * **Extraction is the whole game.**  `1,000` vs `1000`, `\\boxed{x=5}` vs `5` --
    an exact-match reward that does not normalise measures typesetting rather than
    correctness, and the advantage will teach exactly that.
  * **Reward hacking through length.**  Rewarded only for containing the right
    answer, a model learns to enumerate answers.  `length` is DAPO's soft overlong
    penalty: free until `LENGTH_CACHE` tokens from the cap, then linearly to -1.
    It returns a *negative* number, so it takes a positive weight.
"""

from __future__ import annotations

import re

import torch

BOXED = re.compile(r"\\boxed\{([^{}]*)\}")
ANSWER_TAG = re.compile(r"<answer>(.*?)</answer>", re.S)
NUMBER = re.compile(r"-?\d+(?:[.,]\d+)*")
THINK_END = "</think>"

# tokens of slack before the overlong penalty starts biting
LENGTH_CACHE = 512


def extract(text: str) -> str | None:
    """The model's final answer, by the first convention that matches.

    Order matters: `\\boxed{}` is the strongest signal, `<answer>` next, a bare
    trailing number last.  That fallback is what makes this usable on datasets that
    never asked for a format, and is also why `format` exists as a separate reward --
    to pay for the structure instead of relying on the fallback forever.
    """
    boxed = BOXED.findall(text)
    if boxed:
        return boxed[-1].strip()
    m = ANSWER_TAG.search(text)
    if m:
        return m.group(1).strip()
    nums = NUMBER.findall(text)
    return nums[-1] if nums else None


def normalise(s: str) -> str:
    """Strip the decorations that make two identical answers compare unequal."""
    s = s.strip().strip("$").replace(" ", "").replace(",", "")
    s = s.rstrip(".").replace("\\!", "").replace("\\,", "")
    if s.startswith("\\text{") and s.endswith("}"):
        s = s[6:-1]
    # "x=5" and "5" are the same answer to "what is x"
    if "=" in s:
        s = s.split("=")[-1]
    # 5.0 == 5, but only when both really are numbers.  OverflowError is not
    # hypothetical: `\boxed{inf}` parses as a float and then refuses to be an int,
    # and a reward function that raises takes the whole run down with it.
    try:
        f = float(s)
        return repr(int(f)) if f == int(f) else repr(f)
    except (ValueError, OverflowError):
        return s.lower()


def accuracy(text: str, ref: str | None, cfg, n_tokens: int) -> float:
    """1.0 if the extracted answer matches the reference, else 0.0.

    The reference goes through `extract` too, so a dataset storing
    `"solution": "... so \\boxed{42}"` and one storing `"solution": "42"` score the
    same completion identically.
    """
    if ref is None:
        return 0.0
    got, want = extract(text), extract(ref) or ref
    if got is None:
        return 0.0
    return 1.0 if normalise(got) == normalise(want) else 0.0


def math(text: str, ref: str | None, cfg, n_tokens: int) -> float:
    """Symbolic equality via `mathruler.grade_answer`.

    `accuracy` normalises strings and falls back to the last number, which turns a
    symbolic reference like `2 \\sqrt { 221 }` into `221` and scores a bare `221` as
    correct.  This one extracts the boxed answer and defers to mathruler, so
    `2\\sqrt{221}` == `2 \\sqrt { 221 }` while `221` does not, and `\\frac{1}{2}` ==
    `0.5`.  Use it for symbolic references; `accuracy` stays right for plain numbers.

    No `\\boxed{}` means zero -- no last-number fallback here, which is why this reward
    is meant to be paired with `format` and a prompt that asks for a box.  The import
    is local so a run using only `accuracy` never needs mathruler installed.
    """
    if ref is None:
        return 0.0
    from mathruler.grader import extract_boxed_content, grade_answer

    got = extract_boxed_content(text)
    if got in ("", "None"):  # mathruler returns the string "None" when unboxed
        return 0.0
    want = extract_boxed_content(ref)
    if want in ("", "None"):  # a bare reference is the ground truth as-is
        want = ref
    return 1.0 if grade_answer(got, want) else 0.0


def format_ok(text: str, ref: str | None, cfg, n_tokens: int) -> float:
    """1.0 for exactly one closed thinking block followed by a boxed answer.

    The chat template already opens `<think>` (see `template.py`), so what is checked
    is that the model closed it exactly once and then produced a parseable answer --
    the two failure modes being "never stops thinking" and "thinks in the answer".
    """
    if text.count(THINK_END) != 1:
        return 0.0
    tail = text.split(THINK_END, 1)[1]
    return 1.0 if (BOXED.search(tail) or ANSWER_TAG.search(tail)) else 0.0


def length(text: str, ref: str | None, cfg, n_tokens: int) -> float:
    """0.0 until the completion nears `max_completion_length`, then down to -1.0."""
    cap = cfg.max_completion_length
    start = max(cap - LENGTH_CACHE, 1)
    if n_tokens <= start:
        return 0.0
    return -min((n_tokens - start) / (cap - start + 1e-6), 1.0)


REWARDS = {"accuracy": accuracy, "math": math, "format": format_ok, "length": length}


def compute_rewards(
    texts: list[str],
    refs: list[str | None],
    n_tokens: list[int],
    cfg,
) -> tuple[torch.Tensor, dict[str, float]]:
    """-> (`[B]` weighted total, per-rule summed values for logging).

    Weights come from `--reward-weights`; an empty flag means all 1.0.  The per-rule
    sums are logged separately because the total alone cannot tell you that accuracy is
    flat while the format reward carries the whole run -- the most common way a GRPO run
    looks healthy and learns nothing.
    """
    assert len(texts) == len(refs) == len(n_tokens)
    names, weights = cfg.reward_func_names, cfg.reward_weight_values
    assert names, "--reward-funcs is empty; a GRPO step needs at least one reward"
    for name in names:
        assert name in REWARDS, f"unknown reward {name!r}; have {sorted(REWARDS)}"

    total = torch.zeros(len(texts), dtype=torch.float32)
    logged: dict[str, float] = {}
    for name, w in zip(names, weights):
        fn = REWARDS[name]
        vals = torch.tensor(
            [fn(t, r, cfg, n) for t, r, n in zip(texts, refs, n_tokens)],
            dtype=torch.float32,
        )
        logged[f"reward/{name}"] = float(vals.sum())
        total += w * vals
    logged["reward"] = float(total.sum())
    return total, logged