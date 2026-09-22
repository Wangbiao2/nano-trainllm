# Contributing

The point of this project is to be **read**. These conventions therefore matter more than
features: a pull request that makes the code longer and the reader more tired will not be
merged even if it works.

## Before you submit

```bash
python3 -m py_compile $(git ls-files '*.py')
awk 'length > 100 {print FILENAME":"FNR}' $(git ls-files '*.py')
```

Then the six `--dry-run` commands from the README's Quick start, which are the smoke test.
Write their checkpoints outside the tree (`--output-dir /tmp/...`).

`--dry-run` builds a tiny random model with the **real** vocabulary and the real token ids,
so a real encoded batch -- image included -- flows end to end on CPU. It is the whole
smoke test; every stage must pass it. Anything that can only be verified on a GPU should
say so explicitly in the PR, together with the hardware it needs.

## House rules

**One flat `Config`.** No config-of-configs, no YAML, no schema library. Adding a knob
means adding a field in `config.py`; the CLI grows the matching `--flag` on its own
(`cli.py` generates them from `fields(Config)`). **Do not hand-write argparse.** If a new
field's type annotation is not in the `SCALARS` table, `parser()` asserts rather than
guesses -- extend the table, do not work around it.

**A knob needs a user.** Before adding a field, ask whether anything in `examples/` will
ever set it to a non-default value. If the answer is "no, but the paper has this variant",
do not add it -- write the difference into the relevant function's docstring and let the
reader change that line. Already dropped this way: cDPO label smoothing, IPO/hinge, the k1
and k2 KL estimators, the distillation temperature. A knob nobody sets is an untested
branch and a paragraph the reader has to skip.

**Use a bare `assert` to say "unsupported".** Asserts here have two jobs: the runtime check,
and documenting a path we did not make work. Do not wrap them in try/except and do not
downgrade them to warnings. `--packing` with `sdpa` asserts because that combination fails
*silently* -- the full-attention layers would attend across sample boundaries -- and silent
wrongness is far worse than a crash.

**No `print`, no `logging` in the library.** Only `cli.py`'s `report()` and `loop.py`'s
progress bar write anything, and only on rank 0. `Trainer.run()` returns its history as
data; whoever wants numbers takes them. A library that configures logging cannot be
embedded.

**Never let `[B, S, V]` exist.** The vocabulary is ~250k tokens, so one 8k sequence's logits
are ~4 GiB. `Policy.hidden()` returns hidden states only and `logprobs.py` applies `lm_head`
in chunks. A new loss that needs a whole distribution goes through one of the chunked
functions there, or adds one -- never `model(**batch).logits`.

**The denominator is global.** A stage's `denom(batches)` must be computed over the whole
optimiser step and reduced across ranks **before any backward**, which is why `loop.py`
never divides by `grad_accum`. `micro_batch=1 x accum=4` and `micro_batch=4 x accum=1` must
produce exactly the same gradient.

**Metrics must be additive.** `loop.py` does `metrics[k] += v`, so return token counts and
summed NLLs, not means. Divide by `denom` in the display layer.

**One reasoning docstring per module, around 16 lines, 20 at the outside.** Write **why it
is this way** and **how a different spelling fails**; do not write API documentation -- the
signature already says what the arguments are. Inline comments only where the logic is not
self-evident. Line width is 100.

**`__post_init__` must be idempotent.** It runs on construction and again after the CLI
overrides fields, so deriving a value must not depend on whether it has already been
derived.

## Adding a stage

The three questions in `stages/__init__.py` are the entire interface:

1. `micro_batches(units)` -- which forward passes make up one optimiser step. Rollout
   happens here rather than in the loop, which is what keeps `loop.py` free of
   `if stage == ...`.
2. `denom(batches)` -- what the summed loss is divided by.
3. `loss(batch, denom)` -- the scalar to call `.backward()` on, plus a dict of
   already-summed metrics.

Then add a branch in `build()` (**a local import**, so that `--stage sft` never imports the
RL code) and the name to `STAGES` in `config.py`.

## Adding a reward

A function `fn(text, ref, cfg, n_tokens) -> float` in `rl/reward.py`, registered in the
`REWARDS` table; select it by name with `--reward-funcs` and weight it with
`--reward-weights`. Note that a group whose rewards are all equal has an identically zero
advantage and therefore exactly zero gradient. That is GRPO behaving correctly, not a bug
-- do not change the normalisation to manufacture a gradient.
