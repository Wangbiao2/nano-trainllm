"""Reinforcement-learning pieces, split by what they are functions *of*.

  * `reward.py`     completion text            -> scalar reward per completion
  * `advantage.py`  rewards                    -> per-token advantage
  * `loss.py`       (logp, old, ref, advantage)-> the scalar to backward
  * `rollout.py`    prompts                    -> completions

Nothing in here touches a model or a Config field it was not handed, and nothing
in here is `grpo`-specific: `stages/grpo.py` and `stages/opd_rl.py` differ only in
which of these they call and with what.  That split is why the RL code is ~400
lines: there is no trainer class for the stages to inherit from.
"""
