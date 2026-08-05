"""Bridge an rl_osfp ActorCritic checkpoint into the proven BC search shell.

`foundation/search_shell_main.py` is frozen at the bytes that scored 972.0 /
911.9 / 810.8 on the ladder and is never edited, so every adaptation has to
happen on this side of the boundary.

The mismatch is one line.  The shell does

    pol, _v = _NET.forward(...)

while the rl_osfp net returns `(option, count, value)`. It has a count head
and the 160d/5L BC net does not.  Unpacking three values into two names raises
`ValueError` *inside* `_net_scores`, whose deliberately broad `except` swallows
it and returns `None`.  The shell reads `None` as "no model available" and falls
back to handcrafted heuristic priors.

That failure is invisible from the outside: the archive builds, the agent plays
legal games to completion, latency looks normal, and the checkpoint we selected
is simply never consulted.  `verify_bcsearch_submission.py` exists to assert the
priors actually fire, and this shim exists so that they can.

Shipped into the archive as `nn_infer.py`, alongside the real implementation
under `nn_infer_osfp.py`.
"""
from nn_infer_osfp import NumpyNet as _OSFPNet


class NumpyNet(_OSFPNet):
    """`NumpyNet` with the shell's two-value forward contract."""

    def forward(self, kind, card, scal, mask, ctx_id, stype_id):
        option, _count, value = super().forward(
            kind, card, scal, mask, ctx_id, stype_id)
        return option, value
