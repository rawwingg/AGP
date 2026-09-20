"""
The reward basis and the curriculum's weight schedule.

Design summary (docs/agp_mdp_formulation.pdf, section 5):

The reward is ONE fixed list of terms. The curriculum never adds, removes or
redefines a term -- it only changes the weight column. That is the pattern both
multi-stage locomotion papers use, and it means a single environment
implementation serves every stage.

Two of the terms are *potential-based*: the coverage term and the guard-count
term are each the step-to-step change of a function of the state. Their sum over
an episode telescopes exactly into

    w_cov * (final coverage)  -  w_par * (final guard count)

which is the real objective. So those two can keep a fixed weight forever
without biasing the solution. The scaffolding terms are NOT potential-based --
they genuinely change what counts as a good answer -- so they must be annealed
to zero before training ends. That is the rule that decides which column
entries move and which stay put.
"""

from dataclasses import dataclass, fields, replace


# ---------------------------------------------------------------------------
# Weights -- the only thing the curriculum changes
# ---------------------------------------------------------------------------
@dataclass(frozen=True)
class RewardWeights:
    """One column of the schedule table. All values are non-negative magnitudes;
    the sign of each term is baked into the environment, not into the weight."""

    # --- coverage group: the task objective -------------------------------
    cov: float = 1.0      # coverage potential: +delta_plus on Add, -delta_minus on Del
    goal: float = 4.0     # terminal bonus when the gallery ends up covered
    fail: float = 2.0     # terminal penalty when it does not

    # --- parsimony group: use fewer guards --------------------------------
    par: float = 0.0      # guard-count potential: -1 per Add, +1 per Del
    prune: float = 0.0    # bounded bonus for a deletion that costs no coverage
    pars: float = 0.0     # terminal bonus per guard saved against the greedy baseline

    # --- scaffolding group: learning aids, annealed to zero ---------------
    front: float = 0.0    # placed the guard in still-uncovered space
    sweep: float = 0.0    # boundary-first, then inward placement prior
    sep: float = 0.0      # keep guards apart (stops early stacking)

    # --- regularity group: keep the agent inside the rules ----------------
    proj: float = 0.25    # aimed outside the gallery and had to be snapped in
    void: float = 0.0     # Del that was impossible or blocked by the ratchet
    stop: float = 0.0     # Stop called before the gallery was covered

    def blend(self, other: "RewardWeights", alpha: float) -> "RewardWeights":
        """Linear interpolation towards `other`; alpha=0 is self, alpha=1 is other.

        Used to cross a stage boundary gradually instead of switching the reward
        (and therefore every value target the critic has learned) in one step.
        """
        alpha = min(max(alpha, 0.0), 1.0)
        mixed = {f.name: (1 - alpha) * getattr(self, f.name) + alpha * getattr(other, f.name)
                 for f in fields(self)}
        return RewardWeights(**mixed)

    def scaled_scaffolds(self, scale: float) -> "RewardWeights":
        """Return a copy with the three scaffolding weights multiplied by `scale`."""
        return replace(self,
                       front=self.front * scale,
                       sweep=self.sweep * scale,
                       sep=self.sep * scale)


# ---------------------------------------------------------------------------
# Kernel shapes -- fixed across stages, so kept apart from the weights
# ---------------------------------------------------------------------------
@dataclass(frozen=True)
class RewardShape:
    """Parameters of the reward's functional forms, as opposed to its weights."""

    # How much lost coverage still counts as "a free deletion". The prune bonus
    # is exp(-delta_minus / sigma_prune), so 0.005 means half a percent of the
    # gallery is the scale at which the bonus starts to die off.
    sigma_prune: float = 0.005

    # Boundary-inward sweep. The preferred distance-to-wall moves from mu_start
    # to mu_end over the guard budget; both are fractions of the bounding-box
    # diagonal, so the term is scale-free. sigma is the width of the band.
    sweep_mu_start: float = 0.02
    sweep_mu_end: float = 0.25
    sweep_sigma: float = 0.05

    # Two guards closer than this (fraction of the diagonal) count as stacked.
    min_separation: float = 0.15


# ---------------------------------------------------------------------------
# The schedule
# ---------------------------------------------------------------------------
# Stage 1, SATURATE. Cover the gallery; guards are free, not merely cheap
# (par = 0). The delete and stop actions are masked off, so their weights are
# irrelevant here.
#
# Scaffold sizing: front + sweep is at most 0.10 per step, so over a full guard
# budget of 16 an agent that stalls forever can farm about 1.6. Reaching the
# goal is worth 4, avoiding the failure penalty another 2, and the coverage
# potential 1 more. Covering the gallery therefore beats dawdling by a wide
# margin -- which is the calibration that keeps a per-step bonus safe.
SATURATE = RewardWeights(
    cov=1.0, goal=4.0, fail=2.0,
    par=0.0, prune=0.0, pars=0.0,
    front=0.04, sweep=0.06, sep=0.10,
    proj=0.25, void=0.0, stop=0.0,
)

# Stage 2, PRUNE. Delete and stop are unmasked, guards start costing, and a
# clean deletion pays. The coverage weight is unchanged -- it is potential-based,
# so leaving it alone costs nothing and protects against forgetting how to cover.
# All three scaffolds are already at zero.
#
# Weight sizing: a guard costs par = 0.15, so the whole budget of 16 costs 2.4,
# comfortably less than the 6-point swing between solving and failing. Parsimony
# can therefore never make failure look attractive. A free deletion pays
# par + prune = 0.35, which is large next to the per-step noise, so pruning is
# worth doing but never worth risking the goal for.
PRUNE = RewardWeights(
    cov=1.0, goal=4.0, fail=2.0,
    par=0.15, prune=0.20, pars=0.50,
    front=0.0, sweep=0.0, sep=0.0,
    proj=0.25, void=0.10, stop=2.0,
)


def scaffold_scale(progress: float, anneal_from: float = 0.7) -> float:
    """
    Multiplier on the scaffolding weights at `progress` through stage 1.

    Held at 1 until `anneal_from` of the stage has elapsed, then ramped linearly
    to 0. The tail of the stage therefore trains under the unbiased objective,
    giving the policy time to unlearn the prior's distortions before stage 2.
    """
    if progress <= anneal_from:
        return 1.0
    if progress >= 1.0:
        return 0.0
    return (1.0 - progress) / (1.0 - anneal_from)


def saturate_weights(progress: float) -> RewardWeights:
    """Stage-1 weights at a given fraction of the way through the stage."""
    return SATURATE.scaled_scaffolds(scaffold_scale(progress))


def prune_weights(progress: float, blend_frac: float = 0.05) -> RewardWeights:
    """
    Stage-2 weights, easing in from the stage-1 column over the first
    `blend_frac` of the stage so the critic is not hit with a step change.
    """
    alpha = 1.0 if blend_frac <= 0 else min(progress / blend_frac, 1.0)
    return SATURATE.scaled_scaffolds(0.0).blend(PRUNE, alpha)
