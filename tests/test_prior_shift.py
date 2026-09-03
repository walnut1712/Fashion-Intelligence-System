"""Task 1 label-shift correction - the maths, on synthetic data.

These are deliberately synthetic and fast. The real evidence that correction
helps is ``outputs/evaluation/task1_prior_shift_simulation.csv``, which needs a
trained model and a couple of minutes; what is pinned here is the set of
properties that must hold for that evidence to mean anything, and which broke
silently at least once each while the module was being written.
"""

import sys
from pathlib import Path

import numpy as np
import pytest

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.evaluation.prior_shift import (  # noqa: E402
    apply_prior_correction,
    estimate_prior_em,
    importance_weights,
    support_shrink,
)


def _synthetic_posteriors(y_true, num_classes, sharpness=6.0, seed=0):
    """A well-calibrated-ish classifier: the true class gets most of the mass."""
    rng = np.random.default_rng(seed)
    logits = rng.normal(size=(len(y_true), num_classes))
    logits[np.arange(len(y_true)), y_true] += sharpness
    logits -= logits.max(axis=1, keepdims=True)
    probabilities = np.exp(logits)
    return probabilities / probabilities.sum(axis=1, keepdims=True)


def _draw(prior, n, seed=0):
    rng = np.random.default_rng(seed)
    return rng.choice(len(prior), size=n, p=prior)


def test_em_returns_the_train_prior_when_nothing_has_shifted():
    """The null case. If this fails, every positive result is noise."""
    num_classes = 8
    train_prior = np.array([0.30, 0.20, 0.15, 0.12, 0.10, 0.06, 0.04, 0.03])
    y = _draw(train_prior, 4000, seed=1)
    probabilities = _synthetic_posteriors(y, num_classes, seed=1)

    estimated, _, _, converged = estimate_prior_em(probabilities, train_prior)

    assert converged
    total_variation = 0.5 * np.abs(estimated - train_prior).sum()
    assert total_variation < 0.05, f"EM invented a {total_variation:.1%} shift out of nothing"


def test_em_recovers_a_prior_it_was_not_told_about():
    """The whole method rests on this: the target prior is never passed in."""
    num_classes = 8
    train_prior = np.array([0.30, 0.20, 0.15, 0.12, 0.10, 0.06, 0.04, 0.03])
    target_prior = train_prior[::-1].copy()
    y = _draw(target_prior, 6000, seed=2)
    probabilities = _synthetic_posteriors(y, num_classes, seed=2)

    estimated, _, _, _ = estimate_prior_em(probabilities, train_prior, clip=None)

    naive = 0.5 * np.abs(train_prior - target_prior).sum()
    recovered = 0.5 * np.abs(estimated - target_prior).sum()
    assert recovered < naive / 3, (
        f"EM cut the prior error only from {naive:.1%} to {recovered:.1%}")


def test_correction_improves_accuracy_under_a_shift_it_estimated_itself():
    num_classes = 8
    train_prior = np.array([0.40, 0.25, 0.15, 0.08, 0.05, 0.03, 0.02, 0.02])
    target_prior = train_prior[::-1].copy()
    y = _draw(target_prior, 6000, seed=3)
    # Deliberately soft: a saturated classifier has no headroom for a prior to
    # matter, and would make this test pass for the wrong reason.
    probabilities = _synthetic_posteriors(y, num_classes, sharpness=1.5, seed=3)

    estimated, _, _, _ = estimate_prior_em(probabilities, train_prior, clip=None)
    corrected = apply_prior_correction(probabilities, train_prior, estimated,
                                       tau_already_removed=True)

    before = float((probabilities.argmax(1) == y).mean())
    after = float((corrected.argmax(1) == y).mean())
    assert after > before, f"correction did not help: {before:.3f} -> {after:.3f}"


def test_correction_refuses_to_compose_with_tau_adjustment():
    """Both push the prior around. Doing both silently corrects twice."""
    train_prior = np.full(4, 0.25)
    probabilities = np.full((3, 4), 0.25)
    checkpoint = {"logit_adjustment_tau": 0.2}

    with pytest.raises(ValueError, match="logit_adjustment_tau"):
        apply_prior_correction(probabilities, train_prior, train_prior,
                               checkpoint=checkpoint)

    # ... but it is allowed once the caller says the tau path was bypassed.
    apply_prior_correction(probabilities, train_prior, train_prior,
                           checkpoint=checkpoint, tau_already_removed=True)


def test_alpha_zero_is_the_identity():
    rng = np.random.default_rng(4)
    probabilities = rng.dirichlet(np.ones(6), size=50)
    train_prior = np.full(6, 1 / 6)
    target_prior = rng.dirichlet(np.ones(6))

    unchanged = apply_prior_correction(probabilities, train_prior, target_prior,
                                       alpha=0.0, tau_already_removed=True)
    assert np.allclose(unchanged, probabilities, atol=1e-9)


def test_correction_returns_a_distribution():
    rng = np.random.default_rng(5)
    probabilities = rng.dirichlet(np.ones(6), size=50)
    train_prior = rng.dirichlet(np.ones(6))
    target_prior = rng.dirichlet(np.ones(6))

    corrected = apply_prior_correction(probabilities, train_prior, target_prior,
                                       tau_already_removed=True)
    assert np.allclose(corrected.sum(axis=1), 1.0)
    assert (corrected >= 0).all()


def test_support_shrink_damps_starved_classes_and_frees_well_supported_ones():
    """The guard that stopped unguarded EM losing 31 accuracy points."""
    counts = np.array([11, 12, 15, 100, 400, 4843])
    shrink = support_shrink(counts)

    assert ((shrink >= 0) & (shrink <= 1)).all()
    # Non-decreasing, not strictly increasing: the curve saturates at 200 rows,
    # so everything above that is equally trusted and moves freely.
    assert (np.diff(shrink) >= 0).all(), "shrinkage must not fall as support rises"
    below_saturation = counts < 200
    assert (np.diff(shrink[below_saturation]) > 0).all(), (
        "below saturation, more rows must earn more freedom to move")
    assert shrink[0] < 0.5, "a class with 11 rows should be damped hard"
    assert shrink[-1] == pytest.approx(1.0), "a class with 4843 rows should move freely"
    assert shrink[counts == 400][0] == pytest.approx(1.0), "saturation is at 200 rows"


def test_importance_weights_preserve_sample_size_and_reweight_the_right_way():
    train_prior = np.array([0.8, 0.2])
    target_prior = np.array([0.2, 0.8])
    y = np.array([0, 0, 0, 0, 1, 1, 1, 1])

    weights = importance_weights(y, target_prior, train_prior)

    assert weights.sum() == pytest.approx(len(y))
    # The class the target has more of must be weighted up relative to the other.
    assert weights[y == 1].mean() > weights[y == 0].mean()
    assert weights[y == 1].mean() / weights[y == 0].mean() == pytest.approx(16.0)
