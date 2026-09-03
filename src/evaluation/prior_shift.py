"""Task 1 - the graded set is a different population, and the headline metric hides it.

Why this exists
---------------
Every Task 1 number in the repository is measured on a grouped-stratified split
of the labelled catalogue, where the class mix matches training by construction.
The graded submission is not that population. Predicted composition on the 5,829
unlabelled tiles sits at ~20% Personal Care against 2.59% in training, a total
variation distance of roughly 45%, and the classes carrying that mass are the
ones with the least training data in the whole dataset:

    class                    train rows   graded rows   graded share
    Lipstick                        15           294          5.04%
    Kajal and Eyeliner              13           174          2.99%
    Nail Polish                     19           126          2.16%
    Foundation and Primer           12           108          1.85%
    Lip Liner                       12            80          1.37%

17.4% of the submission (1,017 rows) sits on classes holding <=5 rows in the test
split, so their reported per-class F1 - 1.00 for several of them - is a two-row
coin flip. Meanwhile the model's own confidence on the graded set averages 0.699
against 0.867 on the labelled test split, and that confidence is trustworthy:
measured ECE on clean catalogue tiles is 0.018. A calibrated model reporting 0.699
is telling us it expects to be right about 70% of the time, not 87%.

What this module does about it
------------------------------
Three things, in the order they have to happen.

1. **Establishes the precondition.** Prior-correction methods are only licensed
   when the shift is in ``p(y)`` and not in ``p(x|y)``. ``covariate_shift_report``
   measures that directly and writes the table that either licenses the rest of
   this module or forbids it.

2. **Measures the right population.** ``prior_matched_metrics`` re-weights the
   labelled evaluation rows to the deployment prior. Note carefully what this
   does *not* fix: re-weighting the committed per-class recalls raises weighted-F1
   from 87.14 to 87.55, because it multiplies up classes whose recall was
   estimated as 1.00 from two rows. Importance weighting alone is not enough -
   it needs trustworthy per-class recalls first, which is what the starved-class
   cross-validation in ``train_item_type`` is for. Kish effective sample size is
   reported on every row precisely so this is visible: on the test split alone it
   is 353 against n=5,501, and one Lipstick row carries 2.5% of the total weight.

3. **Corrects the predictions.** ``estimate_prior_em`` runs the
   Saerens-Latinne-Decaestecker fixed point over the unlabelled posteriors, and
   ``apply_prior_correction`` re-weights with it. ``simulate_prior_shift``
   validates the whole chain against known priors before any of it is trusted,
   and ``correction_is_safe`` is the four-part guard that has to pass first.

Two hard-won rules are enforced rather than documented
------------------------------------------------------
**Feed the estimators raw posteriors.** ``predict_proba`` applies the checkpoint's
tau logit adjustment by default, and tau is itself a blind push away from the
training prior. Running a measured correction on top of a blind one corrects
twice: tau=0.2 alone already moves Foundation and Primer from 48 predicted rows
to 108. Every function here takes ``adjust=False`` posteriors, and
``apply_prior_correction`` refuses a checkpoint carrying a non-zero tau unless
the caller says it has been removed.

**Do not run EM unrestricted over all 92 classes.** A first pass that did lost 31
accuracy points, because the fixed point happily inflates classes whose
posteriors come from a dozen training rows. Clipping the weight ratio and damping
it by class support recovers it, and both are on by default.
"""

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import accuracy_score, f1_score

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.models.item_type_classifier import (choose_device, load_item_type_model,
                                             predict_proba)

ARTIFACT_DIR = PROJECT_ROOT / "artifacts" / "task1"
RESULTS_DIR = PROJECT_ROOT / "outputs" / "evaluation"
TEST_IMAGES = PROJECT_ROOT / "A2_FashionDataset" / "FashionDataset" / "test" / "images_test"
TRAIN_IMAGES = PROJECT_ROOT / "A2_FashionDataset" / "FashionDataset" / "train" / "images_train"
IMAGE_SUFFIXES = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}

# Shrinkage saturates at 200 training rows: a class with that many has a prior
# the EM step can be trusted to move, one with twelve does not. The curve is
# log-shaped rather than a hard cutoff so there is no cliff to tune around.
SHRINK_SATURATION = 200.0
DEFAULT_CLIP = 5.0
ALPHA_GRID = (0.0, 0.25, 0.5, 0.75, 1.0)

__all__ = [
    "graded_probabilities", "covariate_shift_report", "estimate_prior_em",
    "estimate_prior_bbse", "apply_prior_correction", "importance_weights",
    "prior_matched_metrics", "simulate_prior_shift", "correction_is_safe",
    "support_shrink",
]


# ------------------------------------------------------------------ posteriors
def _image_paths(directory):
    """Graded tiles in id order.

    Sorted numerically, not as strings. Every id here is five digits so the two
    orders agree today, but the failure mode if that ever changes is silent row
    misalignment against the submission template, which is worth one line to rule
    out.
    """
    paths = [p for p in Path(directory).iterdir() if p.suffix.lower() in IMAGE_SUFFIXES]
    return sorted(paths, key=lambda p: int(p.stem) if p.stem.isdigit() else p.stem)


def graded_probabilities(checkpoint_path=ARTIFACT_DIR / "task1_cnn.pt",
                         images_dir=TEST_IMAGES, ingest="squash", force=False,
                         batch_size=256, verbose=True):
    """Raw and tau-adjusted posteriors for the graded set, cached to disk.

    A full TTA pass over the 5,829 tiles costs about two minutes; everything
    downstream of it costs milliseconds. Both matrices are stored because the
    label-shift estimators need the raw one and the published predictions come
    from the adjusted one, and having only one of them on disk is how the two get
    confused for each other.
    """
    checkpoint_path = Path(checkpoint_path)
    cache = ARTIFACT_DIR / "graded_probabilities_{}.npz".format(checkpoint_path.stem)
    paths = _image_paths(images_dir)
    ids = np.array([int(p.stem) for p in paths], dtype=np.int64)

    if cache.exists() and not force:
        stored = np.load(cache, allow_pickle=True)
        if stored["ids"].shape == ids.shape and (stored["ids"] == ids).all():
            if verbose:
                print("cached posteriors: {}".format(cache.name))
            return (stored["ids"], stored["raw"], stored["adjusted"],
                    [str(c) for c in stored["class_names"]])

    device = choose_device()
    model, checkpoint = load_item_type_model(checkpoint_path, device=device)
    tta = bool(checkpoint.get("tta", False))
    if verbose:
        print("scoring {} tiles with {} (tta={}, ingest={})".format(
            len(paths), checkpoint_path.name, tta, ingest))

    raw = predict_proba(model, checkpoint, paths, batch_size=batch_size, device=device,
                        tta=tta, ingest=ingest, adjust=False)
    adjusted = predict_proba(model, checkpoint, paths, batch_size=batch_size,
                             device=device, tta=tta, ingest=ingest, adjust=True)
    class_names = list(checkpoint["class_names"])

    np.savez_compressed(cache, ids=ids, raw=raw.astype(np.float32),
                        adjusted=adjusted.astype(np.float32),
                        class_names=np.array(class_names, dtype=object),
                        run_id=str(checkpoint.get("run_id", "")), tta=tta, ingest=ingest)
    if verbose:
        print("wrote {}".format(cache))
    return ids, raw, adjusted, class_names


# ---------------------------------------------------------------- precondition
def _tile_statistics(paths):
    from PIL import Image
    rows = []
    for path in paths:
        array = np.asarray(Image.open(path).convert("RGB"), dtype=np.float32)
        border = np.concatenate([array[0], array[-1], array[:, 0], array[:, -1]])
        rows.append({
            "white_fraction": float((array.min(-1) >= 230).mean()),
            "border_white_fraction": float((border.min(-1) >= 230).mean()),
            "mean": float(array.mean()),
            "std": float(array.std()),
        })
    return pd.DataFrame(rows)


def covariate_shift_report(n=1200, seed=0, out=None, verbose=True):
    """Is the graded set the same *kind* of image as the training set?

    This is the table that licenses every other function in this module. Label
    shift assumes ``p(x|y)`` is unchanged and only ``p(y)`` moves; if the graded
    tiles were photographs rather than catalogue cutouts, prior correction would
    be the wrong tool and the photo-path model would be the right one.

    Tolerances are deliberately loose. The question is not whether the two
    populations are identical - they cannot be, the class mix differs - but
    whether they are the same imaging pipeline. A white-background cutout and an
    on-model lifestyle photograph differ on these statistics by an order of
    magnitude more than the tolerance.
    """
    rng = np.random.default_rng(seed)
    train_paths = _image_paths(TRAIN_IMAGES)
    graded_paths = _image_paths(TEST_IMAGES)
    train_sample = [train_paths[i] for i in rng.choice(len(train_paths), size=min(n, len(train_paths)), replace=False)]
    graded_sample = [graded_paths[i] for i in rng.choice(len(graded_paths), size=min(n, len(graded_paths)), replace=False)]

    train_stats = _tile_statistics(train_sample)
    graded_stats = _tile_statistics(graded_sample)

    tolerances = {"white_fraction": 0.15, "border_white_fraction": 0.05,
                  "mean": 20.0, "std": 20.0}
    rows = []
    for column, tolerance in tolerances.items():
        a, b = float(train_stats[column].mean()), float(graded_stats[column].mean())
        rows.append({
            "statistic": column, "train": round(a, 4), "graded": round(b, 4),
            "abs_difference": round(abs(a - b), 4), "tolerance": tolerance,
            "pass": bool(abs(a - b) <= tolerance),
        })
    a = float((train_stats["border_white_fraction"] > 0.9).mean())
    b = float((graded_stats["border_white_fraction"] > 0.9).mean())
    rows.append({"statistic": "fraction_border_white_over_0.9", "train": round(a, 4),
                 "graded": round(b, 4), "abs_difference": round(abs(a - b), 4),
                 "tolerance": 0.10, "pass": bool(abs(a - b) <= 0.10)})

    frame = pd.DataFrame(rows)
    frame["n_train"] = len(train_sample)
    frame["n_graded"] = len(graded_sample)
    out = Path(out) if out else RESULTS_DIR / "task1_covariate_shift_check.csv"
    out.parent.mkdir(parents=True, exist_ok=True)
    frame.to_csv(out, index=False)
    if verbose:
        print(frame.to_string(index=False))
        print("\nlabel-shift precondition: {}".format(
            "HOLDS" if bool(frame["pass"].all()) else "VIOLATED"))
        print("wrote {}".format(out))
    return frame


# -------------------------------------------------------------- prior estimation
def support_shrink(train_counts, saturation=SHRINK_SATURATION):
    """Per-class exponent in [0, 1] damping the correction by training support.

    A class with 200+ training rows has a prior the model estimates well enough
    to move all the way; a class with twelve does not, and letting EM move it
    freely is what cost 31 accuracy points on the first attempt. The exponent is
    applied to the weight, so 1.0 is the full correction and 0.0 is none.
    """
    counts = np.asarray(train_counts, dtype=float)
    return np.clip(np.log1p(counts) / np.log1p(saturation), 0.0, 1.0)


def estimate_prior_em(probabilities, train_prior, max_iter=500, tol=1e-7,
                      clip=DEFAULT_CLIP, shrink=None, init=None):
    """Saerens-Latinne-Decaestecker (2002) EM for the target prior.

    Alternates a soft assignment of the unlabelled rows under the current prior
    estimate with a re-estimate of that prior from the assignment. Uses only the
    posteriors, so it needs no labels for the target set - which is the entire
    point, because the graded set has none.

    ``clip`` bounds the weight ratio and ``shrink`` damps it per class. Both
    default to on. Unrestricted EM over 92 classes with a dozen training rows in
    some of them does not merely fail to help, it actively destroys the
    predictions.

    Returns ``(prior, weights, n_iter, converged)``.
    """
    probabilities = np.asarray(probabilities, dtype=np.float64)
    train_prior = np.asarray(train_prior, dtype=np.float64)
    train_prior = train_prior / train_prior.sum()
    weights = np.ones_like(train_prior) if init is None else np.asarray(init, dtype=np.float64)

    converged, used = False, max_iter
    for iteration in range(1, max_iter + 1):
        scaled = probabilities * weights
        posterior = scaled / np.clip(scaled.sum(axis=1, keepdims=True), 1e-12, None)
        updated = posterior.mean(axis=0) / np.clip(train_prior, 1e-12, None)
        updated = updated / np.clip((updated * train_prior).sum(), 1e-12, None)
        if clip is not None:
            updated = np.clip(updated, 1.0 / clip, clip)
        if shrink is not None:
            updated = np.power(updated, shrink)
        if np.abs(updated - weights).max() < tol:
            weights, converged, used = updated, True, iteration
            break
        weights = updated

    prior = weights * train_prior
    return prior / prior.sum(), weights, used, converged


def estimate_prior_bbse(val_probabilities, val_y, target_probabilities, num_classes=None):
    """Black-box shift estimation (Lipton et al. 2018), as an independent check.

    Solves ``C w = q`` where ``C`` is the soft confusion matrix on labelled
    validation rows and ``q`` the mean posterior on the target set. Unlike EM it
    is a single linear solve with no fixed point to converge.

    Expect this to be *unusable* on this problem and report it as such: with 92
    classes and only two validation rows for several of them, ``C`` is close to
    singular. The returned condition number is the evidence for that, and a
    method that visibly fails its own precondition is a better thing to put in a
    report than a method quietly pushed through.
    """
    val_probabilities = np.asarray(val_probabilities, dtype=np.float64)
    val_y = np.asarray(val_y)
    num_classes = num_classes or val_probabilities.shape[1]

    confusion = np.zeros((num_classes, num_classes), dtype=np.float64)
    for label in range(num_classes):
        rows = val_probabilities[val_y == label]
        if len(rows):
            confusion[:, label] = rows.sum(axis=0)
    confusion /= len(val_probabilities)

    q = np.asarray(target_probabilities, dtype=np.float64).mean(axis=0)
    singular = np.linalg.svd(confusion, compute_uv=False)
    condition = float(singular[0] / singular[-1]) if singular[-1] > 0 else float("inf")
    try:
        weights = np.linalg.solve(confusion, q)
    except np.linalg.LinAlgError:
        weights = np.linalg.lstsq(confusion, q, rcond=None)[0]

    weights = np.clip(weights, 0.0, None)
    train_prior = np.bincount(val_y, minlength=num_classes).astype(float)
    train_prior /= train_prior.sum()
    prior = weights * train_prior
    total = prior.sum()
    prior = prior / total if total > 0 else train_prior
    return prior, weights, condition, float(singular[-1])


def apply_prior_correction(probabilities, train_prior, target_prior, alpha=1.0,
                           checkpoint=None, tau_already_removed=False):
    """Re-weight posteriors from the training prior onto the target prior.

    ``p'(y|x) proportional to p(y|x) * (target_y / train_y) ** alpha``.

    ``alpha`` shrinks the correction toward doing nothing, which hedges against
    an imperfect prior estimate. It is swept exactly the way the checkpoint's tau
    is swept, and for the same reason.

    The guard matters. Post-hoc tau adjustment and prior correction are the same
    operation: tau pushes blindly toward uniform, this pushes toward a measured
    target. Applying both corrects twice and is silently wrong rather than
    loudly wrong, so it is refused rather than warned about.
    """
    if checkpoint is not None and not tau_already_removed:
        tau = float(checkpoint.get("logit_adjustment_tau") or 0.0)
        if tau > 0.0:
            raise ValueError(
                "checkpoint carries logit_adjustment_tau={} - prior correction and "
                "tau adjustment are the same operation and must not compose. Pass "
                "adjust=False posteriors and tau_already_removed=True.".format(tau))

    probabilities = np.asarray(probabilities, dtype=np.float64)
    ratio = np.clip(target_prior, 1e-12, None) / np.clip(train_prior, 1e-12, None)
    scaled = probabilities * np.power(ratio, alpha)
    return scaled / np.clip(scaled.sum(axis=1, keepdims=True), 1e-12, None)


# -------------------------------------------------------------- prior-matched metrics
def importance_weights(y_true, target_prior, train_prior):
    """Row weights that re-express a labelled sample as the target population."""
    y_true = np.asarray(y_true)
    ratio = np.clip(target_prior, 1e-12, None) / np.clip(train_prior, 1e-12, None)
    weights = ratio[y_true]
    return weights * (len(weights) / weights.sum())


def prior_matched_metrics(y_true, probabilities, target_prior, train_prior,
                          bootstrap=1000, seed=0):
    """Accuracy / weighted-F1 under the deployment prior, with an honest error bar.

    ``macro_f1`` is reported unweighted on purpose and labelled as the
    prior-invariant control: macro-F1 does not move under label shift by
    construction, so if a change shows up in the weighted numbers but not in
    macro-F1, the change is an artefact of the re-weighting rather than a real
    improvement.

    ``kish_ess`` is the number that stops this table being over-read. Against the
    graded prior it is 353 on the test split alone and 750 on val+test pooled,
    versus nominal sample sizes of 5,501 and 10,998.
    """
    y_true = np.asarray(y_true)
    probabilities = np.asarray(probabilities)
    predicted = probabilities.argmax(axis=1)
    weights = importance_weights(y_true, target_prior, train_prior)
    labels = np.unique(y_true)

    result = {
        "accuracy": accuracy_score(y_true, predicted) * 100,
        "weighted_f1": f1_score(y_true, predicted, average="weighted", labels=labels,
                                zero_division=0) * 100,
        "dep_accuracy": accuracy_score(y_true, predicted, sample_weight=weights) * 100,
        "dep_weighted_f1": f1_score(y_true, predicted, average="weighted", labels=labels,
                                    sample_weight=weights, zero_division=0) * 100,
        "macro_f1": f1_score(y_true, predicted, average="macro", labels=labels,
                             zero_division=0) * 100,
        "kish_ess": float(weights.sum() ** 2 / np.square(weights).sum()),
        "max_row_weight_share": float(weights.max() / weights.sum()),
        "n": int(len(y_true)),
    }

    if bootstrap:
        rng = np.random.default_rng(seed)
        accuracies, f1s = [], []
        for _ in range(bootstrap):
            draw = rng.integers(0, len(y_true), size=len(y_true))
            w = weights[draw]
            accuracies.append(accuracy_score(y_true[draw], predicted[draw], sample_weight=w))
            f1s.append(f1_score(y_true[draw], predicted[draw], average="weighted",
                                labels=labels, sample_weight=w, zero_division=0))
        result["dep_accuracy_lo"] = float(np.percentile(accuracies, 2.5) * 100)
        result["dep_accuracy_hi"] = float(np.percentile(accuracies, 97.5) * 100)
        result["dep_weighted_f1_lo"] = float(np.percentile(f1s, 2.5) * 100)
        result["dep_weighted_f1_hi"] = float(np.percentile(f1s, 97.5) * 100)
    return result


# ------------------------------------------------------------------- simulation
def _resample_to_prior(y_true, target_prior, n, rng, min_rows=0):
    """Draw row indices so the sample's class mix matches ``target_prior``."""
    by_class = {}
    for label in np.unique(y_true):
        rows = np.where(y_true == label)[0]
        if len(rows) >= min_rows:
            by_class[int(label)] = rows
    prior = np.array([target_prior[c] if c in by_class else 0.0
                      for c in range(len(target_prior))])
    if prior.sum() <= 0:
        raise ValueError("target prior puts no mass on any class present in the sample")
    prior = prior / prior.sum()
    counts = rng.multinomial(n, prior)
    picks = [rng.choice(by_class[c], size=k, replace=True)
             for c, k in enumerate(counts) if k and c in by_class]
    return np.concatenate(picks)


def simulate_prior_shift(probabilities, y_true, train_prior, train_counts,
                         graded_prior=None, alphas=ALPHA_GRID, repeats=10,
                         n=4000, min_rows=25, seed=0, out=None, verbose=True):
    """Validate the whole correction chain against priors we actually know.

    Four scenarios, in the order they earn trust:

    ``null``
        Resample to the *training* prior. The guard. Correction must not hurt
        when there is nothing to correct; if it does, the method is broken and
        nothing below it means anything.
    ``synthetic_common``
        A shifted prior built only from classes with enough evaluation rows to
        resample without duplicating the same handful of images. This is the
        honest headline: it isolates "does EM recover a prior" from "can the
        model tell these two specific lipstick tiles apart".
    ``reversed``
        The training prior inverted - a deliberately extreme shift, to show the
        method still behaves when the target is far away.
    ``graded``
        The estimated graded composition across every class. Realistic, but it
        demands ~5% Lipstick from a split holding two Lipstick rows, so the same
        images repeat and the model's memorised behaviour on them inflates the
        result. Reported with that caveat in its own column rather than omitted.
    """
    probabilities = np.asarray(probabilities)
    y_true = np.asarray(y_true)
    train_prior = np.asarray(train_prior, dtype=float)
    shrink = support_shrink(train_counts)
    num_classes = probabilities.shape[1]

    counts = np.bincount(y_true, minlength=num_classes)
    common = counts >= min_rows
    inverse = np.where(common, 1.0 / np.clip(train_prior, 1e-9, None), 0.0)

    scenarios = {"null": train_prior.copy()}
    if common.any():
        scenarios["synthetic_common"] = inverse / inverse.sum()
    reversed_prior = 1.0 / np.clip(train_prior, 1e-9, None)
    scenarios["reversed"] = reversed_prior / reversed_prior.sum()
    if graded_prior is not None:
        scenarios["graded"] = np.asarray(graded_prior, dtype=float)

    caveats = {
        "null": "no shift; correction must not hurt",
        "synthetic_common": "classes with >={} eval rows only; duplication-free".format(min_rows),
        "reversed": "extreme shift; stress test",
        "graded": "duplicates rare rows - the split holds 2 Lipstick rows against a ~5% demand",
    }

    rows = []
    for name, target in scenarios.items():
        restrict = min_rows if name == "synthetic_common" else 0
        for repeat in range(repeats):
            rng = np.random.default_rng(seed + repeat)
            index = _resample_to_prior(y_true, target, n, rng, min_rows=restrict)
            sample_p, sample_y = probabilities[index], y_true[index]
            truth = np.bincount(sample_y, minlength=num_classes).astype(float)
            truth /= truth.sum()

            estimated, _, iters, converged = estimate_prior_em(
                sample_p, train_prior, shrink=shrink)
            for alpha in alphas:
                corrected = apply_prior_correction(sample_p, train_prior, estimated,
                                                   alpha=alpha, tau_already_removed=True)
                predicted = corrected.argmax(1)
                labels = np.unique(sample_y)
                rows.append({
                    "scenario": name, "repeat": repeat, "alpha": alpha,
                    "n": len(sample_y),
                    "true_shift_tv": round(50 * np.abs(truth - train_prior).sum(), 2),
                    "estimate_error_tv": round(50 * np.abs(estimated - truth).sum(), 2),
                    "em_iters": iters, "em_converged": converged,
                    "accuracy": round(100 * accuracy_score(sample_y, predicted), 2),
                    "weighted_f1": round(100 * f1_score(sample_y, predicted, average="weighted",
                                                        labels=labels, zero_division=0), 2),
                    "macro_f1": round(100 * f1_score(sample_y, predicted, average="macro",
                                                     labels=labels, zero_division=0), 2),
                    "caveat": caveats[name],
                })

    frame = pd.DataFrame(rows)
    out = Path(out) if out else RESULTS_DIR / "task1_prior_shift_simulation.csv"
    out.parent.mkdir(parents=True, exist_ok=True)
    frame.to_csv(out, index=False)

    summary = (frame.groupby(["scenario", "alpha"])
               [["accuracy", "weighted_f1", "macro_f1", "estimate_error_tv", "true_shift_tv"]]
               .mean().round(2).reset_index())
    if verbose:
        print(summary.to_string(index=False))
        print("wrote {}".format(out))
    return frame, summary


def correction_is_safe(simulation_summary, covariate_frame, probabilities, train_prior,
                       train_counts, noise_floor=0.242, verbose=True):
    """The four-part go/no-go guard. All of it has to pass, or ship tau unchanged.

    A negative result here is a perfectly good outcome and a stronger thing to
    report than a fragile win.
    """
    shrink = support_shrink(train_counts)
    reasons, checks = [], {}

    checks["covariate_precondition"] = bool(covariate_frame["pass"].all())
    if not checks["covariate_precondition"]:
        reasons.append("covariate-shift check failed: the graded set is not the same "
                       "imaging population, so label shift is the wrong model")

    # Identifiability: if the fixed point depends on where it starts, the
    # likelihood surface is flat and the prior is not identified from this data.
    train_prior = np.asarray(train_prior, dtype=float)
    counts = np.bincount(np.asarray(probabilities).argmax(1), minlength=len(train_prior))
    argmax_prior = counts.astype(float) / max(counts.sum(), 1)
    inits = {
        "train": None,
        "uniform": (np.full(len(train_prior), 1.0 / len(train_prior))
                    / np.clip(train_prior, 1e-9, None)),
        "argmax": np.clip(argmax_prior, 1e-9, None) / np.clip(train_prior, 1e-9, None),
    }
    estimates = {}
    for name, init in inits.items():
        prior, _, _, _ = estimate_prior_em(probabilities, train_prior, shrink=shrink, init=init)
        estimates[name] = prior
    pairwise = [0.5 * np.abs(estimates[a] - estimates[b]).sum()
                for a in estimates for b in estimates if a < b]
    worst = max(pairwise) if pairwise else 0.0
    checks["estimator_stable"] = bool(worst < 0.05)
    checks["max_init_disagreement_tv"] = round(float(worst * 100), 3)
    if not checks["estimator_stable"]:
        reasons.append("EM fixed point depends on initialisation (TV {:.1f}% > 5%) - "
                       "the prior is not identified".format(worst * 100))

    null = simulation_summary[simulation_summary["scenario"] == "null"]
    if len(null):
        base = float(null[null["alpha"] == 0.0]["weighted_f1"].iloc[0])
        full = float(null[null["alpha"] == 1.0]["weighted_f1"].iloc[0])
        drop = base - full
        checks["null_regression"] = round(drop, 3)
        checks["null_clean"] = bool(drop <= noise_floor)
        if not checks["null_clean"]:
            reasons.append("correction costs {:.2f} weighted-F1 when there is no shift, "
                           "beyond the {:.3f} seed noise floor".format(drop, noise_floor))

    honest = simulation_summary[simulation_summary["scenario"] == "synthetic_common"]
    if len(honest):
        base = float(honest[honest["alpha"] == 0.0]["weighted_f1"].iloc[0])
        best_row = honest.loc[honest["weighted_f1"].idxmax()]
        gain = float(best_row["weighted_f1"]) - base
        checks["measured_gain"] = round(gain, 3)
        checks["best_alpha"] = float(best_row["alpha"])
        checks["gain_exceeds_noise"] = bool(gain > noise_floor)
        if not checks["gain_exceeds_noise"]:
            reasons.append("best measured gain {:.2f} is inside the {:.3f} noise floor - "
                           "not a gain".format(gain, noise_floor))

    safe = bool(checks.get("covariate_precondition") and checks.get("estimator_stable")
                and checks.get("null_clean", False) and checks.get("gain_exceeds_noise", False))
    checks["safe"] = safe
    checks["reasons"] = reasons
    if verbose:
        print("\nprior-correction guard: {}".format("PASS" if safe else "BLOCK"))
        for key, value in checks.items():
            if key != "reasons":
                print("  {:<28} {}".format(key, value))
        for reason in reasons:
            print("  - {}".format(reason))
    return safe, checks


# ----------------------------------------------------------------------- drivers
def _cached_splits():
    """The adopted split, imported lazily so this module loads without torch."""
    from src.training.train_item_type import load_splits
    return load_splits(verbose=False)


def _split_probabilities(model, checkpoint, frame, images, device, tta):
    """Raw posteriors for one split, straight from the cached uint8 tiles."""
    import torch
    from src.training.train_item_type import predict_split, split_tensors

    mean = torch.tensor(checkpoint["channel_mean"], dtype=torch.float32,
                        device=device).view(1, 3, 1, 1)
    std = torch.tensor(checkpoint["channel_std"], dtype=torch.float32,
                       device=device).view(1, 3, 1, 1)
    tensor = split_tensors(frame, images, device)
    if isinstance(tensor, tuple):
        tensor = tensor[0]
    probabilities = predict_split(model, tensor, mean, std, batch_size=512, tta=tta)
    return probabilities.cpu().numpy() if hasattr(probabilities, "cpu") else np.asarray(probabilities)


def labelled_probabilities(checkpoint_path=ARTIFACT_DIR / "task1_cnn.pt", verbose=True):
    """Raw posteriors and labels for the val and test splits.

    ``predict_split`` never applies logit adjustment, so what comes back is
    already the raw quantity the estimators need.
    """
    train_df, val_df, test_df, class_names, images = _cached_splits()
    device = choose_device()
    model, checkpoint = load_item_type_model(Path(checkpoint_path), device=device)
    tta = bool(checkpoint.get("tta", False))
    lookup = {name: index for index, name in enumerate(class_names)}

    out = {}
    for name, frame in (("val", val_df), ("test", test_df)):
        if verbose:
            print("scoring {} split ({} rows)".format(name, len(frame)))
        out[name] = (_split_probabilities(model, checkpoint, frame, images, device, tta),
                     frame["articleType"].map(lookup).to_numpy())
    train_y = train_df["articleType"].map(lookup).to_numpy()
    return out, train_y, class_names, checkpoint


def deployment_eval(checkpoint_paths, out=None, bootstrap=1000, reference=None,
                    verbose=True):
    """The headline table: the same model scored against two different populations.

    One row per checkpoint x split. The train-prior columns are what the
    repository has always reported; the ``dep_`` columns are what the graded
    submission will actually experience. ``kish_ess`` says how much to trust the
    second pair.

    **The target prior is estimated once and reused for every checkpoint.** An
    earlier version let each model estimate its own, which is wrong for a
    comparison: the importance weights then differ per model, so the rows are
    not even weighted the same way and the numbers are not comparable. Worse, it
    confounds "this model classifies better" with "this model happened to
    estimate the prior differently" - and it reversed the ranking, putting
    tail_sqrt first on its own estimate and third on a shared one.

    The prior is a property of the graded population, not of the model, so it is
    held fixed. ``reference`` names the checkpoint it is estimated from
    (default: the first), and it is recorded in the output so a reader can see
    which one carried it.
    """
    reference = Path(reference) if reference else Path(checkpoint_paths[0])
    shared_prior, shared_counts = None, None

    rows = []
    for path in checkpoint_paths:
        path = Path(path)
        splits, train_y, class_names, checkpoint = labelled_probabilities(path, verbose=verbose)
        num_classes = len(class_names)
        train_counts = np.bincount(train_y, minlength=num_classes)
        train_prior = train_counts.astype(float) / train_counts.sum()
        shrink = support_shrink(train_counts)

        if shared_prior is None:
            _, reference_raw, _, _ = graded_probabilities(reference, verbose=verbose)
            shared_prior, _, _, converged = estimate_prior_em(
                reference_raw, train_prior, shrink=shrink)
            shared_counts = train_counts
        graded_prior = shared_prior

        pooled_p = np.concatenate([splits["val"][0], splits["test"][0]])
        pooled_y = np.concatenate([splits["val"][1], splits["test"][1]])
        for split_name, (probabilities, y_true) in (
                ("test", splits["test"]), ("val+test", (pooled_p, pooled_y))):
            metrics = prior_matched_metrics(y_true, probabilities, graded_prior,
                                            train_prior, bootstrap=bootstrap)
            metrics.update({"checkpoint": path.name, "split": split_name,
                            "run_id": str(checkpoint.get("run_id", "")),
                            "target_prior": "EM(graded, shrunk) from {}".format(reference.name),
                            "em_converged": converged})
            rows.append(metrics)

    frame = pd.DataFrame(rows)
    ordered = ["checkpoint", "run_id", "split", "accuracy", "weighted_f1", "macro_f1",
               "dep_accuracy", "dep_weighted_f1", "dep_accuracy_lo", "dep_accuracy_hi",
               "dep_weighted_f1_lo", "dep_weighted_f1_hi", "kish_ess",
               "max_row_weight_share", "n", "target_prior", "em_converged"]
    frame = frame[[c for c in ordered if c in frame.columns]].round(3)
    out = Path(out) if out else RESULTS_DIR / "task1_deployment_prior_metrics.csv"
    out.parent.mkdir(parents=True, exist_ok=True)
    frame.to_csv(out, index=False)
    if verbose:
        print(frame.to_string(index=False))
        print("wrote {}".format(out))
    return frame


def starved_class_table(checkpoint_path=ARTIFACT_DIR / "task1_cnn.pt", out=None, verbose=True):
    """Per class: what we trained on, what we can measure, what we are submitting.

    ``graded_rows_at_risk`` is the column that converts "F1 = 1.00 is meaningless"
    into a number of rows of the deliverable that are probably wrong.
    """
    splits, train_y, class_names, _ = labelled_probabilities(checkpoint_path, verbose=verbose)
    num_classes = len(class_names)
    train_counts = np.bincount(train_y, minlength=num_classes)
    test_p, test_y = splits["test"]
    test_counts = np.bincount(test_y, minlength=num_classes)
    predicted = test_p.argmax(1)

    ids, raw, adjusted, _ = graded_probabilities(checkpoint_path, verbose=verbose)
    graded_pred = adjusted.argmax(1)
    graded_counts = np.bincount(graded_pred, minlength=num_classes)
    graded_conf = np.zeros(num_classes)
    for index in range(num_classes):
        mask = graded_pred == index
        graded_conf[index] = float(adjusted[mask].max(1).mean()) if mask.any() else np.nan

    rows = []
    for index, name in enumerate(class_names):
        support = int(test_counts[index])
        hits = int(((test_y == index) & (predicted == index)).sum())
        recall = hits / support if support else np.nan
        share = graded_counts[index] / len(graded_pred)
        rows.append({
            "articleType": name,
            "train_rows": int(train_counts[index]),
            "test_rows": support,
            "test_recall": round(recall, 4) if support else None,
            "recall_is_measurable": support >= 25,
            "graded_rows": int(graded_counts[index]),
            "graded_share_pct": round(100 * share, 3),
            "graded_mean_confidence": round(float(graded_conf[index]), 4)
                if not np.isnan(graded_conf[index]) else None,
            "graded_rows_at_risk": round(graded_counts[index] * (1 - recall), 1)
                if support else None,
        })

    frame = pd.DataFrame(rows).sort_values("graded_rows", ascending=False)
    out = Path(out) if out else RESULTS_DIR / "task1_starved_classes.csv"
    out.parent.mkdir(parents=True, exist_ok=True)
    frame.to_csv(out, index=False)
    if verbose:
        unmeasurable = frame[frame["test_rows"] <= 5]
        print(frame.head(15).to_string(index=False))
        print("\n{} classes hold <=5 test rows and carry {} graded rows ({:.1f}% of the "
              "submission)".format(len(unmeasurable), int(unmeasurable["graded_rows"].sum()),
                                   100 * unmeasurable["graded_rows"].sum() / len(graded_pred)))
        print("wrote {}".format(out))
    return frame


def graded_prior_table(checkpoint_path=ARTIFACT_DIR / "task1_cnn.pt", out=None, verbose=True):
    """Every estimate of the graded class prior, side by side.

    The ``argmax_adjusted`` column exists to expose a trap: the published
    predictions are post-tau, and tau alone moves Foundation and Primer from 48
    predicted rows to 108. Any composition table read off the shipped CSV is
    reading the adjustment as much as the data.
    """
    splits, train_y, class_names, _ = labelled_probabilities(checkpoint_path, verbose=verbose)
    num_classes = len(class_names)
    train_counts = np.bincount(train_y, minlength=num_classes)
    train_prior = train_counts.astype(float) / train_counts.sum()
    shrink = support_shrink(train_counts)

    ids, raw, adjusted, _ = graded_probabilities(checkpoint_path, verbose=verbose)
    em_prior, _, iters, converged = estimate_prior_em(raw, train_prior, shrink=shrink)
    em_free, _, _, _ = estimate_prior_em(raw, train_prior, clip=None, shrink=None)

    val_p, val_y = splits["val"]
    bbse_prior, _, condition, smallest = estimate_prior_bbse(val_p, val_y, raw, num_classes)

    raw_counts = np.bincount(raw.argmax(1), minlength=num_classes)
    adj_counts = np.bincount(adjusted.argmax(1), minlength=num_classes)
    frame = pd.DataFrame({
        "articleType": class_names,
        "train_rows": train_counts,
        "train_prior_pct": (100 * train_prior).round(4),
        "argmax_raw_pct": (100 * raw_counts / raw_counts.sum()).round(4),
        "argmax_adjusted_pct": (100 * adj_counts / adj_counts.sum()).round(4),
        "em_prior_pct": (100 * em_prior).round(4),
        "em_unguarded_pct": (100 * em_free).round(4),
        "bbse_prior_pct": (100 * bbse_prior).round(4),
    }).sort_values("em_prior_pct", ascending=False)
    frame["bbse_condition_number"] = round(condition, 1)
    frame["bbse_smallest_singular"] = round(smallest, 8)

    out = Path(out) if out else RESULTS_DIR / "task1_graded_prior_estimates.csv"
    out.parent.mkdir(parents=True, exist_ok=True)
    frame.to_csv(out, index=False)
    if verbose:
        print(frame.head(15).to_string(index=False))
        print("\nEM converged={} in {} iters; shift from train prior TV={:.1f}%".format(
            converged, iters, 50 * np.abs(em_prior - train_prior).sum()))
        print("BBSE condition number {:.1f} (smallest singular value {:.2e}) - "
              "{}".format(condition, smallest,
                          "usable" if condition < 1e6 else "SINGULAR, not usable"))
        print("wrote {}".format(out))
    return frame


def dropped_class_cost(out=None, verbose=True):
    """How many graded rows carry a true class the model has no logit for.

    ``task1_summary.json`` records 18.3, computed as
    ``len(graded) * dropped_rows / total_rows`` - i.e. under the *training*
    prior. That assumes the graded set has the same composition as training,
    which is the assumption this whole module exists to reject.

    Re-estimating per masterCategory instead: within each category, the fraction
    of its rows that the ``min_class_size=10`` floor drops, times that category's
    share of the graded set. Personal Care is the category the floor hits hardest
    *and* the category the graded set is full of, so the two multiply rather than
    cancel - which is exactly what the flat estimate misses.
    """
    metadata = pd.read_csv(PROJECT_ROOT / "A2_FashionDataset" / "processed"
                           / "clean_train_metadata.csv")
    counts = metadata["articleType"].value_counts()
    dropped = set(counts[counts < 10].index)

    ids, raw, adjusted, class_names = graded_probabilities(verbose=verbose)
    predicted = np.asarray(class_names)[adjusted.argmax(1)]
    category_of = (metadata.drop_duplicates("articleType")
                   .set_index("articleType")["masterCategory"])
    graded_category = pd.Series(predicted).map(category_of)

    rows = []
    for category, group in metadata.groupby("masterCategory"):
        drop_rate = group["articleType"].isin(dropped).mean()
        graded_share = float((graded_category == category).mean())
        rows.append({
            "masterCategory": category,
            "train_rows": int(len(group)),
            "train_drop_rate_pct": round(100 * drop_rate, 3),
            "graded_share_pct": round(100 * graded_share, 3),
            "estimated_affected_rows": round(drop_rate * graded_share * len(predicted), 1),
        })

    frame = pd.DataFrame(rows).sort_values("estimated_affected_rows", ascending=False)
    total = float(frame["estimated_affected_rows"].sum())
    flat = len(predicted) * len(metadata[metadata["articleType"].isin(dropped)]) / len(metadata)
    frame = pd.concat([frame, pd.DataFrame([{
        "masterCategory": "TOTAL (prior-aware)", "train_rows": int(len(metadata)),
        "train_drop_rate_pct": None, "graded_share_pct": 100.0,
        "estimated_affected_rows": round(total, 1)}, {
        "masterCategory": "TOTAL (flat train prior, as recorded)",
        "train_rows": int(len(metadata)), "train_drop_rate_pct": None,
        "graded_share_pct": 100.0, "estimated_affected_rows": round(flat, 1)}])],
        ignore_index=True)

    out = Path(out) if out else ARTIFACT_DIR / "dropped_class_graded_cost.csv"
    out.parent.mkdir(parents=True, exist_ok=True)
    frame.to_csv(out, index=False)
    if verbose:
        print(frame.to_string(index=False))
        print("\nprior-aware estimate {:.1f} rows ({:.2f}% of the submission) against "
              "the recorded flat estimate {:.1f} - {:.1f}x".format(
                  total, 100 * total / len(predicted), flat, total / max(flat, 1e-9)))
        print("wrote {}".format(out))
    return frame


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    parser.add_argument("--checkpoints", nargs="+",
                        default=[str(ARTIFACT_DIR / "task1_cnn.pt")])
    parser.add_argument("--covariate-check", action="store_true")
    parser.add_argument("--graded-prior", action="store_true")
    parser.add_argument("--starved-classes", action="store_true")
    parser.add_argument("--deployment-eval", action="store_true")
    parser.add_argument("--simulate", action="store_true")
    parser.add_argument("--dropped-cost", action="store_true",
                        help="re-estimate how many graded rows carry a dropped true "
                             "class, under the deployment prior rather than the flat "
                             "training prior task1_summary.json assumes")
    parser.add_argument("--guard", action="store_true",
                        help="run the covariate check, the simulation and the go/no-go guard")
    parser.add_argument("--all", action="store_true")
    parser.add_argument("--repeats", type=int, default=10)
    parser.add_argument("--bootstrap", type=int, default=1000)
    parser.add_argument("--force", action="store_true", help="recompute cached posteriors")
    args = parser.parse_args(argv)

    if args.all:
        args.covariate_check = args.graded_prior = args.starved_classes = True
        args.deployment_eval = args.guard = args.dropped_cost = True

    if args.force:
        for path in args.checkpoints:
            graded_probabilities(path, force=True)

    covariate = None
    if args.covariate_check or args.guard:
        print("\n=== covariate-shift precondition ===")
        covariate = covariate_shift_report()

    if args.graded_prior:
        print("\n=== graded class-prior estimates ===")
        graded_prior_table(args.checkpoints[0])

    if args.dropped_cost:
        print("\n=== cost of the dropped classes, under the deployment prior ===")
        dropped_class_cost()

    if args.starved_classes:
        print("\n=== starved classes on the deliverable ===")
        starved_class_table(args.checkpoints[0])

    if args.deployment_eval:
        print("\n=== train-prior vs deployment-prior metrics ===")
        deployment_eval(args.checkpoints, bootstrap=args.bootstrap)

    if args.simulate or args.guard:
        print("\n=== prior-shift simulation ===")
        splits, train_y, class_names, _ = labelled_probabilities(args.checkpoints[0])
        num_classes = len(class_names)
        train_counts = np.bincount(train_y, minlength=num_classes)
        train_prior = train_counts.astype(float) / train_counts.sum()
        pooled_p = np.concatenate([splits["val"][0], splits["test"][0]])
        pooled_y = np.concatenate([splits["val"][1], splits["test"][1]])
        ids, raw, adjusted, _ = graded_probabilities(args.checkpoints[0])
        graded_prior, _, _, _ = estimate_prior_em(
            raw, train_prior, shrink=support_shrink(train_counts))

        frame, summary = simulate_prior_shift(pooled_p, pooled_y, train_prior, train_counts,
                                              graded_prior=graded_prior, repeats=args.repeats)
        if args.guard and covariate is not None:
            safe, checks = correction_is_safe(summary, covariate, raw, train_prior, train_counts)
            payload = {"safe": safe, "checks": {k: v for k, v in checks.items()},
                       "prior": [float(v) for v in graded_prior],
                       "class_names": list(class_names),
                       "checkpoint": Path(args.checkpoints[0]).name}
            target = ARTIFACT_DIR / "graded_prior.json"
            target.write_text(json.dumps(payload, indent=2, default=str), encoding="utf-8")
            print("wrote {}".format(target))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
