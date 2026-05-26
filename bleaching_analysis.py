"""
bleaching_analysis.py
=====================
Fluorescence photobleaching step analysis for single-molecule spots.

Analyzes colocalized AND unmatched spots from colocalize_tif.py output.
Each spot is reviewed individually per-channel with full interactive
breakpoint editing.

Classification scheme
---------------------
  1 step   → monomer
  2 steps  → dimer
  3+ steps → aggregate  (includes uncertain / bad fits)

Complex classes (colocalized pairs)
------------------------------------
  1:1, 1:2, 2:1, 2:2  — based on (ch1 steps):(ch2 steps)
  aggregate            — if either channel is aggregate

Review UI controls
------------------
  Left-click on trace  → add breakpoint at that frame
  Right-click on trace → remove nearest breakpoint
  [c]                  → reset to auto-detected breakpoints
  [←/→]               → navigate prev/next spot
  [q]                  → done with this pass

Output files
------------
Two-channel mode:
  {stem}_bleaching_coloc.csv         — colocalized pairs
  {stem}_bleaching_unmatched_ch1.csv — unmatched ch1 spots
  {stem}_bleaching_unmatched_ch2.csv — unmatched ch2 spots

Single-channel mode:
  {stem}_bleaching_spots.csv         — all spots with step counts and classes

Usage
-----
  python bleaching_analysis.py --mode single --input movie.tif \
      --coloc colocalization_output/movie_colocalized.csv
"""

import argparse
import sys
import time
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.widgets import Button

try:
    import tifffile
except ImportError:
    sys.exit("Please install tifffile:  pip install tifffile")

try:
    import ruptures as rpt
except ImportError:
    sys.exit("Please install ruptures:  pip install ruptures")


# ===========================================================================
# PARAMETERS
# ===========================================================================

FRAMES_PER_CHANNEL   = 350
BLANK_FRAMES         = 10

PSF_RADIUS_PX        = 1.5

BG_ANNULUS_INNER_PX  = 4
BG_ANNULUS_OUTER_PX  = 8
BG_SPOT_MASK_PX      = 3

PELT_PENALTY         = 15
PELT_MIN_SIZE        = 5

# ===========================================================================


def load_tif_stack(tif_path):
    with tifffile.TiffFile(tif_path) as tif:
        stack = np.stack([p.asarray().astype(np.float32) for p in tif.pages], axis=0)
    return stack


# Gaussian sigma for spot intensity extraction — matched to LabVIEW pipeline
GAUSSIAN_SIGMA = 2.20

def extract_trace(stack, cx, cy, frame_start, frame_end, psf_radius,
                  gaussian_sigma=GAUSSIAN_SIGMA):
    """
    Extract per-frame spot intensity using Gaussian-weighted convolution,
    matching the LabVIEW SiMPull pipeline (sigma=2.20).

    A Gaussian kernel is evaluated over a square patch centred on (cx, cy)
    and used as weights for the weighted average pixel intensity per frame.
    This matches PSF shape better than a flat circular disk average.
    """
    row_c = int(round(cy)); col_c = int(round(cx))
    h, w  = stack.shape[1], stack.shape[2]

    # Kernel half-size: 3*sigma covers >99% of Gaussian weight
    r = int(np.ceil(3.0 * gaussian_sigma))
    dy, dx = np.mgrid[-r:r+1, -r:r+1].astype(float)

    # 2D Gaussian weights (unnormalized, will normalize below)
    weights_2d = np.exp(-(dx**2 + dy**2) / (2.0 * gaussian_sigma**2))

    # Flatten and get absolute pixel coordinates
    off_rows = dy.ravel().astype(int)
    off_cols = dx.ravel().astype(int)
    weights   = weights_2d.ravel()

    abs_rows = row_c + off_rows
    abs_cols = col_c + off_cols
    valid = ((abs_rows >= 0) & (abs_rows < h) &
             (abs_cols >= 0) & (abs_cols < w))
    abs_rows = abs_rows[valid]
    abs_cols = abs_cols[valid]
    weights  = weights[valid]
    weights  = weights / weights.sum()   # normalize so weighted mean is correct

    # Extract: shape (n_frames, n_pixels), then weighted sum over pixels
    pixels = stack[frame_start:frame_end, abs_rows, abs_cols]  # (n_frames, n_px)
    return (pixels * weights[np.newaxis, :]).sum(axis=1)


def annular_background_trace(stack, cx, cy, frame_start, frame_end,
                              inner_radius, outer_radius,
                              all_spot_xs, all_spot_ys, spot_mask_radius):
    h, w = stack.shape[1], stack.shape[2]
    cx_i, cy_i = int(round(cx)), int(round(cy))
    r_out = int(np.ceil(outer_radius))
    dy, dx = np.mgrid[-r_out:r_out+1, -r_out:r_out+1]
    dist2 = dy**2 + dx**2
    annulus = (dist2 > inner_radius**2) & (dist2 <= outer_radius**2)
    off_rows = dy[annulus]; off_cols = dx[annulus]
    abs_rows = cy_i + off_rows; abs_cols = cx_i + off_cols
    in_bounds = (abs_rows >= 0) & (abs_rows < h) & (abs_cols >= 0) & (abs_cols < w)
    abs_rows = abs_rows[in_bounds]; abs_cols = abs_cols[in_bounds]
    if len(all_spot_xs) > 0 and len(abs_rows) > 0:
        dr = abs_rows[:, None] - all_spot_ys[None, :]
        dc = abs_cols[:, None] - all_spot_xs[None, :]
        min_dist = np.sqrt(dr**2 + dc**2).min(axis=1)
        keep = min_dist > spot_mask_radius
        abs_rows = abs_rows[keep]; abs_cols = abs_cols[keep]
    if len(abs_rows) > 0:
        # Vectorized: shape (n_frames, n_annulus_pixels) → median over pixels
        patch = stack[frame_start:frame_end, abs_rows, abs_cols]
        return np.median(patch, axis=1).astype(np.float32)
    else:
        # Fallback: whole-frame median per frame
        return np.median(
            stack[frame_start:frame_end].reshape(frame_end - frame_start, -1),
            axis=1).astype(np.float32)


def detect_steps(trace, penalty, min_size):
    if len(trace) < min_size * 2:
        return np.array([], dtype=int)
    model = rpt.Pelt(model="rbf", min_size=min_size, jump=1)
    try:
        result = model.fit(trace).predict(pen=penalty)
        return np.array(result[:-1], dtype=int)
    except Exception:
        return np.array([], dtype=int)


def detect_early_burst(signal, breakpoints,
                       burst_frames=3, plateau_frames=10,
                       burst_ratio=1.5,
                       tail_frames=30, tail_ratio=3.0,
                       max_burst_scan=10):
    """
    Detect a fast initial bleaching step (1-5 frames) that PELT missed.

    Two-pass approach:
      Pass 1 (tail baseline): compare peak of first max_burst_scan frames to
        median of last tail_frames frames. More robust than using a plateau
        window — works even when early dimer steps fall in frames 10-50.
        Triggers if peak > tail_ratio x tail_baseline.

      Pass 2 (plateau fallback): original approach comparing first burst_frames
        to the following plateau_frames. Kept as fallback for cases where the
        tail is not fully bleached.

    Breakpoint is placed at the first frame where signal drops back to within
    1.5x of the tail baseline.

    Returns updated breakpoints array (may be unchanged if no burst detected).
    """
    if len(signal) < max_burst_scan + tail_frames:
        return breakpoints

    bps_sorted = sorted(int(b) for b in breakpoints)
    first_bp   = bps_sorted[0] if bps_sorted else len(signal)

    # Only attempt correction if PELT did not already find an early breakpoint
    if first_bp < max_burst_scan:
        return breakpoints

    # --- Pass 1: tail-based detection ---
    tail_baseline = float(np.median(signal[-tail_frames:]))
    early_peak    = float(signal[:max_burst_scan].max())

    if tail_baseline > 0 and early_peak > tail_ratio * tail_baseline:
        # Find the frame where signal first drops back to 1.5x tail baseline
        threshold = tail_baseline * 1.5
        crossing  = max_burst_scan  # fallback
        for i in range(1, min(first_bp, max_burst_scan + 5)):
            if signal[i] <= threshold:
                crossing = i
                break
        new_bps = np.array(sorted(set(list(breakpoints) + [crossing])), dtype=int)
        return new_bps

    # --- Pass 2: plateau-based fallback (original logic) ---
    if len(signal) < burst_frames + plateau_frames:
        return breakpoints

    burst_level   = float(signal[:burst_frames].mean())
    if first_bp < burst_frames + plateau_frames:
        return breakpoints

    plateau_level = float(signal[burst_frames:burst_frames + plateau_frames].mean())
    if plateau_level <= 0 or burst_level < burst_ratio * plateau_level:
        return breakpoints

    midpoint = (burst_level + plateau_level) / 2
    crossing = burst_frames
    for i in range(burst_frames, min(first_bp, burst_frames + plateau_frames)):
        if signal[i] <= midpoint:
            crossing = i
            break

    new_bps = np.array(sorted(set(list(breakpoints) + [crossing])), dtype=int)
    return new_bps

def compute_step_levels(signal, breakpoints):
    """Return one level per segment, using HMM-smoothed estimates."""
    _, seg_levels = hmm_level_estimates(signal, breakpoints)
    return seg_levels


def detect_upward_steps(signal: np.ndarray, breakpoints,
                        noise_multiplier: float = 2.0) -> bool:
    """
    Return True if any PELT level transition is upward by more than
    noise_multiplier × trace noise.

    An upward step means the segment *after* a breakpoint has a higher
    mean intensity than the segment *before* it — i.e. the fluorophore
    blinked back on or a new molecule entered the spot.  This is
    definitionally incompatible with clean photobleaching and the trace
    should be excluded from stoichiometry counts.

    noise_multiplier=2.0 is a conservative threshold that avoids
    flagging noise fluctuations while catching genuine intensity increases.
    """
    bps = sorted(int(b) for b in breakpoints)
    if not bps:
        return False
    levels = compute_step_levels(signal, bps)
    if len(levels) < 2:
        return False
    # Estimate noise as std of the last (most-bleached) segment
    last_seg = signal[bps[-1]:]
    noise = float(last_seg.std()) if len(last_seg) >= 3 else float(signal.std())
    threshold = noise_multiplier * noise
    for i in range(len(levels) - 1):
        if levels[i + 1] - levels[i] > threshold:
            return True
    return False


# Human-readable labels for stoichiometry counts
_STOICH_LABELS = {
    1: "monomer",
    2: "dimer",
    3: "trimer",
    4: "tetramer",
    5: "pentamer",
    6: "hexamer",
    7: "heptamer",
    8: "octamer",
}

def stoich_label(n: int) -> str:
    """Return a human-readable label for a stoichiometry count."""
    return _STOICH_LABELS.get(n, f"{n}-mer")


def classify_steps(n_steps: int, max_stoichiometry: int = 2) -> str:
    """Classify a step count given the experiment's max expected stoichiometry.

    Steps > max_stoichiometry are classified as 'aggregate'.
    Defaults to max_stoichiometry=2 (monomer/dimer/aggregate) to reproduce
    the original behaviour when the parameter is not supplied.
    """
    if n_steps > max_stoichiometry:
        return "aggregate"
    return stoich_label(n_steps)


def complex_class(cls1, cls2, n1, n2, cls3=None, n3=None):
    """Return complex class string for 2- or 3-channel colocalized spots."""
    if cls3 is not None:
        # Three-channel mode: N:M:P
        if cls1 == "aggregate" or cls2 == "aggregate" or cls3 == "aggregate":
            return "aggregate"
        return f"{n1}:{n2}:{n3}"
    # Two-channel mode: N:M
    if cls1 == "aggregate" or cls2 == "aggregate":
        return "aggregate"
    return f"{n1}:{n2}"


def check_incomplete_bleaching(signal, tail_frames=10, tail_threshold=0.20,
                               tail_abs_floor=10.0):
    """
    Returns True if the spot likely did not fully bleach.
    Criteria: mean of last tail_frames > tail_threshold * mean of first tail_frames.
    AND mean of last tail_frames > tail_abs_floor (absolute noise floor).
    The absolute floor prevents dim spots that bleach cleanly to near-zero
    from being incorrectly flagged.
    """
    if len(signal) < tail_frames * 2:
        return False
    initial = signal[:tail_frames].mean()
    if initial <= 0:
        return False
    tail = float(signal[-tail_frames:].mean())
    if tail <= tail_abs_floor:
        return False
    return tail > tail_threshold * initial


_timing = {"extract": 0.0, "bg": 0.0, "pelt": 0.0, "n": 0}

def prepare_spot(row, stack, frame_start, frame_end,
                 xcol, ycol, all_xs, all_ys, args):
    import time as _t
    t0 = _t.perf_counter()
    trace = extract_trace(stack, row[xcol], row[ycol],
                          frame_start, frame_end, args.psf_radius)
    t1 = _t.perf_counter()
    bg = annular_background_trace(
        stack, row[xcol], row[ycol], frame_start, frame_end,
        args.bg_annulus_inner, args.bg_annulus_outer,
        all_xs, all_ys, args.bg_spot_mask)
    t2 = _t.perf_counter()
    trace    = trace - bg
    signal   = trace[args.blank_frames:]
    auto_bps = detect_steps(signal, args.pelt_penalty, args.pelt_min_size)
    t3 = _t.perf_counter()
    auto_bps = detect_early_burst(signal, auto_bps,
                                  burst_frames=args.burst_frames,
                                  plateau_frames=args.burst_plateau_frames,
                                  burst_ratio=args.burst_ratio,
                                  tail_frames=args.burst_tail_frames,
                                  tail_ratio=args.burst_tail_ratio,
                                  max_burst_scan=args.burst_max_scan)
    incomplete = check_incomplete_bleaching(signal,
                                            args.incomplete_tail_frames,
                                            args.incomplete_tail_threshold,
                                            args.incomplete_abs_floor)
    _timing["extract"] += t1 - t0
    _timing["bg"]      += t2 - t1
    _timing["pelt"]    += t3 - t2
    _timing["n"]       += 1
    upward = detect_upward_steps(signal, auto_bps)
    return {
        "x": float(row[xcol]), "y": float(row[ycol]),
        "trace":            trace,
        "signal":           signal,
        "auto_bps":         auto_bps.copy(),
        "bps":              auto_bps.copy(),
        "incomplete":       incomplete,
        "class_override":   None,
        "bad_trace":        False,
        "good_trace":       False,
        "has_upward_step":  upward,
    }


def compute_trace_features(signal: np.ndarray, auto_bps: np.ndarray,
                           anomalous: bool = False,
                           partial_bleach_confidence: float = 0.0,
                           pop_step_size: float = 0.0) -> dict:
    """
    Compute summary statistics from a background-subtracted signal trace
    and PELT-detected breakpoints. Used as ML training features for the
    bleaching step predictor.

    Parameters
    ----------
    signal                    : 1-D float array, background-subtracted,
                                blank frames removed
    auto_bps                  : 1-D int array of PELT breakpoint indices
    anomalous                 : Stage 4 anomalous flag (0/1) — helps model
                                distinguish genuinely bad fits from clean traces
    partial_bleach_confidence : Stage 5 partial bleach confidence (0–1) —
                                helps model recognise incomplete bleaching cases
    pop_step_size             : Stage 3 population step size estimate —
                                provides a per-movie brightness scale reference

    Returns
    -------
    dict of scalar features, all JSON-serialisable floats/ints.
    """
    n = len(signal)
    if n == 0:
        return {
            "trace_mean": 0.0, "trace_std": 0.0, "trace_snr": 0.0,
            "trace_length": 0,
            "initial_intensity": 0.0, "final_intensity": 0.0,
            "n_auto_steps": 0,
            "min_step_size": 0.0, "max_step_size": 0.0, "mean_step_size": 0.0,
            "step_size_to_noise_ratio": 0.0,
            "early_step_detected": False,
            "anomalous": 0,
            "partial_bleach_confidence": 0.0,
            "pop_step_size": 0.0,
        }

    tail = max(10, n // 10)
    head = max(10, n // 10)

    trace_mean        = float(signal.mean())
    trace_std         = float(signal.std())
    trace_snr         = float(trace_mean / (trace_std + 1e-6))
    initial_intensity = float(signal[:head].mean())
    final_intensity   = float(signal[-tail:].mean())

    bps_sorted   = sorted(int(b) for b in auto_bps)
    n_auto_steps = len(bps_sorted)

    # Compute step sizes between adjacent segment levels (HMM-smoothed)
    _, segment_means = hmm_level_estimates(signal, bps_sorted)

    step_sizes = [abs(segment_means[i] - segment_means[i + 1])
                  for i in range(len(segment_means) - 1)]

    if step_sizes:
        min_step  = float(min(step_sizes))
        max_step  = float(max(step_sizes))
        mean_step = float(np.mean(step_sizes))
    else:
        min_step = max_step = mean_step = 0.0

    step_to_noise = float(mean_step / (trace_std + 1e-6))

    # Whether any breakpoint falls in the first 10% of the trace
    early_step = any(b < n * 0.10 for b in bps_sorted) if bps_sorted else False

    return {
        "trace_mean":               trace_mean,
        "trace_std":                trace_std,
        "trace_snr":                trace_snr,
        "trace_length":             n,
        "initial_intensity":        initial_intensity,
        "final_intensity":          final_intensity,
        "n_auto_steps":             n_auto_steps,
        "min_step_size":            min_step,
        "max_step_size":            max_step,
        "mean_step_size":           mean_step,
        "step_size_to_noise_ratio": step_to_noise,
        "early_step_detected":      early_step,
        "anomalous":                int(bool(anomalous)),
        "partial_bleach_confidence": float(partial_bleach_confidence),
        "pop_step_size":            float(pop_step_size),
    }


# ---------------------------------------------------------------------------
# Stage 6a — HMM level estimation utility
# ---------------------------------------------------------------------------

def hmm_level_estimates(
        signal: np.ndarray,
        breakpoints: "list[int] | np.ndarray",
        min_seg_frames: int = 3,
        n_em_iter: int = 10,
        em_tol: float = 1e-4,
) -> "tuple[np.ndarray, list[float]]":
    """
    Estimate per-segment fluorescence levels using a Gaussian-emission HMM
    with PELT breakpoints as the structural prior.

    Replaces the simple segment-mean (``seg.mean()``) approach used in Stages
    3–5.  For short segments or low-SNR traces the plain mean is pulled by
    noise spikes; the HMM smooths within each level and enforces the known
    step structure.

    Algorithm
    ---------
    1.  Build an initial state sequence from the PELT breakpoints (each frame
        gets the index of its segment).
    2.  Initialise per-state Gaussian parameters (mean, variance) from the
        segment sample statistics.
    3.  Run a small number of Baum-Welch EM iterations **with the transition
        matrix held fixed** to the PELT structure.  Only the emission
        parameters (mean and variance per state) are updated.  This prevents
        the HMM from wandering to solutions that contradict PELT's step
        positions while still fitting cleaner level estimates.
    4.  Return (a) a per-frame state-mean array (same length as ``signal``)
        and (b) a list of one mean per segment — a direct drop-in for the
        ``segment_means`` lists computed elsewhere.

    Graceful degradation
    --------------------
    If the signal is too short, the HMM fails numerically, or any segment
    after EM has fewer than ``min_seg_frames`` frames, the function falls
    back to plain segment means so callers always get a valid result.

    Parameters
    ----------
    signal          : 1-D float array, background-subtracted, blank frames
                      removed.
    breakpoints     : PELT breakpoint indices (may be empty).
    min_seg_frames  : minimum frames per segment; segments shorter than this
                      are excluded from HMM fitting and filled with their
                      plain mean instead.
    n_em_iter       : maximum Baum-Welch iterations (default 10 — enough for
                      level convergence on typical single-molecule traces).
    em_tol          : convergence tolerance on log-likelihood change.

    Returns
    -------
    level_per_frame : np.ndarray, shape (len(signal),) — smoothed level at
                      each frame (the HMM state mean for that frame).
    segment_levels  : list[float] — one mean per segment in breakpoint order.
                      Length == len(breakpoints) + 1.
    """
    sig   = np.asarray(signal, dtype=np.float64)
    n     = len(sig)
    bps   = sorted(int(b) for b in breakpoints)
    n_seg = len(bps) + 1

    boundaries = [0] + bps + [n]

    # ── Plain-mean fallback ───────────────────────────────────────────────
    def _plain_means() -> "tuple[np.ndarray, list[float]]":
        seg_means = []
        lpf = np.empty(n, dtype=np.float64)
        for i in range(n_seg):
            lo, hi = boundaries[i], boundaries[i + 1]
            m = float(sig[lo:hi].mean()) if hi > lo else 0.0
            seg_means.append(m)
            lpf[lo:hi] = m
        return lpf, seg_means

    # Degenerate cases — fall back immediately
    if n < 4 or n_seg < 2:
        return _plain_means()

    # Check all segments are long enough for EM to be meaningful
    seg_lengths = [boundaries[i + 1] - boundaries[i] for i in range(n_seg)]
    if any(l < min_seg_frames for l in seg_lengths):
        return _plain_means()

    # ── Initialise emission parameters ───────────────────────────────────
    # mu[k] and sigma[k] from the sample statistics of each segment.
    mu    = np.array([sig[boundaries[i]:boundaries[i+1]].mean()
                      for i in range(n_seg)])
    sigma = np.array([max(sig[boundaries[i]:boundaries[i+1]].std(), 1.0)
                      for i in range(n_seg)])

    # ── Fixed transition matrix (enforces PELT structure) ─────────────────
    # States only transition to the next state (or stay).  We set a small
    # self-transition probability so the HMM doesn't collapse to a single
    # state, but the structure is dominated by the PELT prior.
    self_prob = 0.99
    A = np.zeros((n_seg, n_seg), dtype=np.float64)
    for k in range(n_seg - 1):
        A[k, k]     = self_prob
        A[k, k + 1] = 1.0 - self_prob
    A[-1, -1] = 1.0   # absorbing last state

    # Initial state distribution (start in state 0)
    pi = np.zeros(n_seg, dtype=np.float64)
    pi[0] = 1.0

    # ── Baum-Welch EM (emission parameters only) ──────────────────────────
    try:
        prev_ll = -np.inf
        for _ in range(n_em_iter):
            # ── Forward pass ──────────────────────────────────────────────
            log_b = np.zeros((n, n_seg), dtype=np.float64)
            for k in range(n_seg):
                diff       = sig - mu[k]
                log_b[:, k] = (-0.5 * (diff / sigma[k])**2
                               - np.log(sigma[k])
                               - 0.5 * np.log(2.0 * np.pi))

            log_alpha = np.full((n, n_seg), -np.inf)
            log_alpha[0] = np.log(pi + 1e-300) + log_b[0]
            log_A = np.log(A + 1e-300)

            for t in range(1, n):
                for k in range(n_seg):
                    log_alpha[t, k] = (
                        np.logaddexp.reduce(log_alpha[t-1] + log_A[:, k])
                        + log_b[t, k]
                    )

            ll = np.logaddexp.reduce(log_alpha[-1])

            # ── Backward pass ─────────────────────────────────────────────
            log_beta = np.zeros((n, n_seg), dtype=np.float64)
            for t in range(n - 2, -1, -1):
                for k in range(n_seg):
                    log_beta[t, k] = np.logaddexp.reduce(
                        log_A[k] + log_b[t+1] + log_beta[t+1]
                    )

            # ── Posterior (gamma) ─────────────────────────────────────────
            log_gamma = log_alpha + log_beta
            log_gamma -= np.logaddexp.reduce(log_gamma, axis=1, keepdims=True)
            gamma = np.exp(log_gamma)

            # ── M-step: update emission means and variances ───────────────
            gamma_sum = gamma.sum(axis=0) + 1e-300
            mu_new    = (gamma * sig[:, None]).sum(axis=0) / gamma_sum
            diff2     = (sig[:, None] - mu_new) ** 2
            var_new   = (gamma * diff2).sum(axis=0) / gamma_sum
            sigma_new = np.sqrt(np.maximum(var_new, 1.0))

            mu    = mu_new
            sigma = sigma_new

            # ── Convergence check ─────────────────────────────────────────
            if abs(ll - prev_ll) < em_tol:
                break
            prev_ll = ll

        # ── Segment means from final posterior ────────────────────────────
        # Use the dominant state in each PELT segment (weighted by gamma)
        # rather than the argmax state per frame, to reduce noise.
        seg_levels = []
        for i in range(n_seg):
            lo, hi = boundaries[i], boundaries[i + 1]
            if hi <= lo:
                seg_levels.append(0.0)
                continue
            # Weighted mean of HMM state means, weighted by posterior mass
            # in this segment — gives a smooth level even for short windows.
            gamma_seg = gamma[lo:hi].mean(axis=0)   # (n_seg,) mean posteriors
            seg_levels.append(float(gamma_seg @ mu))

        # Build per-frame level array
        level_per_frame = np.empty(n, dtype=np.float64)
        for i in range(n_seg):
            lo, hi = boundaries[i], boundaries[i + 1]
            level_per_frame[lo:hi] = seg_levels[i]

        return level_per_frame, seg_levels

    except Exception:
        # Any numerical failure → plain means
        return _plain_means()


# ---------------------------------------------------------------------------
# Stage 3 — Population step size estimation
# ---------------------------------------------------------------------------

def estimate_step_size_population(
        spots: list,
        max_stoichiometry: int = 2,
        min_pool: int = 20,
        outlier_multiple: float = 2.5,
        n_iter: int = 3,
        noise_floor_multiplier: float = 0.5,
        session_pool: "list | None" = None,
) -> dict:
    """
    Estimate the single-fluorophore step size from the population of all
    PELT-detected steps across all spots in one channel of one movie.

    Strategy
    --------
    - Collect every individual downward step amplitude from every spot.
    - Use the median of those raw steps as a robust starting estimate of the
      single-fluorophore brightness (the median is far less sensitive to
      simultaneous multi-fluorophore bleaching events than the mean, which
      gets pulled upward by 2×, 3×, … steps).
    - Iterate: discard steps above outlier_multiple × current estimate (these
      are almost certainly simultaneous bleaching of 2+ fluorophores) and
      recompute the median.  Converges in 2-3 iterations.
    - Apply a noise floor: discard steps whose amplitude is below
      noise_floor_multiplier × current estimate (these are near-noise
      artefacts, not real steps).
    - If the resulting pool is smaller than min_pool, fall back to
      session_pool (steps pooled from earlier movies in the same session)
      if provided.  If that is also too small, return None for the estimate
      so callers can degrade gracefully.

    Parameters
    ----------
    spots              : list of spot dicts (as returned by prepare_spot)
    max_stoichiometry  : used to set the outlier ceiling — steps larger than
                         outlier_multiple × estimate are presumed to be
                         simultaneous events from ≥2 fluorophores and excluded
    min_pool           : minimum number of steps needed for a reliable estimate
    outlier_multiple   : steps above this × current estimate are excluded
    n_iter             : number of refinement iterations
    noise_floor_multiplier : steps below this × current estimate are excluded
    session_pool       : optional flat list of step amplitudes from prior
                         movies in this session, used as fallback if the
                         per-movie pool is too small

    Returns
    -------
    dict with keys:
        step_size_estimate  : float or None — estimated single-step amplitude
        step_size_std       : float or None — std of the refined step pool
        step_size_iqr       : float or None — IQR of the refined step pool
        n_steps_used        : int  — number of steps in the refined pool
        n_steps_raw         : int  — total steps before filtering
        used_session_pool   : bool — True if fallback to session pool occurred
        all_steps_raw       : list[float] — all raw step amplitudes (for
                              accumulating a session pool externally)
    """
    # ── Collect all downward step amplitudes from every spot ──────────────
    all_steps_raw: list[float] = []
    for s in spots:
        bps  = sorted(int(b) for b in s.get("auto_bps", []))
        sig  = s["signal"]
        if not bps:
            continue
        _, segment_means = hmm_level_estimates(sig, bps)
        for i in range(len(segment_means) - 1):
            delta = segment_means[i] - segment_means[i + 1]
            if delta > 0:          # only downward steps
                all_steps_raw.append(delta)

    n_raw = len(all_steps_raw)

    # ── Iterative refinement ──────────────────────────────────────────────
    def _refine(steps: list) -> "tuple[float, list]":
        """One iteration: compute median, drop outliers & noise floor."""
        if not steps:
            return 0.0, []
        arr      = np.array(steps, dtype=np.float64)
        estimate = float(np.median(arr))
        if estimate <= 0:
            return estimate, steps
        ceiling  = outlier_multiple * estimate
        floor    = noise_floor_multiplier * estimate
        kept     = [s for s in steps if floor <= s <= ceiling]
        return estimate, kept

    working = list(all_steps_raw)
    estimate = None

    if working:
        for _ in range(n_iter):
            est, working = _refine(working)
            if not working:
                break
            estimate = est

    n_refined = len(working)
    used_session = False

    # ── Session-pool fallback ─────────────────────────────────────────────
    if n_refined < min_pool and session_pool:
        print(f"  [step size] Per-movie pool too small ({n_refined} steps) "
              f"— falling back to session pool ({len(session_pool)} steps).")
        working  = list(session_pool)
        used_session = True
        for _ in range(n_iter):
            est, working = _refine(working)
            if not working:
                break
            estimate = est
        n_refined = len(working)

    # ── Final statistics ──────────────────────────────────────────────────
    if not working or estimate is None or estimate <= 0:
        print(f"  [step size] Could not estimate step size "
              f"(pool too small: {n_refined} steps after filtering).")
        return {
            "step_size_estimate": None,
            "step_size_std":      None,
            "step_size_iqr":      None,
            "n_steps_used":       n_refined,
            "n_steps_raw":        n_raw,
            "used_session_pool":  used_session,
            "all_steps_raw":      all_steps_raw,
        }

    arr = np.array(working, dtype=np.float64)
    # Final estimate: median of refined pool
    final_estimate = float(np.median(arr))
    final_std      = float(arr.std())
    q25, q75       = np.percentile(arr, [25, 75])
    final_iqr      = float(q75 - q25)

    print(f"  [step size] estimate={final_estimate:.2f}  "
          f"std={final_std:.2f}  IQR={final_iqr:.2f}  "
          f"n={n_refined}/{n_raw} steps used")

    return {
        "step_size_estimate": final_estimate,
        "step_size_std":      final_std,
        "step_size_iqr":      final_iqr,
        "n_steps_used":       n_refined,
        "n_steps_raw":        n_raw,
        "used_session_pool":  used_session,
        "all_steps_raw":      all_steps_raw,
    }


def attach_step_size_estimate(spots: list, step_size_meta: dict) -> None:
    """
    Store the population step size estimate on every spot dict in-place.
    Downstream stages (anomalous trace filtering, partial bleach detection)
    read spot["pop_step_size"] and spot["pop_step_size_std"].
    """
    est = step_size_meta.get("step_size_estimate")
    std = step_size_meta.get("step_size_std")
    iqr = step_size_meta.get("step_size_iqr")
    for s in spots:
        s["pop_step_size"]     = est
        s["pop_step_size_std"] = std
        s["pop_step_size_iqr"] = iqr



# ---------------------------------------------------------------------------
# Stage 4 — Anomalous trace filtering
# ---------------------------------------------------------------------------

def flag_anomalous_traces(
        spots: list,
        drift_slope_threshold: float = 0.05,
        integer_multiple_tolerance: float = 0.40,
        min_signal_snr: float = 1.5,
) -> None:
    """
    Flag traces that do not show clean stepwise photobleaching in-place.
    Sets spot["anomalous"] = True/False and spot["anomalous_reason"] = str.

    Anomalous signatures detected
    ------------------------------
    1. Slow drift — trace has a significantly non-zero linear slope relative
       to its amplitude, and PELT found few or no steps.  Typical of large
       aggregates or photobleaching continua.

    2. No steps, signal above background — PELT found 0 steps but the trace
       mean is clearly above noise (SNR > min_signal_snr).  The spot is
       fluorescing but never steps — unresolvable aggregate or stuck dye.

    3. Inconsistent step sizes — when a population step size estimate is
       available (pop_step_size), check whether every detected PELT step is
       within integer_multiple_tolerance of an integer multiple of that
       estimate.  Steps at e.g. 3.7× or 6.2× the single-step size indicate
       the spot is an aggregate whose steps are unresolvable mixtures.

    Parameters
    ----------
    spots                     : list of spot dicts (after prepare_spot and
                                 attach_step_size_estimate)
    drift_slope_threshold     : |slope / amplitude| above which a 0-step or
                                 1-step trace is considered drifting.
                                 Default 0.05 (5% of amplitude per frame).
    integer_multiple_tolerance: fractional tolerance for integer-multiple
                                 step size check.  0.40 = ±40% of the
                                 single-step estimate.  Deliberately generous
                                 because dim fluorophores have noisy steps.
    min_signal_snr            : mean/std ratio above which a 0-step trace
                                 is considered to have real signal (and
                                 therefore anomalous).

    Note: spots flagged anomalous are still shown in the review UI with a
    distinct purple tint so the researcher can confirm or dismiss the flag.
    They are NOT automatically excluded — the flag is advisory only.
    """
    for s in spots:
        sig  = s["signal"]
        bps  = sorted(int(b) for b in s.get("auto_bps", []))
        n    = len(sig)

        s["anomalous"]        = False
        s["anomalous_reason"] = ""

        if n < 10:
            continue   # trace too short to assess

        # ── Check 1: slow drift ────────────────────────────────────────────
        if len(bps) <= 1:
            # Fit a line to the whole signal
            x      = np.arange(n, dtype=np.float64)
            coeffs = np.polyfit(x, sig.astype(np.float64), 1)
            slope  = coeffs[0]
            amplitude = float(sig.max() - sig.min())
            if amplitude > 0:
                rel_slope = abs(slope) / amplitude
                if rel_slope > drift_slope_threshold:
                    s["anomalous"]        = True
                    s["anomalous_reason"] = (
                        f"drift (rel_slope={rel_slope:.3f}, "
                        f"threshold={drift_slope_threshold})")
                    continue

        # ── Check 2: no steps but clear signal ────────────────────────────
        if len(bps) == 0:
            sig_mean = float(sig.mean())
            sig_std  = float(sig.std()) + 1e-6
            snr      = sig_mean / sig_std
            if snr > min_signal_snr and sig_mean > 0:
                s["anomalous"]        = True
                s["anomalous_reason"] = (
                    f"no steps, signal present (SNR={snr:.2f})")
                continue

        # ── Check 3: step sizes inconsistent with integer multiples ───────
        pop_step = s.get("pop_step_size")
        if pop_step and pop_step > 0 and len(bps) >= 1:
            _, segment_means = hmm_level_estimates(sig, bps)
            bad_steps = []
            for i in range(len(segment_means) - 1):
                delta = segment_means[i] - segment_means[i + 1]
                if delta <= 0:
                    continue   # upward steps handled separately
                ratio     = delta / pop_step
                nearest_n = round(ratio)
                if nearest_n < 1:
                    nearest_n = 1
                deviation = abs(ratio - nearest_n) / nearest_n
                if deviation > integer_multiple_tolerance:
                    bad_steps.append(
                        f"Δ{delta:.1f} ({ratio:.2f}× step, "
                        f"expected {nearest_n}×)")
            if bad_steps:
                s["anomalous"]        = True
                s["anomalous_reason"] = (
                    f"non-integer step sizes: {'; '.join(bad_steps)}")

    n_flagged = sum(1 for s in spots if s["anomalous"])
    if n_flagged:
        print(f"  [anomalous] {n_flagged}/{len(spots)} spots flagged "
              f"as anomalous traces")


# ---------------------------------------------------------------------------
# Stage 5 — Partial bleach detection
# ---------------------------------------------------------------------------

def detect_partial_bleach(
        spots: list,
        max_stoichiometry: int = 2,
        min_post_step_frames: int = 30,
        residual_tolerance: float = 0.40,
        slope_threshold: float = 0.03,
        pre_step_tolerance: float = 0.40,
        pre_step_min_frames: int = 10,
) -> None:
    """
    Detect incomplete traces where the residual fluorescence level after the
    last PELT step is consistent with N unbleached fluorophores.  Updates
    each spot dict in-place.

    For a spot to be called a partial bleach:
      1. It must already be flagged incomplete (spot["incomplete"] == True).
      2. The last PELT breakpoint must leave at least min_post_step_frames
         frames remaining.  Shorter windows give unreliable residual estimates.
      3. The post-step signal slope must be near-zero (stable residual, not
         a slow bleacher still in progress).
      4. The residual level above background (estimated from the tail of the
         post-step window) must be within residual_tolerance of
         N × pop_step_size for some positive integer N such that
         (observed_steps + N) <= max_stoichiometry.
      5. Optionally, if a clean pre-step window exists (>= pre_step_min_frames
         before the first breakpoint), the pre-step level provides a
         corroborating check: it should ~= (observed_steps + N) × pop_step_size.
         Agreement boosts confidence; disagreement reduces it.

    A pop_step_size must be available on the spot dict (set by Stage 3).
    If it is None, partial bleach detection is skipped for that spot.

    Sets on each spot dict
    ----------------------
    partial_bleach           : bool  — True if a partial bleach was detected
    n_unbleached_inferred    : int   — N unbleached fluorophores (0 if none)
    partial_bleach_confidence: float — 0.0-1.0 confidence score
    partial_bleach_n_mer     : str   — inferred total stoichiometry label
    partial_bleach_reclassified: bool — True when incomplete flag was cleared

    Parameters
    ----------
    spots                 : list of spot dicts (after Stages 3 and 4)
    max_stoichiometry     : upper bound on total inferred stoichiometry
    min_post_step_frames  : minimum frames after last step for a reliable
                            residual estimate (default 30)
    residual_tolerance    : fractional tolerance for residual ~= N x step_size
                            (default 0.40 = +/-40%)
    slope_threshold       : |slope / residual_mean| above which the post-step
                            signal is considered still drifting (default 0.03)
    pre_step_tolerance    : fractional tolerance for pre-step level check
                            (default 0.40)
    pre_step_min_frames   : minimum frames before first breakpoint to use the
                            pre-step level as corroborating evidence (default 10)
    """
    n_detected = 0

    for s in spots:
        # Initialise output keys on every spot
        s["partial_bleach"]              = False
        s["n_unbleached_inferred"]       = 0
        s["partial_bleach_confidence"]   = 0.0
        s["partial_bleach_n_mer"]        = ""
        s["partial_bleach_reclassified"] = False

        # Only consider incomplete spots
        if not s.get("incomplete", False):
            continue

        pop_step = s.get("pop_step_size")
        if pop_step is None or pop_step <= 0:
            continue   # no step size reference — cannot assess

        sig   = s["signal"]
        bps   = sorted(int(b) for b in s["bps"])
        n_obs = len(bps)
        n_sig = len(sig)

        if n_sig < min_post_step_frames + 5:
            continue   # trace too short to assess

        # ── Post-step window ──────────────────────────────────────────────
        last_bp     = bps[-1] if bps else 0
        post_frames = n_sig - last_bp
        if post_frames < min_post_step_frames:
            continue   # too few frames after last step for reliable estimate

        post_signal = sig[last_bp:]

        # ── Slope check — residual must be stable ─────────────────────────
        x      = np.arange(len(post_signal), dtype=np.float64)
        coeffs = np.polyfit(x, post_signal.astype(np.float64), 1)
        slope  = coeffs[0]
        residual_mean = float(post_signal.mean())
        if residual_mean > 0:
            rel_slope = abs(slope) / residual_mean
            if rel_slope > slope_threshold:
                continue   # still drifting — not a stable residual

        # ── HMM-refined segment levels ────────────────────────────────────
        # Run HMM on the full trace (all segments incl. the post-step window)
        # so the final-segment level estimate benefits from the fitted model.
        # Fall back to post_signal.mean() if HMM has too few frames.
        _, hmm_seg_levels = hmm_level_estimates(sig, bps)
        hmm_residual = hmm_seg_levels[-1] if hmm_seg_levels else residual_mean

        # ── Residual level above background ───────────────────────────────
        # Background: last fraction of post-step window (plain mean — this
        # window has no further steps, so the HMM adds nothing here).
        bg_tail           = max(5, len(post_signal) // 5)
        bg_est            = float(post_signal[-bg_tail:].mean())
        residual_above_bg = hmm_residual - bg_est

        if residual_above_bg <= 0:
            continue   # residual is at or below estimated background

        # ── Find best-matching integer N ──────────────────────────────────
        best_n         = None
        best_deviation = float("inf")
        for n_unbleached in range(1, max_stoichiometry - n_obs + 1):
            if n_obs + n_unbleached > max_stoichiometry:
                break
            expected  = n_unbleached * pop_step
            deviation = abs(residual_above_bg - expected) / expected
            if deviation < residual_tolerance and deviation < best_deviation:
                best_deviation = deviation
                best_n         = n_unbleached

        if best_n is None:
            continue   # no integer multiple within tolerance

        # ── Pre-step corroboration ────────────────────────────────────────
        corroboration_score = 1.0

        first_bp = bps[0] if bps else n_sig
        if first_bp >= pre_step_min_frames:
            # Use HMM first-segment level as pre-step estimate
            pre_level      = hmm_seg_levels[0] if hmm_seg_levels else float(sig[:first_bp].mean())
            total_inferred = n_obs + best_n
            expected_pre   = total_inferred * pop_step
            if expected_pre > 0:
                pre_deviation = abs(pre_level - expected_pre) / expected_pre
                if pre_deviation <= pre_step_tolerance:
                    corroboration_score = min(1.0,
                        1.0 - 0.5 * (pre_deviation / pre_step_tolerance))
                else:
                    corroboration_score = max(0.2,
                        1.0 - (pre_deviation / pre_step_tolerance))

        # ── Confidence score ──────────────────────────────────────────────
        base_confidence = 1.0 - (best_deviation / residual_tolerance)
        length_factor   = min(1.0, post_frames / (min_post_step_frames * 3))
        confidence      = float(base_confidence * corroboration_score * length_factor)
        confidence      = max(0.0, min(1.0, confidence))

        # ── Record result ─────────────────────────────────────────────────
        total_stoich                    = n_obs + best_n
        s["partial_bleach"]             = True
        s["n_unbleached_inferred"]      = best_n
        s["partial_bleach_confidence"]  = confidence
        s["partial_bleach_n_mer"]       = stoich_label(total_stoich)

        # Clear the incomplete flag — this spot is now classified
        s["incomplete"]                  = False
        s["partial_bleach_reclassified"] = True
        n_detected += 1

    if n_detected:
        print(f"  [partial bleach] {n_detected}/{len(spots)} incomplete spots "
              f"reclassified as partial bleach")


# ---------------------------------------------------------------------------

CLS_COLORS = {
    "monomer":   "#2ecc71",
    "dimer":     "#3498db",
    "trimer":    "#9b59b6",
    "tetramer":  "#e67e22",
    "pentamer":  "#1abc9c",
    "hexamer":   "#f39c12",
    "heptamer":  "#16a085",
    "octamer":   "#8e44ad",
    "aggregate": "#e74c3c",
    "bad_trace": "#888888",
}

def cls_color(cls: str) -> str:
    """Return a display color for a classification label."""
    return CLS_COLORS.get(cls, "#888888")


class TraceReviewUI:
    def __init__(self, spots, title, blank_frames, channel_label,
                 max_stoichiometry: int = 2):
        self.spots   = spots
        self.title   = title
        self.blank   = blank_frames
        self.ch_lbl  = channel_label
        self.idx     = 0
        self.total   = len(spots)
        self.max_stoich = max_stoichiometry

        # Two-panel layout: left = annotated, right = raw (no steps)
        self.fig, (self.ax_ann, self.ax_raw) = plt.subplots(
            1, 2, figsize=(20, 5), sharey=False)
        self.fig.subplots_adjust(bottom=0.20, top=0.87, wspace=0.08)
        self.fig.canvas.manager.set_window_title(title)

        ax_prev     = self.fig.add_axes([0.04, 0.04, 0.06, 0.07])
        ax_next     = self.fig.add_axes([0.11, 0.04, 0.06, 0.07])
        ax_clear    = self.fig.add_axes([0.24, 0.04, 0.07, 0.07])
        ax_complete = self.fig.add_axes([0.33, 0.04, 0.11, 0.07])
        ax_override = self.fig.add_axes([0.46, 0.04, 0.11, 0.07])
        ax_bad      = self.fig.add_axes([0.59, 0.04, 0.09, 0.07])
        ax_good     = self.fig.add_axes([0.70, 0.04, 0.09, 0.07])
        ax_quit     = self.fig.add_axes([0.88, 0.04, 0.07, 0.07])

        self.btn_prev     = Button(ax_prev,     "◄ Prev")
        self.btn_next     = Button(ax_next,     "Next ►")
        self.btn_clear    = Button(ax_clear,    "Reset [c]")
        self.btn_complete = Button(ax_complete, "Mark complete [m]",
                                   color="#ffe0cc", hovercolor="#ffcba4")
        self.btn_override = Button(ax_override, "Class: Monomer ↻ [o]",
                                   color="#e8e8f8", hovercolor="#d0d0f0")
        self.btn_bad      = Button(ax_bad,      "Bad trace [x]",
                                   color="#f0f0f0", hovercolor="#e0e0e0")
        self.btn_good     = Button(ax_good,     "Good trace [y]",
                                   color="#f0fff0", hovercolor="#d0f0d0")
        self.btn_quit     = Button(ax_quit,     "Done ✓ [q]")

        self.btn_prev.on_clicked(    lambda e: self._navigate(-1))
        self.btn_next.on_clicked(    lambda e: self._navigate(+1))
        self.btn_clear.on_clicked(   lambda e: self._reset())
        self.btn_complete.on_clicked(lambda e: self._toggle_complete())
        self.btn_override.on_clicked(lambda e: self._cycle_override())
        self.btn_bad.on_clicked(     lambda e: self._toggle_bad_trace())
        self.btn_good.on_clicked(    lambda e: self._toggle_good_trace())
        self.btn_quit.on_clicked(    lambda e: self._quit())

        self.fig.canvas.mpl_connect("key_press_event",    self._on_key)
        self.fig.canvas.mpl_connect("button_press_event", self._on_click)
        self._draw()

    def _draw(self):
        spot  = self.spots[self.idx]
        trace = spot["trace"]
        sig   = spot["signal"]
        bps   = sorted(int(b) for b in spot["bps"])
        n     = len(bps)
        incomplete      = spot["incomplete"]
        bad_trace       = spot.get("bad_trace", False)
        good_trace      = spot.get("good_trace", False)
        class_override  = spot.get("class_override", None)
        # Recompute upward-step flag live whenever bps change
        upward_step = detect_upward_steps(sig, bps)
        spot["has_upward_step"] = upward_step
        if bad_trace:
            cls = "bad_trace"
        elif class_override is not None:
            cls = class_override
        elif incomplete:
            cls = "aggregate"
        else:
            cls = classify_steps(n, self.max_stoich)
        color  = cls_color(cls)
        frames = np.arange(len(trace))
        n_full = len(trace)

        def _draw_base(ax, annotate):
            ax.clear()
            # Blank region
            if self.blank > 0:
                ax.axvspan(0, self.blank - 0.5, color="#eeeeee", alpha=0.7, zorder=0)
                ax.axvline(self.blank - 0.5, color="#aaaaaa",
                           linewidth=1.0, linestyle=":", zorder=1)
            # Incomplete tail highlight (last 10 frames of signal)
            if incomplete:
                tail_start = n_full - 10
                ax.axvspan(tail_start, n_full - 1,
                           color="#ff6600", alpha=0.25, zorder=0)
                ax.axvline(tail_start, color="#ff6600",
                           linewidth=1.5, linestyle="-", alpha=0.7, zorder=2)
            # Raw trace
            ax.plot(frames, trace, color="#bbbbbb", linewidth=0.9, zorder=2)
            ax.spines["top"].set_visible(False)
            ax.spines["right"].set_visible(False)
            ax.set_xlabel("Frame", fontsize=10)

        # Compute y-axis limits from the full trace with padding
        sig_min = float(trace.min())
        sig_max = float(trace.max())
        pad = (sig_max - sig_min) * 0.08 if sig_max > sig_min else 1.0
        y_lo = sig_min - pad
        y_hi = sig_max + pad

        # --- Left panel: annotated with steps ---
        _draw_base(self.ax_ann, annotate=True)
        level_per_frame, levels = hmm_level_estimates(sig, bps)
        bounds = [0] + bps + [len(sig)]
        # HMM continuous fit overlay (thin, semi-transparent)
        self.ax_ann.plot(np.arange(len(level_per_frame)) + self.blank,
                         level_per_frame,
                         color=color, linewidth=1.0, alpha=0.35,
                         linestyle="-", zorder=3)
        # Horizontal level lines per segment
        for i, level in enumerate(levels):
            self.ax_ann.hlines(level,
                               bounds[i] + self.blank,
                               bounds[i+1] + self.blank - 1,
                               colors=color, linewidth=2.5, zorder=4)
        for bp in bps:
            self.ax_ann.axvline(bp + self.blank, color=color,
                                linewidth=1.3, linestyle="--", alpha=0.8, zorder=3)

        # Annotate each step with its drop amplitude
        for i, bp in enumerate(bps):
            drop  = levels[i] - levels[i + 1]
            y_mid = (levels[i] + levels[i + 1]) / 2
            x_pos = bp + self.blank + 2
            self.ax_ann.text(
                x_pos, y_mid, f"\u0394{drop:.0f}",
                color=color, fontsize=8, fontweight="bold",
                va="center", ha="left", zorder=5,
                bbox=dict(boxstyle="round,pad=0.15", fc="white",
                          ec="none", alpha=0.7),
            )

        self.ax_ann.set_ylabel("Intensity (bg subtracted)", fontsize=10)
        inc_str      = "  ⚠ INCOMPLETE" if incomplete else ""
        override_str = f"  [override: {class_override.upper()}]" if class_override else ""
        self.btn_complete.label.set_text(
            "Mark complete [m]" if incomplete else "Mark incomplete [m]")
        self.btn_bad.label.set_text("Unmark bad [x]"   if bad_trace  else "Bad trace [x]")
        self.btn_bad.ax.set_facecolor("#ddbbbb" if bad_trace  else "#f0f0f0")
        self.btn_good.label.set_text("Unmark good [y]" if good_trace else "Good trace [y]")
        self.btn_good.ax.set_facecolor("#b3e6b3" if good_trace else "#f0fff0")
        _cycle    = [None] + [stoich_label(i) for i in range(1, self.max_stoich + 1)] + ["aggregate"]
        _cur      = spot.get("class_override", None)
        if _cur not in _cycle:
            _cur = None
        _next     = _cycle[(_cycle.index(_cur) + 1) % len(_cycle)]
        _next_lbl = _next.capitalize() if _next else "Auto"
        self.btn_override.label.set_text(f"Class: {_next_lbl} ↻ [o]")
        self.ax_ann.set_title(
            f"{self.ch_lbl}  —  x={spot['x']:.1f}, y={spot['y']:.1f}  "
            f"|  {n} step{'s' if n != 1 else ''}  →  {cls.upper()}{inc_str}{override_str}",
            fontsize=11, color=color, fontweight="bold")

        # --- Right panel: raw trace only ---
        _draw_base(self.ax_raw, annotate=False)
        self.ax_raw.set_title("Raw trace (no steps)", fontsize=11,
                              color="#888888")

        # Apply matching y-axis limits to both panels
        self.ax_ann.set_ylim(y_lo, y_hi)
        self.ax_raw.set_ylim(y_lo, y_hi)

        # ML edit probability flag
        ml_prob = spot.get("ml_edit_prob")
        ml_flagged = ml_prob is not None and ml_prob >= ML_FLAG_THRESHOLD
        if ml_prob is not None:
            ml_note = f"  |  🔶 ML: {ml_prob:.0%} likely needs edit" if ml_flagged \
                      else f"  |  ML: {ml_prob:.0%}"
        else:
            ml_note = ""

        # Anomalous trace flag (Stage 4)
        anomalous        = spot.get("anomalous", False)
        anomalous_reason = spot.get("anomalous_reason", "")
        anomalous_note   = f"  |  🔴 ANOMALOUS: {anomalous_reason}" if anomalous else ""

        # Partial bleach flag (Stage 5)
        partial_bleach    = spot.get("partial_bleach", False)
        pb_confidence     = spot.get("partial_bleach_confidence", 0.0)
        pb_n_mer          = spot.get("partial_bleach_n_mer", "")
        pb_n_unbleached   = spot.get("n_unbleached_inferred", 0)
        partial_bleach_note = (
            f"  |  🟡 PARTIAL BLEACH: {pb_n_mer} inferred "
            f"({pb_n_unbleached} unbleached, conf={pb_confidence:.0%})"
            if partial_bleach else "")

        # Tint figure background: red > purple > yellow > amber > white
        if upward_step:
            self.fig.patch.set_facecolor("#fff0f0")
        elif anomalous:
            self.fig.patch.set_facecolor("#f5f0ff")
        elif partial_bleach:
            self.fig.patch.set_facecolor("#fffff0")
        elif ml_flagged:
            self.fig.patch.set_facecolor("#fff8ee")
        else:
            self.fig.patch.set_facecolor("white")

        inc_note      = "  |  ⚠ incomplete bleaching" if incomplete else ""
        bad_note      = "  |  ✖ bad trace"            if bad_trace  else ""
        good_note     = "  |  ★ good trace"           if good_trace else ""
        override_note = f"  |  override: {class_override.upper()}" if class_override else ""
        upward_note   = "  |  ↑ UPWARD STEP"          if upward_step else ""
        self.fig.suptitle(
            f"{self.title}   [{self.idx+1} / {self.total}]{inc_note}{bad_note}{good_note}{override_note}{upward_note}{anomalous_note}{partial_bleach_note}{ml_note}\n"
            f"Left-click (left panel): add step   Right-click: remove   "
            f"[c]: reset   [o]: cycle class   [x]: bad trace   [y]: good trace   [←/→]: navigate   [q]: done",
            fontsize=9, color=(
                "#cc0000" if upward_step else
                "#7b2fbe" if anomalous else
                "#888888" if bad_trace else
                "#228822" if good_trace else
                "#b8a000" if partial_bleach else
                "#cc4400" if (incomplete and not class_override) else
                "#b06000" if ml_flagged else
                "#666666"))
        self.fig.canvas.draw_idle()

    def _on_key(self, event):
        if   event.key in ("right",): self._navigate(+1)
        elif event.key in ("left",):  self._navigate(-1)
        elif event.key == "c":        self._reset()
        elif event.key == "m":        self._toggle_complete()
        elif event.key == "o":        self._cycle_override()
        elif event.key == "x":        self._toggle_bad_trace()
        elif event.key == "y":        self._toggle_good_trace()
        elif event.key == "q":        self._quit()

    def _on_click(self, event):
        if event.inaxes is not self.ax_ann:
            return
        frame_sig = int(round(event.xdata)) - self.blank
        sig_len   = len(self.spots[self.idx]["signal"])
        if frame_sig < 0 or frame_sig >= sig_len:
            return
        spot = self.spots[self.idx]
        bps  = list(spot["bps"])
        if event.button == 1:       # add
            if frame_sig not in bps:
                bps.append(frame_sig)
                spot["bps"] = np.array(sorted(bps), dtype=int)
                self._draw()
        elif event.button == 3:     # remove nearest
            if bps:
                nearest = min(bps, key=lambda b: abs(b - frame_sig))
                bps.remove(nearest)
                spot["bps"] = np.array(sorted(bps), dtype=int)
                self._draw()

    def _navigate(self, d):
        next_idx = self.idx + d
        # When moving forward past the last spot, confirm before finishing
        if next_idx >= self.total:
            self._confirm_done()
            return
        # Allow free backward navigation (wrapping only backward to last)
        self.idx = max(0, next_idx)
        self._draw()

    def _confirm_done(self):
        """Show a matplotlib-native confirmation dialog at end of pass."""
        from matplotlib.widgets import Button as MplButton

        # Dim the existing axes to indicate pause
        for ax in (self.ax_ann, self.ax_raw):
            ax.set_facecolor("#f5f5f5")
        self.fig.suptitle(
            f"{self.title}  \u2014  End of pass ({self.total} spots reviewed)\nReady to move on?",
            fontsize=13, color="#333333", fontweight="bold")
        self.fig.canvas.draw_idle()

        # Overlay two buttons in the centre of the figure
        ax_yes = self.fig.add_axes([0.35, 0.42, 0.13, 0.10])
        ax_no  = self.fig.add_axes([0.52, 0.42, 0.13, 0.10])
        btn_yes = MplButton(ax_yes, "Done ✓", color="#2ecc71", hovercolor="#27ae60")
        btn_no  = MplButton(ax_no,  "Go back", color="#e0e0e0", hovercolor="#cccccc")

        def _yes(event):
            ax_yes.remove()
            ax_no.remove()
            self._quit()

        def _no(event):
            ax_yes.remove()
            ax_no.remove()
            # Go back to last spot
            self.idx = self.total - 1
            self._draw()

        btn_yes.on_clicked(_yes)
        btn_no.on_clicked(_no)
        self.fig.canvas.draw_idle()

        # Keep button references alive so they aren't garbage-collected
        self._confirm_buttons = (btn_yes, btn_no)

    def _cycle_override(self):
        """Cycle class override: Auto → Monomer → ... → max_stoich-mer → Aggregate → Auto."""
        spot   = self.spots[self.idx]
        _cycle = [None] + [stoich_label(i) for i in range(1, self.max_stoich + 1)] + ["aggregate"]
        cur    = spot.get("class_override", None)
        if cur not in _cycle:
            cur = None
        spot["class_override"] = _cycle[(_cycle.index(cur) + 1) % len(_cycle)]
        self._draw()

    def _toggle_bad_trace(self):
        """Toggle bad trace flag — traces with intensity increases (non-photobleaching)."""
        spot = self.spots[self.idx]
        spot["bad_trace"] = not spot.get("bad_trace", False)
        self._draw()

    def _toggle_good_trace(self):
        """Toggle good trace flag — bookmark spot as a publication-quality example."""
        spot = self.spots[self.idx]
        spot["good_trace"] = not spot.get("good_trace", False)
        self._draw()

    def _toggle_complete(self):
        """Toggle the incomplete bleaching flag for the current spot."""
        spot = self.spots[self.idx]
        spot["incomplete"] = not spot["incomplete"]
        self._draw()

    def _reset(self):
        s = self.spots[self.idx]
        s["bps"] = s["auto_bps"].copy()
        self._draw()

    def _quit(self):
        plt.close(self.fig)

    def run(self):
        plt.show()


def load_all_spots(coloc_csv, tif_stem):
    coloc_dir = coloc_csv.parent
    coloc_df  = pd.read_csv(coloc_csv)

    def _try(path):
        return pd.read_csv(path) if path.exists() else pd.DataFrame()

    u1_df = _try(coloc_dir / f"{tif_stem}_unmatched_ch1.csv")
    u2_df = _try(coloc_dir / f"{tif_stem}_unmatched_ch2.csv")
    return coloc_df, u1_df, u2_df


def build_mask_coords(coloc_df, u1_df, u2_df):
    def _xy(df, xc, yc):
        if df is not None and len(df) > 0 and xc in df.columns:
            return df[xc].values.astype(float), df[yc].values.astype(float)
        return np.array([]), np.array([])
    xs1c, ys1c = _xy(coloc_df, "x_ch1", "y_ch1")
    xs2c, ys2c = _xy(coloc_df, "x_ch2", "y_ch2")
    xs1u, ys1u = _xy(u1_df,   "x",     "y")
    xs2u, ys2u = _xy(u2_df,   "x",     "y")
    return (np.concatenate([xs1c, xs1u]), np.concatenate([ys1c, ys1u]),
            np.concatenate([xs2c, xs2u]), np.concatenate([ys2c, ys2u]))


# ── Bleaching step correction predictor ──────────────────────────────────────

_BLEACHING_PREDICTOR = None   # loaded once on first use
_SHARED_MODE         = False  # set to True by main() when --shared is passed

def _load_bleaching_predictor():
    """
    Load the bleaching predictor bundle.

    Search order depends on build mode (set via --shared CLI flag):
      shared build   : <scripts_dir>/models/ first, ~/smfret_params/models/ fallback
      internal build : ~/smfret_params/models/ first, <scripts_dir>/models/ fallback

    Accepts both bleaching_predictor.pkl.gz (compressed) and
    bleaching_predictor.pkl (uncompressed), preferring .pkl.gz if both exist.
    """
    global _BLEACHING_PREDICTOR
    if _BLEACHING_PREDICTOR is not None:
        return _BLEACHING_PREDICTOR
    import gzip
    import pickle

    scripts_dir = Path(__file__).parent
    local_dir   = scripts_dir / "models"
    home_dir    = Path.home() / "smfret_params" / "models"

    # _SHARED_MODE is set to True by main() when --shared is passed
    dirs = [local_dir, home_dir] if _SHARED_MODE else [home_dir, local_dir]

    model_path = None
    for d in dirs:
        if (d / "bleaching_predictor.pkl.gz").exists():
            model_path = d / "bleaching_predictor.pkl.gz"
            break
        if (d / "bleaching_predictor.pkl").exists():
            model_path = d / "bleaching_predictor.pkl"
            break

    if model_path is None:
        return None
    try:
        opener = gzip.open if model_path.suffix == ".gz" else open
        with opener(model_path, "rb") as f:
            _BLEACHING_PREDICTOR = pickle.load(f)
        print(f"  [ML] Bleaching predictor loaded ({model_path.name})")
    except Exception as e:
        print(f"  [ML] Could not load bleaching predictor: {e}")
        _BLEACHING_PREDICTOR = None
    return _BLEACHING_PREDICTOR


_BLEACH_FEAT_COLS = [
    "trace_mean", "trace_std", "trace_snr",
    "trace_length", "initial_intensity", "final_intensity",
    "n_auto_steps", "min_step_size", "max_step_size",
    "mean_step_size", "step_size_to_noise_ratio", "early_step_detected",
    "anomalous", "partial_bleach_confidence", "pop_step_size",
]

# Probability threshold above which the UI flags a spot for attention
ML_FLAG_THRESHOLD = 0.55


def score_spots_ml(spots: list, channel: str = "ch1") -> None:
    """
    Score each spot dict in-place with ml_edit_prob (0–1).
    Uses the per-channel model if available, otherwise the 'all' model.
    channel should be 'ch1' or 'ch2'.
    Silently skips if no model is loaded.
    """
    bundle = _load_bleaching_predictor()
    if bundle is None:
        for s in spots:
            s["ml_edit_prob"] = None
        return

    predictors = bundle.get("predictors", {})
    model_info = predictors.get(channel) or predictors.get("all")
    if model_info is None:
        for s in spots:
            s["ml_edit_prob"] = None
        return

    model = model_info["model"]
    import numpy as _np

    X_rows = []
    for s in spots:
        auto_bps = sorted(int(b) for b in s.get("auto_bps", []))
        feat     = compute_trace_features(
            s["signal"], _np.array(auto_bps),
            anomalous                = s.get("anomalous", False),
            partial_bleach_confidence= s.get("partial_bleach_confidence", 0.0),
            pop_step_size            = s.get("pop_step_size") or 0.0,
        )
        X_rows.append([feat[c] for c in _BLEACH_FEAT_COLS])

    X = _np.array(X_rows, dtype=_np.float32)
    try:
        probs = model.predict_proba(X)[:, 1]
        for s, p in zip(spots, probs):
            s["ml_edit_prob"] = float(p)
    except Exception as e:
        print(f"  [ML] Scoring failed: {e}")
        for s in spots:
            s["ml_edit_prob"] = None

    n_flagged = sum(
        1 for s in spots
        if s["ml_edit_prob"] is not None and s["ml_edit_prob"] >= ML_FLAG_THRESHOLD
    )
    if n_flagged:
        print(f"  [ML] {n_flagged}/{len(spots)} spots flagged as likely needing correction "
              f"(prob ≥ {ML_FLAG_THRESHOLD})")


def run_pass(spots, title, blank, ch_label, max_stoichiometry: int = 2):
    if not spots:
        print(f"  No spots for: {title} — skipping.")
        return
    print(f"\n  [{title}]  {len(spots)} spots")
    TraceReviewUI(spots, title, blank, ch_label,
                  max_stoichiometry=max_stoichiometry).run()


def finalize(spots, max_stoichiometry: int = 2):
    rows = []
    for s in spots:
        n              = len(sorted(int(b) for b in s["bps"]))
        incomplete     = s["incomplete"]
        bad_trace      = s.get("bad_trace", False)
        good_trace     = s.get("good_trace", False)
        class_override = s.get("class_override")
        partial_bleach = s.get("partial_bleach", False)
        pb_reclassified= s.get("partial_bleach_reclassified", False)

        if bad_trace:
            cls = "bad_trace"
        elif class_override:
            cls = class_override
        elif pb_reclassified:
            # Partial bleach: classify by total inferred stoichiometry
            total = n + s.get("n_unbleached_inferred", 0)
            cls   = classify_steps(total, max_stoichiometry)
        elif incomplete:
            cls = "aggregate"
        else:
            cls = classify_steps(n, max_stoichiometry)
        bps_sorted      = sorted(int(b) for b in s["bps"])
        auto_bps_sorted = sorted(int(b) for b in s.get("auto_bps", []))
        # HMM-smoothed level per segment (pipe-delimited, parallel to breakpoints)
        _, hmm_levels = hmm_level_estimates(s["signal"], bps_sorted)
        hmm_levels_str = "|".join(f"{v:.2f}" for v in hmm_levels)
        row = {
            "x": s["x"], "y": s["y"], "steps": n, "class": cls,
            "incomplete_bleaching": incomplete,
            "bad_trace":            bad_trace,
            "good_trace":           good_trace,
            "has_upward_step":      s.get("has_upward_step", False),
            "class_override":       class_override is not None,
            "breakpoints":          "|".join(str(b) for b in bps_sorted),
            "auto_breakpoints":     "|".join(str(b) for b in auto_bps_sorted),
            "human_edited_breakpoints": bps_sorted != auto_bps_sorted,
            "hmm_segment_levels":   hmm_levels_str,
            # Population step size estimate (Stage 3)
            "pop_step_size":            s.get("pop_step_size"),
            "pop_step_size_std":        s.get("pop_step_size_std"),
            # Anomalous trace flag (Stage 4)
            "anomalous":                s.get("anomalous", False),
            "anomalous_reason":         s.get("anomalous_reason", ""),
            # Partial bleach (Stage 5)
            "partial_bleach":               s.get("partial_bleach", False),
            "n_unbleached_inferred":        s.get("n_unbleached_inferred", 0),
            "partial_bleach_confidence":    s.get("partial_bleach_confidence", 0.0),
            "partial_bleach_n_mer":         s.get("partial_bleach_n_mer", ""),
        }
        # ML training features — internal build only
        if not _SHARED_MODE:
            feat = compute_trace_features(s["signal"], np.array(auto_bps_sorted))
            row.update({
                "trace_mean":               feat["trace_mean"],
                "trace_std":                feat["trace_std"],
                "trace_snr":                feat["trace_snr"],
                "trace_length":             feat["trace_length"],
                "initial_intensity":        feat["initial_intensity"],
                "final_intensity":          feat["final_intensity"],
                "n_auto_steps":             feat["n_auto_steps"],
                "min_step_size":            feat["min_step_size"],
                "max_step_size":            feat["max_step_size"],
                "mean_step_size":           feat["mean_step_size"],
                "step_size_to_noise_ratio": feat["step_size_to_noise_ratio"],
                "early_step_detected":      feat["early_step_detected"],
            })
        rows.append(row)
    return pd.DataFrame(rows)



def generate_good_traces_output(
        stem: str,
        out_dir: Path,
        cc1: list, cc2: list,
        u1s: list, u2s: list,
        stack,
        frames_per_channel: int,
        blank_frames: int,
        ch1_ref_frame: int,
        ch2_ref_frame: int,
        average_half_window: int,
        psf_radius: float,
        is_single_channel: bool = False,
):
    """
    For every spot flagged as good_trace, write:
      - A per-spot CSV with bg-subtracted intensity vs frame
      - A single PDF with one page per spot showing:
          trace panel(s), full averaged image, zoomed crop — spot circled in both
    Colocalized spots (cc1/cc2 are paired by index) get both channels on one page.
    """
    from matplotlib.backends.backend_pdf import PdfPages
    from skimage.exposure import rescale_intensity as _rescale
    import matplotlib.patches as mpatches

    # ── Collect good spots ────────────────────────────────────────────────
    good_spots = []  # list of dicts

    # Colocalized pairs — flag if either channel marked good
    for i, (s1, s2) in enumerate(zip(cc1, cc2)):
        if s1.get("good_trace") or s2.get("good_trace"):
            good_spots.append({
                "kind":   "colocalized",
                "idx":    i,
                "s1":     s1,
                "s2":     s2,
                "label":  f"Coloc pair {i+1}",
            })

    for ch, spots_list, label_prefix in [
        (1, u1s, "Unmatched Ch1"),
        (2, u2s, "Unmatched Ch2"),
    ]:
        for i, s in enumerate(spots_list):
            if s.get("good_trace"):
                good_spots.append({
                    "kind":  f"unmatched_ch{ch}",
                    "idx":   i,
                    "s1":    s,
                    "s2":    None,
                    "label": f"{label_prefix} spot {i+1}",
                })

    if not good_spots:
        print("  No good traces flagged — skipping good trace output.")
        return

    n = len(good_spots)
    print(f"\n  Generating good trace output for {n} spot(s)...")

    # ── Load averaged images once ─────────────────────────────────────────
    # stack is already in memory; compute averaged frames
    fpc = frames_per_channel
    hw  = average_half_window

    def _avg_frames(stack, centre_0idx, hw):
        start = max(0, centre_0idx - hw)
        stop  = min(stack.shape[0] - 1, centre_0idx + hw)
        return stack[start:stop+1].mean(axis=0)

    img_ch1 = _avg_frames(stack, ch1_ref_frame - 1, hw)
    img_ch2 = _avg_frames(stack, ch2_ref_frame - 1, hw)

    def _stretch(img):
        p1, p99 = np.percentile(img, (1, 99))
        return _rescale(img, in_range=(p1, p99), out_range=(0.0, 1.0))

    disp_ch1 = _stretch(img_ch1)
    disp_ch2 = _stretch(img_ch2)

    crop_half = 30  # px half-size for zoomed crop

    def _crop(disp, cx, cy):
        h, w = disp.shape
        x0 = max(0, int(round(cx)) - crop_half)
        x1 = min(w, int(round(cx)) + crop_half)
        y0 = max(0, int(round(cy)) - crop_half)
        y1 = min(h, int(round(cy)) + crop_half)
        return disp[y0:y1, x0:x1], cx - x0, cy - y0

    # ── Draw one trace panel ──────────────────────────────────────────────
    def _draw_trace(ax, spot, ch_label, blank, color):
        trace = spot["trace"]
        sig   = spot["signal"]
        bps   = sorted(int(b) for b in spot["bps"])
        frames = np.arange(len(trace))
        blank_frames_n = blank

        # Blank region
        if blank_frames_n > 0:
            ax.axvspan(0, blank_frames_n - 0.5, color="#eeeeee", alpha=0.7, zorder=0)
            ax.axvline(blank_frames_n - 0.5, color="#aaaaaa",
                       linewidth=1.0, linestyle=":", zorder=1)

        # Raw trace
        ax.plot(frames, trace, color="#bbbbbb", linewidth=0.9, zorder=2)

        # Step levels (HMM-smoothed)
        level_per_frame, levels = hmm_level_estimates(sig, bps)
        bounds = [0] + bps + [len(sig)]
        # HMM continuous fit overlay
        ax.plot(np.arange(len(level_per_frame)) + blank_frames_n,
                level_per_frame,
                color=color, linewidth=1.0, alpha=0.35,
                linestyle="-", zorder=3)
        for i, level in enumerate(levels):
            ax.hlines(level,
                      bounds[i] + blank_frames_n,
                      bounds[i+1] + blank_frames_n - 1,
                      colors=color, linewidth=2.5, zorder=4)
        for bp in bps:
            ax.axvline(bp + blank_frames_n, color=color,
                       linewidth=1.3, linestyle="--", alpha=0.8, zorder=3)
        for i, bp in enumerate(bps):
            drop  = levels[i] - levels[i+1]
            y_mid = (levels[i] + levels[i+1]) / 2
            ax.text(bp + blank_frames_n + 2, y_mid, f"Δ{drop:.0f}",
                    color=color, fontsize=7, fontweight="bold",
                    va="center", ha="left", zorder=5,
                    bbox=dict(boxstyle="round,pad=0.1", fc="white", ec="none", alpha=0.7))

        n_steps = len(bps)
        bad  = spot.get("bad_trace", False)
        inc  = spot["incomplete"]
        ov   = spot.get("class_override")
        cls  = "bad_trace" if bad else (ov if ov else ("aggregate" if inc else classify_steps(n_steps, 2)))
        ax.set_title(f"{ch_label}  |  {n_steps} step(s)  →  {cls.upper()}",
                     fontsize=9, color=color, fontweight="bold")
        ax.set_xlabel("Frame", fontsize=8)
        ax.set_ylabel("Intensity (bg-subtracted)", fontsize=8)
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)

    # ── Draw image panels (full + crop) ───────────────────────────────────
    def _draw_image_panels(ax_full, ax_crop, disp, cx, cy, circle_color):
        circle_r = psf_radius * 2.5

        # Full image
        ax_full.imshow(disp, cmap="gray", origin="upper")
        ax_full.add_patch(mpatches.Circle(
            (cx, cy), radius=circle_r,
            edgecolor=circle_color, facecolor="none", linewidth=1.5, zorder=3))
        ax_full.set_title("Full image", fontsize=8)
        ax_full.axis("off")

        # Zoomed crop
        crop, lx, ly = _crop(disp, cx, cy)
        ax_crop.imshow(crop, cmap="gray", origin="upper")
        ax_crop.add_patch(mpatches.Circle(
            (lx, ly), radius=circle_r,
            edgecolor=circle_color, facecolor="none", linewidth=1.5, zorder=3))
        ax_crop.set_title("Zoomed", fontsize=8)
        ax_crop.axis("off")

    # ── PDF + per-spot CSVs ───────────────────────────────────────────────
    good_dir = out_dir / "good_traces"
    good_dir.mkdir(exist_ok=True)

    pdf_path = out_dir / f"{stem}_good_traces.pdf"
    with PdfPages(pdf_path) as pdf:
        for entry in good_spots:
            s1    = entry["s1"]
            s2    = entry["s2"]
            label = entry["label"]
            kind  = entry["kind"]
            idx   = entry["idx"]
            is_coloc = (s2 is not None)

            # ── CSV output ───────────────────────────────────────────────
            safe_label = label.replace(" ", "_").replace("/", "-")
            if is_coloc and not is_single_channel:
                n_frames = len(s1["trace"])
                csv_data = {
                    "frame":         np.arange(n_frames),
                    "intensity_ch1": s1["trace"],
                    "intensity_ch2": s2["trace"],
                }
                pd.DataFrame(csv_data).to_csv(
                    good_dir / f"{stem}_{safe_label}.csv", index=False)
            else:
                ch = 1 if (is_single_channel or kind == "unmatched_ch1") else 2
                n_frames = len(s1["trace"])
                csv_data = {
                    "frame":              np.arange(n_frames),
                    f"intensity_ch{ch}":  s1["trace"],
                }
                pd.DataFrame(csv_data).to_csv(
                    good_dir / f"{stem}_{safe_label}.csv", index=False)

            # ── PDF page layout ──────────────────────────────────────────
            # Single-channel: always 1 trace + 1 image row, regardless of is_coloc
            # Two-channel colocalized: 2 trace rows + 2 image rows
            # Two-channel unmatched:   1 trace row  + 1 image row
            if is_coloc and not is_single_channel:
                fig = plt.figure(figsize=(12, 10))
                # Row 0: Ch1 trace (full width)
                ax_t1    = fig.add_axes([0.08, 0.72, 0.88, 0.22])
                # Row 1: Ch2 trace (full width)
                ax_t2    = fig.add_axes([0.08, 0.45, 0.88, 0.22])
                # Row 2: Ch1 full image + Ch1 crop + Ch2 full image + Ch2 crop
                ax_i1f   = fig.add_axes([0.04, 0.05, 0.20, 0.32])
                ax_i1z   = fig.add_axes([0.27, 0.05, 0.18, 0.32])
                ax_i2f   = fig.add_axes([0.52, 0.05, 0.20, 0.32])
                ax_i2z   = fig.add_axes([0.75, 0.05, 0.18, 0.32])

                _draw_trace(ax_t1, s1, "Channel 1", blank_frames, cls_color(
                    "bad_trace" if s1.get("bad_trace") else
                    (s1.get("class_override") or
                     ("aggregate" if s1["incomplete"] else
                      classify_steps(len(s1["bps"]), 2)))))
                _draw_trace(ax_t2, s2, "Channel 2", blank_frames, cls_color(
                    "bad_trace" if s2.get("bad_trace") else
                    (s2.get("class_override") or
                     ("aggregate" if s2["incomplete"] else
                      classify_steps(len(s2["bps"]), 2)))))

                _draw_image_panels(ax_i1f, ax_i1z, disp_ch1,
                                   s1["x"], s1["y"], "#00cc44")
                _draw_image_panels(ax_i2f, ax_i2z, disp_ch2,
                                   s2["x"], s2["y"], "#3399ff")

                fig.text(0.04, 0.40, "Ch1 image:", fontsize=8,
                         color="#00cc44", fontweight="bold")
                fig.text(0.52, 0.40, "Ch2 image:", fontsize=8,
                         color="#3399ff", fontweight="bold")
            else:
                # Single-channel coloc spot, or any unmatched spot
                ch   = 1 if (is_single_channel or kind == "unmatched_ch1") else 2
                disp = disp_ch1 if ch == 1 else disp_ch2
                col  = "#00cc44" if ch == 1 else "#3399ff"
                fig = plt.figure(figsize=(12, 7))
                ax_t   = fig.add_axes([0.08, 0.55, 0.88, 0.35])
                ax_if  = fig.add_axes([0.08, 0.05, 0.35, 0.40])
                ax_iz  = fig.add_axes([0.55, 0.05, 0.35, 0.40])

                _draw_trace(ax_t, s1, f"Channel {ch}", blank_frames,
                            cls_color(
                                "bad_trace" if s1.get("bad_trace") else
                                (s1.get("class_override") or
                                 ("aggregate" if s1["incomplete"] else
                                  classify_steps(len(s1["bps"]), 2)))))
                _draw_image_panels(ax_if, ax_iz, disp, s1["x"], s1["y"], col)

            fig.suptitle(
                f"{stem}  —  {label}  |  x={s1['x']:.1f}, y={s1['y']:.1f}",
                fontsize=11, fontweight="bold")
            pdf.savefig(fig, bbox_inches="tight")
            plt.close(fig)

    print(f"  Good traces PDF  → {pdf_path}")
    print(f"  Good traces CSVs → {good_dir}/")


def write_bleach_file(out_path: Path,
                      all_spots: list,
                      classified: list):
    """
    Write a LabView-compatible bleaching step file.

    Format:
        <N> s       — all spots in this population
        x y
        ...
        0 n         — excluded (always 0)
        <A> 1       — monomers
        x y
        ...
        <B> 2       — dimers
        x y
        ...
        <C> u       — aggregate / bad_trace / incomplete
        x y
        ...

    all_spots : list of (x, y) tuples for every spot in population (for the s section)
    classified: list of dicts with keys x, y, class
                class is one of: monomer, dimer, aggregate, bad_trace
    """
    monomers   = [(s["x"], s["y"]) for s in classified if s["class"] == "monomer"]
    dimers     = [(s["x"], s["y"]) for s in classified if s["class"] == "dimer"]
    aggregates = [(s["x"], s["y"]) for s in classified
                  if s["class"] in ("aggregate", "bad_trace")]

    def _coords(pairs):
        lines = []
        for x, y in pairs:
            lines.append(f"{int(round(x))} {int(round(y))}\r\n")
        return lines

    with open(out_path, "w", newline="") as f:
        # s section — all spots
        f.write(f"{len(all_spots)} s\r\n")
        for x, y in all_spots:
            f.write(f"{int(round(x))} {int(round(y))}\r\n")
        # n section — always empty
        f.write("0 n\r\n")
        # 1 section — monomers
        f.write(f"{len(monomers)} 1\r\n")
        f.writelines(_coords(monomers))
        # 2 section — dimers
        f.write(f"{len(dimers)} 2\r\n")
        f.writelines(_coords(dimers))
        # u section — aggregate/bad
        f.write(f"{len(aggregates)} u\r\n")
        f.writelines(_coords(aggregates))

    print(f"  Bleach file → {out_path}")


def save_bleach_overlay_image(out_path: Path,
                              tif_path: Path,
                              spots: list,
                              frame_start: int,
                              frame_end: int,
                              stack,
                              psf_radius: float,
                              title: str,
                              max_stoichiometry: int = 2):
    """
    Save a PNG showing bleaching step classifications for a spot population.
    Each spot gets a circle coloured by class and a label showing step count (1, 2, u).

    spots: list of prepared spot dicts (with x, y, bps, incomplete, bad_trace,
           class_override keys) — the raw spot dicts from prep_df, not finalized.
    """
    import matplotlib.pyplot as plt
    import matplotlib.patches as mpatches
    from skimage.exposure import rescale_intensity

    # Build averaged reference frame from the stack
    mid = (frame_start + frame_end) // 2
    half = min(1, mid - frame_start, frame_end - mid - 1)
    frames = stack[max(frame_start, mid - half): min(frame_end, mid + half + 1)]
    img = frames.mean(axis=0)

    p1, p99 = np.percentile(img, (1, 99))
    img_display = rescale_intensity(img, in_range=(p1, p99), out_range=(0.0, 1.0))

    fig, ax = plt.subplots(1, 1, figsize=(8, 8))
    ax.imshow(img_display, cmap="gray", origin="upper")
    ax.axis("off")

    circle_r = psf_radius * 1.5

    for s in spots:
        n   = len(s["bps"])
        inc = s["incomplete"]
        bt  = s.get("bad_trace", False)
        ov  = s.get("class_override")

        if bt:
            cls = "bad_trace"
        elif ov:
            cls = ov
        elif inc:
            cls = "aggregate"
        else:
            cls = classify_steps(n, max_stoichiometry)

        color = cls_color(cls)
        label = "u" if cls in ("aggregate", "bad_trace") else str(n)

        circle = mpatches.Circle(
            (s["x"], s["y"]), radius=circle_r,
            edgecolor=color, facecolor="none",
            linewidth=1.2, zorder=3,
        )
        ax.add_patch(circle)
        ax.text(
            s["x"], s["y"] - circle_r - 2,
            label,
            color=color, fontsize=6, fontweight="bold",
            ha="center", va="bottom", zorder=4,
        )

    # Build per-class counts for legend
    class_counts: dict[str, int] = {}
    for s in spots:
        n   = len(s["bps"])
        inc = s["incomplete"]
        bt  = s.get("bad_trace", False)
        ov  = s.get("class_override")
        if bt:
            cls = "bad_trace"
        elif ov:
            cls = ov
        elif inc:
            cls = "aggregate"
        else:
            cls = classify_steps(n, max_stoichiometry)
        class_counts[cls] = class_counts.get(cls, 0) + 1

    import matplotlib.patches as mpatches
    legend_elements = []
    for cls in [stoich_label(i) for i in range(1, max_stoichiometry + 1)] + ["aggregate", "bad_trace"]:
        if cls in class_counts:
            legend_elements.append(
                mpatches.Patch(edgecolor=cls_color(cls), facecolor="none",
                               linewidth=1.5, label=f"{cls.capitalize()} (n={class_counts[cls]})"))
    ax.legend(handles=legend_elements, loc="lower right", fontsize=7, framealpha=0.8)
    ax.set_title(f"{title}  |  {len(spots)} spots", fontsize=10)

    fig.tight_layout()
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  Bleach image → {out_path}")



def build_mask_coords_three_channel(coloc_df):
    """Return (all_xs_ch1, all_ys_ch1, all_xs_ch2, all_ys_ch2, all_xs_ch3, all_ys_ch3)
    from a triple-colocalized DataFrame."""
    def _xy(col_x, col_y):
        if col_x in coloc_df.columns:
            return coloc_df[col_x].values.astype(float), coloc_df[col_y].values.astype(float)
        return np.array([]), np.array([])
    xs1, ys1 = _xy("x_ch1", "y_ch1")
    xs2, ys2 = _xy("x_ch2", "y_ch2")
    xs3, ys3 = _xy("x_ch3", "y_ch3")
    return xs1, ys1, xs2, ys2, xs3, ys3


def process_file_three_channel(tif_path, coloc_csv, args, out_dir):
    """
    Three-channel bleaching analysis.

    Reads triple-colocalized spot coordinates from a _colocalized.csv that has
    x_ch1/y_ch1, x_ch2/y_ch2, x_ch3/y_ch3 columns (output of
    colocalize_tif.py --three-channel).

    TIF layout: ch1 frames | ch2 frames | ch3 frames  (3 × N total).

    Review passes (in order):
      1. Triple-colocalized Ch1
      2. Triple-colocalized Ch2
      3. Triple-colocalized Ch3

    No unmatched spots are analyzed in three-channel mode.

    Output:
      {stem}_bleaching_triple_coloc.csv  — one row per triple with steps/class
                                           for each channel + N:M:P complex class
      {stem}_ch1coloc_bleach             — LabView bleach file for Ch1
      {stem}_ch2coloc_bleach             — LabView bleach file for Ch2
      {stem}_ch3coloc_bleach             — LabView bleach file for Ch3
    """
    print(f"\n  Processing (three-channel): {tif_path.name}")
    stem = tif_path.stem

    coloc_df = pd.read_csv(coloc_csv)
    if len(coloc_df) == 0:
        print("  [SKIP] No colocalized spots."); return

    # Verify this is a triple-coloc CSV
    if "x_ch3" not in coloc_df.columns:
        print("  [SKIP] CSV does not contain x_ch3 column — not a three-channel run.")
        print("         Use --mode single for two-channel data.")
        return

    print(f"  Triple-colocalized: {len(coloc_df)}")

    # --- Spot subsampling (--max-spots) ---
    _max_spots = getattr(args, 'max_spots', 0)
    if _max_spots > 0 and _max_spots < len(coloc_df):
        coloc_df = coloc_df.sample(n=_max_spots).reset_index(drop=True)
        print(f"  Subsampled to {len(coloc_df)} spots (requested {_max_spots})")

    print(f"  Loading stack...")
    stack = load_tif_stack(tif_path)

    n_frames = stack.shape[0]
    fpc      = n_frames // 3
    blank    = args.blank_frames
    print(f"  Stack: {n_frames} frames total — {fpc} frames per channel")

    xs1, ys1, xs2, ys2, xs3, ys3 = build_mask_coords_three_channel(coloc_df)

    import threading, queue

    def prep_df(df, xcol, ycol, fs, fe, axs, ays, label):
        n = len(df)
        print(f"  Extracting {label} ({n} spots)...")
        results = []
        t0 = time.perf_counter()
        for i, (_, row) in enumerate(df.iterrows()):
            results.append(prepare_spot(row, stack, fs, fe, xcol, ycol, axs, ays, args))
            if (i + 1) % 50 == 0 or i == n - 1:
                elapsed = time.perf_counter() - t0
                print(f"    {i+1}/{n}  ({elapsed:.1f}s, {elapsed/(i+1)*1000:.0f}ms/spot)")
        return results

    # Extract ch1 first (synchronously), then ch2+ch3 in background
    cc1 = prep_df(coloc_df, "x_ch1", "y_ch1", 0,       fpc,
                  xs1, ys1, "Triple-Coloc Ch1")

    bg_queue = queue.Queue()

    def _extract_remaining():
        cc2 = prep_df(coloc_df, "x_ch2", "y_ch2", fpc,   fpc * 2,
                      xs2, ys2, "Triple-Coloc Ch2")
        bg_queue.put(("cc2", cc2))
        cc3 = prep_df(coloc_df, "x_ch3", "y_ch3", fpc * 2, fpc * 3,
                      xs3, ys3, "Triple-Coloc Ch3")
        bg_queue.put(("cc3", cc3))

    bg_thread = threading.Thread(target=_extract_remaining, daemon=True)
    bg_thread.start()

    score_spots_ml(cc1, channel="ch1")
    _step_meta_ch1_3 = estimate_step_size_population(
        cc1, max_stoichiometry=args.max_stoichiometry_ch1)
    attach_step_size_estimate(cc1, _step_meta_ch1_3)
    flag_anomalous_traces(cc1)
    detect_partial_bleach(cc1, max_stoichiometry=args.max_stoichiometry_ch1)
    run_pass(cc1, f"{stem} — Triple-Coloc Ch1", blank, "Channel 1",
             max_stoichiometry=args.max_stoichiometry_ch1)

    _, cc2 = bg_queue.get(); print("  [ready] Triple-Coloc Ch2")
    score_spots_ml(cc2, channel="ch2")
    _step_meta_ch2_3 = estimate_step_size_population(
        cc2, max_stoichiometry=args.max_stoichiometry_ch2)
    attach_step_size_estimate(cc2, _step_meta_ch2_3)
    flag_anomalous_traces(cc2)
    detect_partial_bleach(cc2, max_stoichiometry=args.max_stoichiometry_ch2)
    run_pass(cc2, f"{stem} — Triple-Coloc Ch2", blank, "Channel 2",
             max_stoichiometry=args.max_stoichiometry_ch2)

    _, cc3 = bg_queue.get(); print("  [ready] Triple-Coloc Ch3")
    score_spots_ml(cc3, channel="ch2")   # use ch2 model for ch3 (no ch3-specific model yet)
    _step_meta_ch3_3 = estimate_step_size_population(
        cc3, max_stoichiometry=args.max_stoichiometry_ch3)
    attach_step_size_estimate(cc3, _step_meta_ch3_3)
    flag_anomalous_traces(cc3)
    detect_partial_bleach(cc3, max_stoichiometry=args.max_stoichiometry_ch3)
    run_pass(cc3, f"{stem} — Triple-Coloc Ch3", blank, "Channel 3",
             max_stoichiometry=args.max_stoichiometry_ch3)

    out_dir.mkdir(parents=True, exist_ok=True)

    # Build output DataFrame
    rows = []
    for s1, s2, s3 in zip(cc1, cc2, cc3):
        def _finalize_spot(s, max_stoich):
            n   = len(sorted(int(b) for b in s["bps"]))
            inc = s["incomplete"]
            bt  = s.get("bad_trace", False)
            ov  = s.get("class_override")
            cls = "bad_trace" if bt else (ov if ov else ("aggregate" if inc else classify_steps(n, max_stoich)))
            bps      = sorted(int(b) for b in s["bps"])
            auto_bps = sorted(int(b) for b in s.get("auto_bps", []))
            return n, cls, inc, bt, ov is not None, bps, auto_bps, s.get("good_trace", False)

        n1, c1, inc1, bt1, ov1, bps1, abps1, gt1 = _finalize_spot(s1, args.max_stoichiometry_ch1)
        n2, c2, inc2, bt2, ov2, bps2, abps2, gt2 = _finalize_spot(s2, args.max_stoichiometry_ch2)
        n3, c3, inc3, bt3, ov3, bps3, abps3, gt3 = _finalize_spot(s3, args.max_stoichiometry_ch3)

        rows.append({
            "x_ch1": s1["x"], "y_ch1": s1["y"],
            "x_ch2": s2["x"], "y_ch2": s2["y"],
            "x_ch3": s3["x"], "y_ch3": s3["y"],
            "steps_ch1": n1, "steps_ch2": n2, "steps_ch3": n3,
            "class_ch1": c1, "class_ch2": c2, "class_ch3": c3,
            "complex_class": complex_class(c1, c2, n1, n2, c3, n3),
            "incomplete_bleaching_ch1": inc1,
            "incomplete_bleaching_ch2": inc2,
            "incomplete_bleaching_ch3": inc3,
            "bad_trace_ch1": bt1, "bad_trace_ch2": bt2, "bad_trace_ch3": bt3,
            "good_trace_ch1": gt1, "good_trace_ch2": gt2, "good_trace_ch3": gt3,
            "has_upward_step_ch1": s1.get("has_upward_step", False),
            "has_upward_step_ch2": s2.get("has_upward_step", False),
            "has_upward_step_ch3": s3.get("has_upward_step", False),
            "class_override_ch1": ov1, "class_override_ch2": ov2, "class_override_ch3": ov3,
            "breakpoints_ch1":     "|".join(str(b) for b in bps1),
            "breakpoints_ch2":     "|".join(str(b) for b in bps2),
            "breakpoints_ch3":     "|".join(str(b) for b in bps3),
            "auto_breakpoints_ch1": "|".join(str(b) for b in abps1),
            "auto_breakpoints_ch2": "|".join(str(b) for b in abps2),
            "auto_breakpoints_ch3": "|".join(str(b) for b in abps3),
            "human_edited_breakpoints_ch1": bps1 != abps1,
            "human_edited_breakpoints_ch2": bps2 != abps2,
            "human_edited_breakpoints_ch3": bps3 != abps3,
            "hmm_segment_levels_ch1": "|".join(f"{v:.2f}" for v in hmm_level_estimates(s1["signal"], bps1)[1]),
            "hmm_segment_levels_ch2": "|".join(f"{v:.2f}" for v in hmm_level_estimates(s2["signal"], bps2)[1]),
            "hmm_segment_levels_ch3": "|".join(f"{v:.2f}" for v in hmm_level_estimates(s3["signal"], bps3)[1]),
        })

    triple_out = pd.DataFrame(rows)
    p = out_dir / f"{stem}_bleaching_triple_coloc.csv"
    triple_out.to_csv(p, index=False)
    print(f"\n  Saved → {p}")

    # LabView bleach files for each channel
    for ch_idx, cc, max_stoich in [
        (1, cc1, args.max_stoichiometry_ch1),
        (2, cc2, args.max_stoichiometry_ch2),
        (3, cc3, args.max_stoichiometry_ch3),
    ]:
        write_bleach_file(
            out_path  = out_dir / f"{stem}_ch{ch_idx}coloc_bleach",
            all_spots = [(s["x"], s["y"]) for s in cc],
            classified= [{"x": s["x"], "y": s["y"],
                          "class": ("bad_trace" if s.get("bad_trace") else
                                    s.get("class_override") or
                                    ("aggregate" if s["incomplete"] else
                                     classify_steps(len(s["bps"]), max_stoich)))}
                         for s in cc],
        )

    # Bleach overlay images
    for ch_idx, cc, fs, fe, max_stoich in [
        (1, cc1, 0,       fpc,     args.max_stoichiometry_ch1),
        (2, cc2, fpc,     fpc * 2, args.max_stoichiometry_ch2),
        (3, cc3, fpc * 2, fpc * 3, args.max_stoichiometry_ch3),
    ]:
        save_bleach_overlay_image(
            out_path=out_dir / f"{stem}_bleach_ch{ch_idx}coloc.png",
            tif_path=tif_path, spots=cc,
            frame_start=fs, frame_end=fe,
            stack=stack, psf_radius=args.psf_radius,
            title=f"{stem} — Triple-Coloc Ch{ch_idx}",
            max_stoichiometry=max_stoich,
        )

    # Terminal summary
    print(f"\n  === Bleaching Summary (three-channel) ===")
    print(f"  Coloc Ch1: {dict(triple_out['class_ch1'].value_counts())}")
    print(f"  Coloc Ch2: {dict(triple_out['class_ch2'].value_counts())}")
    print(f"  Coloc Ch3: {dict(triple_out['class_ch3'].value_counts())}")
    print(f"  Complex:   {dict(triple_out['complex_class'].value_counts())}")

    # Append to summary .txt
    append_bleaching_summary_three_channel(
        out_path  = out_dir / f"{stem}_summary.txt",
        stem      = stem,
        triple_out = triple_out,
    )

    # Good traces PDF + CSVs
    generate_good_traces_output_three_channel(
        stem               = stem,
        out_dir            = out_dir,
        cc1                = cc1,
        cc2                = cc2,
        cc3                = cc3,
        stack              = stack,
        frames_per_channel = fpc,
        blank_frames       = blank,
        ch1_ref_frame      = args.ch1_ref_frame,
        ch2_ref_frame      = (args.ch2_ref_frame if args.ch2_ref_frame is not None
                              else fpc + 11),
        average_half_window= args.average_half_window,
        psf_radius         = args.psf_radius,
    )


def generate_good_traces_output_three_channel(
        stem: str,
        out_dir: "Path",
        cc1: list, cc2: list, cc3: list,
        stack,
        frames_per_channel: int,
        blank_frames: int,
        ch1_ref_frame: int,
        ch2_ref_frame: int,
        average_half_window: int,
        psf_radius: float,
):
    """
    For every triple-colocalized spot where any channel is flagged good_trace,
    write a CSV with all three channel intensities and a PDF page with
    three trace panels + three zoomed image panels.
    """
    from matplotlib.backends.backend_pdf import PdfPages
    from skimage.exposure import rescale_intensity as _rescale
    import matplotlib.patches as mpatches

    good_spots = []
    for i, (s1, s2, s3) in enumerate(zip(cc1, cc2, cc3)):
        if s1.get("good_trace") or s2.get("good_trace") or s3.get("good_trace"):
            good_spots.append({
                "idx": i, "s1": s1, "s2": s2, "s3": s3,
                "label": f"Triple-coloc pair {i+1}",
            })

    if not good_spots:
        print("  No good traces flagged — skipping good trace output.")
        return

    print(f"\n  Generating good trace output for {len(good_spots)} spot(s)...")

    fpc = frames_per_channel
    hw  = average_half_window

    def _avg_frames(stack, centre_0idx, hw):
        start = max(0, centre_0idx - hw)
        stop  = min(stack.shape[0] - 1, centre_0idx + hw)
        return stack[start:stop+1].mean(axis=0)

    img_ch1 = _avg_frames(stack, ch1_ref_frame - 1, hw)
    img_ch2 = _avg_frames(stack, ch2_ref_frame - 1, hw)
    ch3_ref_0idx = fpc * 2 + ch1_ref_frame - 1   # ch3 uses same offset as ch1
    img_ch3 = _avg_frames(stack, ch3_ref_0idx, hw)

    def _stretch(img):
        p1, p99 = np.percentile(img, (1, 99))
        return _rescale(img, in_range=(p1, p99), out_range=(0.0, 1.0))

    disp = [_stretch(img_ch1), _stretch(img_ch2), _stretch(img_ch3)]
    colors = ["#00cc44", "#3399ff", "#ff6600"]
    crop_half = 30

    def _crop(d, cx, cy):
        h, w = d.shape
        x0 = max(0, int(round(cx)) - crop_half)
        x1 = min(w, int(round(cx)) + crop_half)
        y0 = max(0, int(round(cy)) - crop_half)
        y1 = min(h, int(round(cy)) + crop_half)
        return d[y0:y1, x0:x1], cx - x0, cy - y0

    good_dir = out_dir / "good_traces"
    good_dir.mkdir(exist_ok=True)

    pdf_path = out_dir / f"{stem}_good_traces_triple.pdf"
    with PdfPages(pdf_path) as pdf:
        for entry in good_spots:
            s1, s2, s3 = entry["s1"], entry["s2"], entry["s3"]
            label = entry["label"]

            # CSV
            safe_label = label.replace(" ", "_").replace("/", "-")
            n_frames = len(s1["trace"])
            pd.DataFrame({
                "frame":         np.arange(n_frames),
                "intensity_ch1": s1["trace"],
                "intensity_ch2": s2["trace"],
                "intensity_ch3": s3["trace"],
            }).to_csv(good_dir / f"{stem}_{safe_label}.csv", index=False)

            # PDF page: 3 trace rows + 1 row of 3 zoomed crops
            fig = plt.figure(figsize=(14, 12))
            trace_h = 0.18
            trace_bottoms = [0.78, 0.57, 0.36]
            crop_left = [0.05, 0.37, 0.69]

            for ch_i, (s, bottom, col) in enumerate(
                    zip([s1, s2, s3], trace_bottoms, colors)):
                ax_t = fig.add_axes([0.08, bottom, 0.88, trace_h])
                # Draw trace (reuse helper from generate_good_traces_output)
                trace = s["trace"]
                sig   = s["signal"]
                bps   = sorted(int(b) for b in s["bps"])
                frames_arr = np.arange(len(trace))
                if blank_frames > 0:
                    ax_t.axvspan(0, blank_frames - 0.5, color="#eeeeee", alpha=0.7, zorder=0)
                    ax_t.axvline(blank_frames - 0.5, color="#aaaaaa",
                                 linewidth=1.0, linestyle=":", zorder=1)
                ax_t.plot(frames_arr, trace, color="#bbbbbb", linewidth=0.9, zorder=2)
                level_per_frame, levels = hmm_level_estimates(sig, bps)
                bounds = [0] + bps + [len(sig)]
                ax_t.plot(np.arange(len(level_per_frame)) + blank_frames,
                          level_per_frame,
                          color=col, linewidth=1.0, alpha=0.35,
                          linestyle="-", zorder=3)
                for i, level in enumerate(levels):
                    ax_t.hlines(level, bounds[i] + blank_frames,
                                bounds[i+1] + blank_frames - 1,
                                colors=col, linewidth=2.5, zorder=4)
                for bp in bps:
                    ax_t.axvline(bp + blank_frames, color=col,
                                 linewidth=1.3, linestyle="--", alpha=0.8, zorder=3)
                n_steps = len(bps)
                inc = s["incomplete"]
                bt  = s.get("bad_trace", False)
                ov  = s.get("class_override")
                cls = "bad_trace" if bt else (ov if ov else ("aggregate" if inc else classify_steps(n_steps, 2)))
                ax_t.set_title(f"Ch{ch_i+1}  |  {n_steps} step(s)  →  {cls.upper()}",
                               fontsize=9, color=col, fontweight="bold")
                ax_t.set_xlabel("Frame", fontsize=8)
                ax_t.set_ylabel("Intensity", fontsize=8)
                ax_t.spines["top"].set_visible(False)
                ax_t.spines["right"].set_visible(False)

            # Zoomed crops
            for ch_i, (s, left, col) in enumerate(zip([s1, s2, s3], crop_left, colors)):
                ax_z = fig.add_axes([left, 0.05, 0.26, 0.25])
                crop, lx, ly = _crop(disp[ch_i], s["x"], s["y"])
                ax_z.imshow(crop, cmap="gray", origin="upper")
                import matplotlib.patches as mp
                ax_z.add_patch(mp.Circle((lx, ly), radius=psf_radius * 2.5,
                                         edgecolor=col, facecolor="none",
                                         linewidth=1.5, zorder=3))
                ax_z.set_title(f"Ch{ch_i+1} zoom", fontsize=8)
                ax_z.axis("off")

            fig.suptitle(
                f"{stem}  —  {label}  "
                f"|  x1={s1['x']:.1f},y1={s1['y']:.1f}  "
                f"x2={s2['x']:.1f},y2={s2['y']:.1f}  "
                f"x3={s3['x']:.1f},y3={s3['y']:.1f}",
                fontsize=10, fontweight="bold")
            pdf.savefig(fig, bbox_inches="tight")
            plt.close(fig)

    print(f"  Good traces PDF  → {pdf_path}")
    print(f"  Good traces CSVs → {good_dir}/")


def append_bleaching_summary_three_channel(out_path: "Path", stem: str,
                                           triple_out: "pd.DataFrame"):
    """
    Append three-channel bleaching statistics to the summary .txt file.
    """
    def _pct(n, total):
        return f"{100.0*n/total:.1f}%" if total > 0 else "0.0%"

    def _class_lines(df, col, label, upward_col=None):
        total = len(df)
        n_upward = int(df[upward_col].sum()) if upward_col and upward_col in df.columns else 0
        countable = df[~df[upward_col]].copy() if (upward_col and upward_col in df.columns) else df
        n_countable = len(countable)
        lines = [f"{label} (n={total}):"]
        for cls in ["monomer", "dimer", "aggregate"]:
            n = int((countable[col] == cls).sum())
            lines.append(f"  {cls.capitalize():<10}: {n:>4}  ({_pct(n, n_countable)})")
        if n_upward:
            lines.append(f"  Upward step : {n_upward:>4}  ({_pct(n_upward, total)})  [excluded from stoichiometry]")
        return lines

    section = ["--- Bleaching Analysis (three-channel) ---", ""]

    if len(triple_out) > 0:
        for ch in [1, 2, 3]:
            section += _class_lines(triple_out, f"class_ch{ch}",
                                    f"Triple-Colocalized Ch{ch}",
                                    upward_col=f"has_upward_step_ch{ch}")
            section.append("")
        has_upward = (
            triple_out.get("has_upward_step_ch1", pd.Series(False, index=triple_out.index)) |
            triple_out.get("has_upward_step_ch2", pd.Series(False, index=triple_out.index)) |
            triple_out.get("has_upward_step_ch3", pd.Series(False, index=triple_out.index))
        )
        cc_countable = triple_out[~has_upward]
        n_cc_upward = int(has_upward.sum())
        section.append(f"Complex classes (n={len(cc_countable)} countable, {n_cc_upward} excluded):")
        for cls in sorted(cc_countable["complex_class"].unique()):
            n = int((cc_countable["complex_class"] == cls).sum())
            section.append(f"  {cls:<14}: {n:>4}  ({_pct(n, len(cc_countable))})")
        section.append("")

    mode = "a" if out_path.exists() else "w"
    if mode == "w":
        import datetime
        header = [f"=== Analysis Summary: {stem} ===",
                  f"Date: {datetime.date.today().isoformat()}", "", ""]
        section = header + section

    with open(out_path, mode) as f:
        f.write("\n".join(section))
    print(f"  Summary  → {out_path}")



def process_file(tif_path, coloc_csv, args, out_dir):
    print(f"\n  Processing: {tif_path.name}")
    stem = tif_path.stem

    coloc_df, u1_df, u2_df = load_all_spots(coloc_csv, stem)
    if len(coloc_df) == 0:
        print("  [SKIP] No colocalized spots."); return

    print(f"  Colocalized: {len(coloc_df)}  "
          f"Unmatched ch1: {len(u1_df)}  ch2: {len(u2_df)}")

    # --- Spot subsampling (--max-spots) ---
    _max_spots = getattr(args, 'max_spots', 0)
    _total_spots = len(coloc_df) + len(u1_df) + len(u2_df)
    if _max_spots > 0 and _max_spots < _total_spots:
        _frac = _max_spots / _total_spots
        _n_coloc = max(1, round(len(coloc_df) * _frac)) if len(coloc_df) > 0 else 0
        _n_u1    = max(1, round(len(u1_df)    * _frac)) if len(u1_df)    > 0 else 0
        _n_u2    = max(1, round(len(u2_df)    * _frac)) if len(u2_df)    > 0 else 0
        coloc_df = coloc_df.sample(n=min(_n_coloc, len(coloc_df))).reset_index(drop=True)
        u1_df    = u1_df.sample(n=min(_n_u1, len(u1_df))).reset_index(drop=True) if len(u1_df) > 0 else u1_df
        u2_df    = u2_df.sample(n=min(_n_u2, len(u2_df))).reset_index(drop=True) if len(u2_df) > 0 else u2_df
        print(f"  Subsampled to {len(coloc_df)} coloc + {len(u1_df)} u1 + {len(u2_df)} u2 "
              f"(requested {_max_spots} of {_total_spots} total)")

    print(f"  Loading stack...")
    stack = load_tif_stack(tif_path)

    all_xs_ch1, all_ys_ch1, all_xs_ch2, all_ys_ch2 = \
        build_mask_coords(coloc_df, u1_df, u2_df)

    # Derive frames_per_channel from actual stack length
    fpc   = stack.shape[0] // 2
    blank = args.blank_frames
    print(f"  Stack: {stack.shape[0]} frames total — {fpc} frames per channel")

    import threading, queue

    def prep_df(df, xcol, ycol, fs, fe, axs, ays, label):
        n = len(df)
        print(f"  Extracting {label} ({n} spots)...")
        results = []
        t0 = time.perf_counter()
        for i, (_, row) in enumerate(df.iterrows()):
            results.append(prepare_spot(row, stack, fs, fe, xcol, ycol, axs, ays, args))
            if (i + 1) % 50 == 0 or i == n - 1:
                elapsed = time.perf_counter() - t0
                print(f"    {i+1}/{n}  ({elapsed:.1f}s, {elapsed/(i+1)*1000:.0f}ms/spot)")
        return results

    # Detect single-channel mode: ch1 and ch2 positions are identical
    is_single_channel = (
        len(coloc_df) > 0 and
        "x_ch1" in coloc_df.columns and "x_ch2" in coloc_df.columns and
        np.allclose(coloc_df["x_ch1"].values, coloc_df["x_ch2"].values) and
        np.allclose(coloc_df["y_ch1"].values, coloc_df["y_ch2"].values)
    )
    if is_single_channel:
        print("  Single-channel mode detected — running one pass only.")

    # Parse skip set before any extraction — skipped populations are not
    # extracted at all (no trace pulling from TIF), saving significant time.
    skip = {s.strip() for s in getattr(args, "skip_populations", "").split(",") if s.strip()}
    if skip:
        print(f"  Skipping populations (no extraction): {', '.join(sorted(skip))}")

    # --- Extract and review each population ---
    if is_single_channel:
        # Single channel: one pass only, no cc2/unmatched
        cc1 = prep_df(coloc_df, "x_ch1","y_ch1", 0, fpc,
                      all_xs_ch1, all_ys_ch1, "Colocalized Ch1")
        _step_meta_ch1 = estimate_step_size_population(
            cc1, max_stoichiometry=args.max_stoichiometry_ch1)
        attach_step_size_estimate(cc1, _step_meta_ch1)
        flag_anomalous_traces(cc1)
        detect_partial_bleach(cc1, max_stoichiometry=args.max_stoichiometry_ch1)
        score_spots_ml(cc1, channel="ch1")
        run_pass(cc1, f"{stem} — All Spots", blank, "Channel 1",
                 max_stoichiometry=args.max_stoichiometry_ch1)
        cc2 = cc1   # reuse cc1 for output writing (positions identical)
        u1s = []
        u2s = []
    else:
        # Two-channel: extract cc1 first (synchronously) so review can start
        # immediately; extract remaining passes in a background thread but only
        # for populations that are not skipped.
        if "coloc_ch1" not in skip:
            cc1 = prep_df(coloc_df, "x_ch1","y_ch1", 0, fpc,
                          all_xs_ch1, all_ys_ch1, "Colocalized Ch1")
        else:
            print("  [SKIP] Colocalized Ch1 — not extracted.")
            cc1 = []

        bg_queue = queue.Queue()

        def _extract_remaining():
            if "coloc_ch2" not in skip:
                cc2 = prep_df(coloc_df, "x_ch2","y_ch2", fpc, fpc*2,
                              all_xs_ch2, all_ys_ch2, "Colocalized Ch2")
            else:
                cc2 = []
            bg_queue.put(("cc2", cc2))
            if "unmatched_ch1" not in skip:
                u1s = prep_df(u1_df, "x","y", 0, fpc,
                              all_xs_ch1, all_ys_ch1, "Unmatched Ch1") if len(u1_df) else []
            else:
                u1s = []
            bg_queue.put(("u1s", u1s))
            if "unmatched_ch2" not in skip:
                u2s = prep_df(u2_df, "x","y", fpc, fpc*2,
                              all_xs_ch2, all_ys_ch2, "Unmatched Ch2") if len(u2_df) else []
            else:
                u2s = []
            bg_queue.put(("u2s", u2s))

        bg_thread = threading.Thread(target=_extract_remaining, daemon=True)
        bg_thread.start()

        # Ch1 step size estimate — from cc1 (all ch1 spots available now)
        _step_meta_ch1 = estimate_step_size_population(
            cc1, max_stoichiometry=args.max_stoichiometry_ch1) if cc1 else \
            {"step_size_estimate": None, "step_size_std": None,
             "step_size_iqr": None, "all_steps_raw": []}
        attach_step_size_estimate(cc1, _step_meta_ch1)
        flag_anomalous_traces(cc1)
        detect_partial_bleach(cc1, max_stoichiometry=args.max_stoichiometry_ch1)

        if cc1:
            score_spots_ml(cc1, channel="ch1")
            run_pass(cc1, f"{stem} — Colocalized Ch1", blank, "Channel 1",
                     max_stoichiometry=args.max_stoichiometry_ch1)

        _, cc2 = bg_queue.get()
        # Ch2 step size estimate — from cc2; reuse for u2s below
        _step_meta_ch2 = estimate_step_size_population(
            cc2, max_stoichiometry=args.max_stoichiometry_ch2) if cc2 else \
            {"step_size_estimate": None, "step_size_std": None,
             "step_size_iqr": None, "all_steps_raw": []}
        attach_step_size_estimate(cc2, _step_meta_ch2)
        flag_anomalous_traces(cc2)
        detect_partial_bleach(cc2, max_stoichiometry=args.max_stoichiometry_ch2)
        if cc2:
            print("  [ready] Colocalized Ch2")
            score_spots_ml(cc2, channel="ch2")
            run_pass(cc2, f"{stem} — Colocalized Ch2", blank, "Channel 2",
                     max_stoichiometry=args.max_stoichiometry_ch2)
        else:
            print("  [SKIP] Colocalized Ch2 — not extracted.")

        _, u1s = bg_queue.get()
        # Unmatched Ch1 shares the same fluorophore as cc1 — reuse estimate
        attach_step_size_estimate(u1s, _step_meta_ch1)
        flag_anomalous_traces(u1s)
        detect_partial_bleach(u1s, max_stoichiometry=args.max_stoichiometry_ch1)
        if u1s:
            print("  [ready] Unmatched Ch1")
            score_spots_ml(u1s, channel="ch1")
            run_pass(u1s, f"{stem} — Unmatched Ch1", blank, "Channel 1",
                     max_stoichiometry=args.max_stoichiometry_ch1)
        else:
            print("  [SKIP] Unmatched Ch1 — not extracted.")

        _, u2s = bg_queue.get()
        # Unmatched Ch2 shares the same fluorophore as cc2 — reuse estimate
        attach_step_size_estimate(u2s, _step_meta_ch2)
        flag_anomalous_traces(u2s)
        detect_partial_bleach(u2s, max_stoichiometry=args.max_stoichiometry_ch2)
        if u2s:
            print("  [ready] Unmatched Ch2")
            score_spots_ml(u2s, channel="ch2")
            run_pass(u2s, f"{stem} — Unmatched Ch2", blank, "Channel 2",
                     max_stoichiometry=args.max_stoichiometry_ch2)
        else:
            print("  [SKIP] Unmatched Ch2 — not extracted.")

    out_dir.mkdir(parents=True, exist_ok=True)
    out_dir.mkdir(parents=True, exist_ok=True)

    # Colocalized output — only if both coloc populations were extracted
    coloc_out = pd.DataFrame()
    if cc1 and cc2:
        rows = []
        for s1, s2 in zip(cc1, cc2):
            n1 = len(sorted(int(b) for b in s1["bps"]))
            n2 = len(sorted(int(b) for b in s2["bps"]))
            inc1, inc2 = s1["incomplete"], s2["incomplete"]
            bt1,  bt2  = s1.get("bad_trace", False),  s2.get("bad_trace", False)
            gt1,  gt2  = s1.get("good_trace", False), s2.get("good_trace", False)
            ov1, ov2   = s1.get("class_override"), s2.get("class_override")
            c1 = ("bad_trace" if bt1 else
                  ov1 if ov1 else
                  classify_steps(n1 + s1.get("n_unbleached_inferred", 0), args.max_stoichiometry_ch1)
                  if s1.get("partial_bleach_reclassified") else
                  "aggregate" if inc1 else
                  classify_steps(n1, args.max_stoichiometry_ch1))
            c2 = ("bad_trace" if bt2 else
                  ov2 if ov2 else
                  classify_steps(n2 + s2.get("n_unbleached_inferred", 0), args.max_stoichiometry_ch2)
                  if s2.get("partial_bleach_reclassified") else
                  "aggregate" if inc2 else
                  classify_steps(n2, args.max_stoichiometry_ch2))
            bps1      = sorted(int(b) for b in s1["bps"])
            bps2      = sorted(int(b) for b in s2["bps"])
            auto_bps1 = sorted(int(b) for b in s1.get("auto_bps", []))
            auto_bps2 = sorted(int(b) for b in s2.get("auto_bps", []))
            row = {
                "x_ch1": s1["x"], "y_ch1": s1["y"],
                "x_ch2": s2["x"], "y_ch2": s2["y"],
                "steps_ch1": n1, "steps_ch2": n2,
                "class_ch1": c1, "class_ch2": c2,
                "complex_class": complex_class(c1, c2, n1, n2),
                "incomplete_bleaching_ch1": inc1,
                "incomplete_bleaching_ch2": inc2,
                "bad_trace_ch1": bt1,
                "bad_trace_ch2": bt2,
                "good_trace_ch1": gt1,
                "good_trace_ch2": gt2,
                "has_upward_step_ch1": s1.get("has_upward_step", False),
                "has_upward_step_ch2": s2.get("has_upward_step", False),
                "class_override_ch1": ov1 is not None,
                "class_override_ch2": ov2 is not None,
                "breakpoints_ch1":              "|".join(str(b) for b in bps1),
                "breakpoints_ch2":              "|".join(str(b) for b in bps2),
                "auto_breakpoints_ch1":         "|".join(str(b) for b in auto_bps1),
                "auto_breakpoints_ch2":         "|".join(str(b) for b in auto_bps2),
                "human_edited_breakpoints_ch1": bps1 != auto_bps1,
                "human_edited_breakpoints_ch2": bps2 != auto_bps2,
                "hmm_segment_levels_ch1": "|".join(f"{v:.2f}" for v in hmm_level_estimates(s1["signal"], bps1)[1]),
                "hmm_segment_levels_ch2": "|".join(f"{v:.2f}" for v in hmm_level_estimates(s2["signal"], bps2)[1]),
                # Population step size estimates (Stage 3)
                "pop_step_size_ch1":     s1.get("pop_step_size"),
                "pop_step_size_ch2":     s2.get("pop_step_size"),
                # Anomalous trace flags (Stage 4)
                "anomalous_ch1":         s1.get("anomalous", False),
                "anomalous_ch2":         s2.get("anomalous", False),
                "anomalous_reason_ch1":  s1.get("anomalous_reason", ""),
                "anomalous_reason_ch2":  s2.get("anomalous_reason", ""),
                # Partial bleach (Stage 5)
                "partial_bleach_ch1":            s1.get("partial_bleach", False),
                "partial_bleach_ch2":            s2.get("partial_bleach", False),
                "n_unbleached_inferred_ch1":     s1.get("n_unbleached_inferred", 0),
                "n_unbleached_inferred_ch2":     s2.get("n_unbleached_inferred", 0),
                "partial_bleach_confidence_ch1": s1.get("partial_bleach_confidence", 0.0),
                "partial_bleach_confidence_ch2": s2.get("partial_bleach_confidence", 0.0),
            }
            # ML training features — internal build only
            if not _SHARED_MODE:
                feat1 = compute_trace_features(s1["signal"], np.array(auto_bps1))
                feat2 = compute_trace_features(s2["signal"], np.array(auto_bps2))
                row.update({
                    "trace_mean_ch1":               feat1["trace_mean"],
                    "trace_std_ch1":                feat1["trace_std"],
                    "trace_snr_ch1":                feat1["trace_snr"],
                    "trace_length_ch1":             feat1["trace_length"],
                    "initial_intensity_ch1":        feat1["initial_intensity"],
                    "final_intensity_ch1":          feat1["final_intensity"],
                    "n_auto_steps_ch1":             feat1["n_auto_steps"],
                    "min_step_size_ch1":            feat1["min_step_size"],
                    "max_step_size_ch1":            feat1["max_step_size"],
                    "mean_step_size_ch1":           feat1["mean_step_size"],
                    "step_size_to_noise_ratio_ch1": feat1["step_size_to_noise_ratio"],
                    "early_step_detected_ch1":      feat1["early_step_detected"],
                    "trace_mean_ch2":               feat2["trace_mean"],
                    "trace_std_ch2":                feat2["trace_std"],
                    "trace_snr_ch2":                feat2["trace_snr"],
                    "trace_length_ch2":             feat2["trace_length"],
                    "initial_intensity_ch2":        feat2["initial_intensity"],
                    "final_intensity_ch2":          feat2["final_intensity"],
                    "n_auto_steps_ch2":             feat2["n_auto_steps"],
                    "min_step_size_ch2":            feat2["min_step_size"],
                    "max_step_size_ch2":            feat2["max_step_size"],
                    "mean_step_size_ch2":           feat2["mean_step_size"],
                    "step_size_to_noise_ratio_ch2": feat2["step_size_to_noise_ratio"],
                    "early_step_detected_ch2":      feat2["early_step_detected"],
                })
            rows.append(row)
        coloc_out = pd.DataFrame(rows)
        p = out_dir / f"{stem}_bleaching_coloc.csv"
        coloc_out.to_csv(p, index=False)
        print(f"\n  Saved \u2192 {p}")
    else:
        print("  Colocalized bleaching output skipped (one or both coloc populations not extracted).")

    u1_final = finalize(u1s, max_stoichiometry=args.max_stoichiometry_ch1) if (u1s and not is_single_channel) else None
    u2_final = finalize(u2s, max_stoichiometry=args.max_stoichiometry_ch2) if (u2s and not is_single_channel) else None

    for df, label in [(u1_final, "unmatched_ch1"), (u2_final, "unmatched_ch2")]:
        if df is not None:
            p = out_dir / f"{stem}_bleaching_{label}.csv"
            df.to_csv(p, index=False)
            print(f"  Saved \u2192 {p}")

    # --- Write LabView-compatible bleaching files ---
    if not is_single_channel:
        if cc1:
            write_bleach_file(
                out_path   = out_dir / f"{stem}_ch1coloc_bleach",
                all_spots  = [(s["x"], s["y"]) for s in cc1],
                classified = [{"x": s["x"], "y": s["y"],
                               "class": ("bad_trace" if s.get("bad_trace") else
                                         s.get("class_override") or
                                         ("aggregate" if s["incomplete"] else
                                          classify_steps(len(s["bps"]), args.max_stoichiometry_ch1)))}
                              for s in cc1],
            )
        if cc2:
            write_bleach_file(
                out_path   = out_dir / f"{stem}_ch2coloc_bleach",
                all_spots  = [(s["x"], s["y"]) for s in cc2],
                classified = [{"x": s["x"], "y": s["y"],
                               "class": ("bad_trace" if s.get("bad_trace") else
                                         s.get("class_override") or
                                         ("aggregate" if s["incomplete"] else
                                          classify_steps(len(s["bps"]), args.max_stoichiometry_ch2)))}
                              for s in cc2],
            )
        if u1s:
            write_bleach_file(
                out_path   = out_dir / f"{stem}_ch1unmatched_bleach",
                all_spots  = [(s["x"], s["y"]) for s in u1s],
                classified = [{"x": s["x"], "y": s["y"],
                               "class": ("bad_trace" if s.get("bad_trace") else
                                         s.get("class_override") or
                                         ("aggregate" if s["incomplete"] else
                                          classify_steps(len(s["bps"]), args.max_stoichiometry_ch1)))}
                              for s in u1s],
            )
        if u2s:
            write_bleach_file(
                out_path   = out_dir / f"{stem}_ch2unmatched_bleach",
                all_spots  = [(s["x"], s["y"]) for s in u2s],
                classified = [{"x": s["x"], "y": s["y"],
                               "class": ("bad_trace" if s.get("bad_trace") else
                                         s.get("class_override") or
                                         ("aggregate" if s["incomplete"] else
                                          classify_steps(len(s["bps"]), args.max_stoichiometry_ch2)))}
                              for s in u2s],
            )

        # --- Bleaching overlay images (one per population) ---
        if cc1:
            save_bleach_overlay_image(
                out_path=out_dir / f"{stem}_bleach_ch1coloc.png",
                tif_path=tif_path, spots=cc1,
                frame_start=0, frame_end=fpc,
                stack=stack, psf_radius=args.psf_radius,
                title=f"{stem} \u2014 Colocalized Ch1",
                max_stoichiometry=args.max_stoichiometry_ch1,
            )
        if cc2:
            save_bleach_overlay_image(
                out_path=out_dir / f"{stem}_bleach_ch2coloc.png",
                tif_path=tif_path, spots=cc2,
                frame_start=fpc, frame_end=fpc * 2,
                stack=stack, psf_radius=args.psf_radius,
                title=f"{stem} \u2014 Colocalized Ch2",
                max_stoichiometry=args.max_stoichiometry_ch2,
            )
        if u1s:
            save_bleach_overlay_image(
                out_path=out_dir / f"{stem}_bleach_ch1unmatched.png",
                tif_path=tif_path, spots=u1s,
                frame_start=0, frame_end=fpc,
                stack=stack, psf_radius=args.psf_radius,
                title=f"{stem} \u2014 Unmatched Ch1",
                max_stoichiometry=args.max_stoichiometry_ch1,
            )
        if u2s:
            save_bleach_overlay_image(
                out_path=out_dir / f"{stem}_bleach_ch2unmatched.png",
                tif_path=tif_path, spots=u2s,
                frame_start=fpc, frame_end=fpc * 2,
                stack=stack, psf_radius=args.psf_radius,
                title=f"{stem} \u2014 Unmatched Ch2",
                max_stoichiometry=args.max_stoichiometry_ch2,
            )

    # Print terminal summary
    print(f"\n  === Bleaching Summary ===")
    if is_single_channel:
        print(f"  All spots: {dict(coloc_out['class_ch1'].value_counts())}")
    else:
        if not coloc_out.empty:
            print(f"  Coloc Ch1: {dict(coloc_out['class_ch1'].value_counts())}")
            print(f"  Coloc Ch2: {dict(coloc_out['class_ch2'].value_counts())}")
            print(f"  Complex:   {dict(coloc_out['complex_class'].value_counts())}")
        if u1_final is not None: print(f"  Unmatched Ch1: {dict(u1_final['class'].value_counts())}")
        if u2_final is not None: print(f"  Unmatched Ch2: {dict(u2_final['class'].value_counts())}")
    # Append to summary .txt file
    append_bleaching_summary(
        out_path  = out_dir / f"{stem}_summary.txt",
        stem      = stem,
        coloc_out = coloc_out,
        u1_df     = u1_final,
        u2_df     = u2_final,
    )

    # Generate good trace PDF + CSVs
    generate_good_traces_output(
        stem               = stem,
        out_dir            = out_dir,
        cc1                = cc1,
        cc2                = cc2,
        u1s                = u1s,
        u2s                = u2s,
        stack              = stack,
        frames_per_channel = fpc,
        blank_frames       = blank,
        ch1_ref_frame      = args.ch1_ref_frame,
        ch2_ref_frame      = args.ch2_ref_frame if args.ch2_ref_frame is not None else fpc + 11,
        average_half_window= args.average_half_window,
        psf_radius         = args.psf_radius,
        is_single_channel  = is_single_channel,
    )


def append_bleaching_summary(out_path: Path, stem: str,
                             coloc_out: "pd.DataFrame",
                             u1_df: "pd.DataFrame | None",
                             u2_df: "pd.DataFrame | None"):
    """
    Append bleaching step-counting statistics to the summary .txt file
    created by colocalize_tif.py.  If the file doesn't exist yet (e.g.
    bleaching run standalone), writes a minimal header first.
    """
    def _pct(n, total):
        return f"{100.0*n/total:.1f}%" if total > 0 else "0.0%"

    def _class_lines(df: "pd.DataFrame", col: str, label: str,
                      upward_col: str = None) -> list:
        total = len(df)
        n_upward = int(df[upward_col].sum()) if upward_col and upward_col in df.columns else 0
        # Stoichiometry counts exclude upward-step traces
        countable = df[~df[upward_col]].copy() if (upward_col and upward_col in df.columns) else df
        n_countable = len(countable)
        lines = [f"{label} (n={total}):"]
        for cls in ["monomer", "dimer", "aggregate"]:
            n = int((countable[col] == cls).sum())
            lines.append(f"  {cls.capitalize():<10}: {n:>4}  ({_pct(n, n_countable)})")
        if n_upward:
            lines.append(f"  Upward step : {n_upward:>4}  ({_pct(n_upward, total)})  [excluded from stoichiometry]")
        return lines

    section = ["--- Bleaching Analysis ---", ""]

    if len(coloc_out) > 0:
        section += _class_lines(coloc_out, "class_ch1", "Colocalized Ch1",
                                 upward_col="has_upward_step_ch1")
        section += [""]
        section += _class_lines(coloc_out, "class_ch2", "Colocalized Ch2",
                                 upward_col="has_upward_step_ch2")
        section += [""]

        # Complex classes (exclude pairs where either channel has upward step)
        has_upward = (
            coloc_out.get("has_upward_step_ch1", pd.Series(False, index=coloc_out.index)) |
            coloc_out.get("has_upward_step_ch2", pd.Series(False, index=coloc_out.index))
        )
        cc_countable = coloc_out[~has_upward]
        n_cc_upward = int(has_upward.sum())
        section.append(f"Complex classes (n={len(cc_countable)} countable, {n_cc_upward} excluded):")
        for cls in sorted(cc_countable["complex_class"].unique()):
            n = int((cc_countable["complex_class"] == cls).sum())
            section.append(f"  {cls:<12}: {n:>4}  ({_pct(n, len(cc_countable))})")
        section.append("")

    if u1_df is not None and len(u1_df) > 0:
        section += _class_lines(u1_df, "class", "Unmatched Ch1",
                                 upward_col="has_upward_step")
        section.append("")

    if u2_df is not None and len(u2_df) > 0:
        section += _class_lines(u2_df, "class", "Unmatched Ch2",
                                 upward_col="has_upward_step")
        section.append("")

    mode = "a" if out_path.exists() else "w"
    if mode == "w":
        # Standalone run — write minimal header
        import datetime
        header = [f"=== Analysis Summary: {stem} ===",
                  f"Date: {datetime.date.today().isoformat()}", "", ""]
        section = header + section

    with open(out_path, mode) as f:
        f.write("\n".join(section))
    print(f"  Summary  → {out_path}")


def process_file_single(tif_path, spots_csv, args, out_dir):
    """
    Single-channel bleaching analysis.
    Reads spot coordinates from detect_spots_single.py output,
    extracts traces, runs PELT, launches one review pass,
    and saves results + appends to summary .txt.
    """
    print(f"\n  Processing (single-channel): {tif_path.name}")
    stem = tif_path.stem

    spots_df = pd.read_csv(spots_csv)
    if len(spots_df) == 0:
        print("  [SKIP] No spots found."); return

    print(f"  Spots: {len(spots_df)}")

    # --- Spot subsampling (--max-spots) ---
    _max_spots = getattr(args, 'max_spots', 0)
    if _max_spots > 0 and _max_spots < len(spots_df):
        spots_df = spots_df.sample(n=_max_spots).reset_index(drop=True)
        print(f"  Subsampled to {len(spots_df)} spots (requested {_max_spots})")

    print(f"  Loading stack...")
    stack = load_tif_stack(tif_path)

    all_xs = spots_df["x"].values.astype(float)
    all_ys = spots_df["y"].values.astype(float)

    # Single channel occupies all frames
    fpc         = stack.shape[0]
    blank       = args.blank_frames
    frame_start = 0
    frame_end   = fpc
    print(f"  Stack: {fpc} frames total")

    n = len(spots_df)
    print(f"  Extracting {n} spot traces...")
    import time as _t
    spots_prepared = []
    t0 = _t.perf_counter()
    for i, (_, row) in enumerate(spots_df.iterrows()):
        spots_prepared.append(
            prepare_spot(row, stack, frame_start, frame_end,
                         "x", "y", all_xs, all_ys, args))
        if (i + 1) % 50 == 0 or i == n - 1:
            elapsed = _t.perf_counter() - t0
            print(f"    {i+1}/{n}  ({elapsed:.1f}s)")

    _step_meta_sc = estimate_step_size_population(
        spots_prepared, max_stoichiometry=args.max_stoichiometry_ch1)
    attach_step_size_estimate(spots_prepared, _step_meta_sc)
    flag_anomalous_traces(spots_prepared)
    detect_partial_bleach(spots_prepared, max_stoichiometry=args.max_stoichiometry_ch1)
    score_spots_ml(spots_prepared, channel="ch1")

    run_pass(spots_prepared,
             f"{stem} — Single Channel",
             blank, "Channel",
             max_stoichiometry=args.max_stoichiometry_ch1)

    out_dir.mkdir(parents=True, exist_ok=True)
    result_df = finalize(spots_prepared, max_stoichiometry=args.max_stoichiometry_ch1)
    p = out_dir / f"{stem}_bleaching_spots.csv"
    result_df.to_csv(p, index=False)
    print(f"\n  Saved \u2192 {p}")

    # Terminal summary
    print(f"\n  === Bleaching Summary (single channel) ===")
    print(f"  {dict(result_df['class'].value_counts())}")

    append_bleaching_summary_single(
        out_path   = out_dir / f"{stem}_summary.txt",
        stem       = stem,
        result_df  = result_df,
    )


def append_bleaching_summary_single(out_path: Path, stem: str,
                                    result_df: "pd.DataFrame"):
    """
    Append single-channel bleaching statistics to the summary .txt file.
    """
    def _pct(n, total):
        return f"{100.0*n/total:.1f}%" if total > 0 else "0.0%"

    total = len(result_df)
    section = ["--- Bleaching Analysis (single channel) ---", ""]
    section.append(f"All spots (n={total}):")
    for cls in ["monomer", "dimer", "aggregate"]:
        n = int((result_df["class"] == cls).sum())
        section.append(f"  {cls.capitalize():<10}: {n:>4}  ({_pct(n, total)})")
    section.append("")

    mode = "a" if out_path.exists() else "w"
    if mode == "w":
        import datetime
        header = [f"=== Analysis Summary: {stem} ===",
                  f"Date: {datetime.date.today().isoformat()}", "", ""]
        section = header + section

    with open(out_path, mode) as f:
        f.write("\n".join(section))
    print(f"  Summary  \u2192 {out_path}")


def build_parser():
    p = argparse.ArgumentParser(
        description="Photobleaching step analysis for single-molecule spots.",
        formatter_class=argparse.RawDescriptionHelpFormatter, epilog=__doc__)
    p.add_argument("--mode",
                   choices=["single", "batch", "single-channel",
                            "batch-single-channel"],
                   required=True)
    p.add_argument("--input",  help="[single/single-channel] Path to .tif file.")
    p.add_argument("--coloc",  help="[single] Path to *_colocalized.csv.")
    p.add_argument("--spots",  help="[single-channel] Path to *_all_spots.csv "
                               "from detect_spots_single.py.")
    p.add_argument("--folder", help="[batch] Folder with .tif + coloc/spots CSVs.")
    p.add_argument("--output", default=None)
    p.add_argument("--three-channel", action="store_true",
                   help="Three-channel mode: TIF has 3×N frames (ch1|ch2|ch3). "
                        "Coloc CSV must have x_ch3/y_ch3 columns.")
    p.add_argument("--skip-populations", default="",
                   help="Comma-separated list of populations to skip interactive review. "
                        "Valid values: coloc_ch1, coloc_ch2, unmatched_ch1, unmatched_ch2. "
                        "Skipped populations still produce output CSVs using auto-detected steps.")
    p.add_argument("--frames-per-channel", type=int,   default=FRAMES_PER_CHANNEL,
                   help="Ignored — derived from stack at runtime. "
                        "Kept for backwards compatibility.")
    p.add_argument("--blank-frames",       type=int,   default=BLANK_FRAMES)
    p.add_argument("--ch1-ref-frame",      type=int,   default=11,
                   help="1-indexed reference frame for Ch1 averaged image. Default: 11")
    p.add_argument("--ch2-ref-frame",      type=int,   default=None,
                   help="1-indexed reference frame for Ch2 averaged image. "
                        "Default: frames_per_channel + 11")
    p.add_argument("--average-half-window", type=int,  default=1,
                   help="Average ± this many frames around the reference. Default: 1")
    p.add_argument("--psf-radius",         type=float, default=PSF_RADIUS_PX)
    p.add_argument("--bg-annulus-inner",   type=float, default=BG_ANNULUS_INNER_PX)
    p.add_argument("--bg-annulus-outer",   type=float, default=BG_ANNULUS_OUTER_PX)
    p.add_argument("--bg-spot-mask",       type=float, default=BG_SPOT_MASK_PX)
    p.add_argument("--pelt-penalty",          type=float, default=PELT_PENALTY)
    p.add_argument("--pelt-min-size",         type=int,   default=PELT_MIN_SIZE)
    # Early burst detection (fast initial bleaching step)
    p.add_argument("--burst-frames",          type=int,   default=3,
                   help="Frames at signal start to measure initial burst level. Default: 3")
    p.add_argument("--burst-plateau-frames",  type=int,   default=10,
                   help="Frames after burst to measure first plateau. Default: 10")
    p.add_argument("--burst-ratio",           type=float, default=1.5,
                   help="Burst/plateau ratio above which an early step is inserted. Default: 1.5")
    p.add_argument("--burst-tail-frames",     type=int,   default=30,
                   help="Frames at end of signal used as baseline for early burst detection. Default: 30")
    p.add_argument("--burst-tail-ratio",      type=float, default=3.0,
                   help="Peak/tail-baseline ratio to trigger early burst detection. Default: 3.0")
    p.add_argument("--burst-max-scan",        type=int,   default=10,
                   help="Max frames from start to scan for an early burst peak. Default: 10")
    # Incomplete bleaching detection
    p.add_argument("--incomplete-tail-frames",    type=int,   default=10,
                   help="Frames at end of signal used to assess incomplete bleaching. Default: 10")
    p.add_argument("--incomplete-tail-threshold", type=float, default=0.20,
                   help="If tail mean > this fraction of initial signal, flag incomplete. Default: 0.20")
    p.add_argument("--incomplete-abs-floor",      type=float, default=10.0,
                   help="Tail mean below this absolute value is never flagged incomplete. Default: 10.0")
    # Stoichiometry range (Stage 1 & 2)
    p.add_argument("--max-stoichiometry-ch1",     type=int,   default=2,
                   help="Max expected stoichiometry for Ch1. Steps above this are "
                        "classified as aggregate. Default: 2 (reproduces original behaviour).")
    p.add_argument("--max-stoichiometry-ch2",     type=int,   default=2,
                   help="Max expected stoichiometry for Ch2. Default: 2.")
    p.add_argument("--max-stoichiometry-ch3",     type=int,   default=2,
                   help="Max expected stoichiometry for Ch3. Default: 2.")
    p.add_argument("--max-spots", type=int, default=0,
                   help="If > 0, randomly subsample this many spots before trace "
                        "extraction. Default: 0 (show dialog to ask).")
    p.add_argument("--shared", action="store_true", default=False,
                   help="Shared/collaborative mode: load models from the local models/ "
                        "folder next to the scripts instead of ~/smfret_params/models/, "
                        "and skip writing ML training features to output CSVs. "
                        "Set automatically by pipeline.py when BUILD_MODE != 'internal'.")
    return p


def main():
    args = build_parser().parse_args()

    # Flip model search order and disable training feature writes in shared mode
    global _SHARED_MODE
    if args.shared:
        _SHARED_MODE = True

    print("\n=== Photobleaching Step Analysis ===")
    print(f"  Mode         : {args.mode}")
    print(f"  PSF radius   : {args.psf_radius} px")
    print(f"  Blank frames : {args.blank_frames} per channel")
    print(f"  PELT penalty : {args.pelt_penalty}")
    print(f"  Max stoich   : ch1={args.max_stoichiometry_ch1}  "
          f"ch2={args.max_stoichiometry_ch2}  ch3={args.max_stoichiometry_ch3}")
    if args.max_spots > 0:
        print(f"  Max spots    : {args.max_spots} (random subsample)")

    if args.mode == "single":
        if not args.input or not args.coloc:
            build_parser().error("--mode single requires --input and --coloc")
        tif_path  = Path(args.input)
        coloc_csv = Path(args.coloc)
        out_dir   = Path(args.output) if args.output \
                    else tif_path.parent / "bleaching_output"
        if args.three_channel:
            process_file_three_channel(tif_path, coloc_csv, args, out_dir)
        else:
            process_file(tif_path, coloc_csv, args, out_dir)

    elif args.mode == "single-channel":
        if not args.input or not args.spots:
            build_parser().error(
                "--mode single-channel requires --input and --spots")
        tif_path  = Path(args.input)
        spots_csv = Path(args.spots)
        out_dir   = Path(args.output) if args.output \
                    else tif_path.parent / "bleaching_output"
        process_file_single(tif_path, spots_csv, args, out_dir)

    elif args.mode == "batch":
        if not args.folder:
            build_parser().error("--mode batch requires --folder")
        folder  = Path(args.folder)
        out_dir = Path(args.output) if args.output else folder / "bleaching_output"
        tifs    = sorted(folder.glob("*.tif")) + sorted(folder.glob("*.tiff"))
        if not tifs: sys.exit(f"No .tif/.tiff files in {folder}")
        coloc_dir = folder / "colocalization_output"
        for tif in tifs:
            csv = coloc_dir / f"{tif.stem}_colocalized.csv"
            if not csv.exists():
                print(f"  [SKIP] No coloc CSV for {tif.name}"); continue
            if args.three_channel:
                process_file_three_channel(tif, csv, args, out_dir)
            else:
                process_file(tif, csv, args, out_dir)

    else:  # batch-single-channel
        if not args.folder:
            build_parser().error("--mode batch-single-channel requires --folder")
        folder  = Path(args.folder)
        out_dir = Path(args.output) if args.output else folder / "bleaching_output"
        tifs    = sorted(folder.glob("*.tif")) + sorted(folder.glob("*.tiff"))
        if not tifs: sys.exit(f"No .tif/.tiff files in {folder}")
        det_dir = folder / "detection_output"
        for tif in tifs:
            csv = det_dir / f"{tif.stem}_all_spots.csv"
            if not csv.exists():
                print(f"  [SKIP] No spots CSV for {tif.name}"); continue
            process_file_single(tif, csv, args, out_dir)


if __name__ == "__main__":
    main()
