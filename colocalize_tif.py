"""
colocalize_tif.py
=================
Full single-molecule colocalization pipeline for two-channel concatenated .tif movies.

Workflow
--------
  1. Load a 700-frame .tif (frames 1-350 = ch1, frames 351-700 = ch2)
  2. Average a small window of frames around a reference frame in each channel
  3. Auto-detect spots using a Difference-of-Gaussian (DoG) blob detector
  4. Match spots between channels by nearest-neighbour distance threshold
  5. Write colocalized pairs + unmatched spots to CSVs for LabView inspection
  6. Write per-channel LabView coloc files ({stem}_ch1coloc, {stem}_ch2coloc)
     Format: <N> s / x y per colocalized spot / <M> n / x y per unmatched spot

Tuning guide (see PARAMETERS section below)
--------------------------------------------
  PSF_RADIUS_PX      Start around 2.0 and increase if spots are being split into
                     multiple detections, decrease if neighbouring spots are merged.

  DETECTION_THRESHOLD  Controls sensitivity. Lower = more spots detected (more false
                     positives). Higher = fewer spots (more misses). Tune per-channel
                     if one channel is much dimmer via CH1_THRESHOLD / CH2_THRESHOLD.

  COLOC_THRESHOLD_PX  Max centre-to-centre distance (px) to call a colocalization.
                     Start at 2-3× PSF_RADIUS_PX and tighten once you are happy
                     with detection.

Usage
-----
Single file:
    python colocalize_tif.py --mode single --input movie.tif

Batch (folder of .tif files):
    python colocalize_tif.py --mode batch --folder ./data

All tunable parameters can also be overridden from the command line —
run with --help to see all options.
"""

import argparse
import sys
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.spatial import cKDTree

import matplotlib
# Backend is set conditionally in main() — Agg for normal runs, interactive for tune mode
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.widgets import TextBox, Button as MplButton

# tifffile is the most reliable library for scientific .tif stacks
try:
    import tifffile
except ImportError:
    sys.exit("Please install tifffile:  pip install tifffile")

# scikit-image for blob detection
try:
    from skimage.feature import blob_dog
    from skimage.exposure import rescale_intensity
except ImportError:
    sys.exit("Please install scikit-image:  pip install scikit-image")

# PyTorch for CNN spot classifier (optional — pipeline works without it)
try:
    import torch
    _TORCH_AVAILABLE = True
except ImportError:
    _TORCH_AVAILABLE = False


# ===========================================================================
# PARAMETERS — edit these to tune detection for your data
# ===========================================================================

# --- Frame layout ---
# frames_per_channel is derived from the .tif at runtime (total_frames // 2)
# so that files of any length work without parameter changes.
# The value below is only used as a fallback if the file length is unavailable.
FRAMES_PER_CHANNEL   = 350        # fallback only — normally derived from file
CH1_REF_FRAME        = 11         # 1-indexed reference frame in ch1 (centre of average window)
CH2_REF_FRAME_OFFSET = 11         # offset from start of ch2 for ref frame (= ch2_start + offset)
AVERAGE_HALF_WINDOW  = 1          # average ± this many frames around the reference
                                   # (1 → 3-frame average: ref-1, ref, ref+1)

# --- Spot detection ---
PSF_RADIUS_PX        = 1.5        # estimated PSF radius in pixels — KEY tuning parameter
                                   # DoG min_sigma = PSF_RADIUS_PX * 0.6
                                   # DoG max_sigma = PSF_RADIUS_PX * 1.4

DETECTION_THRESHOLD  = 0.001      # global detection threshold (fraction of image max,
                                   # after normalisation). Override per-channel below.
CH1_THRESHOLD        = 0.001      # ch1-specific threshold (dimmer channel)
CH2_THRESHOLD        = 0.01       # ch2-specific threshold

EXCLUDE_BORDER_PX    = 5          # ignore spots within this many pixels of the image edge

# --- Colocalization ---
COLOC_THRESHOLD_PX   = 5.0        # max centre-to-centre distance (px) to call a match
                                   # set to 2× PSF_RADIUS_PX — spots are colocalized if
                                   # their PSFs overlap at all (edge-to-edge contact)

# --- Channel shift correction ---
# A two-pass approach is always used:
#   Pass 1: generous threshold to capture true pairs even if channels are shifted
#   Shift estimated from median offset of pass-1 pairs, applied to Ch2 positions
#   Pass 2: corrected Ch2 positions matched at normal COLOC_THRESHOLD_PX
SHIFT_FIRST_PASS_FACTOR = 3.0     # first-pass threshold = COLOC_THRESHOLD_PX × this factor
SHIFT_MIN_PAIRS         = 10      # minimum matched pairs in pass 1 to trust shift estimate
                                   # if fewer pairs, shift correction is skipped
SHIFT_MAX_PX            = 20.0    # sanity check: estimated shift larger than this is
                                   # ignored (likely a detection error, not a real shift)

# --- False positive filtering ---
# Applied in single/batch mode only (not tune mode).
# Spots are flagged (not removed) if they fail either filter.
# Thresholds are per-channel and relative to the estimated background level.

# Intensity filter: flag spots below MIN or above MAX multiple of background
# Background is estimated from the blank frames at the start of each channel
# (first BLANK_FRAMES frames, where the laser is off) — much more accurate than
# using the image median, which is skewed by real spots.
BLANK_FRAMES           = 10     # number of blank frames at the start of each channel

CH1_INTENSITY_MIN_MULT = 1.2    # ch1 spot must be > this × background (dim channel)
CH1_INTENSITY_MAX_MULT = 100.0  # ch1 spot must be < this × background (aggregates)
CH2_INTENSITY_MIN_MULT = 0.0    # ch2 lower bound disabled by default — ch2 is often very dim
CH2_INTENSITY_MAX_MULT = 100.0

# PSF shape filter: fit a 2D Gaussian and check width and fit quality
PSF_FIT_MIN_R2        = 0.3     # minimum R² of Gaussian fit (0–1; lower = worse fit)
PSF_FIT_WIDTH_TOL     = 0.8     # flag if fitted sigma differs from expected by > this fraction
                                  # e.g. 0.8 means flag if fitted sigma < 0.2× or > 1.8× PSF_RADIUS_PX

# ===========================================================================


def load_averaged_frame(tif_path: Path,
                         ref_frame: int,
                         half_window: int,
                         frames_per_channel: int) -> np.ndarray:
    """
    Load a stack from a .tif file and return a float32 average of frames
    [ref_frame - half_window  ..  ref_frame + half_window] (1-indexed).
    Clamps to valid range automatically.
    """
    with tifffile.TiffFile(tif_path) as tif:
        total_frames = len(tif.pages)
        expected = frames_per_channel * 2
        if total_frames != expected:
            warnings.warn(
                f"{tif_path.name}: expected {expected} frames "
                f"({frames_per_channel} per channel), found {total_frames}. "
                f"Proceeding anyway — check your frame layout."
            )

        # Convert to 0-indexed, clamp to valid range
        centre_0 = ref_frame - 1
        start_0  = max(0, centre_0 - half_window)
        stop_0   = min(total_frames - 1, centre_0 + half_window)

        frames = np.stack(
            [tif.pages[i].asarray().astype(np.float32)
             for i in range(start_0, stop_0 + 1)],
            axis=0
        )

    return frames.mean(axis=0)


def detect_spots(image: np.ndarray,
                 psf_radius: float,
                 threshold: float,
                 exclude_border: int) -> pd.DataFrame:
    """
    Detect diffraction-limited spots using Difference-of-Gaussian blob detection.

    Returns a DataFrame with columns: x, y, sigma, intensity
    (x = column, y = row — matching typical image coordinate convention)
    """
    # Normalise to [0, 1] for consistent thresholding across experiments
    img_norm = rescale_intensity(image, out_range=(0.0, 1.0))

    min_sigma = psf_radius * 0.6
    max_sigma = psf_radius * 1.4

    blobs = blob_dog(
        img_norm,
        min_sigma=min_sigma,
        max_sigma=max_sigma,
        threshold=threshold,
        overlap=0.5,        # suppress overlapping detections aggressively
    )
    # blobs columns: row, col, sigma
    if len(blobs) == 0:
        return pd.DataFrame(columns=["x", "y", "sigma", "intensity"])

    rows, cols, sigmas = blobs[:, 0], blobs[:, 1], blobs[:, 2]

    # Border exclusion
    h, w = image.shape
    mask = (
        (rows >= exclude_border) & (rows < h - exclude_border) &
        (cols >= exclude_border) & (cols < w - exclude_border)
    )
    rows, cols, sigmas = rows[mask], cols[mask], sigmas[mask]

    # Sample intensity at each detected centre from the raw (non-normalised) image
    ri = np.round(rows).astype(int)
    ci = np.round(cols).astype(int)
    intensities = image[ri, ci]

    return pd.DataFrame({
        "x": cols,       # x = horizontal = column index
        "y": rows,       # y = vertical   = row index
        "sigma": sigmas,
        "intensity": intensities,
    })


def filter_by_intensity(spots: pd.DataFrame,
                        background: float,
                        min_mult: float,
                        max_mult: float) -> pd.Series:
    """
    Flag spots whose intensity falls outside [min_mult, max_mult] × background.
    Returns a boolean Series: True = flagged as likely false positive.
    """
    if len(spots) == 0:
        return pd.Series([], dtype=bool)
    intensity  = spots["intensity"]
    too_dim    = (intensity < min_mult * background) if min_mult > 0 else pd.Series([False] * len(spots))
    too_bright = intensity > max_mult * background
    return too_dim | too_bright


def fit_gaussian_2d(image: np.ndarray, cx: float, cy: float,
                    psf_radius: float) -> dict:
    """
    Fit a 2D symmetric Gaussian to a small patch around (cx, cy).
    Returns dict with keys: r2, sigma_fit, success.
    """
    from scipy.optimize import curve_fit

    r = int(np.ceil(psf_radius * 2))
    x0, y0 = int(round(cx)), int(round(cy))
    h, w = image.shape

    # Extract patch, clamped to image bounds
    y1, y2 = max(0, y0 - r), min(h, y0 + r + 1)
    x1, x2 = max(0, x0 - r), min(w, x0 + r + 1)
    patch = image[y1:y2, x1:x2].astype(float)

    if patch.size < 9:
        return {"r2": 0.0, "sigma_fit": np.nan, "success": False}

    yy, xx = np.mgrid[0:patch.shape[0], 0:patch.shape[1]].astype(float)
    cy_local = y0 - y1
    cx_local = x0 - x1

    def gaussian_2d(coords, amplitude, sigma, offset):
        y, x = coords
        return (amplitude * np.exp(
            -((x - cx_local)**2 + (y - cy_local)**2) / (2 * sigma**2)
        ) + offset).ravel()

    try:
        p0 = [patch.max() - patch.min(), psf_radius, patch.min()]
        bounds = ([0, 0.3, -np.inf], [np.inf, psf_radius * 4, np.inf])
        popt, _ = curve_fit(gaussian_2d, (yy, xx), patch.ravel(),
                            p0=p0, bounds=bounds, maxfev=400)
        fitted = gaussian_2d((yy, xx), *popt).reshape(patch.shape)
        ss_res = np.sum((patch - fitted) ** 2)
        ss_tot = np.sum((patch - patch.mean()) ** 2)
        r2 = 1.0 - ss_res / ss_tot if ss_tot > 0 else 0.0
        return {"r2": float(r2), "sigma_fit": float(popt[1]), "success": True}
    except Exception:
        return {"r2": 0.0, "sigma_fit": np.nan, "success": False}


def filter_by_psf_shape(spots: pd.DataFrame,
                        image: np.ndarray,
                        psf_radius: float,
                        min_r2: float,
                        width_tol: float) -> pd.Series:
    """
    Flag spots that fail a 2D Gaussian shape fit.
    Flags if: R² < min_r2, OR fitted sigma differs from psf_radius by > width_tol fraction.
    Returns a boolean Series: True = flagged.
    """
    if len(spots) == 0:
        return pd.Series([], dtype=bool)

    flags = []
    for _, row in spots.iterrows():
        result = fit_gaussian_2d(image, row["x"], row["y"], psf_radius)
        if not result["success"]:
            flags.append(True)
            continue
        bad_fit   = result["r2"] < min_r2
        sigma_ok  = (1.0 - width_tol) * psf_radius <= result["sigma_fit"] <= (1.0 + width_tol) * psf_radius
        flags.append(bad_fit or not sigma_ok)
    return pd.Series(flags, dtype=bool)


def apply_filters(spots: pd.DataFrame,
                  image: np.ndarray,
                  background: float,
                  psf_radius: float,
                  intensity_min_mult: float,
                  intensity_max_mult: float,
                  psf_min_r2: float,
                  psf_width_tol: float,
                  run_psf_filter: bool = True) -> pd.DataFrame:
    """
    Apply intensity and (optionally) PSF shape filters to a spot list.
    Adds columns: flagged_intensity, flagged_psf, flagged (either).
    Returns the augmented DataFrame.
    """
    spots = spots.copy()
    spots["flagged_intensity"] = filter_by_intensity(
        spots, background, intensity_min_mult, intensity_max_mult)

    if run_psf_filter and len(spots) > 0:
        spots["flagged_psf"] = filter_by_psf_shape(
            spots, image, psf_radius, psf_min_r2, psf_width_tol)
    else:
        spots["flagged_psf"] = False

    spots["flagged"] = spots["flagged_intensity"] | spots["flagged_psf"]
    return spots


def colocalize(df1: pd.DataFrame, df2: pd.DataFrame,
               threshold_px: float) -> tuple:
    """
    Mutual nearest-neighbour colocalization between two spot lists.

    A pair (A, B) is called colocalized only if:
      - A's nearest neighbour in ch2 is B  (within threshold_px), AND
      - B's nearest neighbour in ch1 is A  (within threshold_px)

    This prevents wrong-partner matches in dense regions where a ch1 spot
    might be closer to an unrelated ch2 spot than to its true partner.
    One-directional nearest-neighbour matching is used for the shift
    estimation pass (estimate_channel_shift) where a generous threshold
    is intentionally used and mutual constraint would be too strict.

    Returns (colocalized, unmatched_ch1, unmatched_ch2).
    """
    if len(df1) == 0 or len(df2) == 0:
        return pd.DataFrame(), df1.copy(), df2.copy()

    coords1 = df1[["x", "y"]].values.astype(float)
    coords2 = df2[["x", "y"]].values.astype(float)

    # Forward: for each ch1 spot, find nearest ch2 spot
    tree2 = cKDTree(coords2)
    dist_1to2, idx_1to2 = tree2.query(coords1, k=1, workers=-1)

    # Reverse: for each ch2 spot, find nearest ch1 spot
    tree1 = cKDTree(coords1)
    dist_2to1, idx_2to1 = tree1.query(coords2, k=1, workers=-1)

    # A pair (i, j) is mutual if:
    #   ch1[i] → ch2[j]  (forward match within threshold)
    #   ch2[j] → ch1[i]  (reverse match within threshold)
    mutual_mask = np.zeros(len(df1), dtype=bool)
    matched_idx2_list = []

    for i in range(len(df1)):
        j = idx_1to2[i]
        if dist_1to2[i] <= threshold_px and idx_2to1[j] == i:
            mutual_mask[i] = True
            matched_idx2_list.append(j)

    matched_idx2 = np.array(matched_idx2_list, dtype=int)

    matched1 = df1[mutual_mask].copy().reset_index(drop=True)
    matched2 = df2.iloc[matched_idx2].copy().reset_index(drop=True)

    colocalized = matched1.add_suffix("_ch1").join(matched2.add_suffix("_ch2"))
    colocalized["distance_px"] = dist_1to2[mutual_mask]

    unmatched_ch1 = df1[~mutual_mask].copy().reset_index(drop=True)

    matched_set2  = set(matched_idx2_list)
    unmatched_ch2 = df2[[i not in matched_set2 for i in range(len(df2))]].copy().reset_index(drop=True)

    return colocalized, unmatched_ch1, unmatched_ch2


def estimate_channel_shift(spots_ch1: pd.DataFrame,
                            spots_ch2: pd.DataFrame,
                            first_pass_threshold: float,
                            min_pairs: int,
                            max_shift: float) -> tuple:
    """
    Estimate the systematic (x, y) offset between Ch1 and Ch2 using a generous
    first-pass colocalization.

    Returns (shift_x, shift_y, n_pairs_used, shift_applied).
    shift_applied is False if there were too few pairs or the shift exceeded max_shift.
    A shift of (0, 0) with shift_applied=False means no correction was made.
    """
    if len(spots_ch1) == 0 or len(spots_ch2) == 0:
        return 0.0, 0.0, 0, False

    # First pass with generous threshold
    coloc_pass1, _, _ = colocalize(spots_ch1, spots_ch2, first_pass_threshold)

    if len(coloc_pass1) < min_pairs:
        return 0.0, 0.0, len(coloc_pass1), False

    # Offset vectors: Ch2 position minus Ch1 position
    dx = coloc_pass1["x_ch2"].values - coloc_pass1["x_ch1"].values
    dy = coloc_pass1["y_ch2"].values - coloc_pass1["y_ch1"].values

    shift_x = float(np.median(dx))
    shift_y = float(np.median(dy))
    magnitude = np.sqrt(shift_x ** 2 + shift_y ** 2)

    if magnitude > max_shift:
        return shift_x, shift_y, len(coloc_pass1), False

    return shift_x, shift_y, len(coloc_pass1), True


def print_summary(name: str, n_ch1: int, n_ch2: int, coloc: pd.DataFrame,
                  shift_x: float = 0.0, shift_y: float = 0.0,
                  shift_applied: bool = False):
    pct1 = 100 * len(coloc) / n_ch1 if n_ch1 > 0 else 0
    pct2 = 100 * len(coloc) / n_ch2 if n_ch2 > 0 else 0
    print(f"\n  [{name}]")
    print(f"    Ch1 spots detected : {n_ch1}")
    print(f"    Ch2 spots detected : {n_ch2}")
    if shift_applied:
        print(f"    Channel shift      : dx={shift_x:+.2f} px, dy={shift_y:+.2f} px  [corrected]")
    else:
        magnitude = np.sqrt(shift_x**2 + shift_y**2)
        if magnitude > 0.01:
            print(f"    Channel shift      : dx={shift_x:+.2f} px, dy={shift_y:+.2f} px  [NOT corrected — insufficient pairs or shift too large]")
        else:
            print(f"    Channel shift      : none detected")
    print(f"    Colocalized pairs  : {len(coloc)}")
    print(f"    Ch1 coloc rate     : {pct1:.1f}%")
    print(f"    Ch2 coloc rate     : {pct2:.1f}%")
    if len(coloc) > 0:
        print(f"    Median distance    : {coloc['distance_px'].median():.2f} px  "
              f"(mean {coloc['distance_px'].mean():.2f} px)")


def write_coloc_summary(out_path: Path, stem: str,
                        n_ch1: int, n_ch2: int,
                        n_coloc: int, n_u1: int, n_u2: int,
                        run_date: str = ""):
    """
    Write (or overwrite) the analysis summary .txt file with colocalization stats.
    The bleaching analysis script will append to this file later.
    """
    pct_coloc_ch1 = 100.0 * n_coloc / n_ch1 if n_ch1 > 0 else 0.0
    pct_coloc_ch2 = 100.0 * n_coloc / n_ch2 if n_ch2 > 0 else 0.0
    pct_u1        = 100.0 * n_u1    / n_ch1 if n_ch1 > 0 else 0.0
    pct_u2        = 100.0 * n_u2    / n_ch2 if n_ch2 > 0 else 0.0

    lines = [
        f"=== Analysis Summary: {stem} ===",
        f"Date: {run_date or 'unknown'}",
        "",
        "--- Colocalization ---",
        "Channel 1:",
        f"  Total spots detected : {n_ch1}",
        f"  Colocalized          : {n_coloc}  ({pct_coloc_ch1:.1f}%)",
        f"  Unmatched            : {n_u1}  ({pct_u1:.1f}%)",
        "",
        "Channel 2:",
        f"  Total spots detected : {n_ch2}",
        f"  Colocalized          : {n_coloc}  ({pct_coloc_ch2:.1f}%)",
        f"  Unmatched            : {n_u2}  ({pct_u2:.1f}%)",
        "",
        f"Colocalized pairs      : {n_coloc}",
        "",
    ]
    with open(out_path, "w") as f:
        f.write("\n".join(lines))
    print(f"  Summary  → {out_path}")


def write_coloc_file(out_path: Path,
                     colocalized_spots: pd.DataFrame,
                     unmatched_spots: pd.DataFrame,
                     coloc_x_col: str,
                     coloc_y_col: str):
    """
    Write a LabView-compatible coloc file with the format:
        <N> s
        x1 y1
        ...
        <M> n
        x1 y1
        ...
    Coordinates are rounded to integers.
    colocalized_spots uses suffixed columns (e.g. x_ch1, y_ch1).
    unmatched_spots uses plain columns (x, y).
    """
    with open(out_path, "w") as f:
        # Colocalized spots
        f.write(f"{len(colocalized_spots)} s\r\n")
        if len(colocalized_spots) > 0:
            xs = colocalized_spots[coloc_x_col].round().astype(int).tolist()
            ys = colocalized_spots[coloc_y_col].round().astype(int).tolist()
            for x, y in zip(xs, ys):
                f.write(f"{x} {y}\r\n")

        # Non-colocalized spots (plain x, y columns)
        f.write(f"{len(unmatched_spots)} n\r\n")
        if len(unmatched_spots) > 0:
            xs = unmatched_spots["x"].round().astype(int).tolist()
            ys = unmatched_spots["y"].round().astype(int).tolist()
            for x, y in zip(xs, ys):
                f.write(f"{x} {y}\r\n")


def save_overlay_image(out_path: Path,
                       img_ch1: np.ndarray,
                       img_ch2: np.ndarray,
                       coloc: pd.DataFrame,
                       unmatched_ch1: pd.DataFrame,
                       unmatched_ch2: pd.DataFrame,
                       psf_radius: float):
    """
    Save a side-by-side PNG of ch1 and ch2 averaged frames with spot overlays:
      - Green circle   : colocalized spot, passed filters
      - Yellow circle  : colocalized spot, flagged by intensity or PSF filter
      - Red X          : unmatched spot, passed filters
      - Orange X       : unmatched spot, flagged
    """
    from skimage.exposure import rescale_intensity

    fig, axes = plt.subplots(1, 2, figsize=(14, 7))
    circle_radius = psf_radius * 1.5

    for ax, img, ch_label, cx_col, cy_col, flag_col, unmatch_df in [
        (axes[0], img_ch1, "Channel 1", "x_ch1", "y_ch1", "flagged_ch1", unmatched_ch1),
        (axes[1], img_ch2, "Channel 2", "x_ch2", "y_ch2", "flagged_ch2", unmatched_ch2),
    ]:
        p1, p99 = np.percentile(img, (1, 99))
        img_display = rescale_intensity(img, in_range=(p1, p99), out_range=(0.0, 1.0))
        ax.imshow(img_display, cmap="gray", origin="upper")

        # Colocalized spots — green (ok) or yellow (flagged)
        if len(coloc) > 0:
            has_flag = flag_col in coloc.columns
            for _, row in coloc.iterrows():
                flagged = has_flag and row[flag_col]
                color = "#ffdd00" if flagged else "#00ff00"
                circle = mpatches.Circle(
                    (row[cx_col], row[cy_col]), radius=circle_radius,
                    edgecolor=color, facecolor="none",
                    linewidth=1.2, zorder=3,
                )
                ax.add_patch(circle)

        # Unmatched spots — red (ok) or orange (flagged)
        if len(unmatch_df) > 0:
            has_flag   = "flagged" in unmatch_df.columns
            flag_vals  = unmatch_df["flagged"].values if has_flag else np.zeros(len(unmatch_df), dtype=bool)
            ok_mask    = ~flag_vals
            flag_mask  = flag_vals
            xs = unmatch_df["x"].values
            ys = unmatch_df["y"].values
            if ok_mask.any():
                ax.scatter(xs[ok_mask], ys[ok_mask],
                           marker="x", s=60, c="#ff3333", linewidths=1.2, zorder=3)
            if flag_mask.any():
                ax.scatter(xs[flag_mask], ys[flag_mask],
                           marker="x", s=60, c="#ff8800", linewidths=1.2, zorder=3)

        n_coloc = len(coloc)
        n_flagged_coloc = int(coloc[flag_col].sum()) if len(coloc) > 0 and flag_col in coloc.columns else 0
        n_flagged_unmat = int(unmatch_df["flagged"].sum()) if len(unmatch_df) > 0 and "flagged" in unmatch_df.columns else 0
        ax.set_title(
            f"{ch_label}  |  {n_coloc} colocalized ({n_flagged_coloc} flagged ●)   "
            f"{len(unmatch_df)} unmatched ({n_flagged_unmat} flagged ✕)",
            fontsize=9,
        )
        ax.axis("off")

    legend_elements = [
        mpatches.Patch(edgecolor="#00ff00", facecolor="none", linewidth=1.5, label="Colocalized"),
        mpatches.Patch(edgecolor="#ffdd00", facecolor="none", linewidth=1.5, label="Colocalized (flagged)"),
        plt.Line2D([0], [0], marker="x", color="#ff3333", linestyle="none",
                   markersize=8, markeredgewidth=1.5, label="Unmatched"),
        plt.Line2D([0], [0], marker="x", color="#ff8800", linestyle="none",
                   markersize=8, markeredgewidth=1.5, label="Unmatched (flagged)"),
    ]
    fig.legend(handles=legend_elements, loc="lower center", ncol=4,
               fontsize=9, frameon=True)

    fig.tight_layout(rect=[0, 0.04, 1, 1])
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close(fig)


def save_coloc_only_image(out_path: Path,
                          img_ch1: np.ndarray,
                          img_ch2: np.ndarray,
                          coloc: pd.DataFrame,
                          psf_radius: float):
    """
    Save a side-by-side PNG showing only colocalized spots (no unmatched X markers).
    Green circle = passed filters, yellow circle = flagged.
    """
    from skimage.exposure import rescale_intensity

    fig, axes = plt.subplots(1, 2, figsize=(14, 7))
    circle_radius = psf_radius * 1.5

    for ax, img, ch_label, cx_col, cy_col, flag_col in [
        (axes[0], img_ch1, "Channel 1", "x_ch1", "y_ch1", "flagged_ch1"),
        (axes[1], img_ch2, "Channel 2", "x_ch2", "y_ch2", "flagged_ch2"),
    ]:
        p1, p99 = np.percentile(img, (1, 99))
        img_display = rescale_intensity(img, in_range=(p1, p99), out_range=(0.0, 1.0))
        ax.imshow(img_display, cmap="gray", origin="upper")

        if len(coloc) > 0:
            has_flag = flag_col in coloc.columns
            for _, row in coloc.iterrows():
                flagged = has_flag and row[flag_col]
                color = "#ffdd00" if flagged else "#00ff00"
                circle = mpatches.Circle(
                    (row[cx_col], row[cy_col]), radius=circle_radius,
                    edgecolor=color, facecolor="none",
                    linewidth=1.2, zorder=3,
                )
                ax.add_patch(circle)

        n_coloc = len(coloc)
        n_flagged = int(coloc[flag_col].sum()) if len(coloc) > 0 and flag_col in coloc.columns else 0
        ax.set_title(
            f"{ch_label}  |  {n_coloc} colocalized ({n_flagged} flagged ●)",
            fontsize=9,
        )
        ax.axis("off")

    legend_elements = [
        mpatches.Patch(edgecolor="#00ff00", facecolor="none", linewidth=1.5, label="Colocalized"),
        mpatches.Patch(edgecolor="#ffdd00", facecolor="none", linewidth=1.5, label="Colocalized (flagged)"),
    ]
    fig.legend(handles=legend_elements, loc="lower center", ncol=2,
               fontsize=9, frameon=True)

    fig.tight_layout(rect=[0, 0.04, 1, 1])
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close(fig)


def _propagate_flags(coloc: pd.DataFrame,
                     spots: pd.DataFrame,
                     x_col: str, y_col: str) -> pd.Series:
    """
    Match colocalized spot positions back to their source spot list
    and return the corresponding flagged values.
    Uses nearest-neighbour matching on (x, y) to handle float rounding.
    """
    if "flagged" not in spots.columns or len(coloc) == 0:
        return pd.Series([False] * len(coloc), dtype=bool)

    from scipy.spatial import cKDTree as _KDTree
    spot_coords = spots[["x", "y"]].values.astype(float)
    coloc_coords = coloc[[x_col, y_col]].values.astype(float)
    tree = _KDTree(spot_coords)
    _, idx = tree.query(coloc_coords, k=1)
    return pd.Series(spots["flagged"].iloc[idx].values, dtype=bool)


def _propagate_cnn(coloc: pd.DataFrame,
                   spots: pd.DataFrame,
                   x_col: str, y_col: str,
                   src_col: str,
                   default) -> pd.Series:
    """
    Match colocalized spot positions back to their source spot list
    and return the values of src_col (e.g. 'cnn_prob' or 'cnn_flagged').
    Falls back to `default` if src_col is absent.
    """
    if src_col not in spots.columns or len(coloc) == 0:
        return pd.Series([default] * len(coloc))

    from scipy.spatial import cKDTree as _KDTree
    spot_coords  = spots[["x", "y"]].values.astype(float)
    coloc_coords = coloc[[x_col, y_col]].values.astype(float)
    tree = _KDTree(spot_coords)
    _, idx = tree.query(coloc_coords, k=1)
    return pd.Series(spots[src_col].iloc[idx].values)


# ===========================================================================
# CNN SPOT CLASSIFIER
# ===========================================================================

CNN_PATCH_SIZE   = 25          # must match extract_patches.py PATCH_SIZE
CNN_MODEL_PATH   = Path.home() / "smfret_params" / "models" / "spot_classifier.pt"
CNN_THRESH_PATH  = Path.home() / "smfret_params" / "models" / "spot_classifier_threshold.json"


def load_spot_classifier():
    """
    Load the TorchScript CNN spot classifier and its optimal threshold.

    Returns (model, threshold) if both files exist and torch is available,
    otherwise returns (None, None) — the pipeline continues unchanged.
    """
    if not _TORCH_AVAILABLE:
        return None, None
    if not CNN_MODEL_PATH.exists() or not CNN_THRESH_PATH.exists():
        return None, None
    try:
        import json
        model = torch.jit.load(str(CNN_MODEL_PATH), map_location="cpu")
        model.eval()
        with open(CNN_THRESH_PATH) as f:
            thresh_data = json.load(f)
        # threshold JSON may be {"threshold": 0.5} or just a bare float
        if isinstance(thresh_data, dict):
            threshold = float(thresh_data.get("threshold", 0.5))
        else:
            threshold = float(thresh_data)
        print(f"  CNN spot classifier loaded  (threshold={threshold:.3f})")
        return model, threshold
    except Exception as e:
        print(f"  [CNN] Could not load spot classifier: {e}")
        return None, None


def score_spots_cnn(spots_df: pd.DataFrame,
                    image: np.ndarray,
                    model,
                    threshold: float,
                    patch_size: int = CNN_PATCH_SIZE) -> pd.DataFrame:
    """
    Score each spot in spots_df using the CNN classifier.

    Extracts a (patch_size × patch_size) image patch centred on each spot,
    normalises it to [0, 1], and runs the model.

    Adds two columns to a copy of spots_df:
      cnn_prob    — float32 probability that the spot is a real fluorescent spot
      cnn_flagged — bool, True if cnn_prob < threshold (likely false positive)

    If spots_df is empty or model is None, returns spots_df with the columns
    added as 0.0 / False so downstream code never needs to check.
    """
    spots_df = spots_df.copy()

    if model is None or len(spots_df) == 0:
        spots_df["cnn_prob"]    = np.float32(0.0)
        spots_df["cnn_flagged"] = False
        return spots_df

    half = patch_size // 2
    h, w = image.shape

    # Pre-normalise the full image once (same normalisation as extract_patches.py)
    img_min = image.min()
    img_max = image.max()
    denom   = float(img_max - img_min) if img_max > img_min else 1.0
    img_norm = (image.astype(np.float32) - img_min) / denom

    patches = []
    for _, row in spots_df.iterrows():
        cx = int(round(float(row["x"])))
        cy = int(round(float(row["y"])))

        x0 = cx - half;  x1 = cx + half + 1
        y0 = cy - half;  y1 = cy + half + 1

        # Compute padding needed
        pad_left  = max(0, -x0);   pad_top  = max(0, -y0)
        pad_right = max(0, x1 - w); pad_bot = max(0, y1 - h)

        xc0 = max(0, x0); xc1 = min(w, x1)
        yc0 = max(0, y0); yc1 = min(h, y1)

        patch = img_norm[yc0:yc1, xc0:xc1]
        if pad_left or pad_top or pad_right or pad_bot:
            patch = np.pad(patch,
                           ((pad_top, pad_bot), (pad_left, pad_right)),
                           mode="edge")

        # Ensure exact size (rounding edge case)
        if patch.shape != (patch_size, patch_size):
            patch = np.zeros((patch_size, patch_size), dtype=np.float32)

        patches.append(patch)

    # Stack → (N, 1, H, W) float32 tensor
    arr = np.stack(patches, axis=0)[:, np.newaxis, :, :]   # (N,1,25,25)
    tensor = torch.from_numpy(arr.astype(np.float32))

    with torch.no_grad():
        logits = model(tensor)                 # (N,1) or (N,)
        probs  = torch.sigmoid(logits).squeeze(-1).cpu().numpy().astype(np.float32)

    spots_df["cnn_prob"]    = probs
    spots_df["cnn_flagged"] = probs < threshold
    return spots_df


# ===========================================================================

def process_file(tif_path: Path, args, out_dir: Path) -> dict | None:
    """Full pipeline for a single .tif file."""
    print(f"\n  Processing: {tif_path.name}")

    # Derive frames_per_channel from the actual file length
    with tifffile.TiffFile(tif_path) as _tif:
        total_frames = len(_tif.pages)
    frames_per_channel = total_frames // 2
    if total_frames % 2 != 0:
        print(f"    Warning: odd total frame count ({total_frames}) — "
              f"using {frames_per_channel} frames per channel.")
    print(f"    Total frames: {total_frames}  "
          f"({frames_per_channel} per channel)")

    # Override args.frames_per_channel with the derived value so all
    # downstream calls within this function use the correct number
    args_fpc = frames_per_channel

    # ch2 ref frame = ch2 start + same offset as ch1 ref frame
    ch1_ref = args.ch1_ref_frame
    ch2_ref = frames_per_channel + args.ch1_ref_frame  # mirror ch1 offset into ch2

    try:
        img_ch1 = load_averaged_frame(
            tif_path,
            ref_frame=ch1_ref,
            half_window=args.average_half_window,
            frames_per_channel=frames_per_channel,
        )
        img_ch2 = load_averaged_frame(
            tif_path,
            ref_frame=ch2_ref,
            half_window=args.average_half_window,
            frames_per_channel=frames_per_channel,
        )
    except Exception as e:
        print(f"    [ERROR] Could not load frames: {e}")
        return None

    thr1 = args.ch1_threshold if args.ch1_threshold is not None else args.detection_threshold
    thr2 = args.ch2_threshold if args.ch2_threshold is not None else args.detection_threshold

    spots_ch1 = detect_spots(img_ch1, args.psf_radius, thr1, args.exclude_border)
    spots_ch2 = detect_spots(img_ch2, args.psf_radius, thr2, args.exclude_border)

    # --- False positive filtering (intensity + PSF shape) ---
    # Background estimated from the blank frames (laser off) at the start of each channel.
    # These frames have no real signal, so their median gives a clean camera baseline.
    def _blank_background(tif_path, start_frame_1idx, n_blank):
        """Return median intensity of the first n_blank frames starting at start_frame_1idx."""
        with tifffile.TiffFile(tif_path) as tif:
            frames = np.stack(
                [tif.pages[start_frame_1idx - 1 + i].asarray().astype(np.float32)
                 for i in range(min(n_blank, len(tif.pages) - start_frame_1idx + 1))],
                axis=0,
            )
        return float(np.median(frames))

    n_blank = getattr(args, 'blank_frames', BLANK_FRAMES)
    bg_ch1 = _blank_background(tif_path, 1,                        n_blank)
    bg_ch2 = _blank_background(tif_path, frames_per_channel + 1,   n_blank)
    print(f"    Background — Ch1: {bg_ch1:.1f}   Ch2: {bg_ch2:.1f}")

    print(f"    Running false positive filters...")
    spots_ch1 = apply_filters(
        spots_ch1, img_ch1, bg_ch1,
        psf_radius=args.psf_radius,
        intensity_min_mult=args.ch1_intensity_min_mult,
        intensity_max_mult=args.ch1_intensity_max_mult,
        psf_min_r2=args.psf_fit_min_r2,
        psf_width_tol=args.psf_fit_width_tol,
        run_psf_filter=True,
    )
    spots_ch2 = apply_filters(
        spots_ch2, img_ch2, bg_ch2,
        psf_radius=args.psf_radius,
        intensity_min_mult=args.ch2_intensity_min_mult,
        intensity_max_mult=args.ch2_intensity_max_mult,
        psf_min_r2=args.psf_fit_min_r2,
        psf_width_tol=args.psf_fit_width_tol,
        run_psf_filter=True,
    )
    n_flagged_ch1 = int(spots_ch1["flagged"].sum()) if "flagged" in spots_ch1.columns else 0
    n_flagged_ch2 = int(spots_ch2["flagged"].sum()) if "flagged" in spots_ch2.columns else 0
    print(f"    Flagged spots — Ch1: {n_flagged_ch1}   Ch2: {n_flagged_ch2}")

    # --- CNN spot classifier (advisory — adds cnn_prob / cnn_flagged columns) ---
    if not getattr(args, 'no_cnn', False):
        _cnn_model, _cnn_thresh = load_spot_classifier()
        if _cnn_model is not None:
            print(f"    Running CNN spot classifier...")
            spots_ch1 = score_spots_cnn(spots_ch1, img_ch1, _cnn_model, _cnn_thresh)
            spots_ch2 = score_spots_cnn(spots_ch2, img_ch2, _cnn_model, _cnn_thresh)
            n_cnn_flag1 = int(spots_ch1["cnn_flagged"].sum())
            n_cnn_flag2 = int(spots_ch2["cnn_flagged"].sum())
            print(f"    CNN flagged — Ch1: {n_cnn_flag1}   Ch2: {n_cnn_flag2}")

    # --- Two-pass colocalization with channel shift correction ---
    first_pass_threshold = args.coloc_threshold * args.shift_first_pass_factor
    shift_x, shift_y, n_shift_pairs, shift_applied = estimate_channel_shift(
        spots_ch1, spots_ch2,
        first_pass_threshold=first_pass_threshold,
        min_pairs=args.shift_min_pairs,
        max_shift=args.shift_max_px,
    )

    spots_ch2_corrected = spots_ch2.copy()
    if shift_applied:
        spots_ch2_corrected["x"] = spots_ch2_corrected["x"] - shift_x
        spots_ch2_corrected["y"] = spots_ch2_corrected["y"] - shift_y

    coloc, u1, u2 = colocalize(spots_ch1, spots_ch2_corrected, args.coloc_threshold)

    # Restore original Ch2 coordinates
    if shift_applied and len(coloc) > 0:
        coloc["x_ch2"] = coloc["x_ch2"] + shift_x
        coloc["y_ch2"] = coloc["y_ch2"] + shift_y
        coloc["shift_x_applied"] = shift_x
        coloc["shift_y_applied"] = shift_y
    else:
        if len(coloc) > 0:
            coloc["shift_x_applied"] = 0.0
            coloc["shift_y_applied"] = 0.0

    # Propagate flag columns into colocalized DataFrame
    if len(coloc) > 0:
        # flagged_ch1: flag from the ch1 spot that was matched
        # flagged_ch2: flag from the ch2 spot that was matched
        # (spots_ch1/ch2 index was reset inside colocalize via reset_index)
        if "flagged" in spots_ch1.columns:
            coloc["flagged_ch1"] = spots_ch1.loc[
                spots_ch1.index[spots_ch1[["x","y"]].apply(tuple,axis=1).isin(
                    coloc[["x_ch1","y_ch1"]].apply(tuple,axis=1))]
            ]["flagged"].values if False else _propagate_flags(
                coloc, spots_ch1, "x_ch1", "y_ch1")
        if "flagged" in spots_ch2.columns:
            coloc["flagged_ch2"] = _propagate_flags(
                coloc, spots_ch2_corrected, "x_ch2", "y_ch2")

    stem = tif_path.stem
    print_summary(stem, len(spots_ch1), len(spots_ch2), coloc,
                  shift_x=shift_x, shift_y=shift_y, shift_applied=shift_applied)

    # Propagate CNN columns into colocalized DataFrame
    if "cnn_prob" in spots_ch1.columns and len(coloc) > 0:
        coloc["cnn_prob_ch1"]    = _propagate_cnn(coloc, spots_ch1, "x_ch1", "y_ch1", "cnn_prob",    default=0.0)
        coloc["cnn_flagged_ch1"] = _propagate_cnn(coloc, spots_ch1, "x_ch1", "y_ch1", "cnn_flagged", default=False)
    if "cnn_prob" in spots_ch2.columns and len(coloc) > 0:
        coloc["cnn_prob_ch2"]    = _propagate_cnn(coloc, spots_ch2_corrected, "x_ch2", "y_ch2", "cnn_prob",    default=0.0)
        coloc["cnn_flagged_ch2"] = _propagate_cnn(coloc, spots_ch2_corrected, "x_ch2", "y_ch2", "cnn_flagged", default=False)

    # Save outputs
    out_dir.mkdir(parents=True, exist_ok=True)
    coloc.to_csv(out_dir / f"{stem}_colocalized.csv", index=False)
    u1.to_csv(out_dir / f"{stem}_unmatched_ch1.csv", index=False)
    u2.to_csv(out_dir / f"{stem}_unmatched_ch2.csv", index=False)
    spots_ch1.to_csv(out_dir / f"{stem}_all_spots_ch1.csv", index=False)
    spots_ch2.to_csv(out_dir / f"{stem}_all_spots_ch2.csv", index=False)

    import datetime
    write_coloc_summary(
        out_path  = out_dir / f"{stem}_summary.txt",
        stem      = stem,
        n_ch1     = len(spots_ch1),
        n_ch2     = len(spots_ch2),
        n_coloc   = len(coloc),
        n_u1      = len(u1),
        n_u2      = len(u2),
        run_date  = datetime.date.today().isoformat(),
    )

    write_coloc_file(
        out_path=out_dir / f"{stem}_ch1coloc",
        colocalized_spots=coloc,
        unmatched_spots=u1,
        coloc_x_col="x_ch1",
        coloc_y_col="y_ch1",
    )
    write_coloc_file(
        out_path=out_dir / f"{stem}_ch2coloc",
        colocalized_spots=coloc,
        unmatched_spots=u2,
        coloc_x_col="x_ch2",
        coloc_y_col="y_ch2",
    )

    overlay_path = out_dir / f"{stem}_overlay.png"
    save_overlay_image(
        out_path=overlay_path,
        img_ch1=img_ch1,
        img_ch2=img_ch2,
        coloc=coloc,
        unmatched_ch1=u1,
        unmatched_ch2=u2,
        psf_radius=args.psf_radius,
    )

    save_coloc_only_image(
        out_path=out_dir / f"{stem}_overlay_coloc_only.png",
        img_ch1=img_ch1,
        img_ch2=img_ch2,
        coloc=coloc,
        psf_radius=args.psf_radius,
    )

    print(f"    Output → {out_dir}/")

    n_ch1, n_ch2 = len(spots_ch1), len(spots_ch2)
    magnitude = float(np.sqrt(shift_x**2 + shift_y**2))
    n_flagged_coloc = int(coloc["flagged_ch1"].sum()) if len(coloc) > 0 and "flagged_ch1" in coloc.columns else 0
    return {
        "file": tif_path.name,
        "ch1_spots": n_ch1,
        "ch2_spots": n_ch2,
        "ch1_flagged": n_flagged_ch1,
        "ch2_flagged": n_flagged_ch2,
        "colocalized_pairs": len(coloc),
        "colocalized_flagged": n_flagged_coloc,
        "ch1_coloc_pct": round(100 * len(coloc) / n_ch1, 2) if n_ch1 else 0,
        "ch2_coloc_pct": round(100 * len(coloc) / n_ch2, 2) if n_ch2 else 0,
        "median_distance_px": round(coloc["distance_px"].median(), 3) if len(coloc) > 0 else float("nan"),
        "shift_x_px": round(shift_x, 3),
        "shift_y_px": round(shift_y, 3),
        "shift_magnitude_px": round(magnitude, 3),
        "shift_applied": shift_applied,
        "shift_pairs_used": n_shift_pairs,
    }



def write_coloc_summary_three_channel(out_path: Path, stem: str,
                                      n_ch1: int, n_ch2: int, n_ch3: int,
                                      n_triple: int,
                                      n_u1: int, n_u2: int, n_u3: int,
                                      run_date: str = ""):
    """Write analysis summary for a three-channel colocalization run."""
    def _pct(n, total):
        return f"{100.0 * n / total:.1f}%" if total > 0 else "0.0%"

    lines = [
        f"=== Analysis Summary: {stem} ===",
        f"Date: {run_date or 'unknown'}",
        "",
        "--- Colocalization (three-channel) ---",
        "Channel 1:",
        f"  Total spots detected : {n_ch1}",
        f"  Triple colocalized   : {n_triple}  ({_pct(n_triple, n_ch1)})",
        f"  Unmatched            : {n_u1}  ({_pct(n_u1, n_ch1)})",
        "",
        "Channel 2:",
        f"  Total spots detected : {n_ch2}",
        f"  Triple colocalized   : {n_triple}  ({_pct(n_triple, n_ch2)})",
        f"  Unmatched            : {n_u2}  ({_pct(n_u2, n_ch2)})",
        "",
        "Channel 3:",
        f"  Total spots detected : {n_ch3}",
        f"  Triple colocalized   : {n_triple}  ({_pct(n_triple, n_ch3)})",
        f"  Unmatched            : {n_u3}  ({_pct(n_u3, n_ch3)})",
        "",
        f"Triple colocalized spots : {n_triple}",
        "",
    ]
    with open(out_path, "w") as f:
        f.write("\n".join(lines))
    print(f"  Summary  → {out_path}")


def save_overlay_image_three_channel(out_path: Path,
                                     img_ch1: np.ndarray,
                                     img_ch2: np.ndarray,
                                     img_ch3: np.ndarray,
                                     triple: pd.DataFrame,
                                     unmatched_ch1: pd.DataFrame,
                                     unmatched_ch2: pd.DataFrame,
                                     unmatched_ch3: pd.DataFrame,
                                     psf_radius: float):
    """
    Save a three-panel PNG (ch1, ch2, ch3) with spot overlays for triple coloc.
      - Green circle : triple colocalized spot, passed filters
      - Yellow circle: triple colocalized spot, flagged
      - Red X        : unmatched spot, passed filters
      - Orange X     : unmatched spot, flagged
    """
    from skimage.exposure import rescale_intensity

    fig, axes = plt.subplots(1, 3, figsize=(21, 7))
    circle_radius = psf_radius * 1.5

    panel_info = [
        (axes[0], img_ch1, "Channel 1", "x_ch1", "y_ch1", "flagged_ch1", unmatched_ch1),
        (axes[1], img_ch2, "Channel 2", "x_ch2", "y_ch2", "flagged_ch2", unmatched_ch2),
        (axes[2], img_ch3, "Channel 3", "x_ch3", "y_ch3", "flagged_ch3", unmatched_ch3),
    ]
    for ax, img, ch_label, cx_col, cy_col, flag_col, unmatch_df in panel_info:
        p1, p99 = np.percentile(img, (1, 99))
        ax.imshow(rescale_intensity(img, in_range=(p1, p99), out_range=(0.0, 1.0)),
                  cmap="gray", origin="upper")

        if len(triple) > 0:
            has_flag = flag_col in triple.columns
            for _, row in triple.iterrows():
                flagged = has_flag and row[flag_col]
                color = "#ffdd00" if flagged else "#00ff00"
                ax.add_patch(mpatches.Circle(
                    (row[cx_col], row[cy_col]), radius=circle_radius,
                    edgecolor=color, facecolor="none", linewidth=1.2, zorder=3))

        if len(unmatch_df) > 0:
            has_flag  = "flagged" in unmatch_df.columns
            flag_vals = unmatch_df["flagged"].values if has_flag else np.zeros(len(unmatch_df), dtype=bool)
            xs = unmatch_df["x"].values; ys = unmatch_df["y"].values
            ok_mask = ~flag_vals
            if ok_mask.any():
                ax.scatter(xs[ok_mask], ys[ok_mask],
                           marker="x", s=60, c="#ff3333", linewidths=1.2, zorder=3)
            if flag_vals.any():
                ax.scatter(xs[flag_vals], ys[flag_vals],
                           marker="x", s=60, c="#ff8800", linewidths=1.2, zorder=3)

        n_triple_spots = len(triple)
        n_flag_coloc = int(triple[flag_col].sum()) if len(triple) > 0 and flag_col in triple.columns else 0
        n_flag_unmat = int(unmatch_df["flagged"].sum()) if len(unmatch_df) > 0 and "flagged" in unmatch_df.columns else 0
        ax.set_title(
            f"{ch_label}  |  {n_triple_spots} triple coloc ({n_flag_coloc} flagged ●)   "
            f"{len(unmatch_df)} unmatched ({n_flag_unmat} flagged ✕)",
            fontsize=9)
        ax.axis("off")

    legend_elements = [
        mpatches.Patch(edgecolor="#00ff00", facecolor="none", linewidth=1.5, label="Triple colocalized"),
        mpatches.Patch(edgecolor="#ffdd00", facecolor="none", linewidth=1.5, label="Triple coloc (flagged)"),
        plt.Line2D([0], [0], marker="x", color="#ff3333", linestyle="none",
                   markersize=8, markeredgewidth=1.5, label="Unmatched"),
        plt.Line2D([0], [0], marker="x", color="#ff8800", linestyle="none",
                   markersize=8, markeredgewidth=1.5, label="Unmatched (flagged)"),
    ]
    fig.legend(handles=legend_elements, loc="lower center", ncol=4, fontsize=9, frameon=True)
    fig.tight_layout(rect=[0, 0.04, 1, 1])
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close(fig)


def process_file_three_channel(tif_path: Path, args, out_dir: Path) -> dict | None:
    """
    Full three-channel colocalization pipeline.

    TIF layout: ch1 frames | ch2 frames | ch3 frames  (3 × N total).
    frames_per_channel derived at runtime as total_frames // 3.

    Matching strategy:
      Pass 1: ch1 ∩ ch2  (with ch1-reference shift correction for ch2)
      Pass 2: (ch1∩ch2) ∩ ch3  (with ch1-reference shift correction for ch3)
    Triple colocalized = spots present in all three channels.
    """
    print(f"\n  Processing (three-channel): {tif_path.name}")

    with tifffile.TiffFile(tif_path) as _tif:
        total_frames = len(_tif.pages)

    if total_frames % 3 != 0:
        print(f"    Warning: total frame count ({total_frames}) not divisible by 3 — "
              f"using floor division.")
    frames_per_channel = total_frames // 3
    print(f"    Total frames: {total_frames}  ({frames_per_channel} per channel)")

    ch1_ref = args.ch1_ref_frame
    ch2_ref = frames_per_channel + args.ch1_ref_frame
    ch3_ref = 2 * frames_per_channel + args.ch1_ref_frame

    try:
        img_ch1 = load_averaged_frame(tif_path, ch1_ref, args.average_half_window, frames_per_channel)
        img_ch2 = load_averaged_frame(tif_path, ch2_ref, args.average_half_window, frames_per_channel)
        img_ch3 = load_averaged_frame(tif_path, ch3_ref, args.average_half_window, frames_per_channel)
    except Exception as e:
        print(f"    [ERROR] Could not load frames: {e}")
        return None

    thr1 = args.ch1_threshold if args.ch1_threshold is not None else args.detection_threshold
    thr2 = args.ch2_threshold if args.ch2_threshold is not None else args.detection_threshold
    thr3 = args.ch3_threshold if args.ch3_threshold is not None else args.detection_threshold

    spots_ch1 = detect_spots(img_ch1, args.psf_radius, thr1, args.exclude_border)
    spots_ch2 = detect_spots(img_ch2, args.psf_radius, thr2, args.exclude_border)
    spots_ch3 = detect_spots(img_ch3, args.psf_radius, thr3, args.exclude_border)

    # Background from blank frames at the start of each channel block
    def _blank_bg(start_1idx, n_blank):
        with tifffile.TiffFile(tif_path) as tif:
            frames = np.stack(
                [tif.pages[start_1idx - 1 + i].asarray().astype(np.float32)
                 for i in range(min(n_blank, total_frames - start_1idx + 1))],
                axis=0)
        return float(np.median(frames))

    n_blank = getattr(args, 'blank_frames', BLANK_FRAMES)
    bg_ch1 = _blank_bg(1, n_blank)
    bg_ch2 = _blank_bg(frames_per_channel + 1, n_blank)
    bg_ch3 = _blank_bg(2 * frames_per_channel + 1, n_blank)
    print(f"    Background — Ch1: {bg_ch1:.1f}   Ch2: {bg_ch2:.1f}   Ch3: {bg_ch3:.1f}")

    print(f"    Running false positive filters...")
    spots_ch1 = apply_filters(spots_ch1, img_ch1, bg_ch1, psf_radius=args.psf_radius,
                               intensity_min_mult=args.ch1_intensity_min_mult,
                               intensity_max_mult=args.ch1_intensity_max_mult,
                               psf_min_r2=args.psf_fit_min_r2,
                               psf_width_tol=args.psf_fit_width_tol, run_psf_filter=True)
    spots_ch2 = apply_filters(spots_ch2, img_ch2, bg_ch2, psf_radius=args.psf_radius,
                               intensity_min_mult=args.ch2_intensity_min_mult,
                               intensity_max_mult=args.ch2_intensity_max_mult,
                               psf_min_r2=args.psf_fit_min_r2,
                               psf_width_tol=args.psf_fit_width_tol, run_psf_filter=True)
    spots_ch3 = apply_filters(spots_ch3, img_ch3, bg_ch3, psf_radius=args.psf_radius,
                               intensity_min_mult=args.ch3_intensity_min_mult,
                               intensity_max_mult=args.ch3_intensity_max_mult,
                               psf_min_r2=args.psf_fit_min_r2,
                               psf_width_tol=args.psf_fit_width_tol, run_psf_filter=True)

    n_flag1 = int(spots_ch1["flagged"].sum()) if "flagged" in spots_ch1.columns else 0
    n_flag2 = int(spots_ch2["flagged"].sum()) if "flagged" in spots_ch2.columns else 0
    n_flag3 = int(spots_ch3["flagged"].sum()) if "flagged" in spots_ch3.columns else 0
    print(f"    Flagged spots — Ch1: {n_flag1}   Ch2: {n_flag2}   Ch3: {n_flag3}")

    # --- CNN spot classifier (advisory) ---
    if not getattr(args, 'no_cnn', False):
        _cnn_model, _cnn_thresh = load_spot_classifier()
        if _cnn_model is not None:
            print(f"    Running CNN spot classifier...")
            spots_ch1 = score_spots_cnn(spots_ch1, img_ch1, _cnn_model, _cnn_thresh)
            spots_ch2 = score_spots_cnn(spots_ch2, img_ch2, _cnn_model, _cnn_thresh)
            spots_ch3 = score_spots_cnn(spots_ch3, img_ch3, _cnn_model, _cnn_thresh)
            print(f"    CNN flagged — Ch1: {int(spots_ch1['cnn_flagged'].sum())}   "
                  f"Ch2: {int(spots_ch2['cnn_flagged'].sum())}   "
                  f"Ch3: {int(spots_ch3['cnn_flagged'].sum())}")

    # --- Pass 1: ch1 ∩ ch2 with shift correction (ch1 as reference) ---
    fp_thr_12 = args.coloc_threshold * args.shift_first_pass_factor
    sx12, sy12, n_pairs12, shift12 = estimate_channel_shift(
        spots_ch1, spots_ch2, first_pass_threshold=fp_thr_12,
        min_pairs=args.shift_min_pairs, max_shift=args.shift_max_px)

    spots_ch2_corr = spots_ch2.copy()
    if shift12:
        spots_ch2_corr["x"] -= sx12
        spots_ch2_corr["y"] -= sy12

    coloc12, u1_12, u2_12 = colocalize(spots_ch1, spots_ch2_corr, args.coloc_threshold)

    # Restore original ch2 coords in coloc12
    if shift12 and len(coloc12) > 0:
        coloc12["x_ch2"] += sx12
        coloc12["y_ch2"] += sy12
        coloc12["shift_x_ch2"] = sx12
        coloc12["shift_y_ch2"] = sy12
    elif len(coloc12) > 0:
        coloc12["shift_x_ch2"] = 0.0
        coloc12["shift_y_ch2"] = 0.0

    print(f"    Ch1∩Ch2: {len(coloc12)} pairs  "
          f"(shift: dx={sx12:+.2f}, dy={sy12:+.2f} px, "
          f"{'applied' if shift12 else 'NOT applied'})")

    # --- Pass 2: (ch1∩ch2) ∩ ch3 with shift correction (ch1 as reference) ---
    fp_thr_13 = args.coloc_threshold_ch3 * args.shift_first_pass_factor
    sx13, sy13, n_pairs13, shift13 = estimate_channel_shift(
        spots_ch1, spots_ch3, first_pass_threshold=fp_thr_13,
        min_pairs=args.shift_min_pairs, max_shift=args.shift_max_px)

    spots_ch3_corr = spots_ch3.copy()
    if shift13:
        spots_ch3_corr["x"] -= sx13
        spots_ch3_corr["y"] -= sy13

    # Match the ch1 positions from coloc12 against corrected ch3 positions
    if len(coloc12) > 0:
        ch1_positions = coloc12[["x_ch1", "y_ch1"]].rename(
            columns={"x_ch1": "x", "y_ch1": "y"})
        coloc12_ch3, u1_from_12, u3_from_match = colocalize(
            ch1_positions, spots_ch3_corr, args.coloc_threshold_ch3)

        # Build full triple coloc DataFrame
        if len(coloc12_ch3) > 0:
            # Match coloc12_ch3 ch1 positions back to coloc12 rows
            from scipy.spatial import cKDTree as _KDT
            tree12 = _KDT(coloc12[["x_ch1", "y_ch1"]].values.astype(float))
            _, idx12 = tree12.query(
                coloc12_ch3[["x_ch1", "y_ch1"]].values.astype(float), k=1)
            triple = coloc12.iloc[idx12].copy().reset_index(drop=True)
            triple["x_ch3"] = coloc12_ch3["x_ch2"].values + (sx13 if shift13 else 0.0)
            triple["y_ch3"] = coloc12_ch3["y_ch2"].values + (sy13 if shift13 else 0.0)
            triple["shift_x_ch3"] = sx13 if shift13 else 0.0
            triple["shift_y_ch3"] = sy13 if shift13 else 0.0
            triple["distance_ch3_px"] = coloc12_ch3["distance_px"].values

            # Propagate ch3 flags
            triple["flagged_ch3"] = _propagate_flags(
                coloc12_ch3, spots_ch3_corr, "x_ch2", "y_ch2")
            # Propagate ch1/ch2 flags from original coloc12
            triple["flagged_ch1"] = _propagate_flags(triple, spots_ch1, "x_ch1", "y_ch1")
            triple["flagged_ch2"] = _propagate_flags(triple, spots_ch2_corr, "x_ch2", "y_ch2")

            # ch1 positions that matched ch2 but not ch3 → unmatched ch1
            matched_ch1_set = set(idx12.tolist())
            u1_not_triple = coloc12.iloc[
                [i for i in range(len(coloc12)) if i not in matched_ch1_set]
            ].copy().reset_index(drop=True)
            # Rescue ch2 spots from those failed-to-triple pairs
            u2_from_12_fail = u1_not_triple[["x_ch2", "y_ch2"]].rename(
                columns={"x_ch2": "x", "y_ch2": "y"}).copy()
            u1_not_triple = u1_not_triple[["x_ch1", "y_ch1"]].rename(
                columns={"x_ch1": "x", "y_ch1": "y"}).copy()
        else:
            triple = pd.DataFrame()
            u1_not_triple = coloc12[["x_ch1", "y_ch1"]].rename(
                columns={"x_ch1": "x", "y_ch1": "y"}).copy()
            u2_from_12_fail = coloc12[["x_ch2", "y_ch2"]].rename(
                columns={"x_ch2": "x", "y_ch2": "y"}).copy()
    else:
        triple = pd.DataFrame()
        u1_not_triple = pd.DataFrame(columns=["x", "y"])
        u2_from_12_fail = pd.DataFrame(columns=["x", "y"])
        u3_from_match = spots_ch3_corr.copy()

    # Restore ch3 coords in unmatched (undo shift correction for storage)
    if shift13 and len(u3_from_match) > 0:
        u3_from_match = u3_from_match.copy()
        u3_from_match["x"] += sx13
        u3_from_match["y"] += sy13

    # Final unmatched lists: original unmatched + spots rescued from failed pairs
    import pandas as _pd_local
    u1_final = _pd_local.concat(
        [u1_12, u1_not_triple], ignore_index=True).drop_duplicates(subset=["x", "y"])
    u2_final = _pd_local.concat(
        [u2_12, u2_from_12_fail], ignore_index=True).drop_duplicates(subset=["x", "y"])
    u3_final = u3_from_match.copy()

    # Apply ch2 spot filter flags to unmatched ch2
    spots_ch2.to_csv  # just accessing to keep reference; flags already in spots_ch2
    if "flagged" in spots_ch2.columns and len(u2_final) > 0:
        u2_final = apply_filters(u2_final, img_ch2, bg_ch2, psf_radius=args.psf_radius,
                                 intensity_min_mult=args.ch2_intensity_min_mult,
                                 intensity_max_mult=args.ch2_intensity_max_mult,
                                 psf_min_r2=args.psf_fit_min_r2,
                                 psf_width_tol=args.psf_fit_width_tol, run_psf_filter=False)

    print(f"    Ch3 shift      : dx={sx13:+.2f}, dy={sy13:+.2f} px  "
          f"({'applied' if shift13 else 'NOT applied'})")
    print(f"\n    [{tif_path.stem}]")
    print(f"    Ch1 spots: {len(spots_ch1)}   Ch2 spots: {len(spots_ch2)}   "
          f"Ch3 spots: {len(spots_ch3)}")
    print(f"    Triple colocalized: {len(triple)}")
    print(f"    Unmatched — Ch1: {len(u1_final)}   Ch2: {len(u2_final)}   "
          f"Ch3: {len(u3_final)}")

    stem = tif_path.stem
    out_dir.mkdir(parents=True, exist_ok=True)

    # Propagate CNN columns into triple coloc DataFrame
    if len(triple) > 0:
        for _ch, _spots, _xcol, _ycol in [
            (1, spots_ch1,       "x_ch1", "y_ch1"),
            (2, spots_ch2_corr,  "x_ch2", "y_ch2"),
            (3, spots_ch3_corr,  "x_ch3", "y_ch3"),
        ]:
            if "cnn_prob" in _spots.columns:
                triple[f"cnn_prob_ch{_ch}"]    = _propagate_cnn(triple, _spots, _xcol, _ycol, "cnn_prob",    0.0)
                triple[f"cnn_flagged_ch{_ch}"] = _propagate_cnn(triple, _spots, _xcol, _ycol, "cnn_flagged", False)

    # Save CSVs
    if len(triple) > 0:
        triple.to_csv(out_dir / f"{stem}_triple_colocalized.csv", index=False)
    else:
        pd.DataFrame().to_csv(out_dir / f"{stem}_triple_colocalized.csv", index=False)
    u1_final.to_csv(out_dir / f"{stem}_unmatched_ch1.csv", index=False)
    u2_final.to_csv(out_dir / f"{stem}_unmatched_ch2.csv", index=False)
    u3_final.to_csv(out_dir / f"{stem}_unmatched_ch3.csv", index=False)
    spots_ch1.to_csv(out_dir / f"{stem}_all_spots_ch1.csv", index=False)
    spots_ch2.to_csv(out_dir / f"{stem}_all_spots_ch2.csv", index=False)
    spots_ch3.to_csv(out_dir / f"{stem}_all_spots_ch3.csv", index=False)

    import datetime
    write_coloc_summary_three_channel(
        out_path=out_dir / f"{stem}_summary.txt", stem=stem,
        n_ch1=len(spots_ch1), n_ch2=len(spots_ch2), n_ch3=len(spots_ch3),
        n_triple=len(triple),
        n_u1=len(u1_final), n_u2=len(u2_final), n_u3=len(u3_final),
        run_date=datetime.date.today().isoformat())

    # LabView coloc files
    _triple_for_lv = triple if len(triple) > 0 else pd.DataFrame()
    write_coloc_file(out_dir / f"{stem}_ch1coloc", _triple_for_lv, u1_final,
                     "x_ch1", "y_ch1")
    write_coloc_file(out_dir / f"{stem}_ch2coloc", _triple_for_lv, u2_final,
                     "x_ch2", "y_ch2")
    # ch3coloc: write triple coloc using ch3 positions, unmatched ch3 as non-coloc
    if len(triple) > 0:
        # write_coloc_file expects coloc_x_col/coloc_y_col to exist in colocalized_spots
        write_coloc_file(out_dir / f"{stem}_ch3coloc", triple, u3_final,
                         "x_ch3", "y_ch3")
    else:
        write_coloc_file(out_dir / f"{stem}_ch3coloc", pd.DataFrame(), u3_final,
                         "x", "y")

    # Overlay PNG
    save_overlay_image_three_channel(
        out_path=out_dir / f"{stem}_overlay.png",
        img_ch1=img_ch1, img_ch2=img_ch2, img_ch3=img_ch3,
        triple=triple if len(triple) > 0 else pd.DataFrame(),
        unmatched_ch1=u1_final, unmatched_ch2=u2_final, unmatched_ch3=u3_final,
        psf_radius=args.psf_radius)

    print(f"    Output → {out_dir}/")
    return {
        "file": tif_path.name,
        "ch1_spots": len(spots_ch1),
        "ch2_spots": len(spots_ch2),
        "ch3_spots": len(spots_ch3),
        "triple_colocalized": len(triple),
        "ch1_coloc_pct": round(100 * len(triple) / len(spots_ch1), 2) if spots_ch1 is not None and len(spots_ch1) else 0,
        "ch2_coloc_pct": round(100 * len(triple) / len(spots_ch2), 2) if spots_ch2 is not None and len(spots_ch2) else 0,
        "ch3_coloc_pct": round(100 * len(triple) / len(spots_ch3), 2) if spots_ch3 is not None and len(spots_ch3) else 0,
        "shift_x_ch2": round(sx12, 3), "shift_y_ch2": round(sy12, 3),
        "shift_x_ch3": round(sx13, 3), "shift_y_ch3": round(sy13, 3),
    }



def _crop_patch(img: np.ndarray, cx: float, cy: float,
                half: int = 25) -> tuple:
    """
    Return a square crop of `img` centred on (cx, cy) and the local
    coordinates of (cx, cy) within that crop.  Pads with edge values
    if the crop extends outside the image.
    """
    h, w = img.shape
    x0 = int(round(cx)) - half
    y0 = int(round(cy)) - half
    x1 = x0 + 2 * half
    y1 = y0 + 2 * half

    pad_left  = max(0, -x0);  pad_top   = max(0, -y0)
    pad_right = max(0, x1-w); pad_bot   = max(0, y1-h)
    x0c = max(0, x0); x1c = min(w, x1)
    y0c = max(0, y0); y1c = min(h, y1)

    patch = img[y0c:y1c, x0c:x1c]
    if pad_left or pad_top or pad_right or pad_bot:
        patch = np.pad(patch, ((pad_top, pad_bot), (pad_left, pad_right)),
                       mode="edge")

    local_x = cx - x0
    local_y = cy - y0
    return patch, float(local_x), float(local_y)


def _build_review_items(coloc: pd.DataFrame,
                        u1: pd.DataFrame,
                        u2: pd.DataFrame,
                        channel: int,
                        shift_x: float = 0.0,
                        shift_y: float = 0.0) -> list:
    """
    Build the ordered list of flagged spots for review in the given channel.
    Each item is a dict with all info needed to display and update flags.
    shift_x/shift_y: the channel shift that was applied to Ch2 during colocalization.
    Used to convert stored (corrected) Ch2 coords back to image coords for display.
    """
    ch    = channel
    other = 3 - ch
    items = []

    # Colocalized spots flagged in this channel
    flag_col = f"flagged_ch{ch}"
    if flag_col in coloc.columns and len(coloc) > 0:
        for idx, row in coloc[coloc[flag_col] == True].iterrows():
            reason = []
            if row.get(f"flagged_intensity_ch{ch}", False): reason.append("intensity")
            if row.get(f"flagged_psf_ch{ch}",       False): reason.append("PSF shape")
            items.append({
                "source":   "colocalized",
                "df":       "coloc",
                "df_idx":   idx,
                "flag_col": flag_col,
                "x":        float(row[f"x_ch{ch}"]),
                "y":        float(row[f"y_ch{ch}"]),
                "img_x":    float(row[f"x_ch{ch}"]),  # coloc coords already in image space
                "img_y":    float(row[f"y_ch{ch}"]),
                "other_x":  float(row[f"x_ch{other}"]),
                "other_y":  float(row[f"y_ch{other}"]),
                "reason":   ", ".join(reason) or "flagged",
                "decision": None,
            })

    # Unmatched spots flagged in this channel
    u_df      = u1 if ch == 1 else u2
    other_u   = u2 if ch == 1 else u1
    u_label   = f"u{ch}"
    if "flagged" in u_df.columns and len(u_df) > 0:
        for idx, row in u_df[u_df["flagged"] == True].iterrows():
            # Find nearest spot in other channel for context.
            # Unmatched Ch2 coords are shift-corrected; undo the shift so the
            # marker aligns with the actual pixel position on the Ch2 image.
            other_x = other_y = None
            if len(other_u) > 0:
                coords = other_u[["x", "y"]].values.astype(float)
                _, ni  = cKDTree(coords).query([[row["x"], row["y"]]], k=1)
                raw_x = float(other_u.iloc[ni[0]]["x"])
                raw_y = float(other_u.iloc[ni[0]]["y"])
                # If reviewing Ch1 unmatched, other is Ch2 (shift was applied to Ch2)
                # If reviewing Ch2 unmatched, other is Ch1 (no shift applied)
                if ch == 1:
                    other_x = raw_x + shift_x
                    other_y = raw_y + shift_y
                else:
                    other_x = raw_x
                    other_y = raw_y
            reason = []
            if row.get("flagged_intensity", False): reason.append("intensity")
            if row.get("flagged_psf",       False): reason.append("PSF shape")
            # Ch2 unmatched coords are shift-corrected — undo shift for display
            # on the original Ch2 image. Ch1 coords need no adjustment.
            # x/y = stored coords (used for CSV index lookups — must not be modified)
            # img_x/img_y = display coords on the channel image (shift undone for Ch2)
            raw_spot_x = float(row["x"])
            raw_spot_y = float(row["y"])
            if ch == 2:
                disp_x = raw_spot_x + shift_x
                disp_y = raw_spot_y + shift_y
            else:
                disp_x = raw_spot_x
                disp_y = raw_spot_y
            items.append({
                "source":   "unmatched",
                "df":       u_label,
                "df_idx":   idx,
                "flag_col": "flagged",
                "x":        raw_spot_x,   # stored coord — do not change
                "y":        raw_spot_y,
                "img_x":    disp_x,       # pixel coord on image for crop/marker
                "img_y":    disp_y,
                "other_x":  other_x,
                "other_y":  other_y,
                "reason":   ", ".join(reason) or "flagged",
                "decision": None,
            })

    return items


def run_flagged_review(tif_path: Path,
                       coloc: pd.DataFrame,
                       u1: pd.DataFrame,
                       u2: pd.DataFrame,
                       out_dir: Path,
                       stem: str,
                       psf_radius: float,
                       ch1_ref_frame: int = CH1_REF_FRAME,
                       average_half_window: int = AVERAGE_HALF_WINDOW,
                       frames_per_channel: int = FRAMES_PER_CHANNEL,
                       crop_half: int = 25,
                       single_channel: bool = False) -> tuple:
    """
    Launch the interactive flagged-spot review UI.
    In two-channel mode: reviews ch1 then ch2 with side-by-side context panels.
    In single-channel mode: reviews ch1 only with a single panel — Good/Bad only.
    Returns updated (coloc, u1, u2) DataFrames.
    """
    from skimage.exposure import rescale_intensity as _rescale

    print(f"\n  Loading images for review...")
    with tifffile.TiffFile(tif_path) as _tif:
        _total_r = len(_tif.pages)

    if single_channel:
        _fpc_r   = _total_r
        _ch1_ref = ch1_ref_frame  # blank_frames + 1, already set correctly
        print(f"    Single-channel: {_total_r} frames  (ref: {_ch1_ref})")
        img_ch1 = load_averaged_frame(tif_path, _ch1_ref, average_half_window, _fpc_r)
        img_ch2 = img_ch1  # unused
    else:
        _fpc_r     = _total_r // 2
        _ch2_ref_r = _fpc_r + ch1_ref_frame
        print(f"    File: {_total_r} frames ({_fpc_r} per channel)")
        print(f"    Ch1 ref: frame {ch1_ref_frame}  "
              f"Ch2 ref: frame {_ch2_ref_r}  (±{average_half_window})")
        img_ch1 = load_averaged_frame(tif_path, ch1_ref_frame, average_half_window, _fpc_r)
        img_ch2 = load_averaged_frame(tif_path, _ch2_ref_r,   average_half_window, _fpc_r)

    def _stretch(img):
        p1, p99 = np.percentile(img, (1, 99))
        return _rescale(img, in_range=(p1, p99), out_range=(0.0, 1.0))

    disp1 = _stretch(img_ch1)
    disp2 = _stretch(img_ch2)

    circle_r = psf_radius * 1.5

    _dfs = {"u1": u1, "u2": u2}

    channels_to_review = [1] if single_channel else [1, 2]

    for channel in channels_to_review:
        u1 = _dfs["u1"]
        u2 = _dfs["u2"]
        ch    = channel
        other = 3 - ch
        # Extract shift from coloc CSV if available
        _shift_x = float(coloc["shift_x_applied"].iloc[0]) if len(coloc) > 0 and "shift_x_applied" in coloc.columns else 0.0
        _shift_y = float(coloc["shift_y_applied"].iloc[0]) if len(coloc) > 0 and "shift_y_applied" in coloc.columns else 0.0
        items = _build_review_items(coloc, u1, u2, ch,
                                    shift_x=_shift_x, shift_y=_shift_y)

        if not items:
            print(f"  No flagged spots in channel {ch} — skipping.")
            continue

        print(f"\n  Channel {ch}: {len(items)} flagged spots to review.")
        print(f"  [g] good  [b] bad  [s] skip  [←/→] navigate  [q] quit & save")

        img_rev   = disp1 if ch == 1 else disp2
        img_other = disp2 if ch == 1 else disp1

        state = {"idx": 0, "quit": False}

        if single_channel:
            fig, ax_rev = plt.subplots(1, 1, figsize=(6, 6))
            ax_other = ax_rev   # alias — not used
            ax_bare  = ax_rev   # alias — not used
        else:
            fig, axes = plt.subplots(1, 3, figsize=(17, 6))
            ax_rev   = axes[0]
            ax_other = axes[1]
            ax_bare  = axes[2]

        fig.subplots_adjust(bottom=0.20, top=0.87, wspace=0.08)

        title_txt    = fig.suptitle("", fontsize=10, fontweight="bold")
        decision_txt = fig.text(0.5, 0.91, "", ha="center", fontsize=12, fontweight="bold")
        status_txt   = fig.text(0.5, 0.02, "", ha="center", fontsize=9, color="#444444")

        ax_prog = fig.add_axes([0.10, 0.10, 0.80, 0.025])
        ax_prog.set_xlim(0, len(items)); ax_prog.set_ylim(0, 1)
        ax_prog.axis("off")
        prog_patch = ax_prog.barh(0, 0, height=1, color="#4dac26")
        prog_lbl   = ax_prog.text(0, -1.8, "", fontsize=8, va="top")

        def draw(i):
            item = items[i]
            ax_rev.cla()
            if not single_channel:
                ax_other.cla(); ax_bare.cla()

            crop_cx = item.get("img_x", item["x"])
            crop_cy = item.get("img_y", item["y"])
            lx_rev  = float(crop_half)
            ly_rev  = float(crop_half)

            patch_r, _, _ = _crop_patch(img_rev, crop_cx, crop_cy, crop_half)
            ax_rev.imshow(patch_r, cmap="gray", origin="upper", vmin=0, vmax=1)
            ax_rev.add_patch(mpatches.Circle(
                (lx_rev, ly_rev), radius=circle_r,
                edgecolor="#ffdd00", facecolor="none", linewidth=2.0, zorder=3))
            ax_rev.set_title(f"Channel {ch}  |  Flagged: {item['reason']}", fontsize=9)
            ax_rev.axis("off")

            if not single_channel:
                has_other = item["other_x"] is not None
                ox = item["other_x"] - (crop_cx - crop_half) if has_other else None
                oy = item["other_y"] - (crop_cy - crop_half) if has_other else None
                is_coloc  = item["source"] == "colocalized"
                rev_marker = "circle" if is_coloc else "x"

                patch_o, _, _ = _crop_patch(img_other, crop_cx, crop_cy, crop_half)
                ax_other.imshow(patch_o, cmap="gray", origin="upper", vmin=0, vmax=1)
                if rev_marker == "circle":
                    ax_other.add_patch(mpatches.Circle(
                        (lx_rev, ly_rev), radius=circle_r,
                        edgecolor="#ffdd00", facecolor="none", linewidth=2.0, zorder=3))
                else:
                    ax_other.scatter([lx_rev], [ly_rev], marker="x", s=120,
                                     c="#ffdd00", linewidths=2.0, zorder=3)
                if has_other:
                    ax_other.add_patch(mpatches.Circle(
                        (ox, oy), radius=circle_r,
                        edgecolor="#00aaff", facecolor="none", linewidth=2.0, zorder=3))
                    ax_other.set_title(
                        f"Channel {other}  ({'paired' if is_coloc else 'nearest'} spot)",
                        fontsize=9)
                else:
                    ax_other.set_title(f"Channel {other}  (no nearby spot)", fontsize=9)
                ax_other.axis("off")

                if not is_coloc:
                    ax_bare.imshow(patch_r, cmap="gray", origin="upper", vmin=0, vmax=1)
                    ax_bare.set_title(f"Channel {ch}  (no markers)", fontsize=9)
                    ax_bare.axis("off")
                    ax_bare.set_visible(True)
                else:
                    ax_bare.set_visible(False)

            dec = item["decision"]
            if dec == "good":
                decision_txt.set_text("✔ GOOD");            decision_txt.set_color("#009900")
            elif dec == "not_colocalized":
                decision_txt.set_text("↔ NOT COLOCALIZED"); decision_txt.set_color("#cc6600")
            elif dec == "not_real":
                decision_txt.set_text("✘ NOT A REAL SPOT"); decision_txt.set_color("#cc0000")
            else:
                decision_txt.set_text("")

            n_done    = sum(1 for it in items if it["decision"] is not None)
            n_good    = sum(1 for it in items if it["decision"] == "good")
            n_notcoloc= sum(1 for it in items if it["decision"] == "not_colocalized")
            n_notreal = sum(1 for it in items if it["decision"] == "not_real")
            n_decided = n_good + n_notcoloc + n_notreal

            prog_patch[0].set_width(i + 1)
            prog_lbl.set_text(f"  {i+1}/{len(items)}  ({n_done} decided)")
            prog_lbl.set_position((i + 1, -1.8))

            if single_channel:
                keys_hint = "[g] good (keep)  [n] not a real spot (remove)  [←/→] navigate  [q] save & close"
                title_txt.set_text(
                    f"Flagged spot review  —  {i+1} of {len(items)}  |  {keys_hint}")
                status_txt.set_text(
                    f"Good: {n_good}   Removed: {n_notreal}   Skipped: {len(items) - n_decided}")
            else:
                has_coloc_items = any(it["source"] == "colocalized" for it in items)
                if has_coloc_items:
                    keys_hint = "[g] good  [b] not colocalized  [n] not a real spot  [s] skip  [←/→] navigate  [q] save & close"
                else:
                    keys_hint = "[g] good  [n] not a real spot  [s] skip  [←/→] navigate  [q] save & close"
                title_txt.set_text(
                    f"Ch{ch} flagged review  —  {i+1} of {len(items)}  |  {keys_hint}")
                status_txt.set_text(
                    f"Good: {n_good}   Not colocalized: {n_notcoloc}   Not real: {n_notreal}   "
                    f"Skipped: {len(items) - n_decided}")
            fig.canvas.draw_idle()


        def on_key(event):
            i = state["idx"]
            if event.key == "g":
                items[i]["decision"] = "good"
                _advance(1)
                _check_all_decided()
            elif event.key == "b":
                # Only meaningful for colocalized spots; ignored for unmatched
                if items[i]["source"] == "colocalized":
                    items[i]["decision"] = "not_colocalized"
                    _advance(1)
                    _check_all_decided()
            elif event.key == "n":
                items[i]["decision"] = "not_real"
                _advance(1)
                _check_all_decided()
            elif event.key == "s":
                _advance(1)
            elif event.key == "right":
                _advance(1)
            elif event.key == "left":
                _advance(-1)
            elif event.key == "q":
                state["quit"] = True
                plt.close(fig)

        def _check_all_decided():
            """If every item has a decision, close the review window automatically."""
            if all(it["decision"] is not None for it in items):
                state["quit"] = True
                plt.close(fig)

        def _advance(d):
            ni = state["idx"] + d
            if 0 <= ni < len(items):
                state["idx"] = ni
                draw(state["idx"])
            elif d > 0:
                # Tried to go past the end — check if we're done
                _check_all_decided()

        fig.canvas.mpl_connect("key_press_event", on_key)
        draw(0)
        plt.show()

        # Apply decisions:
        #   good           → clear the flag, keep as colocalized / unmatched
        #   not_colocalized→ both spots are real but not a pair; both go to unmatched
        #   not_real       → this channel's spot is fake; rescue other to unmatched
        #   skip           → leave unchanged
        n_good     = sum(1 for it in items if it["decision"] == "good")
        n_notcoloc = sum(1 for it in items if it["decision"] == "not_colocalized")
        n_notreal  = sum(1 for it in items if it["decision"] == "not_real")
        print(f"  Ch{ch} review done — Good: {n_good}  Not colocalized: {n_notcoloc}  "
              f"Not real: {n_notreal}  Skipped: {len(items) - n_good - n_notcoloc - n_notreal}")

        bad_coloc_idx   = []
        rescued_to_u1   = []
        rescued_to_u2   = []
        bad_u1_idx      = []
        bad_u2_idx      = []

        for item in items:
            if item["decision"] == "good":
                # Clear the flag — spot confirmed real
                if item["df"] == "coloc":
                    coloc.at[item["df_idx"], item["flag_col"]] = False
                elif item["df"] == "u1":
                    u1.at[item["df_idx"], "flagged"] = False
                elif item["df"] == "u2":
                    u2.at[item["df_idx"], "flagged"] = False

            elif item["decision"] == "not_colocalized" and item["df"] == "coloc":
                # Both spots are real — rescue both to their respective unmatched lists
                row = coloc.loc[item["df_idx"]]
                rescued_to_u1.append({"x": float(row["x_ch1"]),
                                       "y": float(row["y_ch1"]), "flagged": False})
                rescued_to_u2.append({"x": float(row["x_ch2"]),
                                       "y": float(row["y_ch2"]), "flagged": False})
                bad_coloc_idx.append(item["df_idx"])

            elif item["decision"] == "not_real":
                if item["df"] == "coloc":
                    if single_channel:
                        # Single-channel: spot is fake — just remove it entirely
                        bad_coloc_idx.append(item["df_idx"])
                    else:
                        # Two-channel: this channel's spot is fake — rescue the other channel's spot
                        row = coloc.loc[item["df_idx"]]
                        other = 3 - ch
                        rescued = {"x": float(row[f"x_ch{other}"]),
                                   "y": float(row[f"y_ch{other}"]),
                                   "flagged": False}
                        if other == 1:
                            rescued_to_u1.append(rescued)
                        else:
                            rescued_to_u2.append(rescued)
                        bad_coloc_idx.append(item["df_idx"])
                elif item["df"] == "u1":
                    bad_u1_idx.append(item["df_idx"])
                elif item["df"] == "u2":
                    bad_u2_idx.append(item["df_idx"])

        # Remove bad coloc rows and append rescued spots to unmatched lists.
        import pandas as _pd
        if bad_coloc_idx:
            coloc.drop(index=bad_coloc_idx, inplace=True)
            coloc.reset_index(drop=True, inplace=True)
        if rescued_to_u1:
            _dfs["u1"] = _pd.concat([_dfs["u1"], _pd.DataFrame(rescued_to_u1)],
                                     ignore_index=True)
        if rescued_to_u2:
            _dfs["u2"] = _pd.concat([_dfs["u2"], _pd.DataFrame(rescued_to_u2)],
                                     ignore_index=True)
        if bad_u1_idx:
            _dfs["u1"].drop(index=bad_u1_idx, inplace=True)
            _dfs["u1"].reset_index(drop=True, inplace=True)
        if bad_u2_idx:
            _dfs["u2"].drop(index=bad_u2_idx, inplace=True)
            _dfs["u2"].reset_index(drop=True, inplace=True)

    # Re-save CSVs with updated flags
    u1 = _dfs["u1"]
    u2 = _dfs["u2"]
    coloc.to_csv(out_dir / f"{stem}_colocalized.csv",    index=False)
    u1.to_csv(   out_dir / f"{stem}_unmatched_ch1.csv",  index=False)
    u2.to_csv(   out_dir / f"{stem}_unmatched_ch2.csv",  index=False)
    print(f"  Updated CSVs saved.")

    # Regenerate overlay PNG with corrected flags
    if single_channel:
        # Rebuild spots_ch1 from the coloc CSV (ch1 columns = the real spots)
        _sc_spots = coloc[["x_ch1", "y_ch1"]].rename(
            columns={"x_ch1": "x", "y_ch1": "y"}).copy()
        if "flagged_ch1" in coloc.columns:
            _sc_spots["flagged"] = coloc["flagged_ch1"].values
        else:
            _sc_spots["flagged"] = False
        save_overlay_image_single_channel(
            out_path=out_dir / f"{stem}_overlay.png",
            img_ch1=img_ch1,
            spots=_sc_spots,
            psf_radius=psf_radius,
        )
    else:
        save_overlay_image(
            out_path=out_dir / f"{stem}_overlay.png",
            img_ch1=img_ch1, img_ch2=img_ch2,
            coloc=coloc, unmatched_ch1=u1, unmatched_ch2=u2,
            psf_radius=psf_radius,
        )
        save_coloc_only_image(
            out_path=out_dir / f"{stem}_overlay_coloc_only.png",
            img_ch1=img_ch1, img_ch2=img_ch2,
            coloc=coloc,
            psf_radius=psf_radius,
        )
    print(f"  Overlay PNG regenerated.")

    return coloc, u1, u2


def run_tune_mode(tif_path: Path, args):
    """
    Interactive parameter tuning UI.
    Shows averaged frames with detected spots overlaid.
    In single-channel mode shows ch1 only (one panel).
    In three-channel mode shows ch1 + ch2 + ch3 (three panels).
    Type values into the text boxes and press Enter to update detection.
    Press 'Print parameters' to output the current values to the terminal.
    """
    is_single = getattr(args, 'single_channel', False)
    is_three  = getattr(args, 'three_channel', False) and not is_single
    print(f"\n  Loading frames for tuning: {tif_path.name}")
    print(f"  Type a value and press Enter to update detection.")
    with tifffile.TiffFile(tif_path) as _tif:
        _total = len(_tif.pages)

    if is_single:
        _fpc = _total
        n_blank = getattr(args, 'blank_frames', 10)
        _ch1_ref = n_blank + 1
        print(f"  Single-channel mode — {_total} frames total  (ref: {_ch1_ref})")
    elif is_three:
        _fpc = _total // 3
        _ch1_ref = args.ch1_ref_frame
        _ch2_ref = _fpc + args.ch1_ref_frame
        _ch3_ref = 2 * _fpc + args.ch1_ref_frame
        print(f"  Three-channel mode — {_total} frames ({_fpc} per channel)  "
              f"(refs: ch1={_ch1_ref}, ch2={_ch2_ref}, ch3={_ch3_ref})")
    else:
        _fpc = _total // 2
        _ch1_ref = args.ch1_ref_frame
        _ch2_ref = _fpc + args.ch1_ref_frame
        print(f"  File has {_total} frames — {_fpc} per channel  "
              f"(ch1 ref: {_ch1_ref}, ch2 ref: {_ch2_ref})")

    img_ch1 = load_averaged_frame(
        tif_path,
        ref_frame=_ch1_ref,
        half_window=args.average_half_window,
        frames_per_channel=_fpc,
    )
    if not is_single:
        img_ch2 = load_averaged_frame(
            tif_path,
            ref_frame=_ch2_ref,
            half_window=args.average_half_window,
            frames_per_channel=_fpc,
        )
        if is_three:
            img_ch3 = load_averaged_frame(
                tif_path,
                ref_frame=_ch3_ref,
                half_window=args.average_half_window,
                frames_per_channel=_fpc,
            )
    else:
        img_ch2 = img_ch1  # unused

    from skimage.exposure import rescale_intensity

    p1_1, p99_1 = np.percentile(img_ch1, (1, 99))
    disp_ch1 = rescale_intensity(img_ch1, in_range=(p1_1, p99_1), out_range=(0.0, 1.0))
    if not is_single:
        p1_2, p99_2 = np.percentile(img_ch2, (1, 99))
        disp_ch2 = rescale_intensity(img_ch2, in_range=(p1_2, p99_2), out_range=(0.0, 1.0))
        if is_three:
            p1_3, p99_3 = np.percentile(img_ch3, (1, 99))
            disp_ch3 = rescale_intensity(img_ch3, in_range=(p1_3, p99_3), out_range=(0.0, 1.0))
    else:
        disp_ch2 = disp_ch1

    init_thr1 = args.ch1_threshold if args.ch1_threshold is not None else args.detection_threshold
    init_thr2 = args.ch2_threshold if args.ch2_threshold is not None else args.detection_threshold

    params = {
        "psf":         args.psf_radius,
        "thr1":        init_thr1,
        "thr2":        init_thr2,
        "coloc":       args.coloc_threshold,
        "ch1_int_min": args.ch1_intensity_min_mult,
        "ch1_int_max": args.ch1_intensity_max_mult,
        "ch2_int_min": args.ch2_intensity_min_mult,
        "ch2_int_max": args.ch2_intensity_max_mult,
    }

    # --- Build figure layout ---
    # Images occupy top 74% of figure; controls in bottom 20%
    fig = plt.figure(figsize=(10, 10) if is_single else (16, 10))
    fig.canvas.manager.set_window_title(
        f"Tuning ({'single-channel'if is_single else 'two-channel'}) — {tif_path.name}")

    if is_single:
        ax1 = fig.add_axes([0.02, 0.24, 0.96, 0.72])
        ax2 = ax1  # alias — unused for drawing
    else:
        ax1 = fig.add_axes([0.02, 0.24, 0.46, 0.72])
        ax2 = fig.add_axes([0.52, 0.24, 0.46, 0.72])
        ax2.imshow(disp_ch2, cmap="gray", origin="upper")
        ax2.axis("off")

    ax1.imshow(disp_ch1, cmap="gray", origin="upper")
    ax1.axis("off")

    # Marker state — replaced on each redraw
    state = {
        "sc_unmat1_ok": None, "sc_unmat1_flag": None,
        "sc_unmat2_ok": None, "sc_unmat2_flag": None,
        "circles1": [], "circles2": [],
    }

    # Background from blank frames at channel start
    n_blank = getattr(args, 'blank_frames', BLANK_FRAMES)
    with tifffile.TiffFile(tif_path) as _tif:
        _total_frames = len(_tif.pages)

    def _tune_blank_bg(start_1idx, n):
        with tifffile.TiffFile(tif_path) as _tif:
            frames = np.stack(
                [_tif.pages[start_1idx - 1 + i].asarray().astype(np.float32)
                 for i in range(min(n, _total_frames - start_1idx + 1))],
                axis=0)
        return float(np.median(frames))

    bg_ch1 = _tune_blank_bg(1, n_blank)
    if is_single:
        bg_ch2 = bg_ch1
        print(f"  Tune background — Ch1: {bg_ch1:.1f}")
    else:
        _fpc_bg = _total_frames // 2
        bg_ch2  = _tune_blank_bg(_fpc_bg + 1, n_blank)
        print(f"  Tune background — Ch1: {bg_ch1:.1f}   Ch2: {bg_ch2:.1f}")

    # Load CNN model once for the tune session
    if not getattr(args, 'no_cnn', False):
        _tune_cnn_model, _tune_cnn_thresh = load_spot_classifier()
    else:
        _tune_cnn_model, _tune_cnn_thresh = None, None

    # --- Text box layout ---
    row1_y = 0.13; row2_y = 0.05; box_h = 0.045
    lbl1_y = 0.185; lbl2_y = 0.105

    if is_single:
        # Single-channel: PSF radius | threshold | int min | int max
        row1_boxes = [
            ("PSF radius (px)",      "psf",         0.04, 0.18, str(params["psf"])),
            ("Threshold",            "thr1",         0.26, 0.18, str(params["thr1"])),
            ("Int. min ×bg",         "ch1_int_min",  0.48, 0.18, str(params["ch1_int_min"])),
            ("Int. max ×bg",         "ch1_int_max",  0.70, 0.18, str(params["ch1_int_max"])),
        ]
        row2_boxes = []
    else:
        row1_boxes = [
            ("PSF radius (px)",       "psf",   0.04, 0.13, str(params["psf"])),
            ("Ch1 threshold",         "thr1",  0.21, 0.13, str(params["thr1"])),
            ("Ch2 threshold",         "thr2",  0.38, 0.13, str(params["thr2"])),
            ("Coloc threshold (px)",  "coloc", 0.55, 0.13, str(params["coloc"])),
        ]
        row2_boxes = [
            ("Ch1 int. min ×bg",  "ch1_int_min", 0.04, 0.13, str(params["ch1_int_min"])),
            ("Ch1 int. max ×bg",  "ch1_int_max", 0.21, 0.13, str(params["ch1_int_max"])),
            ("Ch2 int. min ×bg",  "ch2_int_min", 0.38, 0.13, str(params["ch2_int_min"])),
            ("Ch2 int. max ×bg",  "ch2_int_max", 0.55, 0.13, str(params["ch2_int_max"])),
        ]

    text_boxes = {}
    feedback_texts = {}

    for label, key, left, width, init_val in row1_boxes:
        fig.text(left + width / 2, lbl1_y, label,
                 ha="center", va="bottom", fontsize=8, fontweight="bold")
        ax_box = fig.add_axes([left, row1_y, width, box_h])
        tb = TextBox(ax_box, "", initial=init_val)
        tb.text_disp.set_fontsize(10)
        text_boxes[key] = tb
        ft = fig.text(left + width / 2, row1_y - 0.025, "",
                      ha="center", va="top", fontsize=7, color="red")
        feedback_texts[key] = ft

    if row2_boxes:
        fig.text(0.04, lbl2_y, "Intensity filter (× background):",
                 ha="left", va="bottom", fontsize=8, color="#555555", style="italic")
        for label, key, left, width, init_val in row2_boxes:
            fig.text(left + width / 2, lbl2_y, label,
                     ha="center", va="bottom", fontsize=8, fontweight="bold")
            ax_box = fig.add_axes([left, row2_y, width, box_h])
            tb = TextBox(ax_box, "", initial=init_val)
            tb.text_disp.set_fontsize(10)
            text_boxes[key] = tb
            ft = fig.text(left + width / 2, row2_y - 0.025, "",
                          ha="center", va="top", fontsize=7, color="red")
            feedback_texts[key] = ft

    # Buttons: Done | PSF filter toggle | Print
    ax_btn_done  = fig.add_axes([0.88, row1_y, 0.10, box_h])
    ax_btn_psf   = fig.add_axes([0.88, (row1_y + row2_y) / 2 + box_h * 0.1, 0.10, box_h])
    ax_btn_print = fig.add_axes([0.88, row2_y, 0.10, box_h])
    btn_done  = MplButton(ax_btn_done,  "Done ✓",         color="#c8f0c8", hovercolor="#90e090")
    btn_psf   = MplButton(ax_btn_psf,   "PSF filter: OFF", color="#f0f0f0", hovercolor="#e0e0ff")
    btn_print = MplButton(ax_btn_print, "Print params")

    psf_filter_state = {"enabled": False}

    def toggle_psf_filter(_):
        psf_filter_state["enabled"] = not psf_filter_state["enabled"]
        btn_psf.label.set_text("PSF filter: ON" if psf_filter_state["enabled"] else "PSF filter: OFF")
        btn_psf.ax.set_facecolor("#c8c8f0" if psf_filter_state["enabled"] else "#f0f0f0")
        redraw()

    btn_psf.on_clicked(toggle_psf_filter)

    def redraw():
        psf   = params["psf"]
        thr1  = params["thr1"]
        _run_psf = psf_filter_state["enabled"]
        if _run_psf:
            print("  Running PSF shape filter (may take a moment)...")

        spots1 = detect_spots(img_ch1, psf, thr1, args.exclude_border)
        spots1 = apply_filters(
            spots1, img_ch1, bg_ch1, psf_radius=psf,
            intensity_min_mult=params["ch1_int_min"],
            intensity_max_mult=params["ch1_int_max"],
            psf_min_r2=args.psf_fit_min_r2,
            psf_width_tol=args.psf_fit_width_tol,
            run_psf_filter=_run_psf,
        )

        if not is_single:
            thr2   = params["thr2"]
            spots2 = detect_spots(img_ch2, psf, thr2, args.exclude_border)
            spots2 = apply_filters(
                spots2, img_ch2, bg_ch2, psf_radius=psf,
                intensity_min_mult=params["ch2_int_min"],
                intensity_max_mult=params["ch2_int_max"],
                psf_min_r2=args.psf_fit_min_r2,
                psf_width_tol=args.psf_fit_width_tol,
                run_psf_filter=_run_psf,
            )
            coloc, u1, u2 = colocalize(spots1, spots2, params["coloc"])
            if len(coloc) > 0:
                coloc["flagged_ch1"] = _propagate_flags(coloc, spots1, "x_ch1", "y_ch1")
                coloc["flagged_ch2"] = _propagate_flags(coloc, spots2, "x_ch2", "y_ch2")
        else:
            spots2 = spots1  # unused
            coloc = pd.DataFrame()
            u1 = spots1  # all spots are "unmatched" — no colocalization concept
            u2 = pd.DataFrame()



        # Clear previous markers
        for c in state["circles1"] + state["circles2"]:
            c.remove()
        state["circles1"].clear()
        state["circles2"].clear()
        for sc_key in ["sc_unmat1_ok", "sc_unmat1_flag", "sc_unmat2_ok", "sc_unmat2_flag"]:
            if state[sc_key]:
                state[sc_key].remove()
                state[sc_key] = None

        circle_r = psf * 1.5

        if is_single:
            # Single-channel: draw all spots as circles (green ok, yellow flagged)
            # No X markers — there is no "unmatched" concept in single-channel mode
            has_flag = "flagged" in spots1.columns
            for _, row in spots1.iterrows():
                flagged = has_flag and row["flagged"]
                color = "#ffdd00" if flagged else "#00ff00"
                c = mpatches.Circle((row["x"], row["y"]), radius=circle_r,
                                    edgecolor=color, facecolor="none",
                                    linewidth=1.2, zorder=3)
                ax1.add_patch(c)
                state["circles1"].append(c)

            n_flag1 = int(spots1["flagged"].sum()) if has_flag else 0
            ax1.set_title(
                f"Channel 1 (single)  |  {len(spots1)} spots  —  "
                f"{len(spots1) - n_flag1} passed ●  {n_flag1} flagged",
                fontsize=9)
        else:
            # Two-channel: colocalized circles + unmatched X markers on both axes
            for ax, cx_col, cy_col, flag_col, circ_key in [
                (ax1, "x_ch1", "y_ch1", "flagged_ch1", "circles1"),
                (ax2, "x_ch2", "y_ch2", "flagged_ch2", "circles2"),
            ]:
                if len(coloc) > 0:
                    has_flag = flag_col in coloc.columns
                    for _, row in coloc.iterrows():
                        flagged = has_flag and row[flag_col]
                        color = "#ffdd00" if flagged else "#00ff00"
                        c = mpatches.Circle((row[cx_col], row[cy_col]), radius=circle_r,
                                            edgecolor=color, facecolor="none",
                                            linewidth=1.2, zorder=3)
                        ax.add_patch(c)
                        state[circ_key].append(c)

            for ax, df, ok_key, flag_key in [
                (ax1, u1, "sc_unmat1_ok", "sc_unmat1_flag"),
                (ax2, u2, "sc_unmat2_ok", "sc_unmat2_flag"),
            ]:
                has_flag  = "flagged" in df.columns
                flag_vals = df["flagged"].values if has_flag else np.zeros(len(df), dtype=bool)
                ok_mask   = ~flag_vals
                flag_mask = flag_vals
                xs = df["x"].values; ys = df["y"].values
                state[ok_key]   = ax.scatter(
                    xs[ok_mask],   ys[ok_mask],
                    marker="x", s=50, c="#ff3333", linewidths=1.0, zorder=3)
                state[flag_key] = ax.scatter(
                    xs[flag_mask], ys[flag_mask],
                    marker="x", s=50, c="#ff8800", linewidths=1.0, zorder=3)

            n_flag1 = int(spots1["flagged"].sum()) if "flagged" in spots1.columns else 0
            n_flag2 = int(spots2["flagged"].sum()) if "flagged" in spots2.columns else 0

            ax1.set_title(
                f"Channel 1  |  {len(spots1)} spots ({n_flag1} flagged ◔)  |  "
                f"{len(coloc)} colocalized   {len(u1)} unmatched", fontsize=9)
            ax2.set_title(
                f"Channel 2  |  {len(spots2)} spots ({n_flag2} flagged ◔)  |  "
                f"{len(coloc)} colocalized   {len(u2)} unmatched", fontsize=9)

        fig.canvas.draw_idle()

    def make_submit(key, min_val, max_val, is_int=False):
        """Return a submit callback for a given parameter key."""
        def on_submit(text):
            feedback_texts[key].set_text("")
            try:
                val = int(text) if is_int else float(text)
                if val <= 0:
                    raise ValueError("must be > 0")
                params[key] = val
                redraw()
            except ValueError as e:
                feedback_texts[key].set_text(f"Invalid: {e}")
                fig.canvas.draw_idle()
        return on_submit

    text_boxes["psf"].on_submit(make_submit("psf",             0.1,  20.0))
    text_boxes["thr1"].on_submit(make_submit("thr1",           0.0,  1.0))
    text_boxes["ch1_int_min"].on_submit(make_submit("ch1_int_min", 0.0, 1000.0))
    text_boxes["ch1_int_max"].on_submit(make_submit("ch1_int_max", 0.0, 1000.0))
    if not is_single:
        text_boxes["thr2"].on_submit(make_submit("thr2",       0.0,  1.0))
        text_boxes["coloc"].on_submit(make_submit("coloc",     0.1,  100.0))
        text_boxes["ch2_int_min"].on_submit(make_submit("ch2_int_min", 0.0, 1000.0))
        text_boxes["ch2_int_max"].on_submit(make_submit("ch2_int_max", 0.0, 1000.0))

    def print_params(_):
        print(f"\n  === Tuned parameters ===")
        print(f"  PSF_RADIUS_PX           = {params['psf']}")
        print(f"  CH1_THRESHOLD           = {params['thr1']}")
        print(f"  CH2_THRESHOLD           = {params['thr2']}")
        print(f"  COLOC_THRESHOLD_PX      = {params['coloc']}")
        print(f"  CH1_INTENSITY_MIN_MULT  = {params['ch1_int_min']}")
        print(f"  CH1_INTENSITY_MAX_MULT  = {params['ch1_int_max']}")
        print(f"  CH2_INTENSITY_MIN_MULT  = {params['ch2_int_min']}")
        print(f"  CH2_INTENSITY_MAX_MULT  = {params['ch2_int_max']}")

    def done_clicked(_):
        """Save tuned params and close all tune windows."""
        import json as _json
        out = {
            "psf_radius":              params["psf"],
            "ch1_threshold":           params["thr1"],
            "ch2_threshold":           params["thr2"],
            "coloc_threshold":         params["coloc"],
            "ch1_intensity_min_mult":  params["ch1_int_min"],
            "ch1_intensity_max_mult":  params["ch1_int_max"],
            "ch2_intensity_min_mult":  params["ch2_int_min"],
            "ch2_intensity_max_mult":  params["ch2_int_max"],
        }
        params_out = getattr(args, "params_out", None)
        if params_out:
            with open(params_out, "w") as _f:
                _json.dump(out, _f, indent=2)
            print(f"\n  Tuned parameters saved -> {params_out}")
        print_params(None)
        plt.close("all")

    btn_print.on_clicked(print_params)
    btn_done.on_clicked(done_clicked)

    fig.text(0.5, 0.98,
             f"Tuning mode — {tif_path.name}  |  "
             f"Type a value and press Enter to update  |  Click 'Done' when finished",
             ha="center", va="top", fontsize=9, color="#555555")

    redraw()
    plt.show()


def build_parser():
    p = argparse.ArgumentParser(
        description="Single-molecule spot colocalization from concatenated .tif movies.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )

    p.add_argument("--mode", choices=["single", "single-channel", "batch", "tune"], required=True,
                   help="'single' for one .tif, 'batch' for a whole folder, "
                        "'tune' for interactive parameter tuning on a single .tif.")
    p.add_argument("--input",  help="[single] Path to the .tif file.")
    p.add_argument("--folder", help="[batch]  Folder containing .tif files.")
    p.add_argument("--output", default=None,
                   help="Output directory. Default: 'colocalization_output' next to input.")
    p.add_argument("--single-channel", action="store_true", default=False,
                   help="[tune/single-channel] Treat TIF as single-channel — "
                        "load ch1 only, show one panel in tune UI.")

    # Frame layout
    p.add_argument("--frames-per-channel", type=int, default=FRAMES_PER_CHANNEL,
                   help="Ignored — frames per channel is now derived from the "
                        ".tif file at runtime. Kept for backwards compatibility.")
    p.add_argument("--ch1-ref-frame",      type=int, default=CH1_REF_FRAME,
                   help=f"Offset (frames) from channel start for averaging reference. "
                        f"Applied to both channels. Default: {CH1_REF_FRAME}. "
                        f"Ch2 ref frame = frames_per_channel + this value.")
    p.add_argument("--average-half-window", type=int, default=AVERAGE_HALF_WINDOW,
                   help=f"Average ± this many frames around the reference. Default: {AVERAGE_HALF_WINDOW} (3-frame average)")

    # Detection
    p.add_argument("--psf-radius",          type=float, default=PSF_RADIUS_PX,
                   help=f"PSF radius in pixels. Default: {PSF_RADIUS_PX}")
    p.add_argument("--detection-threshold", type=float, default=DETECTION_THRESHOLD,
                   help=f"Detection sensitivity (0–1). Lower = more spots. Default: {DETECTION_THRESHOLD}")
    p.add_argument("--ch1-threshold",       type=float, default=CH1_THRESHOLD,
                   help="Override detection threshold for ch1 only.")
    p.add_argument("--ch2-threshold",       type=float, default=CH2_THRESHOLD,
                   help="Override detection threshold for ch2 only.")
    p.add_argument("--exclude-border",      type=int,   default=EXCLUDE_BORDER_PX,
                   help=f"Ignore spots within this many px of the edge. Default: {EXCLUDE_BORDER_PX}")

    # Colocalization
    p.add_argument("--coloc-threshold",     type=float, default=COLOC_THRESHOLD_PX,
                   help=f"Max centre-to-centre distance (px) to call a colocalization. "
                        f"Default: {COLOC_THRESHOLD_PX} (= 2x PSF_RADIUS_PX; spots colocalized if PSFs overlap)")

    # Channel shift correction
    p.add_argument("--shift-first-pass-factor", type=float, default=SHIFT_FIRST_PASS_FACTOR,
                   help=f"First-pass threshold multiplier for shift estimation. Default: {SHIFT_FIRST_PASS_FACTOR}")
    p.add_argument("--shift-min-pairs",         type=int,   default=SHIFT_MIN_PAIRS,
                   help=f"Min matched pairs required to apply shift correction. Default: {SHIFT_MIN_PAIRS}")
    p.add_argument("--shift-max-px",            type=float, default=SHIFT_MAX_PX,
                   help=f"Max plausible shift (px); larger estimates are ignored. Default: {SHIFT_MAX_PX}")

    # Flagged spot review (single mode only)
    p.add_argument("--review-flagged", action="store_true",
                   help="After processing, launch interactive review of flagged spots.")
    p.add_argument("--review-crop-size", type=int, default=25,
                   help="Half-size (px) of the zoom crop in the review UI. Default: 25")

    # False positive filtering
    p.add_argument("--blank-frames", type=int, default=BLANK_FRAMES,
                   help=f"Number of blank (laser-off) frames at start of each channel "
                        f"used for background estimation. Default: {BLANK_FRAMES}")
    p.add_argument("--ch1-intensity-min-mult", type=float, default=CH1_INTENSITY_MIN_MULT,
                   help=f"Ch1 spot must be > this × background. Default: {CH1_INTENSITY_MIN_MULT}")
    p.add_argument("--ch1-intensity-max-mult", type=float, default=CH1_INTENSITY_MAX_MULT,
                   help=f"Ch1 spot must be < this × background. Default: {CH1_INTENSITY_MAX_MULT}")
    p.add_argument("--ch2-intensity-min-mult", type=float, default=CH2_INTENSITY_MIN_MULT,
                   help=f"Ch2 spot must be > this × background. Default: {CH2_INTENSITY_MIN_MULT}")
    p.add_argument("--ch2-intensity-max-mult", type=float, default=CH2_INTENSITY_MAX_MULT,
                   help=f"Ch2 spot must be < this × background. Default: {CH2_INTENSITY_MAX_MULT}")

    # Three-channel mode
    p.add_argument("--three-channel", action="store_true",
                   help="Three-channel mode: TIF has 3×N frames (ch1|ch2|ch3).")
    p.add_argument("--ch3-threshold",       type=float, default=None,
                   help="Override detection threshold for ch3 only.")
    p.add_argument("--ch3-intensity-min-mult", type=float, default=0.0,
                   help="Ch3 spot must be > this × background. Default: 0.0")
    p.add_argument("--ch3-intensity-max-mult", type=float, default=100.0,
                   help="Ch3 spot must be < this × background. Default: 100.0")
    p.add_argument("--coloc-threshold-ch3", type=float, default=5.0,
                   help="Max centre-to-centre distance (px) for ch1-ch3 colocalization. Default: 5.0")

    p.add_argument("--no-cnn", action="store_true", default=False,
                   help="Disable CNN spot classifier even if model file exists.")

    p.add_argument("--params-out", default=None,
                   help="[tune] Path to write tuned parameters as JSON when Done is clicked.")
    p.add_argument("--psf-fit-min-r2",    type=float, default=PSF_FIT_MIN_R2,
                   help=f"Min Gaussian fit R² to pass PSF filter. Default: {PSF_FIT_MIN_R2}")
    p.add_argument("--psf-fit-width-tol", type=float, default=PSF_FIT_WIDTH_TOL,
                   help=f"Max fractional deviation of fitted sigma from PSF radius. Default: {PSF_FIT_WIDTH_TOL}")

    return p



def save_overlay_image_single_channel(out_path: Path,
                                       img_ch1: np.ndarray,
                                       spots: pd.DataFrame,
                                       psf_radius: float):
    """
    Save a single-panel PNG showing the ch1 averaged frame with all detected
    spots marked:
      - Green circle : spot, passed filters
      - Yellow circle: spot, flagged by intensity filter
    """
    from skimage.exposure import rescale_intensity

    fig, ax = plt.subplots(1, 1, figsize=(8, 8))
    p1, p99 = np.percentile(img_ch1, (1, 99))
    ax.imshow(rescale_intensity(img_ch1, in_range=(p1, p99), out_range=(0.0, 1.0)),
              cmap="gray", origin="upper")

    circle_r = psf_radius * 1.5
    n_ok = n_flag = 0
    for _, row in spots.iterrows():
        flagged = bool(row.get("flagged", False))
        color   = "#ffdd00" if flagged else "#00ff00"
        ax.add_patch(mpatches.Circle(
            (row["x"], row["y"]), radius=circle_r,
            edgecolor=color, facecolor="none", linewidth=1.2, zorder=3))
        if flagged: n_flag += 1
        else:       n_ok   += 1

    ax.set_title(
        f"Channel 1  |  {len(spots)} spots total  —  "
        f"{n_ok} passed  ●  {n_flag} flagged",
        fontsize=10)
    ax.axis("off")

    legend_elements = [
        mpatches.Patch(edgecolor="#00ff00", facecolor="none",
                       linewidth=1.5, label="Spot (passed)"),
        mpatches.Patch(edgecolor="#ffdd00", facecolor="none",
                       linewidth=1.5, label="Spot (flagged)"),
    ]
    fig.legend(handles=legend_elements, loc="lower center", ncol=2,
               fontsize=9, frameon=True)
    fig.tight_layout(rect=[0, 0.04, 1, 1])
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close(fig)


def process_file_single_channel(tif_path: Path, args, out_dir: Path) -> dict | None:
    """
    Single-channel spot detection pipeline for stoichiometry experiments.
    Detects and filters spots in ch1 only, then writes output in the same
    format as process_file() so bleaching_analysis.py can consume it unchanged.
    All spots are written as "colocalized" with identical ch1/ch2 positions.
    """
    print(f"\n  Processing (single-channel): {tif_path.name}")

    with tifffile.TiffFile(tif_path) as tif:
        total_frames = len(tif.pages)

    n_blank   = getattr(args, 'blank_frames', 10)
    ref_frame = n_blank + 1

    try:
        img_ch1 = load_averaged_frame(
            tif_path,
            ref_frame=ref_frame,
            half_window=args.average_half_window,
            frames_per_channel=total_frames,
        )
    except Exception as e:
        print(f"    [ERROR] Could not load frame: {e}")
        return None

    thr1 = args.ch1_threshold if args.ch1_threshold is not None else args.detection_threshold
    spots_ch1 = detect_spots(img_ch1, args.psf_radius, thr1, args.exclude_border)

    # Background from blank frames
    with tifffile.TiffFile(tif_path) as tif:
        blank_frames_arr = np.stack(
            [tif.pages[i].asarray().astype(np.float32)
             for i in range(min(n_blank, total_frames))],
            axis=0)
    bg_ch1 = float(np.median(blank_frames_arr))
    print(f"    Background — Ch1: {bg_ch1:.1f}")

    # Intensity filter only — PSF filter is too aggressive for single-channel
    # stoichiometry experiments where spot density and SNR vary widely.
    print(f"    Running intensity filter (PSF shape filter disabled for single-channel)...")
    spots_ch1 = apply_filters(
        spots_ch1, img_ch1, bg_ch1,
        psf_radius=args.psf_radius,
        intensity_min_mult=args.ch1_intensity_min_mult,
        intensity_max_mult=args.ch1_intensity_max_mult,
        psf_min_r2=args.psf_fit_min_r2,
        psf_width_tol=args.psf_fit_width_tol,
        run_psf_filter=False,
    )
    n_flagged = int(spots_ch1["flagged"].sum()) if "flagged" in spots_ch1.columns else 0
    print(f"    Spots detected: {len(spots_ch1)}  ({n_flagged} flagged)")

    # --- CNN spot classifier (advisory — adds cnn_prob / cnn_flagged columns) ---
    if not getattr(args, 'no_cnn', False):
        _cnn_model, _cnn_thresh = load_spot_classifier()
        if _cnn_model is not None:
            print(f"    Running CNN spot classifier...")
            spots_ch1 = score_spots_cnn(spots_ch1, img_ch1, _cnn_model, _cnn_thresh)
            n_cnn_flag1 = int(spots_ch1["cnn_flagged"].sum())
            print(f"    CNN flagged — Ch1: {n_cnn_flag1}")

    stem = tif_path.stem
    out_dir.mkdir(parents=True, exist_ok=True)

    # Write "colocalized" CSV with ch1 positions duplicated into ch2 columns
    coloc = spots_ch1.copy()
    for col in ["x", "y", "sigma", "intensity",
                "flagged_intensity", "flagged_psf", "flagged"]:
        if col in coloc.columns:
            coloc[f"{col}_ch1"] = coloc[col]
            coloc[f"{col}_ch2"] = coloc[col]
    coloc["distance_px"]     = 0.0
    coloc["shift_x_applied"] = 0.0
    coloc["shift_y_applied"] = 0.0
    coloc["flagged_ch1"]     = coloc.get("flagged", False)
    coloc["flagged_ch2"]     = coloc.get("flagged", False)
    # Propagate CNN columns (ch1 scores duplicated into ch2 slots)
    if "cnn_prob" in coloc.columns:
        coloc["cnn_prob_ch1"]    = coloc["cnn_prob"]
        coloc["cnn_prob_ch2"]    = coloc["cnn_prob"]
        coloc["cnn_flagged_ch1"] = coloc["cnn_flagged"]
        coloc["cnn_flagged_ch2"] = coloc["cnn_flagged"]
    drop_cols = [c for c in ["x","y","sigma","intensity",
                              "flagged_intensity","flagged_psf","flagged",
                              "cnn_prob","cnn_flagged"]
                 if c in coloc.columns]
    coloc.drop(columns=drop_cols, inplace=True)

    coloc.to_csv(out_dir / f"{stem}_colocalized.csv", index=False)
    spots_ch1.to_csv(out_dir / f"{stem}_all_spots_ch1.csv", index=False)

    import datetime
    write_coloc_summary(
        out_path=out_dir / f"{stem}_summary.txt",
        stem=stem,
        n_ch1=len(spots_ch1), n_ch2=len(spots_ch1),
        n_coloc=len(spots_ch1), n_u1=0, n_u2=0,
        run_date=datetime.date.today().isoformat(),
    )

    # Single-panel overlay PNG
    save_overlay_image_single_channel(
        out_path=out_dir / f"{stem}_overlay.png",
        img_ch1=img_ch1,
        spots=spots_ch1,
        psf_radius=args.psf_radius,
    )

    print(f"    Output → {out_dir}/")
    n_flagged_coloc = int(coloc["flagged_ch1"].sum()) if "flagged_ch1" in coloc.columns else 0
    return {
        "file": tif_path.name,
        "ch1_spots": len(spots_ch1),
        "ch2_spots": len(spots_ch1),
        "colocalized_pairs": len(spots_ch1),
        "colocalized_flagged": n_flagged_coloc,
        "single_channel": True,
    }


def run_spot_editor(tif_path: Path,
                    out_dir: Path,
                    stem: str,
                    psf_radius: float,
                    frames_per_channel: int = None,
                    blank_frames: int = BLANK_FRAMES,
                    single_channel: bool = False):
    """
    Interactive spot editor for post-hoc correction of colocalization results.
    Builds ML training data by recording human-corrected spot lists.

    Controls (mouse position tracked per panel):
      Hover + [c]  — add colocalized spot at cursor position (both channels, same coords)
      Hover + [x]  — add unmatched spot at cursor position (this channel only)
      Left-click   — remove nearest spot (colocalized removes from both channels)
      [z]          — undo last action
      [q]          — save and close
    """
    import pandas as _pd
    from skimage.exposure import rescale_intensity as _rescale

    # Load CSVs
    coloc_csv = out_dir / f"{stem}_colocalized.csv"
    u1_csv    = out_dir / f"{stem}_unmatched_ch1.csv"
    u2_csv    = out_dir / f"{stem}_unmatched_ch2.csv"

    coloc = _pd.read_csv(coloc_csv) if coloc_csv.exists() else _pd.DataFrame(
        columns=["x_ch1","y_ch1","x_ch2","y_ch2"])
    u1    = _pd.read_csv(u1_csv)    if u1_csv.exists()    else _pd.DataFrame(columns=["x","y"])
    u2    = _pd.read_csv(u2_csv)    if u2_csv.exists()    else _pd.DataFrame(columns=["x","y"])

    # Load averaged frames
    print(f"\n  Loading images for spot editor...")
    with tifffile.TiffFile(tif_path) as _tif:
        _total = len(_tif.pages)
    fpc = _total // 2 if not single_channel else _total
    img_ch1 = load_averaged_frame(tif_path, blank_frames + 1,
                                  1, fpc)
    if not single_channel:
        img_ch2 = load_averaged_frame(tif_path, fpc + blank_frames + 1,
                                      1, fpc)
    else:
        img_ch2 = img_ch1  # unused but keeps later code uniform

    def _stretch(img):
        p1, p99 = np.percentile(img, (1, 99))
        return _rescale(img, in_range=(p1, p99), out_range=(0.0, 1.0))

    disp1 = _stretch(img_ch1)
    disp2 = _stretch(img_ch2)

    # Working copies as lists of dicts for easy mutation
    coloc_spots = [] if single_channel else (
        coloc[["x_ch1","y_ch1","x_ch2","y_ch2"]].to_dict("records")
        if len(coloc) else [])
    u1_spots    = u1[["x","y"]].to_dict("records") if len(u1) else []
    u2_spots    = [] if single_channel else (
        u2[["x","y"]].to_dict("records") if len(u2) else [])

    # Undo stack: each entry is a snapshot of (coloc_spots, u1_spots, u2_spots)
    import copy
    undo_stack  = []

    def _snapshot():
        undo_stack.append((
            copy.deepcopy(coloc_spots),
            copy.deepcopy(u1_spots),
            copy.deepcopy(u2_spots),
        ))

    circle_r    = psf_radius * 1.5
    remove_r    = psf_radius * 3.0   # click within this radius to remove

    # Mouse position tracking per axis
    cursor = {"x": 0.0, "y": 0.0, "ax": None}

    if single_channel:
        fig, ax1 = plt.subplots(1, 1, figsize=(9, 8))
        ax2 = ax1  # unused alias keeps _redraw/_on_click code uniform
    else:
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 8))
    fig.subplots_adjust(top=0.88, bottom=0.06)
    fig.canvas.manager.set_window_title(
        f"Spot Editor ({'single channel' if single_channel else 'two channel'}) — {stem}")

    title_txt  = fig.suptitle("", fontsize=10)
    status_txt = fig.text(0.5, 0.01, "", ha="center", fontsize=9, color="#444444")

    # Marker collections — rebuilt on each redraw
    _artists = []

    def _redraw():
        for a in _artists:
            try: a.remove()
            except Exception: pass
        _artists.clear()

        axes_to_draw = [(ax1, "x_ch1", "y_ch1", u1_spots, 1)]
        if not single_channel:
            axes_to_draw.append((ax2, "x_ch2", "y_ch2", u2_spots, 2))

        for ax, cx_col, cy_col, u_spots, ch in axes_to_draw:
            # Colocalized circles — green (two-channel only)
            if not single_channel:
                for s in coloc_spots:
                    c = mpatches.Circle(
                        (s[cx_col], s[cy_col]), radius=circle_r,
                        edgecolor="#00ff00", facecolor="none",
                        linewidth=1.3, zorder=3)
                    ax.add_patch(c)
                    _artists.append(c)

            # Unmatched X markers — red
            if u_spots:
                xs = [s["x"] for s in u_spots]
                ys = [s["y"] for s in u_spots]
                sc = ax.scatter(xs, ys, marker="x", s=60,
                                c="#ff3333", linewidths=1.3, zorder=3)
                _artists.append(sc)

            n_u = len(u_spots)
            if single_channel:
                ax.set_title(f"Channel 1  |  {n_u} spots", fontsize=9)
            else:
                n_coloc = len(coloc_spots)
                ax.set_title(f"Channel {ch}  |  {n_coloc} colocalized   {n_u} unmatched",
                             fontsize=9)

        n_edits = len(undo_stack)
        if single_channel:
            title_txt.set_text(
                f"Spot Editor (single channel) — {stem}  |  "
                f"Hover + [x]: add spot   Left-click: remove   [z]: undo   [q]: save & close")
            status_txt.set_text(
                f"Spots: {len(u1_spots)}   Edits: {n_edits}")
        else:
            title_txt.set_text(
                f"Spot Editor — {stem}  |  "
                f"Hover + [c]: add coloc   Hover + [x]: add unmatched   "
                f"Left-click: remove   [z]: undo   [q]: save & close")
            status_txt.set_text(
                f"Colocalized: {len(coloc_spots)}   "
                f"Unmatched ch1: {len(u1_spots)}   "
                f"Unmatched ch2: {len(u2_spots)}   "
                f"Edits: {n_edits}")
        fig.canvas.draw_idle()

    def _on_motion(event):
        if event.inaxes in (ax1, ax2):
            cursor["x"]  = event.xdata
            cursor["y"]  = event.ydata
            cursor["ax"] = event.inaxes

    def _nearest_coloc(ax, x, y):
        """Return index of nearest colocalized spot in given axis, or None."""
        if not coloc_spots:
            return None
        cx_col = "x_ch1" if ax is ax1 else "x_ch2"
        cy_col = "y_ch1" if ax is ax1 else "y_ch2"
        dists  = [np.sqrt((s[cx_col]-x)**2 + (s[cy_col]-y)**2)
                  for s in coloc_spots]
        idx    = int(np.argmin(dists))
        return idx if dists[idx] <= remove_r else None

    def _nearest_unmatched(u_spots, x, y):
        """Return index of nearest unmatched spot, or None."""
        if not u_spots:
            return None
        dists = [np.sqrt((s["x"]-x)**2 + (s["y"]-y)**2) for s in u_spots]
        idx   = int(np.argmin(dists))
        return idx if dists[idx] <= remove_r else None

    def _on_key(event):
        if cursor["ax"] is None:
            return
        ax  = cursor["ax"]
        x, y = cursor["x"], cursor["y"]

        if event.key == "c" and not single_channel:
            # Add colocalized spot — same coords in both channels
            _snapshot()
            coloc_spots.append({"x_ch1": x, "y_ch1": y,
                                 "x_ch2": x, "y_ch2": y,
                                 "manually_added": True})
            _redraw()

        elif event.key == "x":
            # Add unmatched spot in this channel only
            _snapshot()
            if ax is ax1:
                u1_spots.append({"x": x, "y": y, "manually_added": True})
            else:
                u2_spots.append({"x": x, "y": y, "manually_added": True})
            _redraw()

        elif event.key == "z":
            # Undo
            if undo_stack:
                prev = undo_stack.pop()
                coloc_spots.clear(); coloc_spots.extend(prev[0])
                u1_spots.clear();    u1_spots.extend(prev[1])
                u2_spots.clear();    u2_spots.extend(prev[2])
                _redraw()

        elif event.key == "q":
            _save_and_close()

    def _on_click(event):
        if event.button != 1 or event.inaxes not in (ax1, ax2):
            return
        ax  = event.inaxes
        x, y = event.xdata, event.ydata

        # Try removing a colocalized spot first, then unmatched
        ci = _nearest_coloc(ax, x, y)
        if ci is not None:
            _snapshot()
            coloc_spots.pop(ci)
            _redraw()
            return

        u_spots = u1_spots if ax is ax1 else u2_spots
        ui = _nearest_unmatched(u_spots, x, y)
        if ui is not None:
            _snapshot()
            u_spots.pop(ui)
            _redraw()

    def _save_and_close():
        # Build corrected DataFrames
        coloc_df_new = _pd.DataFrame(coloc_spots) if (coloc_spots and not single_channel) else \
                       _pd.DataFrame(columns=["x_ch1","y_ch1","x_ch2","y_ch2"])
        u1_df_new    = _pd.DataFrame(u1_spots)    if u1_spots    else \
                       _pd.DataFrame(columns=["x","y"])
        u2_df_new    = _pd.DataFrame(u2_spots)    if (u2_spots and not single_channel) else \
                       _pd.DataFrame(columns=["x","y"])

        # Mark as corrected for ML training
        coloc_df_new["human_corrected"] = True
        u1_df_new["human_corrected"]    = True
        u2_df_new["human_corrected"]    = True
        if single_channel:
            u1_df_new["single_channel"] = True

        # Save corrected CSVs (overwrite originals)
        coloc_df_new.to_csv(coloc_csv, index=False)
        u1_df_new.to_csv(u1_csv,       index=False)
        u2_df_new.to_csv(u2_csv,       index=False)

        # Also save separate _corrected copies for ML training data
        coloc_df_new.to_csv(out_dir / f"{stem}_colocalized_corrected.csv", index=False)
        u1_df_new.to_csv(   out_dir / f"{stem}_unmatched_ch1_corrected.csv", index=False)
        if not single_channel:
            u2_df_new.to_csv(out_dir / f"{stem}_unmatched_ch2_corrected.csv", index=False)

        print(f"  Spot editor: saved corrections → {out_dir}")
        if single_channel:
            print(f"    Spots (single channel): {len(u1_df_new)}")
        else:
            print(f"    Colocalized: {len(coloc_df_new)}  "
                  f"Unmatched ch1: {len(u1_df_new)}  "
                  f"Unmatched ch2: {len(u2_df_new)}")

        # Regenerate overlay PNGs with corrected spots
        if not single_channel:
            save_overlay_image(
                out_path=out_dir / f"{stem}_overlay.png",
                img_ch1=img_ch1, img_ch2=img_ch2,
                coloc=coloc_df_new,
                unmatched_ch1=u1_df_new,
                unmatched_ch2=u2_df_new,
                psf_radius=psf_radius,
            )
            save_coloc_only_image(
                out_path=out_dir / f"{stem}_overlay_coloc_only.png",
                img_ch1=img_ch1, img_ch2=img_ch2,
                coloc=coloc_df_new,
                psf_radius=psf_radius,
            )
        plt.close(fig)

    # Draw base images once
    ax1.imshow(disp1, cmap="gray", origin="upper")
    ax1.axis("off")
    if not single_channel:
        ax2.imshow(disp2, cmap="gray", origin="upper")
        ax2.axis("off")

    fig.canvas.mpl_connect("motion_notify_event", _on_motion)
    fig.canvas.mpl_connect("key_press_event",     _on_key)
    fig.canvas.mpl_connect("button_press_event",  _on_click)

    _redraw()
    plt.show()


def main():
    parser = build_parser()
    args = parser.parse_args()

    # Set matplotlib backend before any plt calls:
    # tune mode needs an interactive window; other modes render to file only
    if args.mode == "tune":
        # Try interactive backends in order of preference
        for _backend in ["TkAgg", "Qt5Agg", "Qt6Agg", "WXAgg", "MacOSX"]:
            try:
                matplotlib.use(_backend)
                break
            except Exception:
                continue
    else:
        matplotlib.use("Agg")

    print("\n=== Single-Molecule Colocalization Pipeline ===")
    print(f"  Mode               : {args.mode}")
    print(f"  PSF radius         : {args.psf_radius} px")
    print(f"  Detection threshold: {args.detection_threshold}"
          + (f"  (ch1={args.ch1_threshold})" if args.ch1_threshold else "")
          + (f"  (ch2={args.ch2_threshold})" if args.ch2_threshold else ""))
    print(f"  Coloc threshold    : {args.coloc_threshold} px")
    print(f"  Ref frame offset   : {args.ch1_ref_frame} frames from channel start  "
          f"(±{args.average_half_window} frame average)")
    print(f"  Frames/channel     : derived from file at runtime")

    if args.mode == "tune":
        if not args.input:
            parser.error("--mode tune requires --input")
        run_tune_mode(Path(args.input), args)
        return

    if args.mode == "single-channel":
        if not args.input:
            parser.error("--mode single-channel requires --input")
        tif_path = Path(args.input)
        out_dir  = Path(args.output) if args.output else tif_path.parent / "colocalization_output"
        process_file_single_channel(tif_path, args, out_dir)
        return

    if args.mode == "single":
        if not args.input:
            parser.error("--mode single requires --input")
        tif_path = Path(args.input)
        out_dir  = Path(args.output) if args.output else tif_path.parent / "colocalization_output"

        # Three-channel dispatch
        if getattr(args, 'three_channel', False):
            process_file_three_channel(tif_path, args, out_dir)
            return

        result   = process_file(tif_path, args, out_dir)

        if args.review_flagged and result is not None:
            stem     = tif_path.stem
            coloc_df = pd.read_csv(out_dir / f"{stem}_colocalized.csv")
            _u1_path = out_dir / f"{stem}_unmatched_ch1.csv"
            _u2_path = out_dir / f"{stem}_unmatched_ch2.csv"
            u1_df    = pd.read_csv(_u1_path) if _u1_path.exists() else pd.DataFrame(columns=["x","y"])
            u2_df    = pd.read_csv(_u2_path) if _u2_path.exists() else pd.DataFrame(columns=["x","y"])

            has_flags = (
                any(c.startswith("flagged") for c in coloc_df.columns) or
                "flagged" in u1_df.columns or
                "flagged" in u2_df.columns
            )
            if not has_flags:
                print("  No flagged spots found — skipping review.")
            else:
                # Switch to interactive backend for the review window
                for _backend in ["TkAgg", "Qt5Agg", "Qt6Agg", "WXAgg", "MacOSX"]:
                    try:
                        matplotlib.use(_backend)
                        break
                    except Exception:
                        continue
                # Derive frames_per_channel from file for the review call
                with tifffile.TiffFile(tif_path) as _t:
                    _fpc_review = len(_t.pages) // 2
                run_flagged_review(
                    tif_path=tif_path,
                    coloc=coloc_df, u1=u1_df, u2=u2_df,
                    out_dir=out_dir, stem=stem,
                    psf_radius=args.psf_radius,
                    ch1_ref_frame=args.ch1_ref_frame,
                    average_half_window=args.average_half_window,
                    frames_per_channel=_fpc_review,
                    crop_half=args.review_crop_size,
                    single_channel=getattr(args, 'single_channel', False),
                )

    else:  # batch
        if not args.folder:
            parser.error("--mode batch requires --folder")
        folder  = Path(args.folder)
        out_dir = Path(args.output) if args.output else folder / "colocalization_output"
        tifs    = sorted(folder.glob("*.tif")) + sorted(folder.glob("*.tiff"))
        if not tifs:
            sys.exit(f"No .tif/.tiff files found in {folder}")

        print(f"\n  Found {len(tifs)} .tif file(s) in {folder}")
        summaries = []
        for tif_path in tifs:
            if getattr(args, 'three_channel', False):
                process_file_three_channel(tif_path, args, out_dir)
            else:
                result = process_file(tif_path, args, out_dir)
                if result:
                    summaries.append(result)

        if summaries:
            summary_df = pd.DataFrame(summaries)
            summary_path = out_dir / "batch_summary.csv"
            summary_df.to_csv(summary_path, index=False)
            print(f"\n  Batch summary → {summary_path}")


if __name__ == "__main__":
    main()
