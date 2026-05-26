#!/usr/bin/env python3
"""
SiMPull Analysis Pipeline
========================
Single-movie pipeline: VSI → TIF → Colocalization → Bleaching Analysis

Run with no arguments to open the GUI:
    python pipeline.py

Or with CLI flags for scripted use:
    python pipeline.py --ch1 ch1.vsi --ch2 ch2.vsi --output movie.tif ...
"""

import os
import sys
import json
import subprocess
import threading
import traceback
import numpy as np
from datetime import date
from pathlib import Path

# ---------------------------------------------------------------------------
# Config / parameter management
# ---------------------------------------------------------------------------

# Location of scripts folder (same directory as this file)
SCRIPTS_DIR = Path(__file__).parent

# Global config defaults (bleaching + acquisition settings)
DEFAULT_CONFIG = {
    # Acquisition
    "single_channel":      False,
    "blank_frames":        10,
    "peak_threshold":      0.90,
    "fiji_path":           "/Applications/Fiji.app",
    # Colocalization
    "psf_radius":              1.5,
    "detection_threshold":     0.001,
    "ch1_threshold":           0.001,
    "ch2_threshold":           0.01,
    "ch3_threshold":           0.01,
    "exclude_border":          5,
    "coloc_threshold":         5.0,
    "coloc_threshold_ch3":     5.0,
    "ch1_intensity_min_mult":  1.2,
    "ch1_intensity_max_mult":  100.0,
    "ch2_intensity_min_mult":  0.0,
    "ch2_intensity_max_mult":  100.0,
    "ch3_intensity_min_mult":  0.0,
    "ch3_intensity_max_mult":  100.0,
    "psf_fit_min_r2":          0.3,
    "psf_fit_width_tol":       0.8,
    "shift_first_pass_factor": 3.0,
    "shift_min_pairs":         10,
    "shift_max_px":            20.0,
    # Bleaching (global, not condition-specific)
    "pelt_penalty":              15,
    "pelt_min_size":             5,
    "bg_annulus_inner":          4.0,
    "bg_annulus_outer":          8.0,
    "bg_spot_mask":              3.0,
    "burst_frames":              3,
    "burst_plateau_frames":      10,
    "burst_ratio":               1.5,
    "burst_tail_frames":         30,
    "burst_tail_ratio":          3.0,
    "burst_max_scan":            10,
    "incomplete_tail_frames":    10,
    "incomplete_tail_threshold": 0.20,
    "incomplete_abs_floor":      10.0,
    # Stoichiometry range
    "max_stoichiometry_ch1":     2,
    "max_stoichiometry_ch2":     2,
    "max_stoichiometry_ch3":     2,
}

# Parameters saved per condition (colocalization only)
CONDITION_PARAM_KEYS = [
    "psf_radius", "detection_threshold", "ch1_threshold", "ch2_threshold",
    "ch3_threshold", "exclude_border", "coloc_threshold", "coloc_threshold_ch3",
    "ch1_intensity_min_mult", "ch1_intensity_max_mult",
    "ch2_intensity_min_mult", "ch2_intensity_max_mult",
    "ch3_intensity_min_mult", "ch3_intensity_max_mult",
    "psf_fit_min_r2", "psf_fit_width_tol",
]


def get_params_dir() -> Path:
    """Returns ~/smfret_params, creating it if needed."""
    d = Path.home() / "smfret_params"
    d.mkdir(exist_ok=True)
    return d


def get_config_path() -> Path:
    return get_params_dir() / "config.json"


def load_config() -> dict:
    p = get_config_path()
    if p.exists():
        with open(p) as f:
            cfg = json.load(f)
        # Fill in any missing keys from defaults
        for k, v in DEFAULT_CONFIG.items():
            cfg.setdefault(k, v)
        return cfg
    return dict(DEFAULT_CONFIG)


def save_config(cfg: dict):
    with open(get_config_path(), "w") as f:
        json.dump(cfg, f, indent=2)


def condition_key(date_str, protein_a, protein_b, other) -> str:
    parts = [date_str, protein_a, protein_b]
    if other.strip():
        parts.append(other.strip())
    return "_".join(p.strip().replace(" ", "_") for p in parts if p.strip())


def params_path(key: str) -> Path:
    return get_params_dir() / f"{key}.json"


def load_condition_params(key: str, config: dict) -> tuple[dict, bool]:
    """
    Returns (params_dict, found).
    If found, loads saved condition params merged with global config.
    If not found, returns global config defaults.
    """
    p = params_path(key)
    if p.exists():
        with open(p) as f:
            saved = json.load(f)
        merged = dict(config)
        merged.update(saved)
        return merged, True
    return dict(config), False


def save_condition_params(key: str, params: dict):
    subset = {k: params[k] for k in CONDITION_PARAM_KEYS if k in params}
    with open(params_path(key), "w") as f:
        json.dump(subset, f, indent=2)


def get_training_log_path() -> Path:
    return get_params_dir() / "training_log.jsonl"


def _compute_channel_stats(tif, ch_start: int, fpc: int, blank: int) -> dict:
    """
    Compute image statistics for one channel from an open TiffFile.

    ch_start : index of the first frame for this channel
    fpc      : frames per channel
    blank    : number of blank (laser-off) frames at the start of each channel
    """
    import tifffile as _tiff

    # Background: median across the blank frames for this channel
    n_blank = min(blank, fpc)
    if n_blank > 0:
        blank_stack = np.stack(
            [tif.pages[ch_start + i].asarray().astype(np.float32)
             for i in range(n_blank)],
            axis=0,
        )
        bg_level = float(np.median(blank_stack))
    else:
        bg_level = None  # filled in below after img is computed

    # Signal: average of 3 frames around the first signal frame
    ref  = blank + 1
    idxs = sorted(set([max(0, ref - 1), ref, min(fpc - 1, ref + 1)]))
    signal_frames = np.stack(
        [tif.pages[ch_start + i].asarray().astype(np.float32) for i in idxs],
        axis=0,
    )
    img = signal_frames.mean(axis=0)

    if bg_level is None:
        bg_level = float(np.median(img))

    pcts = np.percentile(img, [1, 5, 25, 50, 75, 95, 99])
    return {
        "mean":          float(img.mean()),
        "std":           float(img.std()),
        "p01":           float(pcts[0]),
        "p05":           float(pcts[1]),
        "p25":           float(pcts[2]),
        "p50":           float(pcts[3]),
        "p75":           float(pcts[4]),
        "p95":           float(pcts[5]),
        "p99":           float(pcts[6]),
        "bg_estimate":   bg_level,
        "snr_estimate":  float((pcts[5] - bg_level) / (bg_level + 1e-6)),
        "dynamic_range": float(pcts[6] - pcts[0]),
    }


def log_movie_training_data(tif_path: Path, params: dict,
                             coloc_csv: Path, u1_csv: Path, u2_csv: Path,
                             condition: str, human_corrected: bool,
                             single_channel: bool = False):
    """
    Append one record to the per-movie training log.
    Computes image statistics separately for ch1 and ch2 (where available).
    Stats are stored under image_stats: {ch1: {...}, ch2: {...}}.
    """
    import tifffile as _tiff

    record = {
        "date":            date.today().isoformat(),
        "tif":             str(tif_path),
        "condition":       condition,
        "human_corrected": human_corrected,
        "single_channel":  single_channel,
        "params": {k: params[k] for k in CONDITION_PARAM_KEYS if k in params},
        "results": {},
        "image_stats": {},
    }

    # Count spots from CSVs
    try:
        import pandas as _pd
        if coloc_csv.exists():
            _c = _pd.read_csv(coloc_csv)
            record["results"]["n_coloc"] = len(_c)
        if u1_csv.exists():
            _u1 = _pd.read_csv(u1_csv)
            record["results"]["n_unmatched_ch1"] = len(_u1)
        if u2_csv.exists():
            _u2 = _pd.read_csv(u2_csv)
            record["results"]["n_unmatched_ch2"] = len(_u2)
    except Exception:
        pass

    # Compute per-channel image statistics
    try:
        blank = params.get("blank_frames", 10)
        with _tiff.TiffFile(str(tif_path)) as tif:
            total_frames = len(tif.pages)
            fpc = total_frames // 2 if not single_channel else total_frames

            ch1_stats = _compute_channel_stats(tif, ch_start=0,   fpc=fpc, blank=blank)
            if not single_channel and total_frames >= 2 * fpc:
                ch2_stats = _compute_channel_stats(tif, ch_start=fpc, fpc=fpc, blank=blank)
            else:
                ch2_stats = None

        record["image_stats"] = {"ch1": ch1_stats}
        if ch2_stats is not None:
            record["image_stats"]["ch2"] = ch2_stats

    except Exception:
        pass

    # Append to JSONL log
    with open(get_training_log_path(), "a") as f:
        f.write(json.dumps(record) + "\n")



def suggest_output_name(protein_a, protein_b, other, replicate) -> str:
    parts = [p.strip().replace(" ", "") for p in [protein_a, protein_b] if p.strip()]
    if other.strip():
        parts.append(other.strip().replace(" ", ""))
    parts.append(f"movie{replicate}")
    return "_".join(parts)


def suggest_output_folder(date_str, protein_a, protein_b, other, replicate) -> Path | None:
    """
    Build the auto-suggested output folder:
      ~/Documents/Simpull_pipeline_data_analysis/
          {YYYY.MM.DD}/
              {proteinA}_{proteinB}_{other}/
                  movie{N}/
    Returns None if the minimum required fields (date + protein_a) are not filled.
    """
    if not date_str.strip() or not protein_a.strip():
        return None

    # Format date as YYYY.MM.DD
    d = date_str.strip()
    try:
        from datetime import date as _date
        import re
        m = re.match(r"(\d{4})[.\-_]?(\d{2})[.\-_]?(\d{2})", d)
        if m:
            d_fmt = f"{m.group(1)}.{m.group(2)}.{m.group(3)}"
        else:
            d_fmt = d   # leave as-is if unrecognised
    except Exception:
        d_fmt = d

    # Condition folder: proteinA_proteinB_other (no date, no replicate)
    cond_parts = [p.strip().replace(" ", "_") for p in [protein_a, protein_b] if p.strip()]
    if other.strip():
        cond_parts.append(other.strip().replace(" ", "_"))
    cond_folder = "_".join(cond_parts)

    replicate_folder = f"movie{replicate}"

    base = Path.home() / "Documents" / "Simpull_pipeline_data_analysis"
    return base / d_fmt / cond_folder / replicate_folder


# ---------------------------------------------------------------------------
# Pipeline runner
# ---------------------------------------------------------------------------

def try_ml_predict_params(tif_path: Path, params: dict) -> dict:
    """
    Attempt to predict tuning parameters from image statistics using the
    saved random forest model bundle.

    If the model file does not exist, or prediction fails for any reason,
    returns params unchanged — the tune UI will open with condition defaults
    as normal.

    On success, returns a copy of params with predicted values merged in.
    Only params that have a trained predictor are updated; all others are
    left at their condition default values.
    """
    model_path = get_params_dir() / "models" / "param_predictor.pkl"
    if not model_path.exists():
        return params

    try:
        import pickle
        import tifffile as _tiff
        # numpy available as np (top-level import)

        # Load model bundle
        with open(model_path, "rb") as f:
            bundle = pickle.load(f)

        predictors = bundle.get("predictors", {})
        if not predictors:
            return params

        # Compute per-channel image stats from the TIF
        blank = params.get("blank_frames", 10)
        single_ch = params.get("single_channel", False)

        with _tiff.TiffFile(str(tif_path)) as tif:
            total_frames = len(tif.pages)
            fpc = total_frames // 2 if not single_ch else total_frames

            ch1_stats = _compute_channel_stats(tif, ch_start=0, fpc=fpc, blank=blank)
            if not single_ch and total_frames >= 2 * fpc:
                ch2_stats = _compute_channel_stats(tif, ch_start=fpc, fpc=fpc, blank=blank)
            else:
                ch2_stats = {}

        image_stats = {"ch1": ch1_stats, "ch2": ch2_stats}

        # Flatten to feature vector matching training format
        _STAT_KEYS = ["mean", "std", "p01", "p05", "p25", "p50", "p75",
                      "p95", "p99", "bg_estimate", "snr_estimate"]
        flat = {}
        for ch in ("ch1", "ch2"):
            ch_s = image_stats.get(ch, {})
            for s in _STAT_KEYS:
                flat[f"{ch}_{s}"] = ch_s.get(s, np.nan)

        # Predict each param and merge into a copy of params
        updated = dict(params)
        for param, info in predictors.items():
            feature_cols = info["feature_cols"]
            x = np.array([[flat.get(c, np.nan) for c in feature_cols]],
                          dtype=np.float32)
            if np.isnan(x).any():
                print(f"  DEBUG ML: {param} skipped — NaN features", flush=True)
                continue
            val = float(info["model"].predict(x)[0])
            lo = info["target_mean"] - 3 * info["target_std"]
            hi = info["target_mean"] + 3 * info["target_std"]
            clipped = round(float(np.clip(val, lo, hi)), 5)
            print(f"  DEBUG ML: {param} raw={val:.5f} clipped={clipped:.5f} base={params.get(param, '?')}", flush=True)
            updated[param] = clipped

        return updated

    except Exception as _e:
        try:
            import traceback as _tb
            _dbg = get_params_dir() / "ml_debug.txt"
            with open(_dbg, "w") as _f:
                _f.write(f"try_ml_predict_params failed:\n{_tb.format_exc()}\n")
        except Exception:
            pass
        return params


def build_coloc_args(tif_path: Path, out_dir: Path, params: dict,
                     mode: str, review_flagged: bool,
                     params_out: "Path | None" = None,
                     single_channel: bool = False,
                     three_channel: bool = False) -> list:
    cmd = [
        sys.executable,
        str(SCRIPTS_DIR / "colocalize_tif.py"),
        "--mode", mode,
        "--input", str(tif_path),
        "--output", str(out_dir),
        "--blank-frames",            str(params["blank_frames"]),
        "--psf-radius",              str(params["psf_radius"]),
        "--detection-threshold",     str(params["detection_threshold"]),
        "--ch1-threshold",           str(params["ch1_threshold"]),
        "--ch2-threshold",           str(params["ch2_threshold"]),
        "--exclude-border",          str(params["exclude_border"]),
        "--coloc-threshold",         str(params["coloc_threshold"]),
        "--ch1-intensity-min-mult",  str(params["ch1_intensity_min_mult"]),
        "--ch1-intensity-max-mult",  str(params["ch1_intensity_max_mult"]),
        "--ch2-intensity-min-mult",  str(params["ch2_intensity_min_mult"]),
        "--ch2-intensity-max-mult",  str(params["ch2_intensity_max_mult"]),
        "--psf-fit-min-r2",          str(params["psf_fit_min_r2"]),
        "--psf-fit-width-tol",       str(params["psf_fit_width_tol"]),
        "--shift-first-pass-factor", str(params["shift_first_pass_factor"]),
        "--shift-min-pairs",         str(params["shift_min_pairs"]),
        "--shift-max-px",            str(params["shift_max_px"]),
    ]
    if review_flagged:
        cmd.append("--review-flagged")
    if params_out:
        cmd += ["--params-out", str(params_out)]
    if single_channel:
        cmd.append("--single-channel")
    if three_channel:
        cmd += [
            "--three-channel",
            "--ch3-threshold",           str(params.get("ch3_threshold", 0.01)),
            "--ch3-intensity-min-mult",  str(params.get("ch3_intensity_min_mult", 0.0)),
            "--ch3-intensity-max-mult",  str(params.get("ch3_intensity_max_mult", 100.0)),
            "--coloc-threshold-ch3",     str(params.get("coloc_threshold_ch3", 5.0)),
        ]
    return cmd


def build_detect_single_args(tif_path: Path, out_dir: Path,
                             params: dict, mode: str) -> list:
    return [
        sys.executable,
        str(SCRIPTS_DIR / "detect_spots_single.py"),
        "--mode",               mode,
        "--input",              str(tif_path),
        "--output",             str(out_dir),
        "--blank-frames",       str(params["blank_frames"]),
        "--psf-radius",         str(params["psf_radius"]),
        "--detection-threshold",str(params["detection_threshold"]),
        "--exclude-border",     str(params["exclude_border"]),
        "--intensity-min-mult", str(params["ch1_intensity_min_mult"]),
        "--intensity-max-mult", str(params["ch1_intensity_max_mult"]),
        "--psf-fit-min-r2",     str(params["psf_fit_min_r2"]),
        "--psf-fit-width-tol",  str(params["psf_fit_width_tol"]),
    ]


def build_bleaching_single_args(tif_path: Path, spots_csv: Path,
                                 out_dir: Path, params: dict,
                                 max_spots: int = 0) -> list:
    cmd = [
        sys.executable,
        str(SCRIPTS_DIR / "bleaching_analysis.py"),
        "--mode",   "single-channel",
        "--input",  str(tif_path),
        "--spots",  str(spots_csv),
        "--output", str(out_dir),
        "--blank-frames",             str(params["blank_frames"]),
        "--psf-radius",               str(params["psf_radius"]),
        "--bg-annulus-inner",         str(params["bg_annulus_inner"]),
        "--bg-annulus-outer",         str(params["bg_annulus_outer"]),
        "--bg-spot-mask",             str(params["bg_spot_mask"]),
        "--pelt-penalty",             str(params["pelt_penalty"]),
        "--pelt-min-size",            str(params["pelt_min_size"]),
        "--burst-frames",             str(params["burst_frames"]),
        "--burst-plateau-frames",     str(params["burst_plateau_frames"]),
        "--burst-ratio",              str(params["burst_ratio"]),
        "--burst-tail-frames",        str(params["burst_tail_frames"]),
        "--burst-tail-ratio",         str(params["burst_tail_ratio"]),
        "--burst-max-scan",           str(params["burst_max_scan"]),
        "--incomplete-tail-frames",   str(params["incomplete_tail_frames"]),
        "--incomplete-tail-threshold",str(params["incomplete_tail_threshold"]),
        "--incomplete-abs-floor",     str(params["incomplete_abs_floor"]),
        "--max-stoichiometry-ch1",    str(params.get("max_stoichiometry_ch1", 2)),
    ]
    if max_spots > 0:
        cmd += ["--max-spots", str(max_spots)]
    return cmd


def build_bleaching_args(tif_path: Path, coloc_csv: "Path",
                         out_dir: Path, params: dict,
                         three_channel: bool = False,
                         skip_populations: "list[str] | None" = None,
                         max_spots: int = 0) -> list:
    cmd = [
        sys.executable,
        str(SCRIPTS_DIR / "bleaching_analysis.py"),
        "--mode",   "single",
        "--input",  str(tif_path),
        "--coloc",  str(coloc_csv),
        "--output", str(out_dir),
        "--blank-frames",             str(params["blank_frames"]),
        "--psf-radius",               str(params["psf_radius"]),
        "--bg-annulus-inner",         str(params["bg_annulus_inner"]),
        "--bg-annulus-outer",         str(params["bg_annulus_outer"]),
        "--bg-spot-mask",             str(params["bg_spot_mask"]),
        "--pelt-penalty",             str(params["pelt_penalty"]),
        "--pelt-min-size",            str(params["pelt_min_size"]),
        "--burst-frames",             str(params["burst_frames"]),
        "--burst-plateau-frames",     str(params["burst_plateau_frames"]),
        "--burst-ratio",              str(params["burst_ratio"]),
        "--burst-tail-frames",        str(params["burst_tail_frames"]),
        "--burst-tail-ratio",         str(params["burst_tail_ratio"]),
        "--burst-max-scan",           str(params["burst_max_scan"]),
        "--incomplete-tail-frames",   str(params["incomplete_tail_frames"]),
        "--incomplete-tail-threshold",str(params["incomplete_tail_threshold"]),
        "--incomplete-abs-floor",     str(params["incomplete_abs_floor"]),
        "--max-stoichiometry-ch1",    str(params.get("max_stoichiometry_ch1", 2)),
        "--max-stoichiometry-ch2",    str(params.get("max_stoichiometry_ch2", 2)),
        "--max-stoichiometry-ch3",    str(params.get("max_stoichiometry_ch3", 2)),
    ]
    if three_channel:
        cmd.append("--three-channel")
    if skip_populations:
        cmd += ["--skip-populations", ",".join(skip_populations)]
    if max_spots > 0:
        cmd += ["--max-spots", str(max_spots)]
    return cmd


# ---------------------------------------------------------------------------
# GUI
# ---------------------------------------------------------------------------

def run_gui():
    import tkinter as tk
    from tkinter import ttk, filedialog, messagebox, scrolledtext

    config  = load_config()
    root    = tk.Tk()
    root.title("SiMPull Analysis Pipeline")
    root.resizable(True, True)

    # ── Variables ──────────────────────────────────────────────────────────
    # Restore last-used values from config (date always defaults to today)
    _last = config.get("_last_gui", {})
    date_var       = tk.StringVar(value=date.today().isoformat())
    protein_a_var  = tk.StringVar(value=_last.get("protein_a", ""))
    protein_b_var  = tk.StringVar(value=_last.get("protein_b", ""))
    other_var      = tk.StringVar(value=_last.get("other", ""))
    replicate_var  = tk.IntVar(value=_last.get("replicate", 1))
    condition_var  = tk.StringVar(value="—")

    # channel_mode: "two" | "single" | "three"
    channel_mode_var = tk.StringVar(value=_last.get("channel_mode", "two"))
    # Convenience helpers
    def _is_single_ch():  return channel_mode_var.get() == "single"
    def _is_three_ch():   return channel_mode_var.get() == "three"

    ch1_var        = tk.StringVar()
    ch2_var        = tk.StringVar()
    ch3_var        = tk.StringVar()
    out_folder_var = tk.StringVar()
    out_name_var   = tk.StringVar()

    # Steps to run
    run_prepare_var   = tk.BooleanVar(value=True)
    run_coloc_var     = tk.BooleanVar(value=True)
    run_bleach_var    = tk.BooleanVar(value=True)
    export_dat_var    = tk.BooleanVar(value=True)
    review_flagged_var= tk.BooleanVar(value=True)

    # Bleaching populations to analyse (all on by default)
    bleach_coloc_var        = tk.BooleanVar(value=True)
    bleach_unmatched_ch1_var= tk.BooleanVar(value=True)
    bleach_unmatched_ch2_var= tk.BooleanVar(value=True)

    # Stoichiometry range (loaded from config)
    max_stoich_ch1_var = tk.IntVar(value=config.get("max_stoichiometry_ch1", 2))
    max_stoich_ch2_var = tk.IntVar(value=config.get("max_stoichiometry_ch2", 2))
    max_stoich_ch3_var = tk.IntVar(value=config.get("max_stoichiometry_ch3", 2))
    max_spots_var      = tk.IntVar(value=0)   # 0 = analyze all

    # Existing TIF (when prepare is unchecked)
    existing_tif_var  = tk.StringVar()

    PAD = dict(padx=8, pady=3)

    # ── Helper: update condition key ───────────────────────────────────────
    def _update_condition(*_):
        pb = "" if _is_single_ch() else protein_b_var.get()
        key = condition_key(
            date_var.get(), protein_a_var.get(),
            pb, other_var.get())
        condition_var.set(key if key else "—")
        # Update suggested output name
        _update_out_name()

    def _update_out_name(*_):
        try:
            rep = replicate_var.get()
        except Exception:
            return  # field is empty or invalid — skip update
        pb = "" if _is_single_ch() else protein_b_var.get()
        name = suggest_output_name(
            protein_a_var.get(), pb,
            other_var.get(), rep)
        if name:
            out_name_var.set(name)
        folder = suggest_output_folder(
            date_var.get(), protein_a_var.get(), pb,
            other_var.get(), rep)
        if folder:
            out_folder_var.set(str(folder))

    for v in [date_var, protein_a_var, protein_b_var, other_var]:
        v.trace_add("write", _update_condition)
    replicate_var.trace_add("write", _update_out_name)

    # ── Helper: toggle channel mode ────────────────────────────────────────
    def _toggle_channel_mode(*_):
        sc  = _is_single_ch()
        thc = _is_three_ch()
        # Protein B — irrelevant for single-channel
        if protein_b_entry is not None:
            protein_b_entry.config(state="disabled" if sc else "normal")
        # Ch2 VSI picker
        ch2_entry.config(state="disabled" if sc else "normal")
        ch2_btn.config(state="disabled" if sc else "normal")
        # Ch3 VSI picker — only for three-channel
        ch3_entry.config(state="normal" if thc else "disabled")
        ch3_btn.config(state="normal" if thc else "disabled")
        if sc:
            run_bleach_var.set(True)
        _toggle_prepare()
        _update_condition()
        # Re-evaluate bleaching population checkboxes — unmatched are
        # irrelevant in single-channel mode (no colocalization matching occurs)
        try:
            _toggle_bleach_pops()
        except NameError:
            pass  # _toggle_bleach_pops not yet defined on first build pass

    channel_mode_var.trace_add("write", _toggle_channel_mode)

    # ── Helper: toggle prepare-related widgets ─────────────────────────────
    def _toggle_prepare(*_):
        state = "normal" if run_prepare_var.get() else "disabled"
        for w in vsi_widgets:
            w.config(state=state)
        # Ch3 is only active when both prepare is checked AND three-channel mode
        ch3_state = "normal" if (run_prepare_var.get() and _is_three_ch()) else "disabled"
        ch3_entry.config(state=ch3_state)
        ch3_btn.config(state=ch3_state)
        state2 = "disabled" if run_prepare_var.get() else "normal"
        for w in tif_widgets:
            w.config(state=state2)
        if not run_prepare_var.get():
            pass  # bleaching can still run if a coloc CSV already exists

    def _toggle_coloc(*_):
        _update_bleach_state()

    def _update_bleach_state(*_):
        # Bleaching can run independently as long as a coloc CSV exists
        bleach_cb.config(state="normal")

    # ── Browse helpers ──────────────────────────────────────────────────────
    def _parse_date_from_path(p: str) -> str | None:
        """
        Walk up the directory tree looking for a folder name that looks like
        a date. Supports YYYY.MM.DD, YYYY-MM-DD, YYYY_MM_DD, and YYYYMMDD.
        Returns an ISO date string (YYYY-MM-DD) or None if not found.
        """
        import re
        from datetime import date as _date
        patterns = [
            (r"(\d{4})[.\-_](\d{2})[.\-_](\d{2})", lambda m: (int(m.group(1)), int(m.group(2)), int(m.group(3)))),
            (r"^(\d{4})(\d{2})(\d{2})$",              lambda m: (int(m.group(1)), int(m.group(2)), int(m.group(3)))),
        ]
        for part in Path(p).parts:
            for pattern, extractor in patterns:
                m = re.search(pattern, part)
                if m:
                    try:
                        y, mo, d = extractor(m)
                        return _date(y, mo, d).isoformat()
                    except ValueError:
                        pass
        return None

    def _browse_vsi(var, title):
        p = filedialog.askopenfilename(
            title=title,
            filetypes=[("Olympus VSI", "*.vsi"), ("All files", "*.*")])
        if p:
            var.set(p)
            if var is ch1_var:
                pass   # output folder is auto-set from experiment fields
                # Auto-fill date from folder name if it looks like a date
                detected = _parse_date_from_path(p)
                if detected:
                    date_var.set(detected)

    def _browse_tif():
        p = filedialog.askopenfilename(
            title="Select existing .tif file",
            filetypes=[("TIFF", "*.tif *.tiff"), ("All files", "*.*")])
        if p:
            existing_tif_var.set(p)
            # Auto-fill date from the TIF path (same logic as VSI browse)
            detected = _parse_date_from_path(p)
            if detected:
                date_var.set(detected)
            # Auto-fill output folder and name directly from the TIF file
            tif_p = Path(p)
            # Set output name from TIF stem so CSV lookup works
            out_name_var.set(tif_p.stem)

    def _browse_out_folder():
        p = filedialog.askdirectory(title="Select output folder")
        if p:
            out_folder_var.set(p)

    # ── Main layout ────────────────────────────────────────────────────────
    nb = ttk.Notebook(root)
    nb.pack(fill="both", expand=True, padx=10, pady=10)

    # ─── Tab 1+2 merged: Setup ────────────────────────────────────────────
    tab1 = ttk.Frame(nb, padding=10)
    tab2 = tab1   # keep tab2 as alias so downstream references still work

    # ── Section: Experiment ───────────────────────────────────────────────
    ttk.Label(tab1, text="Experiment",
              font=("", 11, "bold"), foreground="#444444").grid(
        row=0, column=0, columnspan=4, sticky="w", pady=(0, 4))

    # Row 1: Date + Replicate # on the same line
    ttk.Label(tab1, text="Date").grid(row=1, column=0, sticky="w", **PAD)
    ttk.Entry(tab1, textvariable=date_var, width=14).grid(
        row=1, column=1, sticky="w", **PAD)
    ttk.Label(tab1, text="Replicate #", foreground="#555555").grid(
        row=1, column=2, sticky="e", **PAD)
    ttk.Spinbox(tab1, textvariable=replicate_var, from_=1, to=99,
                width=5).grid(row=1, column=3, sticky="w", **PAD)

    # Rows 2-4: Protein A, Protein B, Other
    protein_b_entry = None
    inline_fields = [
        (2, "Protein A",          protein_a_var),
        (3, "Protein B",          protein_b_var),
        (4, "Other / condition",  other_var),
    ]
    for row, label, var in inline_fields:
        ttk.Label(tab1, text=label).grid(row=row, column=0, sticky="w", **PAD)
        e = ttk.Entry(tab1, textvariable=var, width=38)
        e.grid(row=row, column=1, columnspan=3, sticky="ew", **PAD)
        if label == "Protein B":
            protein_b_entry = e

    # Row 5: Condition key badge
    ttk.Label(tab1, text="Condition key:").grid(row=5, column=0, sticky="w", **PAD)
    ttk.Label(tab1, textvariable=condition_var,
              foreground="#336699", font=("Courier", 9, "bold")).grid(
        row=5, column=1, columnspan=3, sticky="w", **PAD)

    ttk.Separator(tab1, orient="horizontal").grid(
        row=6, column=0, columnspan=4, sticky="ew", pady=(8, 4))

    # ── Section: Input files ──────────────────────────────────────────────
    ttk.Label(tab1, text="Input Files",
              font=("", 11, "bold"), foreground="#444444").grid(
        row=7, column=0, columnspan=4, sticky="w", pady=(0, 4))

    # VSI pickers
    ttk.Label(tab1, text="Channel 1 (.vsi)").grid(row=8, column=0, sticky="w", **PAD)
    ch1_entry = ttk.Entry(tab1, textvariable=ch1_var, width=38)
    ch1_entry.grid(row=8, column=1, columnspan=2, sticky="ew", **PAD)
    ch1_btn = ttk.Button(tab1, text="Browse…",
                         command=lambda: _browse_vsi(ch1_var, "Select Channel 1 .vsi"))
    ch1_btn.grid(row=8, column=3, **PAD)

    ttk.Label(tab1, text="Channel 2 (.vsi)").grid(row=9, column=0, sticky="w", **PAD)
    ch2_entry = ttk.Entry(tab1, textvariable=ch2_var, width=38)
    ch2_entry.grid(row=9, column=1, columnspan=2, sticky="ew", **PAD)
    ch2_btn = ttk.Button(tab1, text="Browse…",
                         command=lambda: _browse_vsi(ch2_var, "Select Channel 2 .vsi"))
    ch2_btn.grid(row=9, column=3, **PAD)

    ttk.Label(tab1, text="Channel 3 (.vsi)").grid(row=10, column=0, sticky="w", **PAD)
    ch3_entry = ttk.Entry(tab1, textvariable=ch3_var, width=38, state="disabled")
    ch3_entry.grid(row=10, column=1, columnspan=2, sticky="ew", **PAD)
    ch3_btn = ttk.Button(tab1, text="Browse…",
                         command=lambda: _browse_vsi(ch3_var, "Select Channel 3 .vsi"),
                         state="disabled")
    ch3_btn.grid(row=10, column=3, **PAD)

    vsi_widgets = [ch1_entry, ch1_btn, ch2_entry, ch2_btn]

    # Existing TIF (active when prepare is unchecked)
    ttk.Label(tab1, text="Existing .tif").grid(row=11, column=0, sticky="w", **PAD)
    tif_entry = ttk.Entry(tab1, textvariable=existing_tif_var,
                          width=38, state="disabled")
    tif_entry.grid(row=11, column=1, columnspan=2, sticky="ew", **PAD)
    tif_btn = ttk.Button(tab1, text="Browse…",
                         command=_browse_tif, state="disabled")
    tif_btn.grid(row=11, column=3, **PAD)
    tif_widgets = [tif_entry, tif_btn]

    ttk.Separator(tab1, orient="horizontal").grid(
        row=12, column=0, columnspan=4, sticky="ew", pady=(8, 4))

    # ── Section: Output ───────────────────────────────────────────────────
    ttk.Label(tab1, text="Output",
              font=("", 11, "bold"), foreground="#444444").grid(
        row=13, column=0, columnspan=4, sticky="w", pady=(0, 4))

    out_folder_label_var = tk.StringVar(value="Output folder")
    ttk.Label(tab1, textvariable=out_folder_label_var).grid(
        row=14, column=0, sticky="w", **PAD)
    ttk.Entry(tab1, textvariable=out_folder_var, width=38).grid(
        row=14, column=1, columnspan=2, sticky="ew", **PAD)
    ttk.Button(tab1, text="Browse…",
               command=_browse_out_folder).grid(row=14, column=3, **PAD)

    out_folder_hint_var = tk.StringVar(value="")
    ttk.Label(tab1, textvariable=out_folder_hint_var,
              foreground="#888888", font=("", 9)).grid(
        row=15, column=0, columnspan=4, sticky="w", **PAD)

    def _update_folder_hint(*_):
        coloc   = run_coloc_var.get()
        bleach  = run_bleach_var.get()
        prepare = run_prepare_var.get()
        if bleach and not coloc and not prepare:
            out_folder_label_var.set("Output folder *")
            out_folder_hint_var.set("* Must be the folder containing the existing colocalized CSV")
        else:
            out_folder_label_var.set("Output folder")
            out_folder_hint_var.set("")

    run_coloc_var.trace_add("write",  _update_folder_hint)
    run_bleach_var.trace_add("write", _update_folder_hint)
    run_prepare_var.trace_add("write",_update_folder_hint)

    ttk.Label(tab1, text="Output name").grid(row=16, column=0, sticky="w", **PAD)
    ttk.Entry(tab1, textvariable=out_name_var, width=38).grid(
        row=16, column=1, columnspan=2, sticky="ew", **PAD)
    ttk.Label(tab1, text="(no extension)",
              foreground="#888888", font=("", 9)).grid(row=16, column=3, **PAD)

    tab1.columnconfigure(1, weight=1)
    tab1.columnconfigure(2, weight=1)

    # ─── Tab 3: Steps & Options ───────────────────────────────────────────
    tab3 = ttk.Frame(nb, padding=10)

    ttk.Label(tab3, text="Channel Mode",
              font=("", 13, "bold")).grid(row=0, column=0, columnspan=2,
                                          sticky="w", pady=(0, 4))

    ch_mode_descriptions = [
        ("single", "Single-channel", "single color; stoichiometry"),
        ("two",    "Two-channel",    "two color; colocalization and complex stoichiometry"),
        ("three",  "Three-channel",  "three color; colocalization and complex stoichiometry"),
    ]
    for i, (val, label, desc) in enumerate(ch_mode_descriptions, start=1):
        rb = ttk.Radiobutton(tab3, text=f"{label}  —  {desc}",
                             variable=channel_mode_var, value=val)
        rb.grid(row=i, column=0, columnspan=2, sticky="w", padx=8, pady=2)

    ttk.Separator(tab3, orient="horizontal").grid(
        row=4, column=0, columnspan=2, sticky="ew", pady=8)

    ttk.Label(tab3, text="Pipeline Steps",
              font=("", 13, "bold")).grid(row=5, column=0, columnspan=2,
                                          sticky="w", pady=(0, 8))

    step_checks = [
        ("Prepare TIF  (VSI → TIF + .dat)",  run_prepare_var),
        ("Colocalization",                    run_coloc_var),
        ("Bleaching analysis",                run_bleach_var),
    ]
    for i, (label, var) in enumerate(step_checks, start=6):
        ttk.Checkbutton(tab3, text=label, variable=var).grid(
            row=i, column=0, sticky="w", **PAD)

    bleach_cb = [w for w in tab3.winfo_children()
                 if isinstance(w, ttk.Checkbutton)
                 and "Bleaching" in str(w.cget("text"))][0]
    coloc_cb  = [w for w in tab3.winfo_children()
                 if isinstance(w, ttk.Checkbutton)
                 and "Colocalization" in str(w.cget("text"))][0]

    run_prepare_var.trace_add("write", _toggle_prepare)
    run_coloc_var.trace_add("write",   _toggle_coloc)

    ttk.Separator(tab3, orient="horizontal").grid(
        row=10, column=0, columnspan=2, sticky="ew", pady=8)

    ttk.Label(tab3, text="Options",
              font=("", 13, "bold")).grid(row=11, column=0, columnspan=2,
                                          sticky="w", pady=(0, 4))

    options = [
        ("Export .dat file (for LabView verification)", export_dat_var),
        ("Review flagged spots after colocalization",   review_flagged_var),
    ]
    for i, (label, var) in enumerate(options, start=12):
        ttk.Checkbutton(tab3, text=label, variable=var).grid(
            row=i, column=0, sticky="w", **PAD)

    ttk.Separator(tab3, orient="horizontal").grid(
        row=15, column=0, columnspan=2, sticky="ew", pady=8)

    ttk.Label(tab3, text="Bleaching Populations",
              font=("", 13, "bold")).grid(row=16, column=0, columnspan=2,
                                          sticky="w", pady=(0, 2))
    ttk.Label(tab3, text="Choose which spot groups to review interactively:",
              foreground="#555555", font=("", 9)).grid(
        row=17, column=0, columnspan=2, sticky="w", padx=8)

    bleach_pop_checks = []
    bleach_populations = [
        ("Colocalized spots (Ch1 + Ch2)", bleach_coloc_var),
        ("Unmatched Ch1",                 bleach_unmatched_ch1_var),
        ("Unmatched Ch2",                 bleach_unmatched_ch2_var),
    ]
    for i, (label, var) in enumerate(bleach_populations, start=18):
        cb = ttk.Checkbutton(tab3, text=label, variable=var)
        cb.grid(row=i, column=0, sticky="w", padx=24, pady=1)
        bleach_pop_checks.append(cb)

    def _toggle_bleach_pops(*_):
        """Grey out population checkboxes when bleaching step is disabled.
        Also grey out unmatched checkboxes in single-channel mode (no matching occurs)."""
        bleach_on = run_bleach_var.get()
        sc = _is_single_ch()
        # index 0 = Colocalized, 1 = Unmatched Ch1, 2 = Unmatched Ch2
        for i, cb in enumerate(bleach_pop_checks):
            if not bleach_on:
                cb.config(state="disabled")
            elif sc and i in (1, 2):
                cb.config(state="disabled")
            else:
                cb.config(state="normal")
        for w in _stoich_widgets:
            w.config(state="normal" if bleach_on else "disabled")

    run_bleach_var.trace_add("write", _toggle_bleach_pops)

    # ── Stoichiometry range ────────────────────────────────────────────────
    ttk.Separator(tab3, orient="horizontal").grid(
        row=21, column=0, columnspan=2, sticky="ew", pady=(8, 4))

    ttk.Label(tab3, text="Stoichiometry Range",
              font=("", 13, "bold")).grid(row=22, column=0, columnspan=2,
                                          sticky="w", pady=(0, 2))
    ttk.Label(tab3,
              text="Max expected stoichiometry per channel — steps above this are classified as aggregate.",
              foreground="#555555", font=("", 9)).grid(
        row=23, column=0, columnspan=2, sticky="w", padx=8)

    _stoich_widgets = []

    def _add_stoich_row(row, label, var, state_depends_on_channel=None):
        lbl = ttk.Label(tab3, text=label)
        lbl.grid(row=row, column=0, sticky="w", padx=8, pady=2)
        sb  = ttk.Spinbox(tab3, textvariable=var, from_=1, to=20, width=5)
        sb.grid(row=row, column=1, sticky="w", padx=8, pady=2)
        _stoich_widgets.extend([lbl, sb])
        return lbl, sb

    ch1_stoich_lbl, ch1_stoich_sb = _add_stoich_row(24, "Max stoich  Ch1", max_stoich_ch1_var)
    ch2_stoich_lbl, ch2_stoich_sb = _add_stoich_row(25, "Max stoich  Ch2", max_stoich_ch2_var)
    ch3_stoich_lbl, ch3_stoich_sb = _add_stoich_row(26, "Max stoich  Ch3", max_stoich_ch3_var)

    def _update_stoich_visibility(*_):
        """Show/hide ch2/ch3 stoich spinboxes based on channel mode."""
        sc  = _is_single_ch()
        thc = _is_three_ch()
        for w in [ch2_stoich_lbl, ch2_stoich_sb]:
            w.grid() if not sc else w.grid_remove()
        for w in [ch3_stoich_lbl, ch3_stoich_sb]:
            w.grid() if thc else w.grid_remove()

    channel_mode_var.trace_add("write", _update_stoich_visibility)
    _update_stoich_visibility()

    ttk.Separator(tab3, orient="horizontal").grid(
        row=27, column=0, columnspan=2, sticky="ew", pady=(8, 4))

    ttk.Label(tab3, text="Spot Limit for Bleaching Analysis",
              font=("", 13, "bold")).grid(row=28, column=0, columnspan=2,
                                          sticky="w", pady=(0, 2))
    ttk.Label(tab3,
              text="Max spots to extract traces for. 0 = analyze all spots.",
              foreground="#555555", font=("", 9)).grid(
        row=29, column=0, columnspan=2, sticky="w", padx=8)

    ttk.Label(tab3, text="Max spots").grid(row=30, column=0, sticky="w", padx=8, pady=2)
    ttk.Spinbox(tab3, textvariable=max_spots_var, from_=0, to=99999,
                width=7).grid(row=30, column=1, sticky="w", padx=8, pady=2)
    tab_batch = ttk.Frame(nb, padding=10)

    ttk.Label(tab_batch, text="Batch TIF Creation",
              font=("", 13, "bold")).grid(row=0, column=0, columnspan=3,
                                          sticky="w", pady=(0, 4))

    # Channel mode toggle
    batch_single_ch_var = tk.BooleanVar(value=False)
    ch_mode_frame = ttk.Frame(tab_batch)
    ch_mode_frame.grid(row=1, column=0, columnspan=3, sticky="w", padx=8, pady=(0, 4))
    ttk.Radiobutton(ch_mode_frame, text="Two-channel",
                    variable=batch_single_ch_var, value=False,
                    command=lambda: _batch_update_hint()).pack(side="left", padx=(0, 12))
    ttk.Radiobutton(ch_mode_frame, text="Single-channel",
                    variable=batch_single_ch_var, value=True,
                    command=lambda: _batch_update_hint()).pack(side="left")

    batch_hint_var = tk.StringVar()
    ttk.Label(tab_batch, textvariable=batch_hint_var,
              foreground="#555555", font=("", 9)).grid(
        row=2, column=0, columnspan=3, sticky="w", padx=8, pady=(0, 4))

    batch_export_dat_var = tk.BooleanVar(value=False)
    ttk.Checkbutton(tab_batch, text="Export .dat file for each TIF (for LabView verification)",
                    variable=batch_export_dat_var).grid(
        row=3, column=0, columnspan=3, sticky="w", padx=8, pady=(0, 8))

    def _batch_update_hint(*_):
        if batch_single_ch_var.get():
            batch_hint_var.set(
                "Convention: {ch1_protein}-{ch1_dye}_{condition}_movie{N}_ch1.vsi")
        else:
            batch_hint_var.set(
                "Convention: {ch1_protein}-{ch1_dye}_{ch2_protein}-{ch2_dye}_{condition}_movie{N}_ch1/ch2.vsi")
    _batch_update_hint()

    batch_folder_var = tk.StringVar()
    batch_out_var    = tk.StringVar()

    ttk.Label(tab_batch, text="VSI folder").grid(row=4, column=0, sticky="w", **PAD)
    ttk.Entry(tab_batch, textvariable=batch_folder_var, width=45).grid(
        row=4, column=1, sticky="ew", **PAD)
    ttk.Button(tab_batch, text="Browse…",
               command=lambda: _batch_browse_folder()).grid(row=4, column=2, **PAD)

    ttk.Label(tab_batch, text="Output folder").grid(row=5, column=0, sticky="w", **PAD)
    ttk.Entry(tab_batch, textvariable=batch_out_var, width=45).grid(
        row=5, column=1, sticky="ew", **PAD)
    ttk.Button(tab_batch, text="Browse…",
               command=lambda: _batch_browse_out()).grid(row=5, column=2, **PAD)

    ttk.Button(tab_batch, text="🔍  Scan for files",
               command=lambda: _batch_scan()).grid(
        row=6, column=0, columnspan=3, sticky="w", padx=8, pady=(6, 2))

    # Pairs listbox
    ttk.Label(tab_batch, text="Discovered files:").grid(
        row=7, column=0, columnspan=3, sticky="w", **PAD)

    pair_frame = ttk.Frame(tab_batch)
    pair_frame.grid(row=8, column=0, columnspan=3, sticky="nsew", padx=8, pady=2)
    pair_sb = ttk.Scrollbar(pair_frame, orient="vertical")
    pair_lb = tk.Listbox(pair_frame, height=10, yscrollcommand=pair_sb.set,
                         font=("Courier", 9), selectmode="extended")
    pair_sb.config(command=pair_lb.yview)
    pair_lb.pack(side="left", fill="both", expand=True)
    pair_sb.pack(side="right", fill="y")

    batch_status_var = tk.StringVar(value="")
    ttk.Label(tab_batch, textvariable=batch_status_var,
              foreground="#555555", font=("", 9)).grid(
        row=9, column=0, columnspan=3, sticky="w", **PAD)

    # Batch progress log
    ttk.Label(tab_batch, text="Log:").grid(row=10, column=0, sticky="w", **PAD)
    batch_log_box = scrolledtext.ScrolledText(tab_batch, width=70, height=8,
                                              font=("Courier", 9), state="disabled")
    batch_log_box.grid(row=11, column=0, columnspan=3, sticky="nsew", **PAD)

    btn_batch_run = ttk.Button(tab_batch, text="▶  Create All TIFs", width=20,
                               state="disabled")
    btn_batch_run.grid(row=12, column=0, sticky="w", padx=8, pady=6)

    tab_batch.columnconfigure(1, weight=1)
    tab_batch.rowconfigure(8, weight=1)
    tab_batch.rowconfigure(11, weight=1)

    # Discovered entries: list of (stem, ch1_path, ch2_path_or_None)
    _batch_pairs = []

    def _batch_log(msg):
        batch_log_box.config(state="normal")
        batch_log_box.insert("end", msg + "\n")
        batch_log_box.see("end")
        batch_log_box.config(state="disabled")
        root.update_idletasks()

    def _batch_browse_folder():
        p = filedialog.askdirectory(title="Select folder containing VSI files")
        if p:
            batch_folder_var.set(p)
            if not batch_out_var.get():
                batch_out_var.set(p)

    def _batch_browse_out():
        p = filedialog.askdirectory(title="Select output folder for TIF files")
        if p:
            batch_out_var.set(p)

    def _batch_scan():
        folder = Path(batch_folder_var.get())
        if not folder.exists():
            messagebox.showerror("Batch", "Folder not found.")
            return

        vsi_files = sorted(folder.glob("*.vsi"))
        ch1_files = [f for f in vsi_files if f.stem.endswith("_ch1")]

        _batch_pairs.clear()
        pair_lb.delete(0, "end")

        if batch_single_ch_var.get():
            # Single-channel: accept any .vsi file.
            # Files ending in _ch1 have the suffix stripped for the stem;
            # all others use the full filename stem as-is.
            sc_files = ch1_files if ch1_files else vsi_files
            for f in sc_files:
                stem = f.stem[:-4] if f.stem.endswith("_ch1") else f.stem
                _batch_pairs.append((stem, f, None))
                pair_lb.insert("end", f"  {stem}")
            batch_status_var.set(f"Found {len(sc_files)} single-channel file(s)")
        else:
            # Two-channel: pair ch1 with matching ch2
            paired = []
            unpaired = []
            for f1 in ch1_files:
                stem = f1.stem[:-4]
                f2 = folder / f"{stem}_ch2.vsi"
                if f2.exists():
                    paired.append((stem, f1, f2))
                    pair_lb.insert("end", f"  {stem}")
                else:
                    unpaired.append(f1.name)
            _batch_pairs.extend(paired)
            status = f"Found {len(paired)} pair(s)"
            if unpaired:
                status += f"  |  {len(unpaired)} unpaired: {', '.join(unpaired)}"
            batch_status_var.set(status)

        btn_batch_run.config(state="normal" if _batch_pairs else "disabled")

    def _batch_run():
        if not _batch_pairs:
            return
        out_dir = Path(batch_out_var.get())
        if not out_dir:
            messagebox.showerror("Batch", "Please select an output folder.")
            return
        out_dir.mkdir(parents=True, exist_ok=True)

        btn_batch_run.config(state="disabled")
        cfg = load_config()
        single_ch = batch_single_ch_var.get()
        export_dat = batch_export_dat_var.get()

        def _batch_export_dat(tif_path: Path, is_single: bool):
            DAT_FRAMES_PER_CHANNEL = 350
            try:
                import tifffile as _tiff2
                # numpy available as np (top-level import)2
                stack = _tiff2.imread(str(tif_path))
                total = stack.shape[0]
                fpc = total if is_single else total // 2
                n   = min(DAT_FRAMES_PER_CHANNEL, fpc)
                if is_single:
                    dat = stack[:n]
                else:
                    ch1 = stack[:fpc][:n]
                    ch2 = stack[fpc:][:n]
                    dat = _np2.concatenate([ch1, ch2], axis=0)
                dat_path = tif_path.with_suffix(".dat")
                dat.astype(_np2.uint16).tofile(str(dat_path))
                if n < DAT_FRAMES_PER_CHANNEL:
                    _batch_log(f"  ⚠ .dat: only {fpc} frames/channel — exported {n}/channel")
                else:
                    _batch_log(f"  ✓ .dat → {dat_path.name}")
            except Exception as _e:
                _batch_log(f"  ✗ .dat export failed: {_e}")

        def _do_batch():
            n = len(_batch_pairs)
            for i, (stem, ch1_path, ch2_path) in enumerate(_batch_pairs, 1):
                _batch_log(f"\n[{i}/{n}] {stem}")
                tif_path = out_dir / f"{stem}.tif"
                cmd = [
                    sys.executable,
                    str(SCRIPTS_DIR / "prepare_tif.py"),
                    "--ch1",    str(ch1_path),
                    "--output", str(tif_path),
                    "--blank-frames",       str(cfg["blank_frames"]),
                    "--peak-threshold",     str(cfg["peak_threshold"]),
                    "--fiji",               str(cfg.get("fiji_path", "/Applications/Fiji.app")),
                ]
                if single_ch:
                    cmd.append("--single-channel")
                else:
                    cmd += ["--ch2", str(ch2_path)]
                try:
                    proc = subprocess.Popen(
                        cmd,
                        stdout=subprocess.PIPE,
                        stderr=subprocess.STDOUT,
                        text=True, bufsize=1,
                    )
                    for line in proc.stdout:
                        _batch_log("  " + line.rstrip())
                    proc.wait()
                    if proc.returncode != 0:
                        _batch_log(f"  ✗ Failed (exit {proc.returncode})")
                        root.after(0, lambda idx=i-1: pair_lb.itemconfig(idx, fg="#cc0000"))
                    else:
                        _batch_log(f"  ✓ → {tif_path.name}")
                        root.after(0, lambda idx=i-1: pair_lb.itemconfig(idx, fg="#228B22"))
                        if export_dat:
                            _batch_export_dat(tif_path, single_ch)
                except Exception as e:
                    _batch_log(f"  ✗ Error: {e}")

            _batch_log(f"\n✓ Batch complete — {n} TIF(s) written to {out_dir}")
            root.after(0, lambda: btn_batch_run.config(state="normal"))

        threading.Thread(target=_do_batch, daemon=True).start()

    btn_batch_run.config(command=_batch_run)

    # ─── Summary tab ──────────────────────────────────────────────────────
    tab_summary = ttk.Frame(nb, padding=10)

    ttk.Label(tab_summary, text="Condition Summary",
              font=("TkDefaultFont", 11, "bold")).grid(
        row=0, column=0, columnspan=3, sticky="w", pady=(0, 6))

    ttk.Label(tab_summary,
              text="Select a condition folder to scan all movie subfolders for\n"
                   "*_summary.txt files and export a formatted Excel workbook.",
              foreground="#555555").grid(
        row=1, column=0, columnspan=3, sticky="w", pady=(0, 10))

    ttk.Separator(tab_summary, orient="horizontal").grid(
        row=2, column=0, columnspan=3, sticky="ew", pady=(0, 10))

    # Folder picker row
    summary_folder_var = tk.StringVar()
    ttk.Label(tab_summary, text="Condition folder").grid(
        row=3, column=0, sticky="w", padx=(0, 6), pady=3)
    summary_folder_entry = ttk.Entry(tab_summary, textvariable=summary_folder_var, width=42)
    summary_folder_entry.grid(row=3, column=1, sticky="ew", pady=3)
    ttk.Button(tab_summary, text="Browse…",
               command=lambda: summary_folder_var.set(
                   filedialog.askdirectory(title="Select condition folder") or
                   summary_folder_var.get()
               )).grid(row=3, column=2, sticky="w", padx=(6, 0), pady=3)

    ttk.Separator(tab_summary, orient="horizontal").grid(
        row=4, column=0, columnspan=3, sticky="ew", pady=(10, 10))

    # Status label and Run button
    summary_status_var = tk.StringVar(value="")
    ttk.Label(tab_summary, textvariable=summary_status_var,
              foreground="#2a7a2a", wraplength=420, justify="left").grid(
        row=5, column=0, columnspan=3, sticky="w", pady=(0, 8))

    def _run_summary():
        folder = summary_folder_var.get().strip()
        if not folder:
            summary_status_var.set("⚠  Please select a condition folder first.")
            return
        if not os.path.isdir(folder):
            summary_status_var.set("⚠  Folder not found.")
            return

        summary_status_var.set("Running…")
        tab_summary.update_idletasks()

        script = os.path.join(os.path.dirname(os.path.abspath(__file__)), "make_summary.py")
        if not os.path.isfile(script):
            summary_status_var.set("⚠  make_summary.py not found next to pipeline.py.")
            return

        result = subprocess.run(
            [sys.executable, script, folder],
            capture_output=True, text=True
        )
        if result.returncode == 0:
            # Extract the saved path from stdout for a helpful message
            saved_line = next(
                (l for l in result.stdout.splitlines() if l.startswith("Saved:")), "")
            saved_path = saved_line.replace("Saved:", "").strip()
            summary_status_var.set(
                f"✓  Done.{('  Saved to: ' + saved_path) if saved_path else ''}")
        else:
            err = (result.stderr or result.stdout or "Unknown error").strip()
            summary_status_var.set(f"✗  Error: {err[:300]}")

    ttk.Button(tab_summary, text="Generate Summary Excel",
               command=_run_summary).grid(
        row=6, column=0, columnspan=3, pady=(0, 4))

    tab_summary.columnconfigure(1, weight=1)

    # ─── Progress tab ─────────────────────────────────────────────────────
    tab4 = ttk.Frame(nb, padding=10)
    nb.add(tab_batch, text="  Batch TIF  ")
    nb.add(tab3, text="  Steps & Options  ")
    nb.add(tab1, text="  Setup  ")
    nb.add(tab4, text="  Progress  ")
    nb.add(tab_summary, text="  Summary  ")
    nb.select(tab3)

    ttk.Label(tab4, text="Pipeline Progress",
              font=("", 13, "bold")).grid(row=0, column=0, columnspan=2,
                                          sticky="w", pady=(0, 6))

    # ── Colour palette ────────────────────────────────────────────────────
    STATUS_COLORS = {
        "waiting": "#999999",
        "running": "#1a6fbf",
        "done":    "#228B22",
        "skip":    "#aaaaaa",
        "error":   "#cc2222",
    }
    BOX_BG = {
        "waiting": "#e8e8e8",
        "running": "#d6eaf8",
        "done":    "#d5f0d5",
        "skip":    "#eeeeee",
        "error":   "#fde8e8",
    }
    BOX_BORDER = {
        "waiting": "#bbbbbb",
        "running": "#1a6fbf",
        "done":    "#228B22",
        "skip":    "#cccccc",
        "error":   "#cc2222",
    }

    # ── Pipeline canvas — three boxes connected by arrows ─────────────────
    STEP_NAMES  = ["Prepare TIF", "Colocalization", "Bleaching"]
    STEP_SUBLBL = ["VSI → TIF", "spot detection", "step counting"]
    N_STEPS     = len(STEP_NAMES)
    BOX_H       = 86        # taller boxes
    ARROW_W     = 40
    PAD_X       = 10        # canvas left/right padding
    CANVAS_H    = BOX_H + 22  # extra for pulse dot

    _step_status = ["waiting"] * N_STEPS   # track current status for redraws

    pipeline_canvas = tk.Canvas(tab4, height=CANVAS_H,
                                bg="#f0f2f5", highlightthickness=1,
                                highlightbackground="#cccccc")
    pipeline_canvas.grid(row=1, column=0, columnspan=2,
                         sticky="ew", padx=8, pady=(0, 6))

    # Canvas item IDs — populated in _draw_pipeline()
    box_rects  = [None] * N_STEPS
    box_badges = [None] * N_STEPS   # filled circle behind step number
    box_icons  = [None] * N_STEPS
    box_title  = [None] * N_STEPS
    box_sub    = [None] * N_STEPS
    arrow_lines= [None] * (N_STEPS - 1)
    arrow_heads= [None] * (N_STEPS - 1)
    pulse_dots = [None] * N_STEPS

    def _box_geom(total_w):
        """Return list of (x0, x1) for each box given the canvas width."""
        avail = total_w - 2 * PAD_X - (N_STEPS - 1) * ARROW_W
        bw    = max(80, avail // N_STEPS)
        xs = []
        x  = PAD_X
        for i in range(N_STEPS):
            xs.append((x, x + bw))
            x += bw + ARROW_W
        return xs, bw

    def _draw_pipeline(event=None):
        """Full redraw — called on first render and every resize."""
        w = pipeline_canvas.winfo_width()
        if w < 10:
            return
        pipeline_canvas.delete("all")
        geom, bw = _box_geom(w)

        for i in range(N_STEPS):
            status = _step_status[i]
            x0, x1 = geom[i]
            cx = (x0 + x1) // 2
            y0, y1 = 4, 4 + BOX_H

            # Box with rounded-ish feel via overlapping rects (Tkinter has no rounded rect)
            box_rects[i] = pipeline_canvas.create_rectangle(
                x0, y0, x1, y1,
                fill=BOX_BG[status], outline=BOX_BORDER[status],
                width=2)

            # Step number badge — drawn circle + number for crisp scaling
            badge_r = 14
            bx, by = x0 + 20, y0 + 20
            box_badges[i] = pipeline_canvas.create_oval(
                bx - badge_r, by - badge_r, bx + badge_r, by + badge_r,
                fill=STATUS_COLORS[status], outline="")
            box_icons[i] = pipeline_canvas.create_text(
                bx, by,
                text=str(i + 1), font=("", 13, "bold"),
                fill="white")

            # Step name — bold, larger
            box_title[i] = pipeline_canvas.create_text(
                cx, y0 + 30,
                text=STEP_NAMES[i], font=("", 13, "bold"),
                fill=STATUS_COLORS[status])

            # Sub-label — readable, muted
            box_sub[i] = pipeline_canvas.create_text(
                cx, y0 + 58,
                text=STEP_SUBLBL[i], font=("", 11),
                fill="#888888" if status == "waiting" else STATUS_COLORS[status])

            # Pulse dot below box
            dot_cx = cx
            dot_y0, dot_y1 = y1 + 5, y1 + 13
            pulse_dots[i] = pipeline_canvas.create_oval(
                dot_cx - 4, dot_y0, dot_cx + 4, dot_y1,
                fill="", outline="")

        # Arrows between boxes
        for i in range(N_STEPS - 1):
            _, x_left  = geom[i]
            x_right, _ = geom[i + 1]
            ax0 = x_left  + 4
            ax1 = x_right - 4
            ay  = 4 + BOX_H // 2
            arrow_lines[i] = pipeline_canvas.create_line(
                ax0, ay, ax1, ay,
                fill="#aaaaaa", width=3,
                arrow="last", arrowshape=(14, 17, 5))

    pipeline_canvas.bind("<Configure>", _draw_pipeline)

    # Pulse animation state
    _pulse_active = [False] * N_STEPS
    _pulse_phase  = [0]     * N_STEPS

    def _pulse_tick():
        for i in range(N_STEPS):
            if _pulse_active[i] and pulse_dots[i] is not None:
                _pulse_phase[i] = (_pulse_phase[i] + 1) % 10
                bright = _pulse_phase[i] < 5
                fill = STATUS_COLORS["running"] if bright else "#a8c8e8"
                pipeline_canvas.itemconfig(pulse_dots[i], fill=fill, outline=fill)
        if any(_pulse_active):
            root.after(150, _pulse_tick)

    def _update_box(idx: int, status: str):
        """Update one box's colour; redraw all items for that box."""
        _step_status[idx] = status
        if box_rects[idx] is None:
            return
        pipeline_canvas.itemconfig(box_rects[idx],
                                   fill=BOX_BG[status],
                                   outline=BOX_BORDER[status])
        pipeline_canvas.itemconfig(box_badges[idx], fill=STATUS_COLORS[status])
        for item in (box_icons[idx], box_title[idx]):
            pipeline_canvas.itemconfig(item, fill=STATUS_COLORS[status])
        pipeline_canvas.itemconfig(box_sub[idx],
                                   fill="#888888" if status == "waiting"
                                   else STATUS_COLORS[status])
        was_any = any(_pulse_active)
        _pulse_active[idx] = (status == "running")
        if status != "running" and pulse_dots[idx] is not None:
            pipeline_canvas.itemconfig(pulse_dots[idx], fill="", outline="")
            _pulse_phase[idx] = 0
        if _pulse_active[idx] and not was_any:
            root.after(150, _pulse_tick)

    # ── Colour legend ─────────────────────────────────────────────────────
    legend_frame = ttk.Frame(tab4)
    legend_frame.grid(row=2, column=0, columnspan=2, sticky="w", padx=8, pady=(4, 2))

    legend_items = [
        ("waiting", "Waiting"),
        ("running", "Running"),
        ("done",    "Complete"),
        ("skip",    "Skipped"),
        ("error",   "Error"),
    ]
    for item_val, item_label in legend_items:
        dot_canvas = tk.Canvas(legend_frame, width=16, height=16,
                               highlightthickness=0)
        dot_canvas.pack(side="left", padx=(6, 2))
        dot_canvas.create_oval(1, 1, 15, 15,
                               fill=BOX_BG[item_val],
                               outline=BOX_BORDER[item_val], width=2)
        ttk.Label(legend_frame, text=item_label,
                  foreground=STATUS_COLORS[item_val],
                  font=("", 11)).pack(side="left", padx=(0, 8))

    # ── Log box with tagged stage headers ─────────────────────────────────
    ttk.Separator(tab4, orient="horizontal").grid(
        row=3, column=0, columnspan=2, sticky="ew", pady=(4, 2))

    log_hdr_frame = ttk.Frame(tab4)
    log_hdr_frame.grid(row=4, column=0, columnspan=2, sticky="ew", padx=8)
    ttk.Label(log_hdr_frame, text="Log", font=("", 11, "bold")).pack(side="left")
    ttk.Label(log_hdr_frame, text="timestamps in HH:MM:SS",
              foreground="#aaaaaa", font=("", 10)).pack(side="left", padx=8)

    log_box = scrolledtext.ScrolledText(tab4, width=70, height=16,
                                        font=("Courier", 11), state="disabled",
                                        bg="#1e1e2e", fg="#cdd6f4",
                                        insertbackground="#cdd6f4",
                                        selectbackground="#45475a",
                                        relief="flat", borderwidth=0)
    log_box.grid(row=5, column=0, columnspan=2, sticky="nsew", padx=8, pady=(2, 0))

    # Text tags for log styling
    log_box.tag_configure("ts",       foreground="#6c7086")
    log_box.tag_configure("stage",    foreground="#89b4fa",
                          font=("Courier", 11, "bold"))
    log_box.tag_configure("ok",       foreground="#a6e3a1")
    log_box.tag_configure("warn",     foreground="#f9e2af")
    log_box.tag_configure("err",      foreground="#f38ba8")
    log_box.tag_configure("complete", foreground="#a6e3a1",
                          font=("Courier", 11, "bold"))

    tab4.rowconfigure(5, weight=1)
    tab4.columnconfigure(1, weight=1)

    # ── Run / Archive buttons ──────────────────────────────────────────────
    btn_frame = ttk.Frame(root)
    btn_frame.pack(fill="x", padx=10, pady=(0, 10))

    btn_run     = ttk.Button(btn_frame, text="▶  Run Pipeline", width=20)
    btn_archive = ttk.Button(btn_frame, text="📦  Archive TIF",
                             width=18, state="disabled")
    btn_run.pack(side="left", padx=4)
    btn_archive.pack(side="left", padx=4)

    # ── Logging helpers ────────────────────────────────────────────────────
    import datetime as _dt

    def log(msg: str):
        from datetime import datetime as _now
        ts = _now.now().strftime("%H:%M:%S")
        log_box.config(state="normal")
        start = log_box.index("end-1c")

        # Choose tag based on content
        stripped = msg.strip()
        if stripped.startswith("──") or stripped.startswith("=="):
            tag = "stage"
        elif stripped.startswith("✓") or stripped.startswith("✔"):
            tag = "ok"
        elif stripped.startswith("⚠") or stripped.startswith("Warning"):
            tag = "warn"
        elif stripped.startswith("✗") or stripped.startswith("Error"):
            tag = "err"
        elif "Pipeline complete" in stripped or "complete" in stripped.lower() and "=" in stripped:
            tag = "complete"
        else:
            tag = None

        # Insert timestamp in muted colour, then the message
        log_box.insert("end", f"[{ts}] ", ("ts",))
        end_ts = log_box.index("end-1c")
        if tag:
            log_box.insert("end", msg + "\n", (tag,))
        else:
            log_box.insert("end", msg + "\n")
        log_box.see("end")
        log_box.config(state="disabled")
        root.update_idletasks()

    def set_step(idx: int, status: str, color: str = ""):
        _update_box(idx, status)

    # ── Validate inputs ────────────────────────────────────────────────────
    def _validate() -> "str | None":
        """Returns error string or None if valid."""
        if not protein_a_var.get().strip():
            return "Please enter Protein A."
        if not _is_single_ch() and not protein_b_var.get().strip():
            return "Please enter Protein B."
        if not out_folder_var.get().strip():
            return "Please select an output folder."
        if not out_name_var.get().strip():
            return "Please enter an output filename."
        if run_prepare_var.get():
            if not ch1_var.get().strip() or not Path(ch1_var.get()).exists():
                return "Channel 1 .vsi file not found."
            if not _is_single_ch():
                if not ch2_var.get().strip() or not Path(ch2_var.get()).exists():
                    return "Channel 2 .vsi file not found."
            if _is_three_ch():
                if not ch3_var.get().strip() or not Path(ch3_var.get()).exists():
                    return "Channel 3 .vsi file not found."
        else:
            if not existing_tif_var.get().strip() or \
               not Path(existing_tif_var.get()).exists():
                return "Selected .tif file not found."
        return None

    # ── Archive button ──────────────────────────────────────────────────────
    _tif_path_for_archive = [None]

    def _archive():
        tif = _tif_path_for_archive[0]
        if not tif or not Path(tif).exists():
            messagebox.showerror("Archive", "TIF file not found.")
            return
        dest_dir = filedialog.askdirectory(title="Select archive destination (external drive)")
        if not dest_dir:
            return
        import shutil
        dest = Path(dest_dir) / Path(tif).name
        log(f"\nArchiving {Path(tif).name} → {dest_dir} ...")
        try:
            shutil.move(str(tif), str(dest))
            log(f"✓ Archived. TIF removed from local drive.")
            btn_archive.config(state="disabled")
            messagebox.showinfo("Archive", f"TIF moved to:\n{dest}")
        except Exception as e:
            log(f"✗ Archive failed: {e}")
            messagebox.showerror("Archive error", str(e))

    btn_archive.config(command=_archive)

    # ── Run pipeline ────────────────────────────────────────────────────────
    def _run():
        err = _validate()
        if err:
            messagebox.showerror("Input error", err)
            return

        btn_run.config(state="disabled")
        nb.select(tab4)

        # Reset step indicators and clear log
        log_box.config(state="normal")
        log_box.delete("1.0", "end")
        log_box.config(state="disabled")
        for i in range(3):
            set_step(i, "waiting")

        # Save current field values so they're restored on next launch
        _cfg = load_config()
        _cfg["_last_gui"] = {
            "protein_a":    protein_a_var.get(),
            "protein_b":    protein_b_var.get(),
            "other":        other_var.get(),
            "replicate":    replicate_var.get(),
            "channel_mode": channel_mode_var.get(),
        }
        save_config(_cfg)

        log("=" * 50)
        log(f"SiMPull Pipeline starting")
        log(f"Condition : {condition_var.get()}")
        ch_mode_str = {"two": "Two-channel", "three": "Three-channel", "single": "Single-channel"}.get(channel_mode_var.get(), "?")
        log(f"Mode      : {ch_mode_str}")
        log(f"Output    : {out_folder_var.get()}")
        log("=" * 50)

        def _pipeline():
            try:
                _run_pipeline()
            except Exception:
                log(f"\n✗ Unexpected error:\n{traceback.format_exc()}")
                root.after(0, lambda: btn_run.config(state="normal"))

        threading.Thread(target=_pipeline, daemon=True).start()

    def _run_pipeline():
        cfg     = load_config()
        key     = condition_key(date_var.get(), protein_a_var.get(),
                                protein_b_var.get(), other_var.get())
        params, _ = load_condition_params(key, cfg)
        out_dir = Path(out_folder_var.get())
        out_dir.mkdir(parents=True, exist_ok=True)
        stem    = out_name_var.get().strip()
        tif_path = out_dir / f"{stem}.tif"

        # ── Step 1: Prepare TIF ──────────────────────────────────────────
        if run_prepare_var.get():
            set_step(0, "running")
            log("\n── Step 1: Prepare TIF ──")
            if _is_single_ch():
                cmd = [
                    sys.executable,
                    str(SCRIPTS_DIR / "prepare_tif.py"),
                    "--single-channel",
                    "--ch1",    ch1_var.get(),
                    "--output", str(tif_path),
                    "--blank-frames",       str(cfg["blank_frames"]),
                    "--peak-threshold",     str(cfg["peak_threshold"]),
                    "--fiji",               str(cfg["fiji_path"]),
                ]
            elif _is_three_ch():
                cmd = [
                    sys.executable,
                    str(SCRIPTS_DIR / "prepare_tif.py"),
                    "--ch1",    ch1_var.get(),
                    "--ch2",    ch2_var.get(),
                    "--ch3",    ch3_var.get(),
                    "--output", str(tif_path),
                    "--blank-frames",       str(cfg["blank_frames"]),
                    "--peak-threshold",     str(cfg["peak_threshold"]),
                    "--fiji",               str(cfg["fiji_path"]),
                ]
            else:
                cmd = [
                    sys.executable,
                    str(SCRIPTS_DIR / "prepare_tif.py"),
                    "--ch1",    ch1_var.get(),
                    "--ch2",    ch2_var.get(),
                    "--output", str(tif_path),
                    "--blank-frames",       str(cfg["blank_frames"]),
                    "--peak-threshold",     str(cfg["peak_threshold"]),
                    "--fiji",               str(cfg["fiji_path"]),
                ]
            ok = _run_subprocess(cmd)
            if not ok:
                set_step(0, "error")
                root.after(0, lambda: btn_run.config(state="normal"))
                return
            # Export .dat if requested
            if export_dat_var.get():
                log("  Exporting .dat for LabView...")
                _export_dat(tif_path)
            set_step(0, "done")
            _tif_path_for_archive[0] = str(tif_path)
        else:
            tif_path = Path(existing_tif_var.get())
            set_step(0, "skip")
            log("\n── Step 1: Prepare TIF — skipped (using existing TIF)")
            _tif_path_for_archive[0] = str(tif_path)

        # ── Step 2: Colocalization ───────────────────────────────────────
        if run_coloc_var.get():
            set_step(1, "running")
            log("\n── Step 2: Colocalization ──")

            import tempfile as _tmp, json as _json
            log(f"  Opening tune UI...")

            # ── ML parameter prediction ───────────────────────────────────
            _model_path = get_params_dir() / "models" / "param_predictor.pkl"
            if not _model_path.exists():
                log("  ML model not found — using condition defaults.")
            else:
                _ml_base = dict(cfg)
                _ml_base.update({k: params[k] for k in params
                                 if k not in CONDITION_PARAM_KEYS})
                _ml_base["single_channel"] = _is_single_ch()
                _ml_params = try_ml_predict_params(tif_path, _ml_base)
                _predicted = {k: v for k, v in _ml_params.items()
                              if k in CONDITION_PARAM_KEYS
                              and v != _ml_base.get(k)}
                if _predicted:
                    params.update(_predicted)
                    log("  ML predicted params: " +
                        ", ".join(f"{k}={v:.5f}" for k, v in _predicted.items()))
                else:
                    log("  ML model loaded — predictions at default values.")
            if True:
                # Write tuned params to a temp JSON file so the UI can pass
                # them back to this process after the subprocess exits
                fd, params_out_path = _tmp.mkstemp(suffix=".json", prefix="smfret_tune_")
                import os as _os; _os.close(fd)
                params_out_path = Path(params_out_path)
                tune_cmd = build_coloc_args(tif_path, out_dir, params,
                                            mode="tune",
                                            review_flagged=False,
                                            params_out=params_out_path,
                                            single_channel=_is_single_ch(),
                                            three_channel=_is_three_ch())
                ok = _run_subprocess(tune_cmd)
                params_out_path_local = params_out_path  # capture for lambda
                if not ok:
                    params_out_path_local.unlink(missing_ok=True)
                    set_step(1, "error")
                    root.after(0, lambda: btn_run.config(state="normal"))
                    return
                # Read tuned params back and merge into params dict
                if params_out_path_local.exists():
                    with open(params_out_path_local) as _f:
                        tuned = _json.load(_f)
                    params_out_path_local.unlink(missing_ok=True)
                    # Map tune UI keys -> params dict keys
                    key_map = {
                        "psf_radius":             "psf_radius",
                        "ch1_threshold":          "ch1_threshold",
                        "ch2_threshold":          "ch2_threshold",
                        "coloc_threshold":        "coloc_threshold",
                        "ch1_intensity_min_mult": "ch1_intensity_min_mult",
                        "ch1_intensity_max_mult": "ch1_intensity_max_mult",
                        "ch2_intensity_min_mult": "ch2_intensity_min_mult",
                        "ch2_intensity_max_mult": "ch2_intensity_max_mult",
                    }
                    for tk_key, pk_key in key_map.items():
                        if tk_key in tuned:
                            params[pk_key] = tuned[tk_key]
                    log("  Tuned parameters loaded from UI.")
                else:
                    log("  Warning: tune UI closed without saving — using default parameters.")
                save_condition_params(key, params)

            # Run full colocalization (never pass --review-flagged here;
            # review is handled below in the main process after subprocess exits)
            log("  Running colocalization...")
            coloc_mode = "single-channel" if _is_single_ch() else "single"
            coloc_cmd = build_coloc_args(tif_path, out_dir, params,
                                         mode=coloc_mode,
                                         review_flagged=False,
                                         three_channel=_is_three_ch())
            ok = _run_subprocess(coloc_cmd)
            if not ok:
                set_step(1, "error")
                root.after(0, lambda: btn_run.config(state="normal"))
                return

            # Flagged spot review — must run on the main thread on macOS
            # (NSWindow cannot be created off the main thread).
            # We schedule it via root.after() and use a threading.Event to
            # block the pipeline thread until the review window closes.
            if review_flagged_var.get():
                import pandas as _pd
                _stem     = tif_path.stem
                _coloc_df = _pd.read_csv(out_dir / f"{_stem}_colocalized.csv")
                _u1_path  = out_dir / f"{_stem}_unmatched_ch1.csv"
                _u2_path  = out_dir / f"{_stem}_unmatched_ch2.csv"
                _u1_df    = _pd.read_csv(_u1_path) if _u1_path.exists() else _pd.DataFrame(columns=["x","y"])
                _u2_df    = _pd.read_csv(_u2_path) if _u2_path.exists() else _pd.DataFrame(columns=["x","y"])
                _has_flags = (
                    any(c.startswith("flagged") for c in _coloc_df.columns) or
                    "flagged" in _u1_df.columns or
                    "flagged" in _u2_df.columns
                )
                if _has_flags:
                    log("  Opening flagged spot review...")
                    _review_done  = threading.Event()
                    _review_single_ch = _is_single_ch()   # capture now, before thread

                    def _do_review():
                        try:
                            import sys as _sys
                            if str(SCRIPTS_DIR) not in _sys.path:
                                _sys.path.insert(0, str(SCRIPTS_DIR))
                            from colocalize_tif import run_flagged_review as _run_review
                            _run_review(
                                tif_path=tif_path,
                                coloc=_coloc_df, u1=_u1_df, u2=_u2_df,
                                out_dir=out_dir, stem=_stem,
                                psf_radius=params["psf_radius"],
                                single_channel=_review_single_ch,
                            )
                        finally:
                            _review_done.set()

                    root.after(0, _do_review)
                    _review_done.wait()   # block pipeline thread until review closes
                else:
                    log("  No flagged spots — skipping review.")

            set_step(1, "done")
        else:
            set_step(1, "skip")
            log("\n── Step 2: Colocalization — skipped")

        # ── Step 3: Bleaching Analysis ───────────────────────────────────
        if run_bleach_var.get():
            set_step(2, "running")
            log("\n── Step 3: Bleaching Analysis ──")
            # The colocalized CSV is always named after the TIF stem, not the
            # output name field — use tif_path.stem as the authoritative name.
            tif_stem  = tif_path.stem
            coloc_csv = out_dir / f"{tif_stem}_colocalized.csv"
            if not coloc_csv.exists():
                coloc_csv = out_dir / "colocalization_output" / \
                            f"{tif_stem}_colocalized.csv"
            if not coloc_csv.exists():
                # Last resort: scan folder for any *_colocalized.csv
                candidates = sorted(out_dir.glob("*_colocalized.csv"))
                if not candidates:
                    candidates = sorted(
                        (out_dir / "colocalization_output").glob("*_colocalized.csv"))
                if candidates:
                    coloc_csv = candidates[0]
                    log(f"  ⚠ Using: {coloc_csv.name}")
            if not coloc_csv.exists():
                log(f"  ✗ Colocalization CSV not found in {out_dir}")
                log(f"    (expected: {tif_stem}_colocalized.csv)")
                set_step(2, "error")
                root.after(0, lambda: btn_run.config(state="normal"))
                return

            # Build list of populations to skip based on unchecked checkboxes
            _skip = []
            if not bleach_coloc_var.get():
                _skip.append("coloc_ch1")
                _skip.append("coloc_ch2")
            if not bleach_unmatched_ch1_var.get(): _skip.append("unmatched_ch1")
            if not bleach_unmatched_ch2_var.get(): _skip.append("unmatched_ch2")

            # Inject stoichiometry range from GUI into params
            try:
                params["max_stoichiometry_ch1"] = max_stoich_ch1_var.get()
                params["max_stoichiometry_ch2"] = max_stoich_ch2_var.get()
                params["max_stoichiometry_ch3"] = max_stoich_ch3_var.get()
            except Exception:
                pass  # use defaults already in params

            bleach_cmd = build_bleaching_args(tif_path, coloc_csv,
                                              out_dir, params,
                                              three_channel=_is_three_ch(),
                                              skip_populations=_skip or None,
                                              max_spots=max_spots_var.get())
            ok = _run_subprocess(bleach_cmd)
            if not ok:
                set_step(2, "error")
                root.after(0, lambda: btn_run.config(state="normal"))
                return
            set_step(2, "done")
        else:
            set_step(2, "skip")
            log("\n── Step 3: Bleaching Analysis — skipped")

        # ── Log training data ─────────────────────────────────────────────
        _tif_stem = tif_path.stem
        try:
            log_movie_training_data(
                tif_path        = tif_path,
                params          = params,
                coloc_csv       = out_dir / f"{_tif_stem}_colocalized.csv",
                u1_csv          = out_dir / f"{_tif_stem}_unmatched_ch1.csv",
                u2_csv          = out_dir / f"{_tif_stem}_unmatched_ch2.csv",
                condition       = key,
                human_corrected = False,
                single_channel  = _is_single_ch(),
            )
            log("  Training data logged.")
        except Exception as _e:
            log(f"  Warning: could not log training data: {_e}")

        log("\n" + "=" * 50)
        log("✓ Pipeline complete!")
        log(f"  Output: {out_dir}")
        log("=" * 50)
        root.after(0, lambda: btn_run.config(state="normal"))
        root.after(0, lambda: btn_archive.config(state="normal"))

    def _run_subprocess(cmd: list) -> bool:
        """Run a subprocess, streaming output to the log. Returns True on success."""
        log(f"  $ {' '.join(Path(c).name if i == 1 else c for i, c in enumerate(cmd))}")
        try:
            proc = subprocess.Popen(
                cmd,
                stdout=subprocess.PIPE,
                stderr=subprocess.STDOUT,
                text=True,
                bufsize=1,
            )
            for line in proc.stdout:
                log("  " + line.rstrip())
            proc.wait()
            if proc.returncode != 0:
                log(f"  ✗ Process exited with code {proc.returncode}")
                return False
            return True
        except Exception as e:
            log(f"  ✗ Failed to run: {e}")
            return False

    def _export_dat(tif_path: Path):
        """
        Export a LabView-compatible .dat file from the .tif.
        LabView cannot read more than 700 frames (350 per channel for 2-channel),
        so this always exports exactly the first DAT_FRAMES_PER_CHANNEL frames of
        each channel block, regardless of how long the .tif is.
        For three-channel TIFs, LabView is not supported — a warning is logged.
        """
        DAT_FRAMES_PER_CHANNEL = 350
        try:
            import tifffile
            import numpy as np
            stack = tifffile.imread(str(tif_path))
            total = stack.shape[0]

            if _is_three_ch():
                # Three-channel: LabView does not support this layout.
                # Export ch1|ch2 only (first 700 frames) for partial compatibility.
                fpc = total // 3
                n   = min(DAT_FRAMES_PER_CHANNEL, fpc)
                ch1 = stack[:fpc][:n]
                ch2 = stack[fpc:fpc*2][:n]
                dat = np.concatenate([ch1, ch2], axis=0)
                dat_path = tif_path.with_suffix(".dat")
                dat.astype(np.uint16).tofile(str(dat_path))
                log(f"  ⚠ .dat export (three-channel): LabView does not support "
                    f"three-channel layout — exported Ch1+Ch2 only ({n*2} frames). "
                    f"Ch3 is omitted.")
                log(f"  ✓ Saved {dat_path.name}")
            else:
                fpc = total // 2
                n   = min(DAT_FRAMES_PER_CHANNEL, fpc)
                ch1 = stack[:fpc][:n]
                ch2 = stack[fpc:][:n]
                dat = np.concatenate([ch1, ch2], axis=0)
                dat_path = tif_path.with_suffix(".dat")
                dat.astype(np.uint16).tofile(str(dat_path))
                if n < DAT_FRAMES_PER_CHANNEL:
                    log(f"  ⚠ .dat export: movie only has {fpc} frames/channel — "
                        f"exported {n}/channel instead of {DAT_FRAMES_PER_CHANNEL}")
                else:
                    log(f"  ✓ Saved {dat_path.name}  "
                        f"({n}×2 = {n*2} frames — LabView 700-frame limit)")
        except Exception as e:
            log(f"  ✗ .dat export failed: {e}")

    btn_run.config(command=_run)

    # Trigger initial condition key update
    _update_condition()

    root.minsize(680, 540)
    root.after(50, _draw_pipeline)
    root.mainloop()


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    if len(sys.argv) == 1:
        run_gui()
    else:
        print("CLI mode not yet implemented — run without arguments to open GUI.")
        sys.exit(1)
