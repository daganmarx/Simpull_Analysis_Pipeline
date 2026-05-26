"""
prepare_tif.py
==============
Preprocessing script for Olympus .vsi microscopy files.

Loads two or three single-channel .vsi files, detects the laser-on frame in
each, extracts all available signal frames starting blank_frames before
laser-on, truncates all channels to the shortest, concatenates them, and
saves as a single .tif file ready for colocalize_tif.py.

Frame counts are determined by the actual VSI length rather than a fixed
number — longer acquisitions produce longer .tif files. The downstream
scripts (colocalize_tif.py, bleaching_analysis.py) derive frames_per_channel
from the .tif size at runtime.

VSI reading
-----------
FIJI (Fiji.app) is used to convert each .vsi file to a temporary .tif via
Bio-Formats. This is much faster than the Python bioformats bridge (~30s vs
~20min per channel). FIJI must be installed at /Applications/Fiji.app or
the path set via --fiji.

Laser-on detection
------------------
The laser-on frame is detected as the frame with the largest upward jump in
mean intensity between consecutive frames. This is robust to sparse labelling
where the whole-frame mean barely changes at laser-on.
The extraction window starts blank_frames before that point and runs to the
end of the file, so the output contains blank_frames laser-off frames
followed by all available signal frames per channel.

Output
------
Two-channel:
  Frames   0 - N-1    : Channel 1  (blank_frames blank + all signal frames)
  Frames   N - 2N-1   : Channel 2  (blank_frames blank + all signal frames)
  where N = min(ch1_available_frames, ch2_available_frames).

Three-channel:
  Frames   0 - N-1    : Channel 1  (blank_frames blank + all signal frames)
  Frames   N - 2N-1   : Channel 2  (blank_frames blank + all signal frames)
  Frames  2N - 3N-1   : Channel 3  (blank_frames blank + all signal frames)
  where N = min(ch1_available_frames, ch2_available_frames, ch3_available_frames).

Single-channel:
  All frames from the single VSI file.

Usage
-----
Two-channel (colocalization):
  python prepare_tif.py --ch1 ch1.vsi --ch2 ch2.vsi --output movie.tif

Three-channel (3-color stoichiometry):
  python prepare_tif.py --ch1 ch1.vsi --ch2 ch2.vsi --ch3 ch3.vsi --output movie.tif

Single-channel (stoichiometry):
  python prepare_tif.py --single-channel --ch1 ch1.vsi --output movie.tif
"""

import argparse
import subprocess
import sys
import tempfile
import warnings
from pathlib import Path

import numpy as np

try:
    import tifffile
except ImportError:
    sys.exit("Please install tifffile:  pip install tifffile")


# ===========================================================================
# PARAMETERS
# ===========================================================================

BLANK_FRAMES    = 10
PEAK_THRESHOLD  = 0.90   # kept for CLI compatibility; laser-on uses derivative
FIJI_PATH       = Path("/Applications/Fiji.app")

# ===========================================================================


def find_fiji(fiji_path: Path) -> Path:
    """Return path to the FIJI executable, or exit with a helpful message."""
    candidates = [
        fiji_path / "Contents" / "MacOS" / "ImageJ-macosx",
        fiji_path / "Contents" / "MacOS" / "fiji",
        fiji_path / "ImageJ-linux64",
        fiji_path / "ImageJ-win64.exe",
    ]
    for c in candidates:
        if c.exists():
            return c
    sys.exit(
        f"FIJI executable not found in {fiji_path}.\n"
        f"Install FIJI from https://fiji.sc or pass --fiji /path/to/Fiji.app"
    )


def find_fiji_java(fiji_exe: Path) -> Path | None:
    """
    Find the JDK bundled inside Fiji.app.
    Searches Fiji.app/java/macosx/ for any JDK folder that contains bin/java.
    Works with AdoptOpenJDK, Zulu, Temurin, etc.
    Returns the JAVA_HOME path (i.e. the Contents/Home dir), or None if not found.
    """
    # fiji_exe is e.g. Fiji.app/Contents/MacOS/ImageJ-macosx
    fiji_app = fiji_exe.parent.parent.parent   # -> Fiji.app
    java_root = fiji_app / "java" / "macosx"
    if not java_root.exists():
        return None
    for jdk_dir in sorted(java_root.iterdir()):
        # Check several layout variants:
        #   Zulu:         jdk_dir/jre/Contents/Home/bin/java
        #   AdoptOpenJDK: jdk_dir/Contents/Home/bin/java
        #   Flat:         jdk_dir/bin/java
        for candidate in [
            jdk_dir / "jre" / "Contents" / "Home",
            jdk_dir / "Contents" / "Home",
            jdk_dir / "jre",
            jdk_dir,
        ]:
            if (candidate / "bin" / "java").exists():
                return candidate
    return None


def vsi_to_tif_via_fiji(vsi_path: Path, out_tif: Path, fiji_exe: Path):
    """
    Use FIJI headless + Bio-Formats to export a VSI file to a flat TIFF stack.
    Uses a Jython script with BF.openImagePlus() which works correctly in
    headless mode (unlike the IJ1 macro Bio-Formats Importer dialog).
    """
    import os

    env = os.environ.copy()
    java_home = find_fiji_java(fiji_exe)
    if java_home:
        env["JAVA_HOME"] = str(java_home)
        env["PATH"] = str(java_home / "bin") + os.pathsep + env.get("PATH", "")
        print(f"    Using bundled JVM: {java_home}")
    else:
        print(f"    WARNING: could not find bundled JVM in Fiji.app")

    vsi_str = str(vsi_path).replace("\\", "/")
    out_str = str(out_tif).replace("\\", "/")

    # Jython script — runs via FIJI's Jython engine, has full access to
    # Java classes including BF.openImagePlus() which works headless.
    script = f"""from loci.plugins import BF
from loci.plugins.in import ImporterOptions
from ij import IJ

opts = ImporterOptions()
opts.setId("{vsi_str}")
opts.setSeriesOn(0, True)
opts.setColorMode(ImporterOptions.COLOR_MODE_GRAYSCALE)
opts.setQuiet(True)
opts.setShowMetadata(False)
opts.setShowOMEXML(False)

imps = BF.openImagePlus(opts)
if imps is None or len(imps) == 0:
    raise RuntimeError("Bio-Formats opened no images from: {vsi_str}")

imp = imps[0]
print("opened: " + imp.getTitle())
print("frames: " + str(imp.getNFrames()))
IJ.saveAs(imp, "Tiff", "{out_str}")
print("saved: {out_str}")
imp.close()
"""

    # Write Jython script to output dir (VSI volume may be read-only)
    script_path = out_tif.parent / f"_smfret_export_{vsi_path.stem}.py"
    script_path.parent.mkdir(parents=True, exist_ok=True)
    try:
        script_path.write_text(script)

        base_cmd = [str(fiji_exe)]
        if java_home:
            base_cmd += ["--java-home", str(java_home)]
        base_cmd += ["--headless", "--console", "--run", str(script_path), ""]

        print(f"    Running FIJI Bio-Formats export...")

        result = subprocess.run(
            base_cmd,
            capture_output=True,
            text=True,
            timeout=600,
            env=env,
        )
    finally:
        script_path.unlink(missing_ok=True)

    # Print output for diagnostics
    if result.stdout.strip():
        lines = [l for l in result.stdout.splitlines() if l.strip()
                 and not l.startswith(("|-INFO", "|-WARN", "SLF4J"))]
        if lines:
            print("    FIJI: " + " | ".join(lines[-10:]))
    if result.stderr.strip():
        filtered = [l for l in result.stderr.splitlines()
                    if not any(l.startswith(p) for p in
                               ("SLF4J", "|-INFO", "|-WARN", "The operation",
                                "Please visit"))]
        if filtered:
            print("    FIJI stderr:\n" + "\n".join(f"      {l}" for l in filtered[-20:]))

    if result.returncode != 0:
        raise RuntimeError(f"FIJI failed (exit code {result.returncode})")

    if not out_tif.exists():
        alt = Path(str(out_tif) + ".tif")
        if alt.exists():
            alt.rename(out_tif)
        else:
            raise RuntimeError(f"FIJI ran but output TIF not found: {out_tif}")

    print(f"    FIJI export complete -> {out_tif.name}")

def load_vsi_stack(vsi_path: Path, fiji_exe: Path,
                   tmp_dir: Path) -> np.ndarray:
    """
    Convert VSI to temp TIF via FIJI, load with tifffile, return float32 array
    of shape (n_frames, H, W). Temp TIF is deleted after loading.
    """
    tmp_tif = tmp_dir / f"{vsi_path.stem}_raw.tif"
    vsi_to_tif_via_fiji(vsi_path, tmp_tif, fiji_exe)

    # Resolve symlinks — on macOS /var/folders is a symlink to /private/var/folders
    # FIJI may resolve and save to the canonical path while we have the symlink path
    tmp_tif_real = tmp_tif.resolve()

    # FIJI's saveAs sometimes appends .tif to an already-.tif filename
    candidates = [
        tmp_tif,
        tmp_tif_real,
        tmp_tif.parent / (tmp_tif.stem + ".tif"),   # stem already has .tif -> .tif.tif
        Path(str(tmp_tif) + ".tif"),
    ]
    actual_tif = None
    for c in candidates:
        if c.exists():
            actual_tif = c
            break
    if actual_tif is None:
        # Search more broadly — FIJI may have saved relative to its cwd
        import glob as _glob
        search_dirs = [
            tmp_tif.parent,
            tmp_tif.parent.resolve(),
            Path.cwd(),
            Path.home(),
        ]
        found = []
        for sd in search_dirs:
            found += list(sd.glob(f"{vsi_path.stem}*.tif*"))
        if found:
            actual_tif = found[0]
            print(f"    Warning: expected {tmp_tif}, found at {actual_tif}")
        else:
            # Print directory contents for diagnostics
            td = tmp_tif.parent.resolve()
            contents = list(td.iterdir()) if td.exists() else []
            print(f"    Temp dir contents ({td}): {[f.name for f in contents]}")
            cwd_tifs = list(Path.cwd().glob("*.tif*"))
            print(f"    CWD tifs ({Path.cwd()}): {[f.name for f in cwd_tifs]}")
            raise RuntimeError(f"Expected output TIF not found in {tmp_tif.parent}")

    try:
        stack = tifffile.imread(str(actual_tif)).astype(np.float32)
        # tifffile may return (H, W) for single frame or (N, H, W) for stack
        if stack.ndim == 2:
            stack = stack[np.newaxis]
        elif stack.ndim == 4:
            stack = stack[:, 0]   # (N, 1, H, W) -> (N, H, W)
        return stack
    finally:
        actual_tif.unlink(missing_ok=True)


def compute_mean_intensities(stack: np.ndarray) -> np.ndarray:
    """Return per-frame mean intensity. stack shape: (N, H, W)."""
    return stack.reshape(stack.shape[0], -1).mean(axis=1)


def find_laser_on_frame(means: np.ndarray,
                        peak_threshold: float = 0.90) -> int:
    """
    Find the laser-on frame as the frame with the largest upward jump in
    mean intensity between consecutive frames.
    peak_threshold is kept for backwards compatibility but not used.
    """
    if len(means) < 2:
        return 0
    diffs = np.diff(means.astype(np.float64))
    laser_on = int(np.argmax(diffs)) + 1
    print(f"    Laser-on detection: largest upward jump at frame {laser_on}  "
          f"(delta mean = {diffs[laser_on-1]:+.2f}  "
          f"from {means[laser_on-1]:.1f} -> {means[laser_on]:.1f})")
    return laser_on


def extract_window(stack: np.ndarray, laser_on_frame: int,
                   blank_frames: int, name: str) -> np.ndarray:
    """Extract from (laser_on - blank_frames) to end. Returns (n, H, W)."""
    start = laser_on_frame - blank_frames
    if start < 0:
        warnings.warn(
            f"{name}: laser-on at frame {laser_on_frame} but only "
            f"{laser_on_frame} blank frame(s) available "
            f"(requested {blank_frames}). Using {laser_on_frame} blank frames.",
            stacklevel=2,
        )
        start = 0
    return stack[start:]


def load_and_extract(vsi_path: Path, blank_frames: int,
                     peak_threshold: float,
                     fiji_exe: Path, tmp_dir: Path) -> np.ndarray:
    """
    Full pipeline for one VSI: FIJI export -> load -> detect laser-on -> extract.
    Returns (n_frames, H, W) float32 array.
    """
    print(f"\n  Loading: {vsi_path.name}")
    stack = load_vsi_stack(vsi_path, fiji_exe, tmp_dir)
    n_frames, h, w = stack.shape
    print(f"    Dimensions : {w} x {h} px,  {n_frames} frames")

    if (h, w) != (400, 400):
        warnings.warn(
            f"{vsi_path.name}: image is {w}x{h} px, expected 400x400.",
            stacklevel=2
        )

    means    = compute_mean_intensities(stack)
    laser_on = find_laser_on_frame(means, peak_threshold)
    signal_frames = n_frames - laser_on
    print(f"    Extracting frames {laser_on - blank_frames} - {n_frames - 1}  "
          f"({blank_frames} blank + {signal_frames} signal)")

    return extract_window(stack, laser_on, blank_frames, vsi_path.name)


# ===========================================================================
# Pipeline functions (called by pipeline.py and CLI)
# ===========================================================================

def _run_pipeline(ch1_path, ch2_path, out_path,
                  blank_frames, peak_threshold, fiji_exe):
    """Two-channel pipeline."""
    print("\n=== VSI -> TIF Preprocessing (two-channel) ===")
    print(f"  Ch1            : {ch1_path.name}")
    print(f"  Ch2            : {ch2_path.name}")
    print(f"  Output         : {out_path}")
    print(f"  Blank frames   : {blank_frames}")
    print(f"  FIJI           : {fiji_exe}")

    with tempfile.TemporaryDirectory(prefix="smfret_") as tmp:
        tmp_dir = Path(tmp)
        stack_ch1 = load_and_extract(ch1_path, blank_frames,
                                     peak_threshold, fiji_exe, tmp_dir)
        stack_ch2 = load_and_extract(ch2_path, blank_frames,
                                     peak_threshold, fiji_exe, tmp_dir)

    if stack_ch1.shape[1:] != stack_ch2.shape[1:]:
        raise RuntimeError(
            f"Channel spatial dimensions do not match: "
            f"Ch1={stack_ch1.shape[1:]}  Ch2={stack_ch2.shape[1:]}"
        )

    n = min(stack_ch1.shape[0], stack_ch2.shape[0])
    if stack_ch1.shape[0] != stack_ch2.shape[0]:
        print(f"\n  Channel lengths differ -- truncating to {n} frames/channel "
              f"(Ch1: {stack_ch1.shape[0]}, Ch2: {stack_ch2.shape[0]})")
        stack_ch1 = stack_ch1[:n]
        stack_ch2 = stack_ch2[:n]

    concat = np.concatenate([stack_ch1, stack_ch2], axis=0)
    total_frames, h, w = concat.shape
    fpc = n
    print(f"\n  Concatenated stack : {total_frames} frames  ({w}x{h} px)")
    print(f"  Frames per channel : {fpc}  "
          f"({blank_frames} blank + {fpc - blank_frames} signal)")
    print(f"  Saving -> {out_path} ...")
    out_path.parent.mkdir(parents=True, exist_ok=True)
    tifffile.imwrite(str(out_path), concat.astype(np.uint16),
                     photometric="minisblack")

    print(f"\n  Done.")
    print(f"  Output layout  : Ch1 frames 0-{fpc-1},  "
          f"Ch2 frames {fpc}-{total_frames-1}")
    print(f"  Blank frames   : first {blank_frames} of each channel block")
    print(f"\n  Ready for: python colocalize_tif.py --mode single "
          f"--input {out_path}")


def _run_pipeline_single(ch1_path, out_path,
                         blank_frames, peak_threshold, fiji_exe):
    """Single-channel pipeline for stoichiometry experiments."""
    print("\n=== VSI -> TIF Preprocessing (single-channel) ===")
    print(f"  Input          : {ch1_path.name}")
    print(f"  Output         : {out_path}")
    print(f"  Blank frames   : {blank_frames}")
    print(f"  FIJI           : {fiji_exe}")

    with tempfile.TemporaryDirectory(prefix="smfret_") as tmp:
        tmp_dir = Path(tmp)
        stack = load_and_extract(ch1_path, blank_frames,
                                 peak_threshold, fiji_exe, tmp_dir)

    total_frames, h, w = stack.shape
    print(f"\n  Stack: {total_frames} frames  ({w}x{h} px)")
    print(f"  Saving -> {out_path} ...")
    out_path.parent.mkdir(parents=True, exist_ok=True)
    tifffile.imwrite(str(out_path), stack.astype(np.uint16),
                     photometric="minisblack")

    print(f"\n  Done.")
    print(f"  Output frames  : {total_frames}  "
          f"(frames 0-{blank_frames-1}: blank,  "
          f"frames {blank_frames}-{total_frames-1}: signal)")
    print(f"\n  Ready for: python detect_spots_single.py --mode single "
          f"--input {out_path}")


def _run_pipeline_three_channel(ch1_path, ch2_path, ch3_path, out_path,
                                blank_frames, peak_threshold, fiji_exe):
    """Three-channel pipeline for 3-color stoichiometry experiments."""
    print("\n=== VSI -> TIF Preprocessing (three-channel) ===")
    print(f"  Ch1            : {ch1_path.name}")
    print(f"  Ch2            : {ch2_path.name}")
    print(f"  Ch3            : {ch3_path.name}")
    print(f"  Output         : {out_path}")
    print(f"  Blank frames   : {blank_frames}")
    print(f"  FIJI           : {fiji_exe}")

    with tempfile.TemporaryDirectory(prefix="smfret_") as tmp:
        tmp_dir = Path(tmp)
        stack_ch1 = load_and_extract(ch1_path, blank_frames,
                                     peak_threshold, fiji_exe, tmp_dir)
        stack_ch2 = load_and_extract(ch2_path, blank_frames,
                                     peak_threshold, fiji_exe, tmp_dir)
        stack_ch3 = load_and_extract(ch3_path, blank_frames,
                                     peak_threshold, fiji_exe, tmp_dir)

    shapes = {
        "Ch1": stack_ch1.shape[1:],
        "Ch2": stack_ch2.shape[1:],
        "Ch3": stack_ch3.shape[1:],
    }
    if len(set(shapes.values())) > 1:
        raise RuntimeError(
            f"Channel spatial dimensions do not match: "
            + "  ".join(f"{k}={v}" for k, v in shapes.items())
        )

    n = min(stack_ch1.shape[0], stack_ch2.shape[0], stack_ch3.shape[0])
    lengths = {"Ch1": stack_ch1.shape[0], "Ch2": stack_ch2.shape[0],
               "Ch3": stack_ch3.shape[0]}
    if len(set(lengths.values())) > 1:
        print(f"\n  Channel lengths differ -- truncating to {n} frames/channel  "
              + "  ".join(f"({k}: {v})" for k, v in lengths.items()))
        stack_ch1 = stack_ch1[:n]
        stack_ch2 = stack_ch2[:n]
        stack_ch3 = stack_ch3[:n]

    concat = np.concatenate([stack_ch1, stack_ch2, stack_ch3], axis=0)
    total_frames, h, w = concat.shape
    fpc = n
    print(f"\n  Concatenated stack : {total_frames} frames  ({w}x{h} px)")
    print(f"  Frames per channel : {fpc}  "
          f"({blank_frames} blank + {fpc - blank_frames} signal)")
    print(f"  Saving -> {out_path} ...")
    out_path.parent.mkdir(parents=True, exist_ok=True)
    tifffile.imwrite(str(out_path), concat.astype(np.uint16),
                     photometric="minisblack")

    print(f"\n  Done.")
    print(f"  Output layout  : "
          f"Ch1 frames 0-{fpc-1},  "
          f"Ch2 frames {fpc}-{2*fpc-1},  "
          f"Ch3 frames {2*fpc}-{total_frames-1}")
    print(f"  Blank frames   : first {blank_frames} of each channel block")
    print(f"\n  Ready for: python colocalize_tif.py --three-channel --mode single "
          f"--input {out_path}")




def main():
    if len(sys.argv) == 1:
        # No args -- launch simple standalone GUI
        try:
            import tkinter as tk
            from tkinter import ttk, filedialog, messagebox
        except ImportError:
            sys.exit("tkinter not available. Use CLI flags instead.")

        root = tk.Tk()
        root.title("VSI -> TIF Preprocessor")
        root.resizable(False, False)

        ch1_var       = tk.StringVar()
        ch2_var       = tk.StringVar()
        out_var       = tk.StringVar()
        blank_var     = tk.IntVar(value=BLANK_FRAMES)
        fiji_var      = tk.StringVar(value=str(FIJI_PATH))
        status_var    = tk.StringVar(value="Ready.")
        PAD = dict(padx=6, pady=4)

        def _browse_vsi(var, title):
            p = filedialog.askopenfilename(
                title=title,
                filetypes=[("Olympus VSI", "*.vsi"), ("All files", "*.*")])
            if p:
                var.set(p)

        def _browse_out():
            p = filedialog.asksaveasfilename(
                title="Save output .tif as",
                defaultextension=".tif",
                filetypes=[("TIFF", "*.tif")])
            if p:
                out_var.set(p)

        def _browse_fiji():
            p = filedialog.askdirectory(title="Select Fiji.app folder")
            if p:
                fiji_var.set(p)

        def _run():
            ch1 = ch1_var.get().strip()
            ch2 = ch2_var.get().strip()
            out = out_var.get().strip()
            if not ch1 or not Path(ch1).exists():
                messagebox.showerror("Error", "Channel 1 file not found."); return
            if not ch2 or not Path(ch2).exists():
                messagebox.showerror("Error", "Channel 2 file not found."); return
            if not out:
                messagebox.showerror("Error", "Please specify an output filename."); return
            btn_run.config(state="disabled")
            status_var.set("Running -- see terminal for progress...")
            root.update()
            try:
                fiji_exe = find_fiji(Path(fiji_var.get()))
                _run_pipeline(Path(ch1), Path(ch2), Path(out),
                              blank_var.get(), PEAK_THRESHOLD, fiji_exe)
                status_var.set(f"Done!  Saved -> {Path(out).name}")
                messagebox.showinfo("Done", f"Saved:\n{out}")
            except Exception as e:
                status_var.set(f"Error: {e}")
                messagebox.showerror("Error", str(e))
            finally:
                btn_run.config(state="normal")

        frame = ttk.Frame(root, padding=12)
        frame.grid(row=0, column=0, sticky="nsew")
        ttk.Label(frame, text="VSI -> TIF Preprocessor",
                  font=("", 14, "bold")).grid(row=0, column=0, columnspan=3, pady=(0,10))
        for i, (lbl, var, txt, cmd) in enumerate([
            ("Channel 1 (.vsi)", ch1_var, "Browse...",
             lambda: _browse_vsi(ch1_var, "Select Channel 1")),
            ("Channel 2 (.vsi)", ch2_var, "Browse...",
             lambda: _browse_vsi(ch2_var, "Select Channel 2")),
            ("Output (.tif)",    out_var, "Save as...", _browse_out),
            ("Fiji.app",         fiji_var, "Browse...", _browse_fiji),
        ], start=1):
            ttk.Label(frame, text=lbl).grid(row=i, column=0, sticky="w", **PAD)
            ttk.Entry(frame, textvariable=var, width=50).grid(row=i, column=1, **PAD)
            ttk.Button(frame, text=txt, command=cmd).grid(row=i, column=2, **PAD)
        ttk.Separator(frame, orient="horizontal").grid(
            row=6, column=0, columnspan=3, sticky="ew", pady=8)
        ttk.Label(frame, text="Blank frames").grid(row=7, column=0, sticky="w", **PAD)
        ttk.Entry(frame, textvariable=blank_var, width=8).grid(row=7, column=1, sticky="w", **PAD)
        btn_run = ttk.Button(frame, text="Run", command=_run, width=20)
        btn_run.grid(row=8, column=0, columnspan=3, pady=6)
        ttk.Label(frame, textvariable=status_var, foreground="#444",
                  font=("", 9)).grid(row=9, column=0, columnspan=3)
        root.mainloop()
        return

    parser = argparse.ArgumentParser(
        description="Preprocess Olympus .vsi files into a .tif for analysis.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    parser.add_argument("--single-channel", action="store_true")
    parser.add_argument("--ch1",    required=True)
    parser.add_argument("--ch2",    default=None)
    parser.add_argument("--ch3",    default=None,
                        help="Optional third channel .vsi for 3-color experiments.")
    parser.add_argument("--output", required=True)
    parser.add_argument("--blank-frames",   type=int,   default=BLANK_FRAMES)
    parser.add_argument("--peak-threshold", type=float, default=PEAK_THRESHOLD,
                        help="Unused -- kept for backwards compatibility.")
    parser.add_argument("--fiji", default=str(FIJI_PATH),
                        help=f"Path to Fiji.app. Default: {FIJI_PATH}")

    args = parser.parse_args()
    ch1_path = Path(args.ch1)
    out_path = Path(args.output)

    if not ch1_path.exists():
        sys.exit(f"File not found: {ch1_path}")
    if out_path.suffix.lower() not in (".tif", ".tiff"):
        sys.exit("Output file must have a .tif or .tiff extension.")

    fiji_exe = find_fiji(Path(args.fiji))
    print(f"  FIJI executable: {fiji_exe}")

    if args.single_channel:
        _run_pipeline_single(ch1_path, out_path,
                             args.blank_frames, args.peak_threshold, fiji_exe)
    elif args.ch3:
        # Three-channel mode
        if not args.ch2:
            sys.exit("Three-channel mode requires --ch2.")
        ch2_path = Path(args.ch2)
        ch3_path = Path(args.ch3)
        if not ch2_path.exists():
            sys.exit(f"File not found: {ch2_path}")
        if not ch3_path.exists():
            sys.exit(f"File not found: {ch3_path}")
        _run_pipeline_three_channel(ch1_path, ch2_path, ch3_path, out_path,
                                    args.blank_frames, args.peak_threshold, fiji_exe)
    else:
        if not args.ch2:
            sys.exit("Two-channel mode requires --ch2.")
        ch2_path = Path(args.ch2)
        if not ch2_path.exists():
            sys.exit(f"File not found: {ch2_path}")
        _run_pipeline(ch1_path, ch2_path, out_path,
                      args.blank_frames, args.peak_threshold, fiji_exe)


if __name__ == "__main__":
    main()
