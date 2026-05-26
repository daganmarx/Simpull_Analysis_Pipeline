# SiMPull Analysis Pipeline

A single-molecule pull-down (SiMPull) image analysis pipeline for processing two- and three-channel fluorescence microscopy data. Detects and colocalizes single-molecule spots across channels and counts photobleaching steps to determine protein stoichiometry.

---

## What you need before starting

- A Mac (macOS 11 or later, Intel or Apple Silicon)
- About 20 minutes for the one-time setup
- Your raw `.vsi` microscopy files, or existing `.tif` stacks if you have them already

---

## One-time setup

You only need to do this once. After setup, launching the pipeline takes a single command.

### Step 1 — Install Miniconda

Miniconda gives you the Python environment the pipeline runs in.

1. Go to [https://docs.conda.io/en/latest/miniconda.html](https://docs.conda.io/en/latest/miniconda.html)
2. Download the **macOS** installer (choose the Apple Silicon version if you have an M1/M2/M3 Mac, otherwise Intel)
3. Open the downloaded `.pkg` file and follow the installer
4. When it finishes, **close and reopen Terminal**

To confirm it worked, open Terminal and type:
```
conda --version
```
You should see a version number. If you see "command not found", restart Terminal and try again.

### Step 2 — Download the pipeline

1. On this GitHub page, click the green **Code** button → **Download ZIP**
2. Unzip the downloaded file
3. Move the resulting folder somewhere permanent — your Documents folder works well

### Step 3 — Install Fiji

Fiji is required to convert raw `.vsi` microscopy files to `.tif`. If you already have `.tif` files and don't need VSI conversion, you can skip this step.

1. Go to [https://fiji.sc](https://fiji.sc) and click **Download**
2. Open the downloaded `.dmg` and drag **Fiji.app** into your `/Applications` folder
3. Open Fiji once to confirm it launches, then close it

### Step 4 — Create the pipeline environment

This installs all the Python packages the pipeline needs.

1. Open **Terminal**
2. Navigate to the pipeline folder by typing `cd ` (with a space after), then drag the pipeline folder from Finder into the Terminal window, then press Enter. It should look something like:
   ```
   cd /Users/yourname/Documents/simpull-pipeline
   ```
3. Run:
   ```
   conda env create -f environment_simpull.yml
   ```
   This will take a few minutes. You'll see a lot of text — that's normal.
4. When it finishes, run:
   ```
   conda activate simpull
   ```
   You should see `(simpull)` appear at the start of the line in Terminal.

Setup is complete. You only need to do Steps 1–4 once.

---

## Running the pipeline

Every time you want to use the pipeline:

1. Open **Terminal**
2. Run these two commands:
   ```
   conda activate simpull
   ```
   ```
   python /Users/yourname/Documents/simpull-pipeline/pipeline.py
   ```
   Replace the path with wherever you put the pipeline folder. The graphical interface will open.

**Tip:** You can make a shortcut. Create a plain text file called `run_simpull.command` containing those two lines, save it to your Desktop, and run `chmod +x ~/Desktop/run_simpull.command` in Terminal once. After that, double-clicking the file will launch the pipeline directly.

---

## Workflow

The pipeline has three steps, run in order. Each can be skipped if you've already completed it.

### Step 1 — Prepare TIF
Converts raw `.vsi` files from your microscope into a `.tif` stack the pipeline can read. Supports single-channel, two-channel, and three-channel experiments. If you already have `.tif` files, uncheck this step and point the pipeline at your existing file instead.

### Step 2 — Colocalization
Detects fluorescent spots in each channel and identifies which spots from channel 1 and channel 2 are colocalized (i.e. the same molecule). Before running the full analysis, an interactive tuning window opens so you can check that spots are being detected correctly.

**Tuning the detection parameters:**

| Parameter | What it does | Where to start |
|-----------|-------------|----------------|
| PSF radius (px) | Expected size of one fluorescent spot | 1.5 — increase if spots are splitting, decrease if they're merging |
| Ch1 / Ch2 threshold | Detection sensitivity per channel | 0.001 — lower = more spots (more false positives), higher = fewer spots (more misses) |
| Coloc threshold (px) | Max distance between spot centres to call a colocalization | 5.0 |
| Int. min/max ×bg | Intensity filter relative to local background | 1.2–100 for Ch1; 0–100 for Ch2 |

Type a new value into any box and press **Enter** to update the display. When satisfied, click **Done**.

Detection parameters are saved automatically per experimental condition and reloaded the next time you analyse the same condition, so you only need to tune once per condition.

### Step 3 — Bleaching Analysis
Extracts fluorescence intensity traces for each detected spot and automatically counts photobleaching steps using a changepoint detection algorithm. An interactive review window lets you manually correct any traces where the automatic detection was wrong.

**Review UI controls:**
- **Left-click** on a trace → add a breakpoint at that frame
- **Right-click** on a trace → remove the nearest breakpoint
- **[c]** → reset to the automatically detected breakpoints
- **[← / →]** → move to the previous / next spot
- **[q]** → finish and save

---

## Output files

Results are saved to the output folder you specify in the GUI.

| File | Description |
|------|-------------|
| `*_colocalized.csv` | Colocalized spot pairs with coordinates, step counts, and classifications |
| `*_unmatched_ch1.csv` | Ch1 spots with no Ch2 partner |
| `*_unmatched_ch2.csv` | Ch2 spots with no Ch1 partner |
| `*_overlay.png` | Image with all detected spots marked |
| `*_bleaching_coloc.csv` | Bleaching results for colocalized pairs |
| `*_bleaching_unmatched_ch1.csv` | Bleaching results for unmatched Ch1 spots |
| `*_bleaching_unmatched_ch2.csv` | Bleaching results for unmatched Ch2 spots |

The `class` column in bleaching output uses the following labels:

| Label | Meaning |
|-------|---------|
| `monomer` | 1 bleaching step detected |
| `dimer` | 2 bleaching steps detected |
| `trimer`, `tetramer`, ... | 3+ steps, up to the max stoichiometry set in the GUI |
| `aggregate` | More steps than the expected maximum |
| `bad_trace` | Trace quality too poor to classify |

---

## File structure

```
pipeline.py              — main GUI and pipeline orchestration
colocalize_tif.py        — spot detection, colocalization, tune UI
bleaching_analysis.py    — bleaching step counting and review UI
prepare_tif.py           — VSI → TIF conversion via Fiji
environment_simpull.yml  — conda environment specification
models/                  — pre-trained ML models (do not modify)
  param_predictor.pkl      — predicts good starting detection parameters
  bleaching_predictor.pkl  — flags traces likely to need manual correction
  spot_classifier.pt       — CNN classifier for false positive spots
  spot_classifier_threshold.json
```

---

## Troubleshooting

**"FIJI executable not found"**
Make sure Fiji is installed in `/Applications/Fiji.app`. If it's somewhere else, you can update the path in the pipeline GUI settings. Download Fiji from [https://fiji.sc](https://fiji.sc) if needed.

**"command not found: conda"**
Close Terminal completely, reopen it, and try again. If it still doesn't work, Miniconda may not have installed correctly — re-run the Miniconda installer.

**"Could not find conda environment: simpull"**
You need to create the environment first. Run `conda env create -f environment_simpull.yml` from inside the pipeline folder.

**The GUI doesn't open / crashes immediately**
Make sure you've activated the environment first: `conda activate simpull`. You should see `(simpull)` at the start of the Terminal line before running `python pipeline.py`.

**Spots not detected / far too many spots detected**
Use the interactive tuning UI to adjust the threshold for each channel. Channels with high fluorescent background typically need a higher threshold. Start by changing the channel threshold in steps of 10× (e.g. 0.001 → 0.01) until the spot count looks reasonable.

**Pipeline crashes with an import error**
The `simpull` environment may be incomplete. Try:
```
conda activate simpull
pip install -r requirements.txt
```
Or recreate the environment from scratch:
```
conda env remove -n simpull
conda env create -f environment_simpull.yml
```

---

## Getting help

Contact [your name / lab contact] or open an issue on this repository.
