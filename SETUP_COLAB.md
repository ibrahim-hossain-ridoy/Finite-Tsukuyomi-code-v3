# Colab Setup Guide for v3 Experiment

## Step-by-step setup

### 1. Set runtime to GPU

Go to **Runtime → Change runtime type → T4 GPU** before doing anything else.
The experiment runs on CPU but is 10-20× faster on GPU.

### 2. Mount Google Drive (for checkpoint persistence)

```python
from google.colab import drive
drive.mount('/content/drive')
```

### 3. Upload or clone the codebase

**Option A — Upload a zip:**
```python
# Upload experiment.zip via the Colab file browser, then:
!unzip experiment.zip -d /content/experiment
%cd /content/experiment
```

**Option B — Clone from a repository:**
```bash
!git clone <your-repo-url> /content/experiment
%cd /content/experiment
```

### 4. Install dependencies

```bash
!pip install -r requirements.txt
```

Most dependencies (torch, numpy, sklearn) are already on Colab, so this
should be fast.

### 5. Verify GPU is available

```python
import torch
print(f"CUDA available: {torch.cuda.is_available()}")
print(f"Device: {torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'CPU'}")
```

Expected output: `CUDA available: True`, device name containing "T4".

### 6. Run the debug test first

```bash
!python run_experiment_v3.py --debug
```

This should complete in ~1-2 minutes and confirms the pipeline works end
to end. Check the output for:
- "Stage 1 complete" and "Saved results" messages
- All three metrics (F, w, dmn) with finite values

### 7. Symlink checkpoints to Drive (persistence across sessions)

Colab sessions time out after ~90 minutes of inactivity (or ~12 hours max).
To persist checkpoints and results across sessions:

```bash
# Create directories on Drive
!mkdir -p "/content/drive/MyDrive/v3_experiment/checkpoints"
!mkdir -p "/content/drive/MyDrive/v3_experiment/results_v3"

# Symlink so the experiment writes directly to Drive
!ln -sfn "/content/drive/MyDrive/v3_experiment/checkpoints" /content/experiment/checkpoints
!ln -sfn "/content/drive/MyDrive/v3_experiment/results_v3" /content/experiment/results_v3
```

### 8. Launch the full sweep

```bash
!python run_experiment_v3.py
```

Expected runtime: roughly 2-5 hours total for 5 Stage 1 runs + 30 Stage 2
runs on a T4 GPU (each Stage 2 run takes ~5-10 minutes).

### 9. Resume after a session timeout

If Colab disconnects mid-sweep, simply reconnect, remount Drive, and
re-run:

```bash
# After reconnecting:
from google.colab import drive
drive.mount('/content/drive')

%cd /content/experiment

# Re-create symlinks
!ln -sfn "/content/drive/MyDrive/v3_experiment/checkpoints" /content/experiment/checkpoints
!ln -sfn "/content/drive/MyDrive/v3_experiment/results_v3" /content/experiment/results_v3

# Resume — automatically skips completed runs
!python run_experiment_v3.py
```

The runner checks for existing `.npz` files and Stage 1 checkpoints
before each run. Already-completed work is skipped with a log message.

### 10. Analyze results and generate figures

```bash
!python analyze_results_v3.py
!python make_figures_v3.py
```

Download results from Drive or directly from Colab:
```python
from google.colab import files
files.download('results_v3/summary_by_r_v3.csv')
files.download('results_v3/figures/v3_results.png')
```

## Troubleshooting

**"CUDA out of memory"**: Reduce `BATCH_SIZE` in `config_v3.py` to 64.
The default (128) should fit comfortably on a T4 (16GB), but if other
processes are using GPU memory, reducing batch size helps.

**Session keeps timing out before the sweep finishes**: The sweep can
be split across multiple sessions thanks to resume logic. Each session
will pick up where the last one left off.

**"No .npz files found"**: Make sure the symlinks are set up correctly.
Run `ls results_v3/raw/` to verify files exist.
