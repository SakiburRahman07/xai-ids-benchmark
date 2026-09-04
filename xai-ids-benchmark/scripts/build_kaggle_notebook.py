"""Generate notebooks/run_in_kaggle.ipynb (Kaggle-specific runner).

Kaggle differences vs Colab:
  - Writable space is /kaggle/working (not the repo dir).
  - Datasets are added as "Input" in the notebook UI and mount under
    /kaggle/input/<dataset-slug>/. No manual download / form / Drive needed.
  - Sessions cap at 9h; GPU is P100 (or T4 x2). Internet must be enabled
    (Settings -> Internet -> On) to clone the GitHub repo.
  - Results saved under /kaggle/working/ are auto-included as notebook Output.
"""
import json
from pathlib import Path

cells = []


def md(src):
    cells.append({"cell_type": "markdown", "metadata": {}, "source": src})


def code(src):
    cells.append({"cell_type": "code", "execution_count": None, "metadata": {}, "outputs": [], "source": src})


md([
    "# XAI-IDS Benchmark - Kaggle Runner\n",
    "\n",
    "Runs the full 4 (XAI) x 2 (DL) x 3 (dataset) factorial benchmark from the master's thesis proposal, on Kaggle.\n",
    "\n",
    "**Kaggle setup (do these before running):**\n",
    "1. **Settings -> Internet -> On** (needed to clone the GitHub repo).\n",
    "2. **Settings -> Accelerator -> GPU** (P100 or T4 x2).\n",
    "3. **Add Input datasets** (right panel -> Add Input -> search): add the three IDS datasets (search terms below). They mount under `/kaggle/input/<slug>/` - the notebook auto-detects them.\n",
    "   - Search: `CIC-IDS-2017` or `cicids2017`\n",
    "   - Search: `UNSW-NB15`\n",
    "   - Search: `CIC IoT 2023` or `ciciot2023`\n",
    "\n",
    "Modes:\n",
    "- `--quick` (default): 2k rows, 20 epochs, ~1-3 hours on a P100. Use this first.\n",
    "- Full run: set `QUICK = False`; for the 9h Kaggle limit, also set `MAX_EVAL` and `N_BOOTSTRAP` (suggested: `MAX_EVAL=500`, `N_BOOTSTRAP=30`).\n",
    "\n",
    "Results are written to `/kaggle/working/results/` and auto-saved as notebook Output when you save the notebook.\n",
])
md(["## 1. Configuration"])
code([
    "GITHUB_REPO = 'https://github.com/SakiburRahman07/xai-ids-benchmark.git'\n",
    "BRANCH = 'main'\n",
    "QUICK = True                      # True = smoke run; False = full factorial\n",
    "MAX_EVAL = None                    # None = default (500 quick / 2000 full); for 9h limit try 500\n",
    "N_BOOTSTRAP = None                 # None = default (20 quick / 100 full); for 9h limit try 30\n",
    "DATASETS = None                    # None = all 3 present; or e.g. ['cicids2017']\n",
    "MODELS = None                      # None = both; or ['cnn1d','ft_transformer']\n",
    "METHODS = None                     # None = all enabled; or ['shap','lime']\n",
    "WORK_DIR = '/kaggle/working'        # Kaggle writable space\n",
    "REPO_INPUT_SLUG = ''               # If you upload the repo zip as a Kaggle Dataset (Add Input), put its slug here (e.g., 'youruser/xai-ids-benchmark'). Empty = try git clone first.\n",
    "print(f'Repo: {GITHUB_REPO} | Quick: {QUICK} | max_eval={MAX_EVAL} | n_boot={N_BOOTSTRAP}')",
])
md(["## 2. Get the repo and install dependencies\n", "\n", "Two paths (auto-detected):\n",
    "- **Internet On:** clones from GitHub.\n",
    "- **Internet Off:** uses the repo uploaded as a Kaggle Dataset (set `REPO_INPUT_SLUG` below to the dataset slug you added via Add Input).\n",
    "\n",
    "To get the repo into Kaggle without Internet: download the repo zip from GitHub on your laptop, then in Kaggle right panel → Add Input → New Dataset → upload the zip. Set `REPO_INPUT_SLUG` to the resulting slug."])
code([
    "import os, subprocess, shutil, sys, glob\n",
    "os.chdir(WORK_DIR)\n",
    "REPO_DIR = os.path.join(WORK_DIR, 'xai-ids-benchmark')\n",
    "if os.path.isdir(REPO_DIR):\n",
    "    shutil.rmtree(REPO_DIR)\n",
    "\n",
    "# Path A: try git clone (needs Internet On).\n",
    "cloned = False\n",
    "try:\n",
    "    subprocess.run(['git','clone','--depth','1','-b',BRANCH,GITHUB_REPO,REPO_DIR], check=True, capture_output=True)\n",
    "    cloned = True\n",
    "    print('Cloned from GitHub (Internet is On).')\n",
    "except Exception as e:\n",
    "    print('git clone failed (Internet likely Off):', e.stderr.decode() if hasattr(e,'stderr') and e.stderr else e)\n",
    "\n",
    "# Path B: fall back to repo uploaded as a Kaggle Dataset.\n",
    "if not cloned:\n",
    "    KAGGLE_INPUT = '/kaggle/input'\n",
    "    candidates = []\n",
    "    if REPO_INPUT_SLUG:\n",
    "        candidates.append(os.path.join(KAGGLE_INPUT, REPO_INPUT_SLUG))\n",
    "    # Also scan all inputs for a run_benchmark.py file.\n",
    "    if os.path.isdir(KAGGLE_INPUT):\n",
    "        for slug in os.listdir(KAGGLE_INPUT):\n",
    "            candidates.append(os.path.join(KAGGLE_INPUT, slug))\n",
    "    src_root = None\n",
    "    for c in candidates:\n",
    "        if os.path.exists(os.path.join(c, 'scripts/run_benchmark.py')):\n",
    "            src_root = c; break\n",
    "        if os.path.exists(os.path.join(c, 'xai-ids-benchmark/scripts/run_benchmark.py')):\n",
    "            src_root = os.path.join(c, 'xai-ids-benchmark'); break\n",
    "    if src_root:\n",
    "        # Copy (not symlink) so the working tree is writable.\n",
    "        shutil.copytree(src_root, REPO_DIR)\n",
    "        print(f'Copied repo from Kaggle Input: {src_root}')\n",
    "    else:\n",
    "        raise RuntimeError('No repo found. Enable Internet, or upload the repo zip as a Kaggle Dataset and set REPO_INPUT_SLUG.')\n",
    "\n",
    "os.chdir(REPO_DIR)\n",
    "if not os.path.exists('scripts/run_benchmark.py') and os.path.exists('xai-ids-benchmark/scripts/run_benchmark.py'):\n",
    "    os.chdir('xai-ids-benchmark')\n",
    "print('CWD:', os.getcwd())\n",
    "assert os.path.exists('scripts/run_benchmark.py'), 'run_benchmark.py not found'",
])
code([
    "# Install deps not preinstalled on Kaggle. Kaggle has torch/numpy/pandas/scikit-learn.\n",
    "# NOTE: rtdl is NOT needed - we use a local FT-Transformer implementation.\n",
    "import importlib.util\n",
    "def have(name): return importlib.util.find_spec(name) is not None\n",
    "pkgs = ['shap','lime','dice_ml','captum','statsmodels','seaborn','pyyaml']\n",
    "missing = [p for p in pkgs if not have(p)]\n",
    "if missing:\n",
    "    subprocess.run([sys.executable,'-m','pip','install','-q',*missing])\n",
    "print('deps OK')",
])
code([
    "import torch\n",
    "print('torch', torch.__version__, '| CUDA:', torch.cuda.is_available(),\n",
    "      '|', torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'CPU')",
])
md([
    "## 3. Link Kaggle Input datasets into `data_raw/`\n",
    "\n",
    "Kaggle Input datasets mount under `/kaggle/input/<slug>/`. The cell below auto-detects common IDS dataset mirrors by globbing `/kaggle/input/` for CSVs matching each dataset's expected files, then symlinks them into `data_raw/<dataset>/` so the benchmark loaders find them.\n",
    "\n",
    "If you added the datasets via the right panel (Add Input), this should work automatically. If a dataset is missing, the benchmark will skip its cells gracefully.",
])
code([
    "import glob\n",
    "KAGGLE_INPUT = '/kaggle/input'\n",
    "os.chdir(os.path.dirname(os.path.abspath('scripts/run_benchmark.py')) or '.')\n",
    "REPO_ROOT = os.getcwd()\n",
    "print('REPO_ROOT:', REPO_ROOT)\n",
    "\n",
    "# Find CSVs in any Kaggle input dataset that look like each target.\n",
    "def find_csvs(patterns):\n",
    "    hits = []\n",
    "    for slug in os.listdir(KAGGLE_INPUT) if os.path.isdir(KAGGLE_INPUT) else []:\n",
    "        base = os.path.join(KAGGLE_INPUT, slug)\n",
    "        for pat in patterns:\n",
    "            hits += glob.glob(os.path.join(base, '**', pat), recursive=True)\n",
    "    return sorted(set(hits))\n",
    "\n",
    "mapping = {\n",
    "    'cicids2017': ['*MachineLearning*.csv', '*Friday*.csv', '*Thursday*.csv', '*Tuesday*.csv', '*Wednesday*.csv', '*Monday*.csv'],\n",
    "    'unsw_nb15':  ['UNSW-NB15_*.csv', '*UNSW*NB15*.csv'],\n",
    "    'ciciot2023': ['*.csv'],  # broad; ciciot files are many small CSVs\n",
    "}\n",
    "for ds, pats in mapping.items():\n",
    "    files = find_csvs(pats)\n",
    "    dst = os.path.join(REPO_ROOT, 'data_raw', ds)\n",
    "    os.makedirs(dst, exist_ok=True)\n",
    "    # Heuristic: ciciot2023 '*.csv' is broad - only take if slug looks ciciot.\n",
    "    if ds == 'ciciot2023':\n",
    "        files = [f for f in files if 'iot' in f.lower() or 'ciciot' in f.lower()]\n",
    "    for f in files:\n",
    "        link = os.path.join(dst, os.path.basename(f))\n",
    "        if not os.path.exists(link):\n",
    "            os.symlink(f, link)\n",
    "    print(f'{ds}: linked {len(files)} CSV(s) -> data_raw/{ds}/')\n",
    "print('\\nKaggle input slugs:', os.listdir(KAGGLE_INPUT) if os.path.isdir(KAGGLE_INPUT) else 'none')",
])
code([
    "# Verify datasets are present (prints manual instructions if not).\n",
    "!python scripts/prepare_data.py --base . || echo 'Add the dataset via the right panel -> Add Input.'",
])
md(["## 4. Run the benchmark\n", "\n", "Smoke run uses 2k rows / 20 epochs (1-3 hours on P100). For the full factorial within the 9h Kaggle limit, set `QUICK=False`, `MAX_EVAL=500`, `N_BOOTSTRAP=30`."])
code([
    "OUT = os.path.join(WORK_DIR, 'results')\n",
    "args = [sys.executable, 'scripts/run_benchmark.py', '--config-dir','config', '--base','.', '--out-dir', OUT]\n",
    "if QUICK: args.append('--quick')\n",
    "if MAX_EVAL:     args += ['--max-eval', str(MAX_EVAL)]\n",
    "if N_BOOTSTRAP:  args += ['--n-bootstrap', str(N_BOOTSTRAP)]\n",
    "if DATASETS:     args += ['--datasets', *DATASETS]\n",
    "if MODELS:       args += ['--models', *MODELS]\n",
    "if METHODS:      args += ['--methods', *METHODS]\n",
    "print('Running:', ' '.join(args))\n",
    "subprocess.run(args, check=False)",
])
md(["## 5. Results & analysis (Friedman/Nemenyi, H2 Wilcoxon)"])
code([
    "import pandas as pd, json, os\n",
    "tidy_path = os.path.join(OUT, 'tidy_metrics.csv')\n",
    "if os.path.exists(tidy_path):\n",
    "    tidy = pd.read_csv(tidy_path)\n",
    "    print('Tidy metric rows:', len(tidy))\n",
    "    print(tidy.head(20))\n",
    "else:\n",
    "    print('No tidy_metrics.csv - did the run produce results?')",
])
code([
    "fn_path = os.path.join(OUT, 'friedman_nemenyi.json')\n",
    "if os.path.exists(fn_path):\n",
    "    with open(fn_path) as f:\n",
    "        fn = json.load(f)\n",
    "    for metric, res in fn.items():\n",
    "        print(f'\\n=== {metric} ===')\n",
    "        print('avg ranks:', res.get('avg_ranks'))\n",
    "        print('CD:', round(res.get('critical_distance',0),3), '| p:', res.get('p_value'))\n",
    "        for pair, dec in res.get('pairwise',{}).items():\n",
    "            print('  ', pair, 'diff=', round(dec['rank_diff'],3), 'sig=', dec['significant'])\n",
    "else:\n",
    "    print('No friedman_nemenyi.json - insufficient data.')",
])
code([
    "# Critical distance diagram for a chosen metric.\n",
    "import sys; sys.path.insert(0, 'src')\n",
    "from xai_ids_benchmark.analysis.plots import plot_cd\n",
    "metric_to_plot = 'deletion_auc'\n",
    "if os.path.exists(fn_path) and metric_to_plot in fn:\n",
    "    plot_cd(fn[metric_to_plot]['avg_ranks'], fn[metric_to_plot]['critical_distance'],\n",
    "            output_path=os.path.join(OUT,'cd_diagram.png'), title=f'CD diagram - {metric_to_plot}')\n",
    "    from IPython.display import Image, display\n",
    "    display(Image(os.path.join(OUT,'cd_diagram.png')))\n",
    "else:\n",
    "    print('CD diagram not produced.')",
])
code([
    "# H2: stability degradation on UNSW-NB15 (Wilcoxon).\n",
    "h2_path = os.path.join(OUT, 'h2_wilcoxon.json')\n",
    "if os.path.exists(h2_path):\n",
    "    with open(h2_path) as f:\n",
    "        print(json.dumps(json.load(f), indent=2))\n",
    "else:\n",
    "    print('H2 file not produced (insufficient stability data).')",
])
md(["## 6. Save results\n", "\n", "Results in `/kaggle/working/results/` are auto-saved as notebook Output when you Save the notebook. You can also download them directly."])
code([
    "import shutil\n",
    "zip_path = os.path.join(WORK_DIR, 'results.zip')\n",
    "shutil.make_archive(zip_path.replace('.zip',''), 'zip', OUT)\n",
    "print('Saved:', zip_path)",
])

nb = {
    "nbformat": 4, "nbformat_minor": 0,
    "metadata": {
        "kernelspec": {"name": "python3", "display_name": "Python 3"},
        "language_info": {"name": "python"},
        "accelerator": "GPU",
    },
    "cells": cells,
}
out = Path(__file__).resolve().parent.parent / "notebooks" / "run_in_kaggle.ipynb"
out.parent.mkdir(parents=True, exist_ok=True)
out.write_text(json.dumps(nb, indent=1), encoding="utf-8")
print("wrote", out, out.stat().st_size, "bytes")
