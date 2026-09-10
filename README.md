# Relationformer: A Unified Framework for Image-to-Graph Generation

## Requirements
For the code in this checkout, use the modern environment described in
[Analisi del codice e guida WSL (Italiano)](ANALISI_TRAINING_WSL.md).
It includes the paper comparison, applied fixes, test results, and instructions for an RTX A1000 6 GB.
`requirements.txt` is retained as the historical environment, not the installation path for Python 3.12.

For local WSL, install the matching PyTorch wheels first, then the tested dependencies:

```bash
python -m pip install torch==2.7.1 torchvision==0.22.1 --index-url https://download.pytorch.org/whl/cu128
python -m pip install -r requirements-wsl.txt
```

The active attention implementation uses PyTorch `grid_sample` and activation checkpointing.
No CUDA extension build or local CUDA Toolkit is required for this path. Compiling `models/ops`
alone does not select the native extension.


## Code Usage

## 1. Dataset preparation

The default configuration trains on the patched P&ID dataset. Keep each PNG beside its paired GraphML annotation under the following root:

```
code_root/
└── data/
    PID2Graph/
    └── Patched/
        ├── Dataset PID/
        ├── PID2Graph OPEN100/
        └── PID2Graph Synthetic/
```

Samples from the configured training sources are split by a hash of their source drawing directory:
approximately 95% training and 5% validation, with OPEN100 held out for testing.
The loader reads normalized centers, bounding boxes, node classes, edges, and edge classes from GraphML.

If you want strict paper-style annotation normalization (canonical GraphML keys + class mapping) and an image-consistency check, run:

```bash
python prepare_pid2graph_paper_dataset.py \
  --source-root data/PID2Graph/Patched \
  --output-root data/PID2Graph/Patched_paper \
  --compare-root 'data/P&ID_imgs/PID2Graph/Patched' \
  --copy-images
```

The script writes a detailed report in `data/PID2Graph/Patched_paper/conversion_report.json`.

### Paper replication note

The public PID2Graph archive contains the paper's evaluation datasets, but not the exact
`Synthetic 700` and 60 real-world P&IDs used for training. The default configuration is therefore
a practical available-data experiment: `Dataset PID` and `PID2Graph Synthetic` are split 95:5 by
source drawing for training/validation, while `PID2Graph OPEN100` remains completely held out for
testing. Results from this setup must not be presented as the paper's exact training protocol or as
an independent result on `PID2Graph Synthetic`. The available training sources contain no
`inlet_outlet` instances; that output class is retained for paper compatibility but cannot be learned
without adding the unpublished training data or contaminating the held-out OPEN100 evaluation.

## Kaggle setup

For **Run All**, upload `kaggle_train_relationformer.ipynb` to Kaggle, select two GPUs,
attach the patched dataset, and enable Internet. It includes the current Python sources:
no clone or push is needed. Defaults are batch 4 per GPU, AMP, and at most 9 training hours.
Set an optional resume checkpoint in the first code cell; only one training run is launched.
After editing runtime sources locally, refresh the embedded copy with
`python3 scripts/update_kaggle_snapshot.py` before uploading the notebook again.

If an older run fails during preprocessing with `rebuild_storage_fd: unable to mmap`,
use the updated notebook in a fresh session. Workers now transfer NumPy arrays and the
parent creates ordinary CPU tensors, avoiding one shared-memory mapping per graph tensor.
`RELATIONFORMER_PREPROCESS_WORKERS=1` selects a fully serial fallback. Completed caches
are reused; incomplete preprocessing caches are rebuilt.

Select a GPU accelerator (dual T4 is supported), add the patched dataset as a Kaggle Dataset, and
enable Internet for the initial ImageNet ResNet-101 weight download (or provide the weights in the
torchvision cache). Run these cells from the repository root and keep Kaggle's preinstalled PyTorch
and CUDA packages:

```bash
pip install -q -r requirements-kaggle.txt
```

Point the loader to the directory that directly contains `Dataset PID`, `PID2Graph Synthetic`, and
`PID2Graph OPEN100`. The cache must be outside `/kaggle/input`, which is read-only:

```bash
export PID2GRAPH_DATA_PATH=/kaggle/input/pid2graph-patched/Patched
export RELATIONFORMER_CACHE_DIR=/kaggle/working/relationformer-cache

python - <<'PY'
import torch
print("torch:", torch.__version__)
print("CUDA:", torch.version.cuda)
print("GPUs:", torch.cuda.device_count())
assert torch.cuda.is_available(), "Enable a GPU accelerator in Kaggle"
PY

python train.py --config configs/road_2D.yaml --cuda_visible_device 0 1
```

The first run indexes GraphML files and creates a resized image/graph cache. Checkpoints and logs
are written under `trained_weights/runs/pid2graph_kaggle_10/`. If a dual-T4 run runs out of memory,
reduce `DATA.BATCH_SIZE` from 8 to 6 or 4. Resume with:

```bash
python train.py --config configs/road_2D.yaml --cuda_visible_device 0 1 \
  --resume 'trained_weights/runs/pid2graph_kaggle_10/models/checkpoint_epoch=NN.pt'
```

## 2. Training

#### 2.1 Prepare config file

The Kaggle config is `configs/road_2D.yaml`; the local starting config is `configs/wsl_2D.yaml`.

#### 2.2 Train

For example, the command for training Relationformer is following:

```bash
python train.py --config configs/wsl_2D.yaml --cuda_visible_device 0
```

## 3. Evaluation

Once you have the config file and trained model, run following command to evaluate it on test set:

```bash
python test.py --config configs/wsl_2D.yaml --cuda_visible_device 0 --checkpoint /path/to/checkpoint.pt
```
