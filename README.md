# Relationformer: A Unified Framework for Image-to-Graph Generation

## Requirements
* CUDA>=9.2
* PyTorch>=1.7.1

For other system requirements please follow

```bash
pip install -r requirements.txt
```

### Compiling CUDA operators
```bash
cd ./models/ops
python setup.py install
```


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

Samples are split deterministically by their source drawing directory: 80% training, 10% validation, and 10% test. The loader converts GraphML node bounding boxes to normalized center points and uses GraphML edges as graph targets.

If you want strict paper-style annotation normalization (canonical GraphML keys + class mapping) and an image-consistency check, run:

```bash
python prepare_pid2graph_paper_dataset.py \
  --source-root data/PID2Graph/Patched \
  --output-root data/PID2Graph/Patched_paper \
  --compare-root data/P&ID_imgs/PID2Graph/Patched \
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

The config file can be found at `.configs/road_2D.yaml`. Make custom changes if necessary.

#### 2.2 Train

For example, the command for training Relationformer is following:

```bash
python train.py --config configs/road_2D.yaml --cuda_visible_device 3
```

## 3. Evaluation

Once you have the config file and trained model, run following command to evaluate it on test set:

```bash
python test.py --config configs/road_2D.yaml --cuda_visible_device 3 --checkpoint ./trained_weights/last_checkpoint.pt
```