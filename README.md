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
    P&ID_imgs/
    └── PID2Graph/
        └── Patched/
            ├── Dataset PID/
            ├── PID2Graph OPEN100/
            └── PID2Graph Synthetic/
```

Training samples are split deterministically by source drawing (never by patch), using the paper's 95:5 training-validation ratio. The loader retains normalized bounding boxes, the seven symbol classes, structural nodes (`crossing`, `connector`, `border`), and the two edge classes (`solid`, `non-solid`).

The published PID2Graph download in this repository is benchmark data. It contains Dataset-P&ID and PID2Graph Synthetic, but not the paper's private training corpus: 2,000 Synthetic 700 drawings for pre-training, then 500 Synthetic 700 and 60 real-world drawings for fine-tuning. Training on the benchmark root is therefore rejected by default to prevent test leakage. `DATA.ALLOW_BENCHMARK_TRAINING: true` is available only for explicitly non-reproduction experiments.

## 2. Training

#### 2.1 Prepare config file

The config file can be found at `.configs/road_2D.yaml`. Make custom changes if necessary.

#### 2.2 Train

For example, the command for training Relationformer is following:

```bash
python train.py --config configs/road_2D.yaml --cuda_visible_device 3
```

The configuration matches the disclosed Relationformer hyperparameters: batch size 20, 512x512 input, 80 epochs, ResNet-101, 401 queries, learning rates `1e-4`/`3e-5`, loss weights `2,2,1,4,3`, and randomized edge directions. Set `DATA.DATA_PATH` to the patched Synthetic 700 training corpus for pre-training. For fine-tuning, point it to a prepared mix of 500 synthetic and 60 real drawings in which each real drawing is sampled three times, then initialize a fresh optimizer and scheduler from the pre-training network weights:

```bash
python train.py --config configs/road_2D.yaml --weights trained_weights/runs/pid_pretrain_10/models/<checkpoint>.pt
```

Exact numerical reproduction additionally requires the authors' unavailable training drawings and augmentation ranges/probabilities, which the paper does not publish.

## 3. Evaluation

Once you have the config file and trained model, run following command to evaluate it on test set:

```bash
python test.py --config configs/road_2D.yaml --cuda_visible_device 3 --checkpoint ./trained_weights/last_checkpoint.pt
```