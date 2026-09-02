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

Samples are split deterministically by their source drawing directory: 80% training, 10% validation, and 10% test. The loader converts GraphML node bounding boxes to normalized center points and uses GraphML edges as graph targets.

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