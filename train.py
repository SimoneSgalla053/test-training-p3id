import os
import yaml
import sys
import json
from argparse import ArgumentParser
import numpy as np

DEFAULT_CONFIG = os.path.join(os.path.dirname(os.path.abspath(__file__)), "configs", "road_2D.yaml")

parser = ArgumentParser()
parser.add_argument(
    "--config",
    default=DEFAULT_CONFIG,
    help="config file (.yml) containing the hyper-parameters for training. "
    "If None, use the nnU-Net config. See /config for examples.",
)
parser.add_argument("--resume", default=None, help="checkpoint of the last epoch of the model")
parser.add_argument("--device", default="cuda", help="device to use for training")
parser.add_argument(
    "--cuda_visible_device",
    nargs="*",
    type=int,
    default=[0],
    help="list of index where skip conn will be made",
)


class obj:
    def __init__(self, dict1):
        self.__dict__.update(dict1)


def dict2obj(dict1):
    return json.loads(json.dumps(dict1), object_hook=obj)


def load_config(path, verbose):
    with open(path) as f:
        config = yaml.load(f, Loader=yaml.FullLoader)
    if verbose:
        print("\n*** Config file")
        print(path)
        print(config["log"]["message"])
    return dict2obj(config)


def setup_logging(config, rank):
    import logging

    # stdout is a pipe under `!python` / commit mode: without this, output only shows up in blocks
    sys.stdout.reconfigure(line_buffering=True)
    handlers = [logging.StreamHandler(sys.stdout)]
    if rank == 0:
        log_dir = os.path.join(
            config.TRAIN.SAVE_PATH, "runs", "%s_%d" % (config.log.exp_name, config.DATA.SEED)
        )
        os.makedirs(log_dir, exist_ok=True)
        handlers.append(logging.FileHandler(os.path.join(log_dir, "train.log")))
    logging.basicConfig(
        level=logging.INFO if rank == 0 else logging.WARNING,
        format="%(asctime)s %(levelname)s %(name)s: %(message)s",
        datefmt="%H:%M:%S",
        handlers=handlers,
        force=True,
    )


def main(args):
    os.environ["CUDA_VISIBLE_DEVICES"] = ",".join(map(str, args.cuda_visible_device))
    sys.stdout.reconfigure(line_buffering=True)

    import torch
    import ignite.distributed as idist
    from dataset_road_network import build_road_network_data

    config = load_config(args.config, verbose=True)

    # build index + preprocessing cache once here: doing it inside a rank would
    # leave the other ranks blocked on a collective past the NCCL timeout
    train_ds, val_ds = build_road_network_data(config, mode="split")
    del train_ds, val_ds

    num_gpus = torch.cuda.device_count() if args.device == "cuda" else 0
    backend = "nccl" if num_gpus > 1 else None
    with idist.Parallel(backend=backend, nproc_per_node=num_gpus if backend else None) as parallel:
        parallel.run(training, args)


def training(local_rank, args):
    import itertools
    import torch
    import ignite.distributed as idist
    from torch.utils.data import DataLoader
    from torch.utils.data.distributed import DistributedSampler
    from dataset_road_network import build_road_network_data, image_graph_collate_road_network
    from evaluator import build_evaluator
    from trainer import build_trainer
    from models import build_model
    from tensorboardX import SummaryWriter
    from models.matcher import build_matcher
    from losses import SetCriterion

    rank = idist.get_rank()
    world_size = idist.get_world_size()
    config = load_config(args.config, verbose=False)
    setup_logging(config, rank)

    torch.backends.cudnn.benchmark = True
    torch.backends.cudnn.enabled = True
    torch.multiprocessing.set_sharing_strategy("file_system")
    device = idist.device() if args.device == "cuda" and torch.cuda.is_available() else torch.device("cpu")
    if args.device == "cuda" and device.type != "cuda" and rank == 0:
        print("CUDA is unavailable; training on CPU.")
    use_amp = bool(getattr(config.TRAIN, "AMP", True)) and device.type == "cuda"
    if rank == 0:
        print(f"world_size={world_size} amp={use_amp} device={device}")

    net = build_model(config).to(device)
    if world_size > 1:
        # only rank 0 downloads pretrained weights; sync every rank to its init
        for tensor in itertools.chain(net.parameters(), net.buffers()):
            torch.distributed.broadcast(tensor.data, src=0)

    matcher = build_matcher(config)
    loss = SetCriterion(config, matcher, net)

    # cache was built in main(); every rank just loads it
    train_ds, val_ds = build_road_network_data(config, mode="split")

    train_sampler = DistributedSampler(train_ds, shuffle=True) if world_size > 1 else None
    val_sampler = DistributedSampler(val_ds, shuffle=False) if world_size > 1 else None
    # workers are forked from a CUDA process (~GBs RSS each): keep them few, the
    # memmap dataset makes __getitem__ nearly free anyway
    num_workers = config.DATA.NUM_WORKERS
    loader_kwargs = {
        "batch_size": config.DATA.BATCH_SIZE,
        "num_workers": num_workers,
        "collate_fn": image_graph_collate_road_network,
        "pin_memory": device.type == "cuda",
    }
    train_kwargs = dict(loader_kwargs)
    if num_workers > 0:
        train_kwargs["persistent_workers"] = True
        train_kwargs["prefetch_factor"] = 2

    train_loader = DataLoader(
        train_ds, shuffle=train_sampler is None, sampler=train_sampler, **train_kwargs
    )
    val_loader = DataLoader(val_ds, shuffle=False, sampler=val_sampler, **loader_kwargs)

    param_dicts = [
        {
            "params": [
                p
                for n, p in net.named_parameters()
                if not match_name_keywords(n, ["encoder.0"])
                and not match_name_keywords(n, ["reference_points", "sampling_offsets"])
                and p.requires_grad
            ],
            "lr": float(config.TRAIN.LR),
        },
        {
            "params": [
                p
                for n, p in net.named_parameters()
                if match_name_keywords(n, ["encoder.0"]) and p.requires_grad
            ],
            "lr": float(config.TRAIN.LR_BACKBONE),
        },
        {
            "params": [
                p
                for n, p in net.named_parameters()
                if match_name_keywords(n, ["reference_points", "sampling_offsets"])
                and p.requires_grad
            ],
            "lr": float(config.TRAIN.LR) * 0.1,
        },
    ]

    optimizer = torch.optim.AdamW(
        param_dicts, lr=float(config.TRAIN.LR), weight_decay=float(config.TRAIN.WEIGHT_DECAY)
    )

    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, config.TRAIN.LR_DROP)
    scaler = (
        torch.amp.GradScaler("cuda", enabled=use_amp)
        if hasattr(torch.amp, "GradScaler")
        else torch.cuda.amp.GradScaler(enabled=use_amp)
    )

    if args.resume:
        checkpoint = torch.load(args.resume, map_location="cpu")
        net.load_state_dict(checkpoint["net"])
        optimizer.load_state_dict(checkpoint["optimizer"])
        scheduler.load_state_dict(checkpoint["scheduler"])
        if "scaler" in checkpoint:
            scaler.load_state_dict(checkpoint["scaler"])
        last_epoch = scheduler.last_epoch

    writer = None
    if rank == 0:
        writer = SummaryWriter(
            log_dir=os.path.join(
                config.TRAIN.SAVE_PATH, "runs", "%s_%d" % (config.log.exp_name, config.DATA.SEED)
            ),
        )

    evaluator = build_evaluator(
        val_loader, net, optimizer, scheduler, scaler, writer, config, device, use_amp=use_amp
    )
    trainer = build_trainer(
        train_loader,
        net,
        loss,
        optimizer,
        scheduler,
        scaler,
        writer,
        evaluator,
        config,
        device,
    )

    if args.resume:
        evaluator.state.epoch = last_epoch
        trainer.state.epoch = last_epoch
        trainer.state.iteration = trainer.state.epoch_length * last_epoch

    trainer.run()


def match_name_keywords(n, name_keywords):
    out = False
    for b in name_keywords:
        if b in n:
            out = True
            break
    return out


if __name__ == "__main__":
    args = parser.parse_args()

    import torch.multiprocessing

    torch.multiprocessing.set_sharing_strategy("file_system")

    main(args)
