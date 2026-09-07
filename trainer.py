import os
import time
from monai.engines import SupervisedTrainer
from monai.inferers import SimpleInferer
from monai.handlers import (
    LrScheduleHandler,
    ValidationHandler,
    StatsHandler,
    TensorBoardStatsHandler,
    CheckpointSaver,
)
import torch
import ignite.distributed as idist
from ignite.engine import Events
from ignite.handlers import EarlyStopping


def average_gradients(model):
    """Manual all-reduce instead of DDP: relation_embed is called from the loss, outside forward."""
    world_size = idist.get_world_size()
    if world_size <= 1:
        return
    grads = [p.grad for p in model.parameters() if p.grad is not None]
    handles = [torch.distributed.all_reduce(g, async_op=True) for g in grads]
    for handle in handles:
        handle.wait()
    for grad in grads:
        grad.div_(world_size)


# define customized trainer
class RelationformerTrainer(SupervisedTrainer):
    def __init__(self, *args, scaler, clip_max_norm, **kwargs):
        super().__init__(*args, **kwargs)
        # not `self.scaler`: MONAI's Trainer.run() resets that attribute to None
        self.grad_scaler = scaler
        self.clip_max_norm = clip_max_norm

    def _iteration(self, engine, batchdata):
        images, nodes, edges = batchdata[0], batchdata[1], batchdata[2]
        ids = batchdata[3]

        # inputs, targets = self.get_batch(batchdata, image_keys=IMAGE_KEYS, label_keys="label")
        # inputs = torch.cat(inputs, 1)
        images = images.to(engine.state.device, non_blocking=True)
        nodes = [node.to(engine.state.device, non_blocking=True) for node in nodes]
        edges = [edge.to(engine.state.device, non_blocking=True) for edge in edges]
        target = {"nodes": nodes, "edges": edges}

        self.network.train()
        self.optimizer.zero_grad(set_to_none=True)

        with torch.autocast("cuda", dtype=torch.float16, enabled=self.grad_scaler.is_enabled()):
            h, out = self.network(images)
            losses = self.loss_function(h, out, target)

        self.grad_scaler.scale(losses["total"]).backward()
        average_gradients(self.network)
        self.grad_scaler.unscale_(self.optimizer)
        torch.nn.utils.clip_grad_norm_(self.network.parameters(), self.clip_max_norm)
        self.grad_scaler.step(self.optimizer)
        self.grad_scaler.update()

        return {"images": images, "points": nodes, "edges": edges, "loss": losses}


def build_trainer(
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
):
    rank = idist.get_rank()
    train_handlers = [
        LrScheduleHandler(
            lr_scheduler=scheduler,
            print_lr=rank == 0,
            epoch_level=True,
        ),
        ValidationHandler(
            validator=evaluator, interval=config.TRAIN.VAL_INTERVAL, epoch_level=True
        ),
    ]
    if rank == 0:
        loss_tags = {
            "classification_loss": "class",
            "node_loss": "nodes",
            "edge_loss": "edges",
            "box_loss": "boxes",
            "card_loss": "cards",
            "total_loss": "total",
        }
        train_handlers += [
            StatsHandler(tag_name="train_loss", output_transform=lambda x: x["loss"]["total"]),
            CheckpointSaver(
                save_dir=os.path.join(
                    config.TRAIN.SAVE_PATH,
                    "runs",
                    "%s_%d" % (config.log.exp_name, config.DATA.SEED),
                    "models",
                ),
                save_dict={
                    "net": net,
                    "optimizer": optimizer,
                    "scheduler": scheduler,
                    "scaler": scaler,
                },
                save_interval=1,
                n_saved=1,
            ),
        ] + [
            TensorBoardStatsHandler(
                writer,
                tag_name=tag,
                output_transform=lambda x, key=key: x["loss"][key],
                global_epoch_transform=lambda x: scheduler.last_epoch,
            )
            for tag, key in loss_tags.items()
        ]

    trainer = RelationformerTrainer(
        device=device,
        max_epochs=config.TRAIN.EPOCHS,
        train_data_loader=train_loader,
        network=net,
        optimizer=optimizer,
        loss_function=loss,
        inferer=SimpleInferer(),
        train_handlers=train_handlers,
        scaler=scaler,
        clip_max_norm=float(getattr(config.TRAIN, "CLIP_MAX_NORM", 0.1)),
    )

    max_hours = getattr(config.TRAIN, "MAX_HOURS", None)
    if max_hours:
        budget = float(max_hours) * 3600

        @trainer.on(Events.STARTED)
        def _start_clock(engine):
            engine.state.wall_start = time.time()

        @trainer.on(Events.EPOCH_STARTED)
        def _epoch_clock(engine):
            engine.state.epoch_start = time.time()

        # registered after ValidationHandler, so the epoch time includes validation
        @trainer.on(Events.EPOCH_COMPLETED)
        def _stop_on_budget(engine):
            now = time.time()
            elapsed, epoch_time = now - engine.state.wall_start, now - engine.state.epoch_start
            stop = torch.tensor([elapsed + epoch_time > budget], device=device)
            # ranks must agree, otherwise the survivor hangs in the next all_reduce
            stop = bool(idist.all_reduce(stop.int(), "MAX").item()) if idist.get_world_size() > 1 else bool(stop)
            if stop:
                if rank == 0:
                    print(
                        f"Stopping after epoch {engine.state.epoch}: {elapsed / 3600:.2f}h elapsed, "
                        f"next epoch (~{epoch_time / 60:.0f} min) would exceed MAX_HOURS={max_hours}"
                    )
                engine.terminate()

    patience = getattr(config.TRAIN, "EARLY_STOPPING_PATIENCE", None)
    if patience:
        early_stopping = EarlyStopping(
            patience=patience,
            # val_smd is lower-is-better; EarlyStopping expects higher-is-better
            score_function=lambda engine: -engine.state.metrics["val_smd"],
            trainer=trainer,
            min_delta=float(getattr(config.TRAIN, "EARLY_STOPPING_MIN_DELTA", 0.0)),
        )
        evaluator.add_event_handler(Events.COMPLETED, early_stopping)

    return trainer
