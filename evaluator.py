import os
from pathlib import Path
import matplotlib.pyplot as plt
import numpy as np
import torch
import ignite.distributed as idist
from monai.engines import SupervisedEvaluator
from monai.handlers import StatsHandler, CheckpointSaver, TensorBoardStatsHandler
from metric_smd import MeanSMD
from monai.inferers import SimpleInferer
from monai.transforms import (
    Compose,
    AsDiscreted,
)
from multiprocessing import Pool
import pdb
from inference import relation_infer

from torch.utils.data import DataLoader
from typing import TYPE_CHECKING, Callable, Dict, Iterable, List, Optional, Sequence, Tuple, Union
from monai.utils import ForwardMode, min_version, optional_import
from monai.config import IgniteInfo
from monai.engines.utils import default_metric_cmp_fn, default_prepare_batch
from monai.inferers import Inferer, SimpleInferer
from monai.transforms import Transform
from monai.utils import ForwardMode, min_version, optional_import

if TYPE_CHECKING:
    from ignite.engine import Engine, EventEnum
    from ignite.metrics import Metric
else:
    Engine, _ = optional_import(
        "ignite.engine", IgniteInfo.OPT_IMPORT_VERSION, min_version, "Engine"
    )
    Metric, _ = optional_import(
        "ignite.metrics", IgniteInfo.OPT_IMPORT_VERSION, min_version, "Metric"
    )
    EventEnum, _ = optional_import(
        "ignite.engine", IgniteInfo.OPT_IMPORT_VERSION, min_version, "EventEnum"
    )


# Define customized evaluator
class RelationformerEvaluator(SupervisedEvaluator):
    def __init__(
        self,
        device: torch.device,
        val_data_loader: Union[Iterable, DataLoader],
        network: torch.nn.Module,
        epoch_length: Optional[int] = None,
        non_blocking: bool = False,
        prepare_batch: Callable = default_prepare_batch,
        iteration_update: Optional[Callable] = None,
        inferer: Optional[Inferer] = None,
        postprocessing: Optional[Transform] = None,
        key_val_metric: Optional[Dict[str, Metric]] = None,
        additional_metrics: Optional[Dict[str, Metric]] = None,
        metric_cmp_fn: Callable = default_metric_cmp_fn,
        val_handlers: Optional[Sequence] = None,
        amp: bool = False,
        mode: Union[ForwardMode, str] = ForwardMode.EVAL,
        event_names: Optional[List[Union[str, EventEnum]]] = None,
        event_to_attr: Optional[dict] = None,
        decollate: bool = True,
        **kwargs,
    ) -> None:
        super().__init__(
            device=device,
            val_data_loader=val_data_loader,
            epoch_length=epoch_length,
            non_blocking=non_blocking,
            prepare_batch=prepare_batch,
            iteration_update=iteration_update,
            postprocessing=postprocessing,
            key_val_metric=key_val_metric,
            additional_metrics=additional_metrics,
            metric_cmp_fn=metric_cmp_fn,
            val_handlers=val_handlers,
            amp=amp,
            mode=mode,
            event_names=event_names,
            event_to_attr=event_to_attr,
            decollate=decollate,
            network=network,
            inferer=SimpleInferer() if inferer is None else inferer,
        )

        self.config = kwargs.pop("config")
        self.use_amp = kwargs.pop("use_amp", False)
        self.last_visualization_epoch = None

    def _iteration(self, engine, batchdata):
        images, boxes, edges = batchdata[0], batchdata[1], batchdata[3]

        images = images.to(engine.state.device, non_blocking=True)
        boxes = [box.to(engine.state.device, non_blocking=True) for box in boxes]
        edges = [edge.to(engine.state.device, non_blocking=True) for edge in edges]

        self.network.eval()

        with torch.no_grad(), torch.autocast("cuda", dtype=torch.float16, enabled=self.use_amp):
            h, out = self.network(images)

            pred_nodes, pred_edges = relation_infer(
                h,
                out,
                self.network,
                self.config.MODEL.DECODER.OBJ_TOKEN,
                self.config.MODEL.DECODER.RLN_TOKEN,
                nms=getattr(self.config.INFERENCE, "NMS", False),
                node_threshold=self.config.INFERENCE.NODE_THRESHOLD,
                edge_threshold=self.config.INFERENCE.EDGE_THRESHOLD,
            )

        if (
            self.config.TRAIN.SAVE_VAL
            and idist.get_rank() == 0
            and self.last_visualization_epoch != engine.state.epoch
        ):
            save_graph_comparison(
                images[0],
                boxes[0][..., :2],
                edges[0],
                pred_nodes[0],
                pred_edges[0],
                Path(self.config.TRAIN.SAVE_PATH)
                / "results"
                / "validation"
                / f"epoch_{engine.state.epoch:03d}.png",
            )
            self.last_visualization_epoch = engine.state.epoch

        return {
            "images": images,
            "nodes": [box[..., :2] for box in boxes],
            "edges": edges,
            "pred_nodes": pred_nodes,
            "pred_edges": pred_edges,
        }


def build_evaluator(val_loader, net, optimizer, scheduler, scaler, writer, config, device, use_amp=False):
    val_handlers = []
    if idist.get_rank() == 0:
        val_handlers = [
            StatsHandler(output_transform=lambda x: None),
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
                save_key_metric=True,
                key_metric_n_saved=5,
                save_interval=1,
                key_metric_negative_sign=True,
            ),
            TensorBoardStatsHandler(
                writer,
                tag_name="val_smd",
                output_transform=lambda x: None,
                global_epoch_transform=lambda x: scheduler.last_epoch,
            ),
        ]

    evaluator = RelationformerEvaluator(
        config=config,
        use_amp=use_amp,
        device=device,
        val_data_loader=val_loader,
        network=net,
        inferer=SimpleInferer(),
        key_val_metric={
            "val_smd": MeanSMD(
                output_transform=lambda x: (
                    x["nodes"],
                    x["edges"],
                    x["pred_nodes"],
                    x["pred_edges"],
                ),
            )
        },
        val_handlers=val_handlers,
        amp=False,
    )

    return evaluator


def save_graph_comparison(
    image, target_nodes, target_edges, predicted_nodes, predicted_edges, output_path
):
    output_path.parent.mkdir(parents=True, exist_ok=True)
    image = to_numpy(image[0])
    target_nodes, target_edges = to_numpy(target_nodes), to_numpy(target_edges)
    predicted_nodes, predicted_edges = to_numpy(predicted_nodes), to_numpy(predicted_edges)

    figure, axes = plt.subplots(1, 3, figsize=(15, 5), dpi=150)
    axes[0].imshow(image, cmap="gray")
    axes[0].set_title("Input")
    for axis, title, nodes, edges in (
        (axes[1], "Ground truth", target_nodes, target_edges),
        (axes[2], "Prediction", predicted_nodes, predicted_edges),
    ):
        axis.imshow(image, cmap="gray")
        for source, target in edges:
            axis.plot(
                [nodes[source, 0] * image.shape[1], nodes[target, 0] * image.shape[1]],
                [nodes[source, 1] * image.shape[0], nodes[target, 1] * image.shape[0]],
                color="tab:orange",
                linewidth=1,
            )
        if len(nodes):
            axis.scatter(
                nodes[:, 0] * image.shape[1], nodes[:, 1] * image.shape[0], s=8, c="tab:red"
            )
        axis.set_title(title)
    for axis in axes:
        axis.axis("off")
    figure.tight_layout()
    figure.savefig(output_path, bbox_inches="tight")
    plt.close(figure)


def to_numpy(value):
    if isinstance(value, torch.Tensor):
        return value.detach().cpu().numpy()
    return np.asarray(value)
