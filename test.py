import os
import yaml
import json
from argparse import ArgumentParser
from pathlib import Path

import numpy as np
import networkx as nx
import matplotlib.pyplot as plt

PROJECT_ROOT = Path(__file__).resolve().parent
DEFAULT_CONFIG = PROJECT_ROOT / "configs" / "road_2D.yaml"

parser = ArgumentParser()
parser.add_argument(
    "--config",
    default=str(DEFAULT_CONFIG),
    help="config file (.yml) containing the hyper-parameters for training. "
    "If None, use the nnU-Net config. See /config for examples.",
)
parser.add_argument(
    "--checkpoint",
    default=None,
    help="checkpoint of the model to test. Defaults to the newest saved checkpoint.",
)
parser.add_argument("--device", default="cuda", help="device to use for training")
parser.add_argument(
    "--cuda_visible_device",
    nargs="*",
    type=int,
    default=[0],
    help="list of index where skip conn will be made.",
)
parser.add_argument(
    "--visualize",
    default=None,
    help="path to save visualized graphs to. Defaults to trained_weights/results/test.",
)
parser.add_argument(
    "--num_visualizations", type=int, default=10, help="number of test graph comparisons to save."
)
parser.add_argument(
    "--max_samples",
    type=int,
    default=None,
    help="maximum number of test samples to evaluate; evaluates the full test set by default.",
)
parser.add_argument(
    "--batch_size",
    type=int,
    default=None,
    help="test data-loader batch size; uses the configured batch size by default.",
)


class obj:
    def __init__(self, dict1):
        self.__dict__.update(dict1)


def dict2obj(dict1):
    return json.loads(json.dumps(dict1), object_hook=obj)


def ensure_format(bboxes):
    for bbox in bboxes:
        if bbox[0] > bbox[2]:
            bbox[0], bbox[2] = bbox[2], bbox[0]
        if bbox[1] > bbox[3]:
            bbox[1], bbox[3] = bbox[3], bbox[1]
    return bboxes


def edges_to_boxes(nodes, edges, pad=0.01):
    """Axis-aligned boxes around edges; padded so axis-parallel lines have nonzero area."""
    if len(edges) == 0:
        return np.zeros((0, 4), dtype=np.float32)
    edges = np.asarray(edges, dtype=np.int64)
    boxes = ensure_format(np.hstack([nodes[edges[:, 0]], nodes[edges[:, 1]]]).astype(np.float32))
    boxes[:, :2] -= pad
    boxes[:, 2:] += pad
    return boxes


def plot_val_rel_sample(
    id_, path, image, points1, edges1, points2, edges2, attn_map=None, relative_coords=True
):
    path = Path(path)
    H, W = image.shape[0], image.shape[1]
    fig, ax = plt.subplots(1, 3, figsize=(15, 5), dpi=150)

    image = (-1 * np.transpose(np.flip(image, 0), (1, 0, 2)) + 1) / 2

    # Displaying the image
    ax[0].imshow(image)
    ax[0].axis("off")

    # border_nodes = np.array([[1,1],[1,H-1],[W-1,H-1],[W-1,1]])
    border_nodes = np.array([[-1, -1], [-1, H + 1], [W + 1, H + 1], [W + 1, -1]])
    border_edges = [(0, 1), (1, 2), (2, 3), (3, 0)]

    G1 = nx.Graph()
    G1.add_nodes_from(list(range(len(border_nodes))))
    coord_dict = {}
    tmp = [coord_dict.update({i: (pts[1], pts[0])}) for i, pts in enumerate(border_nodes)]
    for n, p in coord_dict.items():
        G1.nodes[n]["pos"] = p
    G1.add_edges_from(border_edges)

    pos = nx.get_node_attributes(G1, "pos")
    nx.draw(
        G1,
        pos,
        ax=ax[1],
        node_size=1,
        node_color="darkgrey",
        edge_color="darkgrey",
        width=1.5,
        font_size=12,
        with_labels=False,
    )
    nx.draw(
        G1,
        pos,
        ax=ax[2],
        node_size=1,
        node_color="darkgrey",
        edge_color="darkgrey",
        width=1.5,
        font_size=12,
        with_labels=False,
    )

    G = nx.Graph()
    edges = [tuple(rel) for rel in edges1]
    nodes = list(np.unique(np.array(edges)))
    coord_dict = {}
    tmp = [
        coord_dict.update({nodes[i]: (W * pts[1], H - H * pts[0])})
        for i, pts in enumerate(points1[nodes, :])
    ]
    G.add_nodes_from(nodes)
    for n, p in coord_dict.items():
        G.nodes[n]["pos"] = p
    G.add_edges_from(edges)

    pos = nx.get_node_attributes(G, "pos")
    nx.draw(
        G,
        pos,
        ax=ax[1],
        node_size=10,
        node_color="lightcoral",
        edge_color="mediumorchid",
        width=1.5,
        font_size=12,
        with_labels=False,
    )

    G = nx.Graph()
    edges = [tuple(rel) for rel in edges2]
    nodes = list(np.unique(np.array(edges)))
    coord_dict = {}
    tmp = [
        coord_dict.update({nodes[i]: (W * pts[1], H - H * pts[0])})
        for i, pts in enumerate(points2[nodes, :])
    ]
    G.add_nodes_from(nodes)
    for n, p in coord_dict.items():
        G.nodes[n]["pos"] = p
    G.add_edges_from(edges)
    pos = nx.get_node_attributes(G, "pos")
    nx.draw(
        G,
        pos,
        ax=ax[2],
        node_size=10,
        node_color="lightcoral",
        edge_color="mediumorchid",
        width=1.5,
        font_size=12,
        with_labels=False,
    )

    plt.savefig(path / f"sample{id_}.png", bbox_inches="tight")


def test(args):

    # Load the config files
    with open(args.config) as f:
        print("\n*** Config file")
        print(args.config)
        config = yaml.load(f, Loader=yaml.FullLoader)
        print(config["log"]["message"])
    config = dict2obj(config)
    os.environ["CUDA_VISIBLE_DEVICES"] = ",".join(map(str, args.cuda_visible_device))

    import torch
    from monai.data import DataLoader
    from tqdm import tqdm
    import numpy as np

    from dataset_road_network import build_road_network_data, image_graph_collate_road_network
    from evaluator import save_graph_comparison
    from models import build_model
    from inference import relation_infer
    from metric_smd import StreetMoverDistance
    from metric_map import BBoxEvaluator
    from metric_edge_map import EdgeMeanAveragePrecision
    from metric_topo.topo import compute_topo
    from box_ops_2D import box_cxcywh_to_xyxy_np

    torch.backends.cudnn.benchmark = True
    torch.backends.cudnn.enabled = True
    torch.multiprocessing.set_sharing_strategy("file_system")
    device = (
        torch.device("cuda")
        if args.device == "cuda" and torch.cuda.is_available()
        else torch.device("cpu")
    )
    if args.device == "cuda" and device.type != "cuda":
        print("CUDA is unavailable; testing on CPU.")

    net = build_model(config).to(device)

    test_ds = build_road_network_data(config, mode="test")
    if args.max_samples is not None:
        if args.max_samples <= 0:
            raise ValueError("--max_samples must be greater than zero.")
        test_ds = torch.utils.data.Subset(test_ds, range(min(args.max_samples, len(test_ds))))
    batch_size = args.batch_size or config.DATA.BATCH_SIZE
    if batch_size <= 0:
        raise ValueError("--batch_size must be greater than zero.")

    test_loader = DataLoader(
        test_ds,
        batch_size=batch_size,
        shuffle=False,
        num_workers=config.DATA.NUM_WORKERS,
        collate_fn=image_graph_collate_road_network,
        pin_memory=True,
    )

    checkpoint_path = Path(args.checkpoint) if args.checkpoint else find_latest_checkpoint(config)
    if checkpoint_path is None:
        raise FileNotFoundError(
            "No checkpoint found. Train the model first or pass --checkpoint PATH."
        )
    print(f"Loading checkpoint: {checkpoint_path}")
    checkpoint = torch.load(checkpoint_path, map_location="cpu")
    net.load_state_dict(checkpoint["net"])
    net.eval()

    visualization_path = (
        Path(args.visualize)
        if args.visualize
        else PROJECT_ROOT / "trained_weights" / "results" / "test"
    )
    if args.num_visualizations > 0:
        visualization_path.mkdir(parents=True, exist_ok=True)

    # init metric
    # metric = StreetMoverDistance(eps=1e-7, max_iter=100, reduction=MetricReduction.MEAN)
    metric_smd = StreetMoverDistance(eps=1e-5, max_iter=10, reduction="none")
    smd_results = []

    symbol_classes = [
        "general",
        "tank",
        "valve",
        "instrumentation",
        "pump",
        "inlet/outlet",
        "arrow",
    ]
    metric_symbol_map = BBoxEvaluator(symbol_classes, max_detections=100)
    metric_node_map = BBoxEvaluator(["node"], max_detections=100)
    metric_edge_map = EdgeMeanAveragePrecision(config.MODEL.NUM_EDGE_CLASSES)

    topo_results = []
    with torch.no_grad():
        print("Started processing test set.")
        for id_, batchdata in enumerate(tqdm(test_loader)):

            # extract data and put to device
            images, boxes, node_labels, edges, edge_labels = batchdata[:5]
            images = images.to(device, non_blocking=False)
            boxes = [box.to(device, non_blocking=False) for box in boxes]
            nodes = [box[..., :2] for box in boxes]
            node_labels = [label.to(device, non_blocking=False) for label in node_labels]
            edges = [edge.to(device, non_blocking=False) for edge in edges]
            edge_labels = [label.to(device, non_blocking=False) for label in edge_labels]

            h, out = net(images)
            (
                pred_nodes,
                pred_edges,
                pred_nodes_box,
                pred_nodes_box_score,
                pred_nodes_box_class,
                pred_edges_box_score,
                pred_edges_box_class,
            ) = relation_infer(
                h.detach(),
                out,
                net,
                config.MODEL.DECODER.OBJ_TOKEN,
                config.MODEL.DECODER.RLN_TOKEN,
                nms=config.INFERENCE.NMS,
                map_=True,
                node_threshold=config.INFERENCE.NODE_THRESHOLD,
                edge_threshold=config.INFERENCE.EDGE_THRESHOLD,
            )

            # Save visualization
            if id_ < args.num_visualizations:
                save_graph_comparison(
                    images[0],
                    nodes[0],
                    edges[0],
                    pred_nodes[0],
                    pred_edges[0],
                    visualization_path / f"sample{id_}.png",
                )

            # Add smd of current batch elem
            ret = metric_smd(nodes, edges, pred_nodes, pred_edges)
            smd_results += ret.tolist()

            symbol_pred_boxes = []
            symbol_pred_classes = []
            symbol_pred_scores = []
            symbol_gt_boxes = []
            symbol_gt_classes = []
            for predicted_boxes, predicted_classes, predicted_scores, target_boxes, target_classes in zip(
                pred_nodes_box,
                pred_nodes_box_class,
                pred_nodes_box_score,
                boxes,
                node_labels,
            ):
                predicted_mask = predicted_classes <= len(symbol_classes)
                target_mask = target_classes <= len(symbol_classes)
                symbol_pred_boxes.append(box_cxcywh_to_xyxy_np(predicted_boxes[predicted_mask]))
                symbol_pred_classes.append(predicted_classes[predicted_mask])
                symbol_pred_scores.append(predicted_scores[predicted_mask])
                symbol_gt_boxes.append(
                    box_cxcywh_to_xyxy_np(target_boxes[target_mask].cpu().numpy())
                )
                symbol_gt_classes.append(target_classes[target_mask].cpu().numpy())
            metric_symbol_map.add(
                pred_boxes=symbol_pred_boxes,
                pred_classes=symbol_pred_classes,
                pred_scores=symbol_pred_scores,
                gt_boxes=symbol_gt_boxes,
                gt_classes=symbol_gt_classes,
            )

            metric_node_map.add(
                pred_boxes=[box_cxcywh_to_xyxy_np(box) for box in pred_nodes_box],
                pred_classes=[np.ones(len(box), dtype=np.int64) for box in pred_nodes_box],
                pred_scores=pred_nodes_box_score,
                gt_boxes=[
                    box_cxcywh_to_xyxy_np(boxes_.cpu().numpy()) for boxes_ in boxes
                ],
                gt_classes=[np.ones(len(box), dtype=np.int64) for box in boxes],
            )

            for values in zip(
                pred_nodes_box,
                pred_edges,
                pred_edges_box_score,
                pred_edges_box_class,
                boxes,
                edges,
                edge_labels,
            ):
                metric_edge_map.add(
                    values[0],
                    values[1],
                    values[2],
                    values[3],
                    values[4].cpu().numpy(),
                    values[5].cpu().numpy(),
                    values[6].cpu().numpy(),
                )

            for node_, edge_, pred_node_, pred_edge_ in zip(nodes, edges, pred_nodes, pred_edges):
                topo_results.append(compute_topo(node_.cpu(), edge_.cpu(), pred_node_, pred_edge_))

    # Determine smd
    smd_mean = torch.tensor(smd_results).mean().item()
    smd_std = torch.tensor(smd_results).std().item()
    print(f"smd value: mean {smd_mean}, std {smd_std}\n")

    symbol_metric_scores = metric_symbol_map.eval()
    print(f"symbol AP_IoU_0.50_MaxDet_100 {symbol_metric_scores['AP_IoU_0.50_MaxDet_100']}")

    # Determine class-agnostic node box AP / AR
    node_metric_scores = metric_node_map.eval()
    print(
        f"node mAP_IoU_0.50_0.95_0.05_MaxDet_100 {node_metric_scores['mAP_IoU_0.50_0.95_0.05_MaxDet_100']}"
    )
    print(f"node AP_IoU_0.10_MaxDet_100 {node_metric_scores['AP_IoU_0.10_MaxDet_100']}")
    print(f"node AP_IoU_0.20_MaxDet_100 {node_metric_scores['AP_IoU_0.20_MaxDet_100']}")
    print(f"node AP_IoU_0.30_MaxDet_100 {node_metric_scores['AP_IoU_0.30_MaxDet_100']}")
    print(f"node AP_IoU_0.40_MaxDet_100 {node_metric_scores['AP_IoU_0.40_MaxDet_100']}")
    print(f"node AP_IoU_0.50_MaxDet_100 {node_metric_scores['AP_IoU_0.50_MaxDet_100']}")
    print(f"node AP_IoU_0.60_MaxDet_100 {node_metric_scores['AP_IoU_0.60_MaxDet_100']}")
    print(f"node AP_IoU_0.70_MaxDet_100 {node_metric_scores['AP_IoU_0.70_MaxDet_100']}")
    print(f"node AP_IoU_0.80_MaxDet_100 {node_metric_scores['AP_IoU_0.80_MaxDet_100']}")
    print(f"node AP_IoU_0.90_MaxDet_100 {node_metric_scores['AP_IoU_0.90_MaxDet_100']}\n")

    print(
        f"node mAR_IoU_0.50_0.95_0.05_MaxDet_100 {node_metric_scores['mAR_IoU_0.50_0.95_0.05_MaxDet_100']}"
    )
    print(f"node AR_IoU_0.10_MaxDet_100 {node_metric_scores['AR_IoU_0.10_MaxDet_100']}")
    print(f"node AR_IoU_0.20_MaxDet_100 {node_metric_scores['AR_IoU_0.20_MaxDet_100']}")
    print(f"node AR_IoU_0.30_MaxDet_100 {node_metric_scores['AR_IoU_0.30_MaxDet_100']}")
    print(f"node AR_IoU_0.40_MaxDet_100 {node_metric_scores['AR_IoU_0.40_MaxDet_100']}")
    print(f"node AR_IoU_0.50_MaxDet_100 {node_metric_scores['AR_IoU_0.50_MaxDet_100']}")
    print(f"node AR_IoU_0.60_MaxDet_100 {node_metric_scores['AR_IoU_0.60_MaxDet_100']}")
    print(f"node AR_IoU_0.70_MaxDet_100 {node_metric_scores['AR_IoU_0.70_MaxDet_100']}")
    print(f"node AR_IoU_0.80_MaxDet_100 {node_metric_scores['AR_IoU_0.80_MaxDet_100']}")
    print(f"node AR_IoU_0.90_MaxDet_100 {node_metric_scores['AR_IoU_0.90_MaxDet_100']}\n")

    edge_map, edge_ap = metric_edge_map.compute()
    print(f"edge mAP {edge_map}")
    print(f"edge AP per class (solid, non-solid): {edge_ap}\n")

    # Determine topo
    print(np.array(topo_results).mean(axis=0))


def find_latest_checkpoint(config):
    model_dir = (
        PROJECT_ROOT
        / config.TRAIN.SAVE_PATH
        / "runs"
        / ("%s_%d" % (config.log.exp_name, config.DATA.SEED))
        / "models"
    )
    checkpoints = list(model_dir.glob("*.pt"))
    return max(checkpoints, key=lambda path: path.stat().st_mtime) if checkpoints else None


if __name__ == "__main__":
    args = parser.parse_args()
    test(args)
