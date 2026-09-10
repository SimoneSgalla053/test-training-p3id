import os
import yaml
import json
from argparse import ArgumentParser
from pathlib import Path

import numpy as np
import networkx as nx
import matplotlib.pyplot as plt

from dataset_road_network import PID_EDGE_CLASS_TO_ID, PID_NODE_CLASS_TO_ID

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

    config.MODEL.ENCODER.PRETRAINED = False
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
    checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=True)
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

    node_class_names = [
        class_name
        for class_name, _ in sorted(PID_NODE_CLASS_TO_ID.items(), key=lambda item: item[1])
    ]
    edge_class_names = [
        class_name
        for class_name, _ in sorted(PID_EDGE_CLASS_TO_ID.items(), key=lambda item: item[1])
    ]
    metric_node_map = BBoxEvaluator(node_class_names, max_detections=100)
    metric_edge_map = BBoxEvaluator(edge_class_names, max_detections=100)

    topo_results = []
    with torch.no_grad():
        print("Started processing test set.")
        for id_, batchdata in enumerate(tqdm(test_loader)):

            # extract data and put to device
            images, nodes, edges = batchdata[0], batchdata[1], batchdata[2]
            node_classes = batchdata[4] if len(batchdata) > 4 else None
            edge_classes = batchdata[5] if len(batchdata) > 5 else None
            node_boxes = batchdata[6] if len(batchdata) > 6 else None
            images = images.to(device, non_blocking=False)
            nodes = [node.to(device, non_blocking=False) for node in nodes]
            edges = [edge.to(device, non_blocking=False) for edge in edges]
            if node_classes is not None:
                node_classes = [node_class.to(device, non_blocking=False) for node_class in node_classes]
            if edge_classes is not None:
                edge_classes = [edge_class.to(device, non_blocking=False) for edge_class in edge_classes]
            if node_boxes is not None:
                node_boxes = [box.to(device, non_blocking=False) for box in node_boxes]

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
                edge_chunk_size=getattr(config.INFERENCE, "EDGE_CHUNK_SIZE", 4096),
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

            # Add elements of current batch elem to node map evaluator
            metric_node_map.add(
                pred_boxes=[box_cxcywh_to_xyxy_np(box) for box in pred_nodes_box],
                pred_classes=pred_nodes_box_class,
                pred_scores=pred_nodes_box_score,
                gt_boxes=[
                    box_cxcywh_to_xyxy_np(
                        boxes_.cpu().numpy()
                        if node_boxes is not None
                        else np.concatenate(
                            [nodes_.cpu().numpy(), np.ones_like(nodes_.cpu()) * 0.2], axis=1
                        )
                    )
                    for nodes_, boxes_ in zip(
                        nodes,
                        node_boxes if node_boxes is not None else [None] * len(nodes),
                    )
                ],
                gt_classes=[
                    node_class.cpu().numpy() if node_classes is not None else np.ones((nodes_.shape[0],))
                    for nodes_, node_class in zip(
                        nodes,
                        node_classes if node_classes is not None else [None] * len(nodes),
                    )
                ],
            )

            # Add elements of current batch elem to edge map evaluator
            pred_edges_box = [
                edges_to_boxes(nodes_.cpu().numpy(), edges_)
                for edges_, nodes_ in zip(pred_edges, pred_nodes)
            ]
            gt_edges_box = [
                edges_to_boxes(nodes_.cpu().numpy(), edges_.cpu().numpy())
                for edges_, nodes_ in zip(edges, nodes)
            ]

            metric_edge_map.add(
                pred_boxes=pred_edges_box,
                pred_classes=pred_edges_box_class,
                pred_scores=pred_edges_box_score,
                gt_boxes=gt_edges_box,
                gt_classes=[
                    edge_class.cpu().numpy() if edge_classes is not None else np.ones((edges_.shape[0],))
                    for edges_, edge_class in zip(
                        edges,
                        edge_classes if edge_classes is not None else [None] * len(edges),
                    )
                ],
            )

            for node_, edge_, pred_node_, pred_edge_ in zip(nodes, edges, pred_nodes, pred_edges):
                topo_results.append(compute_topo(node_.cpu(), edge_.cpu(), pred_node_, pred_edge_))

    # Determine smd
    smd_mean = torch.tensor(smd_results).mean().item()
    smd_std = torch.tensor(smd_results).std().item()
    print(f"smd value: mean {smd_mean}, std {smd_std}\n")

    # Determine node box ap / ar
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

    # Determine edge box ap / ar
    edge_metric_scores = metric_edge_map.eval()
    print(
        f"edge mAP_IoU_0.50_0.95_0.05_MaxDet_100 {edge_metric_scores['mAP_IoU_0.50_0.95_0.05_MaxDet_100']}"
    )
    print(f"edge AP_IoU_0.10_MaxDet_100 {edge_metric_scores['AP_IoU_0.10_MaxDet_100']}")
    print(f"edge AP_IoU_0.20_MaxDet_100 {edge_metric_scores['AP_IoU_0.20_MaxDet_100']}")
    print(f"edge AP_IoU_0.30_MaxDet_100 {edge_metric_scores['AP_IoU_0.30_MaxDet_100']}")
    print(f"edge AP_IoU_0.40_MaxDet_100 {edge_metric_scores['AP_IoU_0.40_MaxDet_100']}")
    print(f"edge AP_IoU_0.50_MaxDet_100 {edge_metric_scores['AP_IoU_0.50_MaxDet_100']}")
    print(f"edge AP_IoU_0.60_MaxDet_100 {edge_metric_scores['AP_IoU_0.60_MaxDet_100']}")
    print(f"edge AP_IoU_0.70_MaxDet_100 {edge_metric_scores['AP_IoU_0.70_MaxDet_100']}")
    print(f"edge AP_IoU_0.80_MaxDet_100 {edge_metric_scores['AP_IoU_0.80_MaxDet_100']}")
    print(f"edge AP_IoU_0.90_MaxDet_100 {edge_metric_scores['AP_IoU_0.90_MaxDet_100']}\n")

    print(
        f"edge mAR_IoU_0.50_0.95_0.05_MaxDet_100 {edge_metric_scores['mAR_IoU_0.50_0.95_0.05_MaxDet_100']}"
    )
    print(f"edge AR_IoU_0.10_MaxDet_100 {edge_metric_scores['AR_IoU_0.10_MaxDet_100']}")
    print(f"edge AR_IoU_0.20_MaxDet_100 {edge_metric_scores['AR_IoU_0.20_MaxDet_100']}")
    print(f"edge AR_IoU_0.30_MaxDet_100 {edge_metric_scores['AR_IoU_0.30_MaxDet_100']}")
    print(f"edge AR_IoU_0.40_MaxDet_100 {edge_metric_scores['AR_IoU_0.40_MaxDet_100']}")
    print(f"edge AR_IoU_0.50_MaxDet_100 {edge_metric_scores['AR_IoU_0.50_MaxDet_100']}")
    print(f"edge AR_IoU_0.60_MaxDet_100 {edge_metric_scores['AR_IoU_0.60_MaxDet_100']}")
    print(f"edge AR_IoU_0.70_MaxDet_100 {edge_metric_scores['AR_IoU_0.70_MaxDet_100']}")
    print(f"edge AR_IoU_0.80_MaxDet_100 {edge_metric_scores['AR_IoU_0.80_MaxDet_100']}")
    print(f"edge AR_IoU_0.90_MaxDet_100 {edge_metric_scores['AR_IoU_0.90_MaxDet_100']}\n")

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
