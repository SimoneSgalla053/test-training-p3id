"""Run RelationFormer inference on one P&ID image."""

import json
from argparse import ArgumentParser
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from PIL import Image
import torch
import yaml

from inference import relation_infer
from models import build_model


PROJECT_ROOT = Path(__file__).resolve().parent
DEFAULT_CONFIG = PROJECT_ROOT / "configs" / "road_2D.yaml"

parser = ArgumentParser(description=__doc__)
parser.add_argument("image", type=Path, help="PNG, JPEG, or other image to process")
parser.add_argument("--checkpoint", type=Path, required=True, help="trained model checkpoint (.pt)")
parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG, help="model configuration file")
parser.add_argument("--output-dir", type=Path, default=PROJECT_ROOT / "trained_weights" / "results" / "single_image")
parser.add_argument("--device", choices=("cuda", "cpu"), default="cuda")
parser.add_argument("--node-threshold", type=float, default=None)
parser.add_argument("--edge-threshold", type=float, default=None)


class ConfigObject:
    def __init__(self, values):
        self.__dict__.update(values)


def load_config(path):
    with path.open() as config_file:
        return json.loads(json.dumps(yaml.safe_load(config_file)), object_hook=ConfigObject)


def load_image(path, image_size):
    with Image.open(path) as image:
        original = image.convert("L")
        original_size = original.size
        resized = original.resize(tuple(image_size), Image.BILINEAR)
        image_array = np.asarray(resized, dtype=np.float32) / 255.0
    return torch.from_numpy(image_array)[None, None], original_size


def save_visualization(image_path, nodes, edges, output_path):
    with Image.open(image_path) as image:
        image_array = np.asarray(image.convert("L"))
    height, width = image_array.shape

    figure, axis = plt.subplots(figsize=(12, 12), dpi=150)
    axis.imshow(image_array, cmap="gray")
    for first, second in edges:
        axis.plot(
            [nodes[first][0] * width, nodes[second][0] * width],
            [nodes[first][1] * height, nodes[second][1] * height],
            color="deepskyblue",
            linewidth=1.2,
        )
    if nodes:
        axis.scatter(
            [node[0] * width for node in nodes],
            [node[1] * height for node in nodes],
            s=12,
            c="crimson",
        )
    axis.axis("off")
    figure.savefig(output_path, bbox_inches="tight", pad_inches=0)
    plt.close(figure)


def main(args):
    if not args.image.is_file():
        raise FileNotFoundError(f"Image not found: {args.image}")
    if not args.checkpoint.is_file():
        raise FileNotFoundError(f"Checkpoint not found: {args.checkpoint}")

    config = load_config(args.config)
    device = torch.device("cuda" if args.device == "cuda" and torch.cuda.is_available() else "cpu")
    if args.device == "cuda" and device.type != "cuda":
        print("CUDA is unavailable; running on CPU.")

    model = build_model(config).to(device)
    checkpoint = torch.load(args.checkpoint, map_location="cpu")
    model.load_state_dict(checkpoint["net"])
    model.eval()

    image, original_size = load_image(args.image, config.DATA.IMG_SIZE)
    node_threshold = args.node_threshold or config.INFERENCE.NODE_THRESHOLD
    edge_threshold = args.edge_threshold or config.INFERENCE.EDGE_THRESHOLD
    with torch.no_grad():
        hidden, output = model(image.to(device))
        predicted_nodes, predicted_edges = relation_infer(
            hidden,
            output,
            model,
            config.MODEL.DECODER.OBJ_TOKEN,
            config.MODEL.DECODER.RLN_TOKEN,
            nms=config.INFERENCE.NMS,
            node_threshold=node_threshold,
            edge_threshold=edge_threshold,
        )

    nodes = predicted_nodes[0].cpu().tolist()
    edges = predicted_edges[0].tolist()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    stem = args.image.stem
    graph_path = args.output_dir / f"{stem}_graph.json"
    preview_path = args.output_dir / f"{stem}_prediction.png"
    with graph_path.open("w") as graph_file:
        json.dump(
            {
                "image": str(args.image),
                "original_size": {"width": original_size[0], "height": original_size[1]},
                "nodes": nodes,
                "edges": edges,
            },
            graph_file,
            indent=2,
        )
    save_visualization(args.image, nodes, edges, preview_path)
    print(f"Found {len(nodes)} nodes and {len(edges)} edges.")
    print(f"Graph: {graph_path}")
    print(f"Preview: {preview_path}")


if __name__ == "__main__":
    main(parser.parse_args())