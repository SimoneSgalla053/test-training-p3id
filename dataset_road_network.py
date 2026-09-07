"""Functionality for 2D road network dataset."""

import torch
from torch.utils.data import Dataset
import torchvision.transforms.functional as tvf
from PIL import Image

import time
import pickle
import random
import yaml
import json
import hashlib
from pathlib import Path
from xml.etree import ElementTree


class ToulouseRoadNetworkDataset(Dataset):
    """
    Generates a subclass of the PyTorch torch.utils.data.Dataset class
    """

    def __init__(self, root_path="data/", split="valid", use_raw_images=False):
        """
        :param root_path: root data path
        :param split: data split in {"train", "valid", "test", "augment"}
        :param max_prev_node: only return the last previous 'max_prev_node' elements in the adjacency row of a node
            default is 4, which corresponds to the 95th percentile in the data
        :param step: step size used in the data generation, default is 0.001° (around 110 metres per datapoint)
        :param use_raw_images: loads raw images if yes, otherwise faster and more compact numpy array representations
        :param return_coordinates: returns coordinates on the real map for each datapoint, used for qualitative studies
        """
        assert split in {"train", "valid", "test", "augment"}
        print(f"Started loading the data ({split})...")
        start_time = time.time()

        dataset_path = f"{root_path}/{split}.pickle"
        images_path = f"{root_path}/{split}_images.pickle"
        images_raw_path = f"{root_path}/{split}/images/"

        ids, list_nodes, list_edges = load_dataset(dataset_path)

        self.ids = ["{:0>7d}".format(int(i)) for i in ids]
        self.nodes = list_nodes
        self.edges = list_edges

        print(f"Started loading the images...")

        if use_raw_images:
            self.images = load_raw_images(ids, images_raw_path)
        else:
            self.images = load_images(ids, images_path)

        print(f"Dataset loading completed, took {round(time.time() - start_time, 2)} seconds!")
        print(f"Dataset size: {len(self)}\n")

    def __len__(self):
        r"""
        :return: data length
        """
        return len(self.ids)

    def __getitem__(self, idx):
        r"""
        :param idx: index in the data
        :return: chosen data point
        """
        return self.images[idx][None], self.nodes[idx], self.edges[idx], self.ids[idx]


class PatchedPIDDataset(Dataset):
    """Loads patched P&ID PNG images and their paired GraphML annotations."""

    SPLIT_RANGES = {
        "train": (0, 80),
        "valid": (80, 90),
        "test": (90, 100),
    }

    def __init__(self, root_path, split="train", image_size=(512, 512), split_seed=10, max_nodes=None):
        if split not in self.SPLIT_RANGES:
            raise ValueError(f"Unsupported P&ID split: {split}")

        root = Path(root_path)
        if not root.is_dir():
            raise FileNotFoundError(f"P&ID dataset directory does not exist: {root}")

        self.image_size = tuple(image_size)
        self.samples = index_pid_samples(root, split_seed, max_nodes)[split]

        if not self.samples:
            raise RuntimeError(f"No paired P&ID samples found for the {split} split under {root}")

        print(f"Loaded {len(self.samples)} P&ID samples for {split}.")

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        image_path, graph_path, sample_id = self.samples[idx]
        image = Image.open(image_path).convert("L").resize(self.image_size, Image.BILINEAR)
        image = tvf.to_tensor(image)
        nodes, edges = load_pid_graph(graph_path, image_path)
        return image[None], nodes, edges, sample_id


def index_pid_samples(root, split_seed, max_nodes, cache_dir=None):
    """Scans the dataset once and returns {split: [(image, graph, id), ...]}.

    Results are cached on disk keyed by the root path, seed and node limit,
    so subsequent runs skip the expensive rglob + GraphML parsing.
    """
    root = Path(root)
    cache_dir = Path(cache_dir) if cache_dir else Path.home() / ".cache" / "relationformer"
    cache_key = hashlib.sha1(f"{root.resolve()}:{split_seed}:{max_nodes}".encode()).hexdigest()[:16]
    cache_path = cache_dir / f"pid_index_{cache_key}.pickle"

    if cache_path.is_file():
        with open(cache_path, "rb") as cache_file:
            cached = pickle.load(cache_file)
        print(f"Loaded P&ID sample index from {cache_path}")
        return {
            split: [(root / img, root / graph, sample_id) for img, graph, sample_id in samples]
            for split, samples in cached.items()
        }

    print(f"Indexing P&ID samples under {root} (first run, this can take a few minutes)...")
    start_time = time.time()
    graph_paths = sorted(root.rglob("*.graphml"))
    print(f"Found {len(graph_paths)} GraphML files, filtering...")

    splits = {split: [] for split in PatchedPIDDataset.SPLIT_RANGES}
    skipped_graphs = 0
    for index, graph_path in enumerate(graph_paths, 1):
        image_path = graph_path.with_suffix(".png")
        if not image_path.is_file():
            continue

        sample_key = graph_path.relative_to(root).as_posix()
        split_value = int(hashlib.sha1(f"{split_seed}:{sample_key}".encode()).hexdigest(), 16) % 100
        for split, (split_start, split_end) in PatchedPIDDataset.SPLIT_RANGES.items():
            if split_start <= split_value < split_end:
                if is_trainable_pid_graph(graph_path, max_nodes=max_nodes):
                    splits[split].append(
                        (
                            image_path.relative_to(root).as_posix(),
                            graph_path.relative_to(root).as_posix(),
                            graph_path.relative_to(root).with_suffix("").as_posix(),
                        )
                    )
                else:
                    skipped_graphs += 1
                break

        if index % 5000 == 0:
            print(f"  {index}/{len(graph_paths)} graphs processed ({time.time() - start_time:.0f}s)")

    print(
        f"Indexing done in {time.time() - start_time:.0f}s "
        f"({skipped_graphs} unusable graphs skipped): "
        + ", ".join(f"{split}={len(samples)}" for split, samples in splits.items())
    )

    cache_dir.mkdir(parents=True, exist_ok=True)
    with open(cache_path, "wb") as cache_file:
        pickle.dump(splits, cache_file)

    return {
        split: [(root / img, root / graph, sample_id) for img, graph, sample_id in samples]
        for split, samples in splits.items()
    }


def load_pid_graph(graph_path, image_path):
    """Converts GraphML bounding boxes to normalized node centers and edge indices.

    Nodes that take part in no edge are dropped, since only line connectivity is learned.
    """
    with Image.open(image_path) as image:
        width, height = image.size

    namespace = "{http://graphml.graphdrawing.org/xmlns}"
    graph = ElementTree.parse(graph_path).getroot().find(f"{namespace}graph")
    node_centers = {}
    for node in graph.findall(f"{namespace}node"):
        attributes = {data.get("key"): data.text for data in node.findall(f"{namespace}data")}
        xmin = attributes.get("d1", attributes.get("d5"))
        ymin = attributes.get("d2", attributes.get("d6"))
        xmax = attributes.get("d3", attributes.get("d7"))
        ymax = attributes.get("d4", attributes.get("d8"))
        if None in (xmin, ymin, xmax, ymax):
            raise ValueError(f"Node {node.get('id')} in {graph_path} has no bounding box")

        node_centers[node.get("id")] = (
            (float(xmin) + float(xmax)) / (2 * width),
            (float(ymin) + float(ymax)) / (2 * height),
        )

    edge_pairs = set()
    for edge in graph.findall(f"{namespace}edge"):
        source, target = edge.get("source"), edge.get("target")
        if source in node_centers and target in node_centers and source != target:
            edge_pairs.add((source, target) if source < target else (target, source))

    if not edge_pairs:
        raise ValueError(f"Graph {graph_path} has no usable edges")

    connected_ids = sorted({node_id for pair in edge_pairs for node_id in pair})
    node_indices = {node_id: index for index, node_id in enumerate(connected_ids)}
    nodes = torch.tensor([node_centers[node_id] for node_id in connected_ids], dtype=torch.float32)
    edges = torch.tensor(
        sorted((node_indices[source], node_indices[target]) for source, target in edge_pairs),
        dtype=torch.long,
    )
    return nodes, edges


def is_trainable_pid_graph(graph_path, max_nodes=None):
    namespace = "{http://graphml.graphdrawing.org/xmlns}"
    graph = ElementTree.parse(graph_path).getroot().find(f"{namespace}graph")
    node_ids = set()
    for node in graph.findall(f"{namespace}node"):
        attributes = {data.get("key"): data.text for data in node.findall(f"{namespace}data")}
        bounding_box = (
            attributes.get("d1", attributes.get("d5")),
            attributes.get("d2", attributes.get("d6")),
            attributes.get("d3", attributes.get("d7")),
            attributes.get("d4", attributes.get("d8")),
        )
        if None in bounding_box:
            return False
        node_ids.add(node.get("id"))

    if max_nodes is not None and len(node_ids) > max_nodes:
        return False

    return any(
        edge.get("source") in node_ids
        and edge.get("target") in node_ids
        and edge.get("source") != edge.get("target")
        for edge in graph.findall(f"{namespace}edge")
    )


def image_graph_collate_road_network(batch):
    images = torch.cat([item[0] for item in batch], 0).contiguous()
    nodes = [item[1] for item in batch]
    edges = [item[2] for item in batch]
    ids = [item[3] for item in batch]
    return [images, nodes, edges, ids]


def load_dataset(dataset_path):
    """
    Loads the chosen split of the data

    :param dataset_path: path of the data split pickle
    :param max_prev_node: only return the last previous 'max_prev_node' elements in the adjacency row of a node
    :param return_coordinates: returns coordinates on the real map for each datapoint
    :return:
    """
    with open(dataset_path, "rb") as pickled_file:
        dataset = pickle.load(pickled_file)

    list_nodes = []
    list_edges = []
    ids = list(dataset.keys())
    random.Random(42).shuffle(
        ids
    )  # permute to remove any correlation between consecutive datapoints

    for id in ids:
        datapoint = dataset[id]

        # Retrieve from dataset
        nodes = (torch.FloatTensor(datapoint["nodes"]) + 1) / 2
        edges = torch.tensor((datapoint["edges"]))[:, :2]

        # Sort edges
        edges_sorted = torch.sort(edges, 1)[0]
        edges_sorted = torch.stack(sorted(edges_sorted, key=lambda a: (a[0], a[1])))

        # Get rid of duplicate edges and nodes that do not participate in edges
        edges_clean = torch.unique(edges_sorted, dim=0)
        participating_nodes = torch.unique(edges_clean)

        for node in torch.arange(nodes.shape[0] - 1, -1, -1):
            if node not in participating_nodes:
                edges_clean[edges_clean > node] -= 1

        nodes_clean = nodes[participating_nodes]

        list_nodes.append(nodes_clean)
        list_edges.append(edges_clean)

    return ids, list_nodes, list_edges


def load_images(ids, images_path):
    """
    Load images from arrays in pickle files

    :param ids: ids of the images in the data order
    :param images_path: path of the pickle file
    :return: the images, as pytorch tensors
    """
    images = []
    with open(images_path, "rb") as pickled_file:
        images_features = pickle.load(pickled_file)
    for id in ids:
        img = torch.FloatTensor(images_features["{:0>7d}".format(int(id))])
        assert img.shape[1] == img.shape[2]
        assert img.shape[1] in {64}
        images.append(img)

    return images


def load_raw_images(ids, images_path):
    """
    Load images from raw files

    :param ids: ids of the images in the data order
    :param images_path: path of the raw images
    :return: the images, as pytorch tensors
    """
    images = []
    for count, id in enumerate(ids):
        # if count % 10000 == 0:
        #     print(count)
        image_path = images_path + "{:0>7d}".format(int(id)) + ".png"
        img = Image.open(image_path).convert("L")
        img = tvf.to_tensor(img)
        assert img.shape[1] == img.shape[2]
        assert img.shape[1] in {64, 128}
        images.append(img)
    return images


def build_road_network_data(config, mode="split"):
    if config.DATA.DATASET == "patched-pid-2D":
        dataset_kwargs = {
            "root_path": config.DATA.DATA_PATH,
            "image_size": config.DATA.IMG_SIZE,
            "split_seed": config.DATA.SEED,
            "max_nodes": config.MODEL.DECODER.OBJ_TOKEN,
        }
        if mode == "split":
            return PatchedPIDDataset(split="train", **dataset_kwargs), PatchedPIDDataset(
                split="valid", **dataset_kwargs
            )
        if mode == "test":
            return PatchedPIDDataset(split="test", **dataset_kwargs)
        raise ValueError(f"Unsupported data build mode: {mode}")

    if mode == "split":
        train_ds = ToulouseRoadNetworkDataset(root_path=config.DATA.DATA_PATH, split="train")
        val_ds = ToulouseRoadNetworkDataset(root_path=config.DATA.DATA_PATH, split="valid")

        return train_ds, val_ds
    elif mode == "test":
        test_ds = ToulouseRoadNetworkDataset(root_path=config.DATA.DATA_PATH, split="test")
        return test_ds


if __name__ == "__main__":
    import cv2
    import numpy as np
    from torch.utils.data import DataLoader
    from torch.nn.utils.rnn import pad_sequence

    def custom_collate_fn(batch):
        """
        Custom collate function ordering the element in a batch by descending length

        :param batch: batch from pytorch dataloader
        :return: the ordered batch
        """
        x_adj, x_coord, y_adj, y_coord, img, seq_len, ids = zip(*batch)

        x_adj = pad_sequence(x_adj, batch_first=True, padding_value=0)
        x_coord = pad_sequence(x_coord, batch_first=True, padding_value=0)
        y_adj = pad_sequence(y_adj, batch_first=True, padding_value=0)
        y_coord = pad_sequence(y_coord, batch_first=True, padding_value=0)
        img, seq_len = torch.stack(img), torch.stack(seq_len)

        seq_len, perm_index = seq_len.sort(0, descending=True)
        x_adj = x_adj[perm_index]
        x_coord = x_coord[perm_index]
        y_adj = y_adj[perm_index]
        y_coord = y_coord[perm_index]
        img = img[perm_index]
        ids = [ids[perm_index[i]] for i in range(perm_index.shape[0])]

        return x_adj, x_coord, y_adj, y_coord, img, seq_len, ids

    class obj:
        def __init__(self, dict1):
            self.__dict__.update(dict1)

    def dict2obj(dict1):
        return json.loads(json.dumps(dict1), object_hook=obj)

    config = "configs/road_2D_deform_detr.yaml"
    with open(config) as f:
        config = yaml.load(f, Loader=yaml.FullLoader)
    config = dict2obj(config)

    train_ds, val_ds = build_road_network_data(config, mode="split")
    # dataloader = DataLoader(train_ds, batch_size=16, shuffle=False, collate_fn=custom_collate_fn)

    for i in [14, 2, 4, 6, 30, 26, 43, 24, 69173, 48360, 60201]:
        ret = train_ds[i]  # some strange cases 14, 2, 4, 6, 30, 26, 43, 24
        print(ret[-1])

        nodes_pixels = (ret[1] * ret[0].shape[-1]).type(torch.int32).numpy()

        image = ret[0].squeeze().cpu().numpy()
        image = np.flip(image, 0).copy()

        for node in nodes_pixels:
            image = cv2.circle(image, node, 5, (0, 0, 0), 2)
            cv2.imshow("testing", image)
            cv2.waitKey()

        print(ret[-2])
