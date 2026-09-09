"""Prepare a paper-aligned patched PID2Graph dataset.

The script does three things:
1) validates every PNG/GraphML pair,
2) rewrites GraphML with canonical keys + canonical class labels,
3) optionally compares image files against another dataset root.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import shutil
import time
from collections import Counter
from pathlib import Path
from xml.etree import ElementTree

PID_NODE_CLASS_TO_ID = {
    "general": 1,
    "tank": 2,
    "valve": 3,
    "instrumentation": 4,
    "pump": 5,
    "inlet_outlet": 6,
    "arrow": 7,
    "crossing": 8,
    "ankle": 9,
    "border": 10,
}

PID_NODE_LABEL_ALIASES = {
    "general": "general",
    "tank": "tank",
    "tank_vessel": "tank",
    "tank/vessel": "tank",
    "vessel": "tank",
    "valve": "valve",
    "instrumentation": "instrumentation",
    "instrument": "instrumentation",
    "pump": "pump",
    "compressor": "pump",
    "pump_compressor": "pump",
    "pump/compressor": "pump",
    "inlet_outlet": "inlet_outlet",
    "inlet/outlet": "inlet_outlet",
    "inlet": "inlet_outlet",
    "outlet": "inlet_outlet",
    "arrow": "arrow",
    "crossing": "crossing",
    "connector": "ankle",
    "ankle": "ankle",
    "border": "border",
    "border_node": "border",
}

PID_EDGE_CLASS_TO_ID = {
    "solid": 1,
    "non_solid": 2,
}

PID_EDGE_LABEL_ALIASES = {
    "solid": "solid",
    "non_solid": "non_solid",
    "non-solid": "non_solid",
    "nonsolid": "non_solid",
    "dashed": "non_solid",
    "dash": "non_solid",
}

NAMESPACE = "{http://graphml.graphdrawing.org/xmlns}"


def _normalize(label: str) -> str:
    return label.strip().lower().replace(" ", "_")


def _node_id_to_name(class_id: int) -> str:
    for class_name, mapped_id in PID_NODE_CLASS_TO_ID.items():
        if mapped_id == class_id:
            return class_name
    raise KeyError(class_id)


def _edge_id_to_name(class_id: int) -> str:
    for class_name, mapped_id in PID_EDGE_CLASS_TO_ID.items():
        if mapped_id == class_id:
            return class_name
    raise KeyError(class_id)


def _key_map(root: ElementTree.Element) -> dict[str, str]:
    mapping = {}
    for key in root.findall(f"{NAMESPACE}key"):
        key_id = key.get("id")
        key_name = key.get("attr.name")
        if key_id and key_name:
            mapping[key_id] = key_name
    return mapping


def _attrs(element: ElementTree.Element, key_map: dict[str, str]) -> dict[str, str]:
    out = {}
    for data in element.findall(f"{NAMESPACE}data"):
        key_name = key_map.get(data.get("key"))
        if key_name and data.text is not None and key_name not in out:
            out[key_name] = data.text
    return out


def _map_node_label(raw: str) -> str:
    normalized = _normalize(raw)
    canonical = PID_NODE_LABEL_ALIASES.get(normalized)
    if canonical is None:
        raise ValueError(f"Unsupported node label: {raw}")
    return canonical


def _map_edge_label(raw: str | None) -> str:
    normalized = _normalize(raw or "solid")
    canonical = PID_EDGE_LABEL_ALIASES.get(normalized)
    if canonical is None:
        raise ValueError(f"Unsupported edge label: {raw}")
    return canonical


def _sha1(path: Path) -> str:
    digest = hashlib.sha1()
    with open(path, "rb") as file_handle:
        while True:
            chunk = file_handle.read(1024 * 1024)
            if not chunk:
                break
            digest.update(chunk)
    return digest.hexdigest()


def _log_progress(phase: str, current: int, total: int, started_at: float, interval: int) -> None:
    if current != total and current % interval != 0:
        return

    elapsed = max(time.monotonic() - started_at, 1e-6)
    rate = current / elapsed
    remaining = max(total - current, 0)
    eta_seconds = remaining / rate if rate > 0 else 0
    percentage = 100 * current / total if total else 100
    print(
        f"[{phase}] {current}/{total} ({percentage:.1f}%) | "
        f"{rate:.1f} file/s | elapsed {elapsed / 60:.1f} min | ETA {eta_seconds / 60:.1f} min",
        flush=True,
    )


def parse_graph(graph_path: Path) -> tuple[list[dict], list[dict]]:
    root = ElementTree.parse(graph_path).getroot()
    graph = root.find(f"{NAMESPACE}graph")
    if graph is None:
        raise ValueError(f"Missing <graph> in {graph_path}")
    keys = _key_map(root)

    nodes = []
    for node in graph.findall(f"{NAMESPACE}node"):
        attributes = _attrs(node, keys)
        xmin = attributes.get("xmin")
        ymin = attributes.get("ymin")
        xmax = attributes.get("xmax")
        ymax = attributes.get("ymax")
        label = attributes.get("label")
        if None in (xmin, ymin, xmax, ymax, label):
            raise ValueError(f"Incomplete node data in {graph_path}, node={node.get('id')}")

        nodes.append(
            {
                "id": node.get("id"),
                "label": _map_node_label(label),
                "xmin": float(xmin),
                "ymin": float(ymin),
                "xmax": float(xmax),
                "ymax": float(ymax),
            }
        )

    edges = []
    for edge in graph.findall(f"{NAMESPACE}edge"):
        attributes = _attrs(edge, keys)
        source = edge.get("source")
        target = edge.get("target")
        if source is None or target is None or source == target:
            continue
        edges.append(
            {
                "source": source,
                "target": target,
                "label": _map_edge_label(attributes.get("edge_label")),
            }
        )

    if not edges:
        raise ValueError(f"Graph has no valid edges: {graph_path}")

    return nodes, edges


def write_graph(graph_path: Path, nodes: list[dict], edges: list[dict]) -> None:
    graphml = ElementTree.Element("graphml")
    graphml.set("xmlns", "http://graphml.graphdrawing.org/xmlns")
    graphml.set("xmlns:xsi", "http://www.w3.org/2001/XMLSchema-instance")
    graphml.set(
        "xsi:schemaLocation",
        "http://graphml.graphdrawing.org/xmlns http://graphml.graphdrawing.org/xmlns/1.0/graphml.xsd",
    )

    key_node_label = ElementTree.SubElement(
        graphml, "key", {"id": "d0", "for": "node", "attr.name": "label", "attr.type": "string"}
    )
    key_node_xmin = ElementTree.SubElement(
        graphml, "key", {"id": "d1", "for": "node", "attr.name": "xmin", "attr.type": "double"}
    )
    key_node_ymin = ElementTree.SubElement(
        graphml, "key", {"id": "d2", "for": "node", "attr.name": "ymin", "attr.type": "double"}
    )
    key_node_xmax = ElementTree.SubElement(
        graphml, "key", {"id": "d3", "for": "node", "attr.name": "xmax", "attr.type": "double"}
    )
    key_node_ymax = ElementTree.SubElement(
        graphml, "key", {"id": "d4", "for": "node", "attr.name": "ymax", "attr.type": "double"}
    )
    key_edge_label = ElementTree.SubElement(
        graphml, "key", {"id": "d5", "for": "edge", "attr.name": "edge_label", "attr.type": "string"}
    )

    _ = (
        key_node_label,
        key_node_xmin,
        key_node_ymin,
        key_node_xmax,
        key_node_ymax,
        key_edge_label,
    )

    graph_elem = ElementTree.SubElement(graphml, "graph", {"edgedefault": "undirected"})

    for node in nodes:
        node_elem = ElementTree.SubElement(graph_elem, "node", {"id": node["id"]})
        ElementTree.SubElement(node_elem, "data", {"key": "d0"}).text = node["label"]
        ElementTree.SubElement(node_elem, "data", {"key": "d1"}).text = f"{node['xmin']:.6f}"
        ElementTree.SubElement(node_elem, "data", {"key": "d2"}).text = f"{node['ymin']:.6f}"
        ElementTree.SubElement(node_elem, "data", {"key": "d3"}).text = f"{node['xmax']:.6f}"
        ElementTree.SubElement(node_elem, "data", {"key": "d4"}).text = f"{node['ymax']:.6f}"

    for edge in edges:
        edge_elem = ElementTree.SubElement(
            graph_elem,
            "edge",
            {"source": edge["source"], "target": edge["target"]},
        )
        ElementTree.SubElement(edge_elem, "data", {"key": "d5"}).text = edge["label"]

    graph_path.parent.mkdir(parents=True, exist_ok=True)
    ElementTree.ElementTree(graphml).write(graph_path, encoding="utf-8", xml_declaration=True)


def compare_images(
    source_root: Path,
    compare_root: Path,
    relative_pngs: list[Path],
    log_interval: int,
) -> dict:
    report = {
        "checked": 0,
        "missing_in_compare": 0,
        "different_content": 0,
        "same_content": 0,
        "examples_missing": [],
        "examples_different": [],
    }

    started_at = time.monotonic()
    total = len(relative_pngs)
    for index, relative_png in enumerate(relative_pngs, 1):
        source_path = source_root / relative_png
        compare_path = compare_root / relative_png
        report["checked"] += 1
        if not compare_path.is_file():
            report["missing_in_compare"] += 1
            if len(report["examples_missing"]) < 10:
                report["examples_missing"].append(relative_png.as_posix())
            _log_progress("image comparison", index, total, started_at, log_interval)
            continue

        if source_path.stat().st_size == compare_path.stat().st_size and _sha1(source_path) == _sha1(compare_path):
            report["same_content"] += 1
        else:
            report["different_content"] += 1
            if len(report["examples_different"]) < 10:
                report["examples_different"].append(relative_png.as_posix())
        _log_progress("image comparison", index, total, started_at, log_interval)

    return report


def main() -> None:
    parser = argparse.ArgumentParser(description="Convert patched PID2Graph GraphML annotations to paper format.")
    parser.add_argument("--source-root", default="data/PID2Graph/Patched", help="input patched dataset root")
    parser.add_argument("--output-root", default="data/PID2Graph/Patched_paper", help="converted dataset root")
    parser.add_argument(
        "--compare-root",
        default=None,
        help="optional second root used to verify whether PNG files are identical",
    )
    parser.add_argument("--copy-images", action="store_true", help="copy PNG files into output-root")
    parser.add_argument(
        "--log-interval",
        type=int,
        default=500,
        help="print progress every N files (default: 500)",
    )
    args = parser.parse_args()

    if args.log_interval <= 0:
        parser.error("--log-interval must be greater than zero")

    source_root = Path(args.source_root).resolve()
    output_root = Path(args.output_root).resolve()
    compare_root = Path(args.compare_root).resolve() if args.compare_root else None

    if not source_root.is_dir():
        raise FileNotFoundError(f"Source dataset root not found: {source_root}")

    graph_paths = sorted(source_root.rglob("*.graphml"))
    if not graph_paths:
        raise RuntimeError(f"No GraphML files found under {source_root}")

    node_counts = Counter()
    edge_counts = Counter()
    converted = 0
    skipped = 0
    failures = []
    relative_pngs = []

    print(f"Found {len(graph_paths)} GraphML files under {source_root}", flush=True)
    conversion_started_at = time.monotonic()
    for index, graph_path in enumerate(graph_paths, 1):
        relative = graph_path.relative_to(source_root)
        image_path = graph_path.with_suffix(".png")
        if not image_path.is_file():
            skipped += 1
            _log_progress(
                "annotation conversion",
                index,
                len(graph_paths),
                conversion_started_at,
                args.log_interval,
            )
            continue

        try:
            nodes, edges = parse_graph(graph_path)
        except Exception as ex:  # noqa: BLE001
            failures.append({"file": relative.as_posix(), "error": str(ex)})
            _log_progress(
                "annotation conversion",
                index,
                len(graph_paths),
                conversion_started_at,
                args.log_interval,
            )
            continue

        for node in nodes:
            node_counts[node["label"]] += 1
        for edge in edges:
            edge_counts[edge["label"]] += 1

        out_graph = output_root / relative
        write_graph(out_graph, nodes, edges)
        if args.copy_images:
            out_image = output_root / relative.with_suffix(".png")
            out_image.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(image_path, out_image)

        relative_pngs.append(relative.with_suffix(".png"))
        converted += 1
        _log_progress(
            "annotation conversion",
            index,
            len(graph_paths),
            conversion_started_at,
            args.log_interval,
        )

    image_comparison = None
    if compare_root is not None:
        if not compare_root.is_dir():
            raise FileNotFoundError(f"Compare root not found: {compare_root}")
        image_comparison = compare_images(
            source_root,
            compare_root,
            relative_pngs,
            args.log_interval,
        )

    report = {
        "source_root": source_root.as_posix(),
        "output_root": output_root.as_posix(),
        "compare_root": compare_root.as_posix() if compare_root else None,
        "total_graphml": len(graph_paths),
        "converted_graphml": converted,
        "skipped_missing_png": skipped,
        "failed_graphs": len(failures),
        "failure_examples": failures[:20],
        "node_class_to_id": PID_NODE_CLASS_TO_ID,
        "edge_class_to_id": PID_EDGE_CLASS_TO_ID,
        "node_class_counts": dict(sorted(node_counts.items())),
        "edge_class_counts": dict(sorted(edge_counts.items())),
        "image_comparison": image_comparison,
    }

    output_root.mkdir(parents=True, exist_ok=True)
    report_path = output_root / "conversion_report.json"
    with open(report_path, "w", encoding="utf-8") as report_file:
        json.dump(report, report_file, indent=2)

    print(f"Converted graphs: {converted}/{len(graph_paths)}")
    print(f"Skipped graphs without PNG: {skipped}")
    print(f"Failed graphs: {len(failures)}")
    if image_comparison is not None:
        print("Image comparison:", json.dumps(image_comparison, indent=2))
    print(f"Report written to: {report_path}")

    # sanity: verify class ids are contiguous and stable
    expected_node_ids = list(range(1, len(PID_NODE_CLASS_TO_ID) + 1))
    expected_edge_ids = list(range(1, len(PID_EDGE_CLASS_TO_ID) + 1))
    node_ids = sorted(PID_NODE_CLASS_TO_ID.values())
    edge_ids = sorted(PID_EDGE_CLASS_TO_ID.values())
    if node_ids != expected_node_ids:
        raise RuntimeError(
            f"Node class ids must be contiguous starting from 1, found {node_ids}"
        )
    if edge_ids != expected_edge_ids:
        raise RuntimeError(
            f"Edge class ids must be contiguous starting from 1, found {edge_ids}"
        )

    print(
        "Node classes:",
        {idx: _node_id_to_name(idx) for idx in expected_node_ids},
    )
    print(
        "Edge classes:",
        {idx: _edge_id_to_name(idx) for idx in expected_edge_ids},
    )


if __name__ == "__main__":
    main()
