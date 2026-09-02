"""Render GraphML annotations over every image in the reduced P&ID dataset."""

from pathlib import Path
from xml.etree import ElementTree

from PIL import Image, ImageDraw

PROJECT_ROOT = Path(__file__).resolve().parent
DATASET_ROOT = PROJECT_ROOT / "data" / "P&ID_imgs" / "PID2Graph" / "PID2Grpah-reducted"
REVIEW_ROOT = PROJECT_ROOT / "annotation_review"
NAMESPACE = "{http://graphml.graphdrawing.org/xmlns}"


def get_bounding_box(node):
    attributes = {data.get("key"): data.text for data in node.findall(f"{NAMESPACE}data")}
    xmin = attributes.get("d1", attributes.get("d5"))
    ymin = attributes.get("d2", attributes.get("d6"))
    xmax = attributes.get("d3", attributes.get("d7"))
    ymax = attributes.get("d4", attributes.get("d8"))
    if None in (xmin, ymin, xmax, ymax):
        return None
    return tuple(map(float, (xmin, ymin, xmax, ymax)))


def render_annotation(graph_path):
    image_path = graph_path.with_suffix(".png")
    if not image_path.is_file():
        return False

    graph = ElementTree.parse(graph_path).getroot().find(f"{NAMESPACE}graph")
    nodes = {}
    for node in graph.findall(f"{NAMESPACE}node"):
        bounding_box = get_bounding_box(node)
        if bounding_box is not None:
            nodes[node.get("id")] = bounding_box

    relative_path = graph_path.relative_to(DATASET_ROOT).with_suffix(".png")
    output_path = REVIEW_ROOT / relative_path
    output_path.parent.mkdir(parents=True, exist_ok=True)

    with Image.open(image_path) as source:
        image = source.convert("RGB")
    draw = ImageDraw.Draw(image)
    for edge in graph.findall(f"{NAMESPACE}edge"):
        source = nodes.get(edge.get("source"))
        target = nodes.get(edge.get("target"))
        if source is not None and target is not None:
            draw.line(
                (
                    (source[0] + source[2]) / 2,
                    (source[1] + source[3]) / 2,
                    (target[0] + target[2]) / 2,
                    (target[1] + target[3]) / 2,
                ),
                fill="orange",
                width=4,
            )
    for xmin, ymin, xmax, ymax in nodes.values():
        draw.rectangle((xmin, ymin, xmax, ymax), outline="red", width=3)
        center_x, center_y = (xmin + xmax) / 2, (ymin + ymax) / 2
        draw.ellipse((center_x - 5, center_y - 5, center_x + 5, center_y + 5), fill="cyan")
    image.save(output_path)
    return True


def main():
    if not DATASET_ROOT.is_dir():
        raise FileNotFoundError(f"Reduced dataset directory does not exist: {DATASET_ROOT}")

    graph_paths = sorted(DATASET_ROOT.rglob("*.graphml"))
    rendered = sum(render_annotation(graph_path) for graph_path in graph_paths)
    print(f"Rendered {rendered} annotated images to {REVIEW_ROOT}")


if __name__ == "__main__":
    main()
