import numpy as np
from scipy.optimize import linear_sum_assignment
from sklearn.metrics import average_precision_score


def _box_area(boxes):
    return np.clip(boxes[:, 2] - boxes[:, 0], 0, None) * np.clip(
        boxes[:, 3] - boxes[:, 1], 0, None
    )


def _cxcywh_to_xyxy(boxes):
    centers = boxes[:, :2]
    half_size = boxes[:, 2:] / 2
    return np.concatenate((centers - half_size, centers + half_size), axis=1)


def _generalized_iou(boxes1, boxes2):
    area1 = _box_area(boxes1)
    area2 = _box_area(boxes2)
    intersection_min = np.maximum(boxes1[:, None, :2], boxes2[None, :, :2])
    intersection_max = np.minimum(boxes1[:, None, 2:], boxes2[None, :, 2:])
    intersection_size = np.clip(intersection_max - intersection_min, 0, None)
    intersection = intersection_size[..., 0] * intersection_size[..., 1]
    union = area1[:, None] + area2[None, :] - intersection
    iou = intersection / np.maximum(union, 1e-12)

    enclosing_min = np.minimum(boxes1[:, None, :2], boxes2[None, :, :2])
    enclosing_max = np.maximum(boxes1[:, None, 2:], boxes2[None, :, 2:])
    enclosing_size = np.clip(enclosing_max - enclosing_min, 0, None)
    enclosing_area = enclosing_size[..., 0] * enclosing_size[..., 1]
    return iou - (enclosing_area - union) / np.maximum(enclosing_area, 1e-12)


def _pair(edge):
    source, target = map(int, edge)
    return (source, target) if source < target else (target, source)


class EdgeMeanAveragePrecision:
    """Edge mAP from Algorithm A1 of the P&ID Relationformer paper."""

    def __init__(self, num_classes):
        self.num_classes = num_classes
        self.labels = [[] for _ in range(num_classes)]
        self.scores = [[] for _ in range(num_classes)]

    def add(
        self,
        predicted_boxes,
        predicted_edges,
        predicted_edge_scores,
        predicted_edge_classes,
        target_boxes,
        target_edges,
        target_edge_classes,
    ):
        predicted_boxes = np.asarray(predicted_boxes)
        target_boxes = np.asarray(target_boxes)
        predicted_edges = np.asarray(predicted_edges, dtype=np.int64).reshape(-1, 2)
        target_edges = np.asarray(target_edges, dtype=np.int64).reshape(-1, 2)
        predicted_edge_scores = np.asarray(predicted_edge_scores)
        predicted_edge_classes = np.asarray(predicted_edge_classes, dtype=np.int64)
        target_edge_classes = np.asarray(target_edge_classes, dtype=np.int64)

        predicted_to_target = {}
        target_to_predicted = {}
        if len(predicted_boxes) and len(target_boxes):
            similarity = _generalized_iou(
                _cxcywh_to_xyxy(target_boxes), _cxcywh_to_xyxy(predicted_boxes)
            )
            target_indices, predicted_indices = linear_sum_assignment(-similarity)
            predicted_to_target = dict(zip(predicted_indices.tolist(), target_indices.tolist()))
            target_to_predicted = dict(zip(target_indices.tolist(), predicted_indices.tolist()))

        target_edge_lookup = {
            _pair(edge): int(edge_class)
            for edge, edge_class in zip(target_edges, target_edge_classes)
        }
        predicted_edge_lookup = {_pair(edge) for edge in predicted_edges}

        for edge, score, edge_class in zip(
            predicted_edges, predicted_edge_scores, predicted_edge_classes
        ):
            class_index = int(edge_class) - 1
            mapped_pair = None
            if int(edge[0]) in predicted_to_target and int(edge[1]) in predicted_to_target:
                mapped_pair = _pair(
                    (predicted_to_target[int(edge[0])], predicted_to_target[int(edge[1])])
                )
            is_true_positive = (
                mapped_pair in target_edge_lookup
                and target_edge_lookup[mapped_pair] == int(edge_class)
            )
            self.labels[class_index].append(int(is_true_positive))
            self.scores[class_index].append(float(score))

        for edge, edge_class in zip(target_edges, target_edge_classes):
            source, target = map(int, edge)
            corresponding_edge = None
            if source in target_to_predicted and target in target_to_predicted:
                corresponding_edge = _pair((target_to_predicted[source], target_to_predicted[target]))
            if corresponding_edge not in predicted_edge_lookup:
                class_index = int(edge_class) - 1
                self.labels[class_index].append(1)
                self.scores[class_index].append(0.0)

    def compute(self):
        per_class = []
        for labels, scores in zip(self.labels, self.scores):
            if not labels or not any(labels):
                per_class.append(float("nan"))
            else:
                per_class.append(float(average_precision_score(labels, scores)))
        return float(np.nanmean(per_class)), per_class
