"""Convert object/relation tokens into an undirected graph."""

import numpy as np
import torch
from torchvision.ops import batched_nms

from box_ops_2D import box_cxcywh_to_xyxy


@torch.no_grad()
def relation_infer(
    h, out, model, obj_token, rln_token, nms=False, map_=False,
    node_threshold=0.5, edge_threshold=0.5, edge_chunk_size=4096,
):
    """Score all unordered pairs with bounded relation-head memory.

    Preserve the mean-of-two-direction-logits rule and output ordering.
    Boxes use normalized (cx, cy, width, height); edges index the kept nodes.
    """
    if not 0 <= node_threshold <= 1 or not 0 <= edge_threshold <= 1:
        raise ValueError("Node and edge thresholds must lie in [0, 1]")
    if edge_chunk_size < 1:
        raise ValueError("edge_chunk_size must be positive")
    if rln_token not in (0, 1):
        raise ValueError("The relation head supports zero or one relation token")

    object_token = h[..., :obj_token, :]
    node_probs = out["pred_logits"].softmax(-1)
    node_scores, node_classes = node_probs[..., 1:].max(-1)
    node_classes = node_classes + 1
    valid_token = node_scores >= node_threshold

    pred_nodes, pred_edges = [], []
    boxes, scores, classes, edge_scores_out, edge_classes_out = [], [], [], [], []
    for batch_id in range(h.shape[0]):
        node_id = torch.nonzero(valid_token[batch_id]).squeeze(1)
        if nms and node_id.numel():
            kept = batched_nms(
                box_cxcywh_to_xyxy(out["pred_nodes"][batch_id, node_id].float()),
                node_scores[batch_id, node_id].float(),
                node_classes[batch_id, node_id],
                iou_threshold=0.90,
            )
            node_id = node_id[kept].sort().values

        pred_nodes.append(out["pred_nodes"][batch_id, node_id, :2].detach())
        if map_:
            boxes.append(out["pred_nodes"][batch_id, node_id].detach().cpu().numpy())
            scores.append(node_scores[batch_id, node_id].cpu().numpy())
            classes.append(node_classes[batch_id, node_id].cpu().numpy())

        pairs = torch.triu_indices(len(node_id), len(node_id), offset=1, device=h.device).T
        kept_edges, kept_scores, kept_classes = [], [], []
        for chunk in pairs.split(edge_chunk_size):
            if not len(chunk):
                continue
            first = object_token[batch_id, node_id[chunk[:, 0]]]
            second = object_token[batch_id, node_id[chunk[:, 1]]]
            forward_parts, reverse_parts = [first, second], [second, first]
            if rln_token:
                relation = h[batch_id, obj_token:obj_token + 1].expand(len(chunk), -1)
                forward_parts.append(relation)
                reverse_parts.append(relation)
            logits = model.relation_embed(torch.cat(forward_parts, dim=1))
            reverse_logits = model.relation_embed(torch.cat(reverse_parts, dim=1))
            probabilities = ((logits + reverse_logits) / 2.0).softmax(-1)
            edge_scores, edge_classes = probabilities[:, 1:].max(-1)
            keep = edge_scores >= edge_threshold
            kept_edges.append(chunk[keep].cpu())
            kept_scores.append(edge_scores[keep].cpu())
            kept_classes.append((edge_classes[keep] + 1).cpu())

        pred_edges.append(
            torch.cat(kept_edges).numpy() if kept_edges else np.empty((0, 2), dtype=np.int64)
        )
        if map_:
            edge_scores_out.append(
                torch.cat(kept_scores).numpy() if kept_scores else np.empty(0, dtype=np.float32)
            )
            edge_classes_out.append(
                torch.cat(kept_classes).numpy() if kept_classes else np.empty(0, dtype=np.int64)
            )

    if map_:
        return pred_nodes, pred_edges, boxes, scores, classes, edge_scores_out, edge_classes_out
    return pred_nodes, pred_edges
