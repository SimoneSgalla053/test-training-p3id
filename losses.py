import torch
import torch.nn.functional as F
from torch import nn
import itertools
import pdb
import box_ops_2D
import numpy as np


def sigmoid_focal_loss(inputs, targets, num_boxes, alpha: float = 0.25, gamma: float = 2):
    """
    Loss used in RetinaNet for dense detection: https://arxiv.org/abs/1708.02002.
    Args:
        inputs: A float tensor of arbitrary shape.
                The predictions for each example.
        targets: A float tensor with the same shape as inputs. Stores the binary
                 classification label for each element in inputs
                (0 for the negative class and 1 for the positive class).
        alpha: (optional) Weighting factor in range (0,1) to balance
                positive vs negative examples. Default = -1 (no weighting).
        gamma: Exponent of the modulating factor (1 - p_t) to
               balance easy vs hard examples.
    Returns:
        Loss tensor
    """
    prob = inputs.sigmoid()
    ce_loss = F.binary_cross_entropy_with_logits(inputs, targets, reduction="none")
    p_t = prob * targets + (1 - prob) * (1 - targets)
    loss = ce_loss * ((1 - p_t) ** gamma)

    if alpha >= 0:
        alpha_t = alpha * targets + (1 - alpha) * (1 - targets)
        loss = alpha_t * loss

    return loss.sum() / num_boxes


@torch.no_grad()
def accuracy(output, target, topk=(1,)):
    """Computes the precision@k for the specified values of k"""
    if target.numel() == 0:
        return [torch.zeros([], device=output.device)]
    maxk = max(topk)
    batch_size = target.size(0)

    _, pred = output.topk(maxk, 1, True, True)
    pred = pred.t()
    correct = pred.eq(target.view(1, -1).expand_as(pred))

    res = []
    for k in topk:
        correct_k = correct[:k].view(-1).float().sum(0)
        res.append(correct_k.mul_(100.0 / batch_size))
    return res


class SetCriterion(nn.Module):
    """This class computes the loss for Graphformer.
    The process happens in two steps:
        1) we compute hungarian assignment between ground truth boxes and the outputs of the model
        2) we supervise each pair of matched ground-truth / prediction (supervise class and box)
    """

    def __init__(self, config, matcher, net):
        """Create the criterion.
        Parameters:
            num_classes: number of object categories, omitting the special no-object category
            matcher: module able to compute a matching between targets and proposals
            weight_dict: dict containing as key the names of the losses and as values their relative weight.
            eos_coef: relative classification weight applied to the no-object category
            losses: list of all the losses to be applied. See get_loss for list of available losses.
        """
        super().__init__()
        self.matcher = matcher
        self.net = net
        self.rln_token = config.MODEL.DECODER.RLN_TOKEN
        self.obj_token = config.MODEL.DECODER.OBJ_TOKEN
        self.randomize_edge_directions = config.TRAIN.RANDOMIZE_EDGE_DIRECTIONS
        self.max_pos_edges = getattr(config.TRAIN, "MAX_POS_EDGES", None)
        self.neg_edge_ratio = getattr(config.TRAIN, "NEG_EDGE_RATIO", 4)
        self.min_neg_edges = getattr(config.TRAIN, "MIN_NEG_EDGES", 20)
        self.num_classes = config.MODEL.NUM_CLASSES
        self.num_edge_classes = getattr(config.MODEL, "NUM_EDGE_CLASSES", 3)
        self.losses = config.TRAIN.LOSSES
        self.weight_dict = {
            "boxes": config.TRAIN.W_BBOX,
            "class": config.TRAIN.W_CLASS,
            "cards": config.TRAIN.W_CARD,
            "nodes": config.TRAIN.W_NODE,
            "edges": config.TRAIN.W_EDGE,
        }

    def loss_class(self, outputs, indices):
        """Compute the losses related to the bounding boxes, the L1 regression loss and the GIoU loss
        targets dicts must contain the key "boxes" containing a tensor of dim [nb_target_boxes, 4]
        The target boxes are expected in format (center_x, center_y, w, h), normalized by the image size.
        """
        idx = self._get_src_permutation_idx(indices)
        targets = torch.zeros(outputs[..., 0].shape, dtype=torch.long).to(outputs.device)
        matched_target_labels = [labels[i] for labels, (_, i) in zip(self._target_node_classes, indices)]
        if matched_target_labels:
            targets[idx] = torch.cat(matched_target_labels, dim=0)
        class_weights = torch.ones(self.num_classes, device=outputs.device)
        class_weights[0] = 0.2
        loss = F.cross_entropy(
            outputs.permute(0, 2, 1),
            targets,
            weight=class_weights,
            reduction="mean",
        )

        return loss

    def loss_cardinality(self, outputs, indices):
        """Compute the cardinality error, ie the absolute error in the number of predicted non-empty boxes
        This is not really a loss, it is intended for logging purposes only. It doesn't propagate gradients
        """
        idx = self._get_src_permutation_idx(indices)
        targets = torch.zeros(outputs[..., 0].shape, dtype=torch.long).to(outputs.device)
        if len(indices):
            target_labels = torch.cat(
                [labels[i] for labels, (_, i) in zip(self._target_node_classes, indices)], dim=0
            )
            targets[idx] = target_labels

        tgt_lengths = torch.as_tensor([(t > 0).sum() for t in targets], device=outputs.device)
        # Count the number of predictions that are not background.
        card_pred = (outputs.argmax(-1) > 0).sum(1)

        loss = F.l1_loss(card_pred.float(), tgt_lengths.float(), reduction="sum") / (
            outputs.shape[0] * outputs.shape[1]
        )

        return loss

    def loss_nodes(self, outputs, targets, indices):
        """Classification loss (NLL)
        targets dicts must contain the key "labels" containing a tensor of dim [nb_target_boxes]
        """
        num_nodes = sum(len(t) for t in targets)

        idx = self._get_src_permutation_idx(indices)
        pred_nodes = outputs[idx]
        target_nodes = torch.cat([t[i] for t, (_, i) in zip(targets, indices)], dim=0)

        loss = F.l1_loss(
            pred_nodes, target_nodes, reduction="none"
        )  # TODO: check detr for loss function

        loss = loss.sum() / num_nodes

        return loss

    def loss_boxes(self, outputs, targets, indices):
        """Compute the losses related to the bounding boxes, the L1 regression loss and the GIoU loss
        targets dicts must contain the key "boxes" containing a tensor of dim [nb_target_boxes, 4]
        The target boxes are expected in format (center_x, center_y, w, h), normalized by the image size.
        """
        num_boxes = sum(len(t) for t in targets)
        idx = self._get_src_permutation_idx(indices)
        src_boxes = outputs[idx]

        target_boxes = torch.cat([t[i] for t, (_, i) in zip(targets, indices)], dim=0)

        loss = 1 - torch.diag(
            box_ops_2D.generalized_box_iou(
                box_ops_2D.box_cxcywh_to_xyxy(src_boxes),
                box_ops_2D.box_cxcywh_to_xyxy(target_boxes),
            )
        )
        loss = loss.sum() / num_boxes
        return loss

    def loss_edges(self, h, target_nodes, target_edges, target_edge_classes, indices):
        """Relation loss on all ground-truth edges plus randomly sampled negative pairs."""
        object_token = h[..., : self.obj_token, :]
        if self.rln_token > 0:
            relation_token = h[..., self.obj_token : self.rln_token + self.obj_token, :]

        edge_labels = []
        relation_feature = []
        for batch_id, (edges, edge_classes, (src_idx, tgt_idx)) in enumerate(
            zip(target_edges, target_edge_classes, indices)
        ):
            # object tokens in matcher order; position k corresponds to GT node tgt_idx[k]
            rearranged_object_token = object_token[batch_id, src_idx, :]
            matched_node_count = rearranged_object_token.shape[0]

            # remap GT node ids -> matched position, dropping edges touching unmatched nodes
            edges = edges.cpu()
            edge_classes = edge_classes.cpu()
            remap = torch.full((int(edges.max()) + 1 if edges.numel() else 1,), -1, dtype=torch.long)
            remap[tgt_idx] = torch.arange(matched_node_count)
            pos_edge = remap[edges]
            valid_pos_mask = (pos_edge >= 0).all(1)
            pos_edge = pos_edge[valid_pos_mask]
            pos_edge_classes = edge_classes[valid_pos_mask]

            full_adj = torch.ones((matched_node_count, matched_node_count)) - torch.eye(
                matched_node_count
            )
            full_adj[pos_edge[:, 0], pos_edge[:, 1]] = 0
            full_adj[pos_edge[:, 1], pos_edge[:, 0]] = 0
            neg_edges = torch.nonzero(torch.triu(full_adj))

            if self.max_pos_edges is not None and pos_edge.shape[0] > self.max_pos_edges:
                keep_pos = torch.randperm(pos_edge.shape[0])[: self.max_pos_edges]
                pos_edge = pos_edge[keep_pos]
                pos_edge_classes = pos_edge_classes[keep_pos]

            take_neg = min(
                neg_edges.shape[0],
                max(self.neg_edge_ratio * pos_edge.shape[0], self.min_neg_edges),
            )
            neg_edges = neg_edges[torch.randperm(neg_edges.shape[0])[:take_neg]]

            all_edges_ = torch.cat((pos_edge, neg_edges), 0)
            if self.randomize_edge_directions:
                flip = torch.rand(all_edges_.shape[0]) > 0.5
                all_edges_[flip] = all_edges_[flip][:, [1, 0]]
            all_edges_ = all_edges_.to(h.device)

            edge_labels.append(
                torch.cat(
                    (
                        pos_edge_classes.to(torch.long),
                        torch.zeros(take_neg, dtype=torch.long),
                    ),
                    0,
                )
            )

            feature_parts = [
                rearranged_object_token[all_edges_[:, 0], :],
                rearranged_object_token[all_edges_[:, 1], :],
            ]
            if self.rln_token > 0:
                feature_parts.append(
                    relation_token[batch_id, ...].repeat(all_edges_.shape[0], 1)
                )
            relation_feature.append(torch.cat(feature_parts, 1))

        relation_feature = torch.cat(relation_feature, 0)
        edge_labels = torch.cat(edge_labels, 0).to(h.device)
        relation_pred = self.net.relation_embed(relation_feature)
        return F.cross_entropy(relation_pred, edge_labels, reduction="mean")

    def _get_src_permutation_idx(self, indices):
        # permute predictions following indices
        batch_idx = torch.cat([torch.full_like(src, i) for i, (src, _) in enumerate(indices)])
        src_idx = torch.cat([src for (src, _) in indices])
        return batch_idx, src_idx

    def _get_tgt_permutation_idx(self, indices):
        # permute targets following indices
        batch_idx = torch.cat([torch.full_like(tgt, i) for i, (_, tgt) in enumerate(indices)])
        tgt_idx = torch.cat([tgt for (_, tgt) in indices])
        return batch_idx, tgt_idx

    def forward(self, h, out, target):
        """This performs the loss computation.
        Parameters:
             outputs: dict of tensors, see the output specification of the model for the format
             targets: list of dicts, such that len(targets) == batch_size.
                      The expected keys in each dict depends on the losses applied, see each loss' doc
        """

        # Retrieve the matching between the outputs of the last layer and the targets
        self._target_node_classes = target.get(
            "node_classes",
            [torch.ones((nodes.shape[0],), dtype=torch.long, device=nodes.device) for nodes in target["nodes"]],
        )
        target_edge_classes = target.get(
            "edge_classes",
            [torch.ones((edges.shape[0],), dtype=torch.long, device=edges.device) for edges in target["edges"]],
        )
        target_boxes = target.get(
            "boxes",
            [
                torch.cat(
                    [nodes, 0.2 * torch.ones_like(nodes)],
                    dim=-1,
                )
                for nodes in target["nodes"]
            ],
        )
        target["node_classes"] = self._target_node_classes
        target["edge_classes"] = target_edge_classes

        indices = self.matcher(out, target)

        losses = {}
        losses["class"] = self.loss_class(out["pred_logits"], indices)
        losses["nodes"] = self.loss_nodes(out["pred_nodes"][..., :2], target["nodes"], indices)
        losses["boxes"] = self.loss_boxes(out["pred_nodes"], target_boxes, indices)
        losses["edges"] = self.loss_edges(
            h,
            target["nodes"],
            target["edges"],
            target_edge_classes,
            indices,
        )
        losses["cards"] = self.loss_cardinality(out["pred_logits"], indices)

        losses["total"] = sum([losses[key] * self.weight_dict[key] for key in self.losses])

        return losses
