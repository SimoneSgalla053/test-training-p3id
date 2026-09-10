"""Regression tests use synthetic graphs and never download pretrained weights."""
import itertools
from types import SimpleNamespace

import numpy as np
import pytest
import torch
from ignite.engine import Engine

from inference import relation_infer
from metric_smd import MeanSMD


@pytest.mark.parametrize('relation_tokens', [0, 1])
@pytest.mark.parametrize('chunk_size', [1, 4, 4096])
def test_chunked_edges_match_individual_pair_scoring(relation_tokens, chunk_size):
    torch.manual_seed(4)
    n, dim = 6, 8
    head = torch.nn.Sequential(torch.nn.Linear(dim * (2 + relation_tokens), 12),
                               torch.nn.ReLU(), torch.nn.Linear(12, 3))
    model = SimpleNamespace(relation_embed=head)
    hidden = torch.randn(1, n + relation_tokens, dim)
    output = {'pred_logits': torch.tensor([[[0., 4., 1.]] * n]),
              'pred_nodes': torch.rand(1, n, 4)}
    expected_pairs, expected_scores, expected_classes = [], [], []
    for i, j in itertools.combinations(range(n), 2):
        relation = [hidden[0, n]] if relation_tokens else []
        forward = head(torch.cat([hidden[0, i], hidden[0, j], *relation]))
        reverse = head(torch.cat([hidden[0, j], hidden[0, i], *relation]))
        probs = ((forward + reverse) / 2).softmax(-1)
        score, label = probs[1:].max(0)
        if score >= 0.35:
            expected_pairs.append((i, j))
            expected_scores.append(score.item())
            expected_classes.append(label.item() + 1)
    result = relation_infer(hidden, output, model, n, relation_tokens, map_=True,
                            edge_threshold=0.35, edge_chunk_size=chunk_size)
    np.testing.assert_array_equal(result[1][0], np.asarray(expected_pairs).reshape(-1, 2))
    np.testing.assert_allclose(result[5][0], expected_scores, rtol=1e-6)
    np.testing.assert_array_equal(result[6][0], expected_classes)


def test_empty_graph_and_class_aware_nms_do_not_modify_boxes():
    hidden = torch.zeros(1, 4, 4)
    output = {'pred_logits': torch.tensor([[[0., 9., 0.], [0., 8., 0.], [0., 0., 9.]]]),
              'pred_nodes': torch.tensor([[[.5, .5, .1, .1]] * 3])}
    original = output['pred_nodes'].clone()
    model = SimpleNamespace(relation_embed=torch.nn.Linear(12, 3))
    result = relation_infer(hidden, output, model, 3, 1, nms=True, map_=True)
    assert len(result[0][0]) == 2  # Same-class duplicate removed; different class kept.
    torch.testing.assert_close(output['pred_nodes'], original)
    empty = relation_infer(hidden, output, model, 3, 1, node_threshold=1., map_=True)
    assert empty[1][0].shape == (0, 2)
    assert empty[1][0].dtype == np.int64
    assert empty[5][0].shape == (0,)
    with pytest.raises(ValueError):
        relation_infer(hidden, output, model, 3, 1, edge_chunk_size=0)


def test_smd_attaches_and_evaluates_an_empty_prediction():
    nodes = torch.tensor([[.1, .2], [.8, .2]])
    edges = torch.tensor([[0, 1]])
    batch = ([nodes], [edges], [torch.empty(0, 2)], [np.empty((0, 2), dtype=np.int64)])
    engine = Engine(lambda engine, data: data)
    MeanSMD().attach(engine, 'smd')
    state = engine.run([batch])
    assert state.metrics['smd'] == 100.


def test_offline_model_training_validation_checkpoint_and_resume(tmp_path, monkeypatch):
    from train import load_config
    from models import build_model
    from models.matcher import build_matcher
    from losses import SetCriterion
    from evaluator import build_evaluator
    from trainer import build_trainer
    from tensorboardX import SummaryWriter
    import torchvision.models

    torch.set_num_threads(2)
    config = load_config('configs/wsl_2D.yaml', verbose=False)
    config.MODEL.ENCODER.PRETRAINED = False
    config.MODEL.DECODER.OBJ_TOKEN = 8
    config.MODEL.DECODER.ENC_LAYERS = 1
    config.MODEL.DECODER.DEC_LAYERS = 1
    config.TRAIN.EPOCHS = 1
    config.TRAIN.SAVE_PATH = str(tmp_path)
    config.TRAIN.EARLY_STOPPING_PATIENCE = None
    config.TRAIN.LOG_INTERVAL = 1
    config.TRAIN.AMP = False
    config.INFERENCE.NODE_THRESHOLD = 1.
    builder = torchvision.models.resnet101

    def offline_builder(**kwargs):
        assert kwargs.get('weights') is None
        return builder(**kwargs)

    monkeypatch.setattr(torchvision.models, 'resnet101', offline_builder)
    net = build_model(config)
    nodes = torch.tensor([[.2, .3], [.7, .3]])
    edges = torch.tensor([[0, 1]])
    boxes = torch.cat([nodes, torch.full_like(nodes, .1)], dim=1)
    batch = [torch.rand(1, 1, 64, 64), [nodes], [edges], ['synthetic'],
             [torch.tensor([1, 3])], [torch.tensor([1])], [boxes]]
    loss = SetCriterion(config, build_matcher(config), net)
    optimizer = torch.optim.AdamW(net.parameters(), lr=1e-4)
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, 60)
    scaler = torch.amp.GradScaler('cuda', enabled=False)
    device = torch.device('cpu')
    writer = SummaryWriter(str(tmp_path / 'events'))
    evaluator = build_evaluator([batch], net, optimizer, scheduler, scaler, writer, config, device)
    assert evaluator.metric_cmp_fn(1., 2.)
    trainer = build_trainer([batch], net, loss, optimizer, scheduler, scaler, writer,
                            evaluator, config, device)
    before = net.relation_embed.layers[-1].weight.detach().clone()
    trainer.run()
    assert not torch.equal(before, net.relation_embed.layers[-1].weight)
    assert np.isfinite(trainer.state.output['loss']['total'].item())
    assert evaluator.state.metrics['val_smd'] == 100.
    paths = list(tmp_path.rglob('*epoch=1.pt'))
    assert paths
    checkpoint = torch.load(paths[0], map_location='cpu', weights_only=True)
    assert {'net', 'optimizer', 'scheduler', 'scaler'} <= checkpoint.keys()
    net.load_state_dict(checkpoint['net'])
    optimizer.load_state_dict(checkpoint['optimizer'])
    scheduler.load_state_dict(checkpoint['scheduler'])
    scaler.load_state_dict(checkpoint['scaler'])
    trainer.state.max_epochs = 2
    trainer.state.epoch = scheduler.last_epoch
    trainer.state.iteration = trainer.state.epoch_length * scheduler.last_epoch
    trainer.run()
    assert trainer.state.epoch == 2
    assert scheduler.last_epoch == 2
    writer.close()

    # Exercise the image-to-JSON path, including an explicit zero threshold.
    import json
    from PIL import Image
    import yaml
    from predict_image import main

    config_path = tmp_path / 'config.yaml'
    config_path.write_text(yaml.safe_dump(json.loads(json.dumps(config, default=vars))))
    image_path = tmp_path / 'patch.png'
    Image.new('L', (64, 64), color=255).save(image_path)
    # Keep the test small; the actual local configuration retains 512x512.
    values = yaml.safe_load(config_path.read_text())
    values['DATA']['IMG_SIZE'] = [64, 64]
    config_path.write_text(yaml.safe_dump(values))
    latest_checkpoint = next(tmp_path.rglob('*epoch=2.pt'))
    main(SimpleNamespace(image=image_path, checkpoint=latest_checkpoint, config=config_path,
                         output_dir=tmp_path / 'predictions', device='cpu',
                         node_threshold=0., edge_threshold=0., edge_chunk_size=3))
    graph = json.loads((tmp_path / 'predictions' / 'patch_graph.json').read_text())
    assert len(graph['nodes']) == len(graph['node_scores']) == len(graph['node_classes']) == 8
    assert len(graph['edges']) == len(graph['edge_scores']) == len(graph['edge_classes']) == 28
    assert graph['thresholds'] == {'node': 0., 'edge': 0.}
