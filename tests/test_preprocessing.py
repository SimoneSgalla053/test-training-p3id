"""Exercise real worker IPC and recovery after interrupted cache construction."""
import pickle
from xml.etree import ElementTree

import numpy as np
import pytest
import torch
from PIL import Image

import dataset_road_network as dataset


@pytest.fixture
def samples(tmp_path):
    graph = '''<?xml version="1.0"?>
    <graphml xmlns="http://graphml.graphdrawing.org/xmlns">
      <key id="label" for="node" attr.name="label" attr.type="string"/>
      <key id="xmin" for="node" attr.name="xmin" attr.type="double"/>
      <key id="xmax" for="node" attr.name="xmax" attr.type="double"/>
      <key id="ymin" for="node" attr.name="ymin" attr.type="double"/>
      <key id="ymax" for="node" attr.name="ymax" attr.type="double"/>
      <key id="edge_label" for="edge" attr.name="edge_label" attr.type="string"/>
      <graph edgedefault="undirected">
        <node id="a"><data key="label">valve</data><data key="xmin">2</data>
          <data key="xmax">6</data><data key="ymin">4</data><data key="ymax">8</data></node>
        <node id="b"><data key="label">connector</data><data key="xmin">20</data>
          <data key="xmax">24</data><data key="ymin">4</data><data key="ymax">8</data></node>
        <node id="c"><data key="label">pump</data><data key="xmin">30</data>
          <data key="xmax">34</data><data key="ymin">14</data><data key="ymax">18</data></node>
        <edge source="a" target="b"><data key="edge_label">solid</data></edge>
        <edge source="b" target="c"><data key="edge_label">non_solid</data></edge>
      </graph>
    </graphml>'''
    result = []
    for index, value in enumerate([30, 120, 240]):
        image = tmp_path / f'{index}.png'
        annotation = tmp_path / f'{index}.graphml'
        Image.new('L', (48, 24), value).save(image)
        annotation.write_text(graph)
        result.append((image, annotation, str(index)))
    return result


@pytest.mark.parametrize('workers', [1, 2])
def test_preprocessing_returns_private_tensors_and_reuses_cache(samples, tmp_path, monkeypatch, workers):
    # Cross multiple Pool chunks and check unordered results land at the right index.
    repeated = [samples[index % 3] for index in range(137)]
    prefix = tmp_path / 'cache' / 'samples'
    if workers == 1:
        monkeypatch.setattr(dataset, 'Pool', lambda *a, **kw: pytest.fail('Serial mode opened a Pool'))
    images, graphs = dataset.preprocess_pid_samples(repeated, (12, 8), prefix, num_workers=workers)
    assert images.shape == (137, 8, 12)
    expected = dataset.load_pid_graph(samples[0][1], width=48, height=24)
    for index, graph in enumerate(graphs):
        assert np.all(images[index] == [30, 120, 240][index % 3])
        for tensor, target in zip(graph, expected):
            assert not tensor.is_shared(), 'Parent retained a shared-memory Torch storage'
            torch.testing.assert_close(tensor, target, rtol=0, atol=0)
    assert dataset._WORKER_IMAGES is None

    def should_not_rebuild(*args, **kwargs):
        pytest.fail('Valid cache was rebuilt')
    monkeypatch.setattr(dataset, '_preprocess_pid_sample', should_not_rebuild)
    cached_images, cached_graphs = dataset.preprocess_pid_samples(repeated, (12, 8), prefix, num_workers=workers)
    np.testing.assert_array_equal(images, cached_images)
    for graph, cached_graph in zip(graphs, cached_graphs):
        for tensor, cached_tensor in zip(graph, cached_graph):
            assert not cached_tensor.is_shared()
            torch.testing.assert_close(tensor, cached_tensor, rtol=0, atol=0)


@pytest.mark.parametrize('workers', [1, 2])
def test_incomplete_cache_and_worker_failure_can_be_retried(samples, tmp_path, workers):
    prefix = tmp_path / 'retry'
    images_path = tmp_path / 'retry_images.npy'
    graphs_path = tmp_path / 'retry_graphs.pickle'
    np.save(images_path, np.zeros((3, 8, 12), dtype=np.uint8))
    graphs_path.write_bytes(b'')  # Interrupted old pickle write.
    images, graphs = dataset.preprocess_pid_samples(samples, (12, 8), prefix, num_workers=workers)
    assert int(images[0, 0, 0]) == 30
    assert len(pickle.loads(graphs_path.read_bytes())) == 3
    del images, graphs

    broken = tmp_path / 'broken.graphml'
    broken.write_text('not XML')
    expanded = samples + [(samples[0][0], broken, 'broken')]
    with pytest.raises(ElementTree.ParseError):
        dataset.preprocess_pid_samples(expanded, (12, 8), prefix, num_workers=workers)
    assert not graphs_path.exists(), 'Stale completion marker survived failed rebuild'
    assert dataset._WORKER_IMAGES is None
    fixed = samples + [samples[0]]
    images, graphs = dataset.preprocess_pid_samples(fixed, (12, 8), prefix, num_workers=workers)
    assert images.shape[0] == len(graphs) == 4
    assert not list(tmp_path.glob('*.tmp'))


def test_failed_pickle_write_does_not_publish_completion_marker(samples, tmp_path, monkeypatch):
    prefix = tmp_path / 'pickle-failure'
    def interrupted_write(*args, **kwargs):
        raise OSError('simulated interrupted write')
    with monkeypatch.context() as m:
        m.setattr(dataset.pickle, 'dump', interrupted_write)
        with pytest.raises(OSError, match='simulated interrupted write'):
            dataset.preprocess_pid_samples(samples, (12, 8), prefix, num_workers=1)
    assert not tmp_path.joinpath('pickle-failure_graphs.pickle').exists()
    assert not list(tmp_path.glob('*.tmp'))
    _, graphs = dataset.preprocess_pid_samples(samples, (12, 8), prefix, num_workers=1)
    assert len(graphs) == 3
