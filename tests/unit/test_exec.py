__copyright__ = "Copyright (c) 2020-2021 Jina AI Limited. All rights reserved."
__license__ = "Apache-2.0"

from operator import itemgetter
import pytest

from jina import Executor, Document, DocumentArray
import cv2

from yolov5_segmenter import YoloV5Segmenter


def test_load():
    segmenter = Executor.load_config('config.yml')
    assert segmenter.model_name_or_path == 'yolov5s'


@pytest.mark.parametrize(
    'model_path', [
        'tests/data/models/yolov5s.pt',
        'tests/data/models/yolov5m.pt',
        'yolov5s',
        'yolov5m',
        'https://github.com/ultralytics/yolov5/releases/download/v5.0/yolov5s.pt'
    ]
)
def test_model_name_or_path(build_da, model_path):
    da = build_da()
    segmenter = YoloV5Segmenter(model_name_or_path=model_path)
    segmenter.segment(da, parameters={})
    for doc in da:
        assert len(doc.chunks) > 0
        for chunk in doc.chunks:
            assert chunk.blob.ndim == 3
            assert chunk.tags.get("label")
            assert chunk.tags.get("conf")


@pytest.mark.parametrize(
    'model_path, expected_detections', [
        ('tests/data/models/yolov5s.pt', {'bus.jpg': 5, 'zidane.jpg': 3, 'man.jpg': 3}),
        ('tests/data/models/yolov5m.pt', {'bus.jpg': 6, 'zidane.jpg': 3, 'man.jpg': 3}),
    ]
)
def test_n_detections(build_da, model_path, expected_detections):
    da = build_da()
    segmenter = YoloV5Segmenter(model_name_or_path=model_path)
    segmenter.segment(da, parameters={})
    for doc in da:
        assert len(doc.chunks) == expected_detections[doc.tags["filename"]]


@pytest.mark.parametrize(
    'confidence_threshold, expected_detections', [
        (0.3, {'bus.jpg': 6, 'zidane.jpg': 3, 'man.jpg': 3}),
        (0.5, {'bus.jpg': 5, 'zidane.jpg': 3, 'man.jpg': 3}),
        (0.8, {'bus.jpg': 3, 'zidane.jpg': 2, 'man.jpg': 0}),
    ]
)
def test_confidence_threshold(build_da, confidence_threshold, expected_detections):
    da = build_da()
    segmenter = YoloV5Segmenter(model_name_or_path='tests/data/models/yolov5m.pt',
                                default_confidence_threshold=confidence_threshold)
    segmenter.segment(da, parameters={})
    for doc in da:
        assert len(doc.chunks) == expected_detections[doc.tags["filename"]]
        assert all(chunk.tags["conf"] >= confidence_threshold for chunk in doc.chunks)


def test_traversal_paths():
    da = DocumentArray([
        Document(
            id='root',
            blob=cv2.imread('tests/data/img/man.jpg'),
        ),
    ])

    segmenter = YoloV5Segmenter(model_name_or_path='tests/data/models/yolov5m.pt')
    segmenter.segment(da, parameters={})

    # detects 2 persons and 1 cell phone
    assert len(da[0].chunks) == 3
    assert all(
        label in ['person', 'cell phone']
        for label in
        map(itemgetter('label'), da[0].chunks.get_attributes('tags'))
    )

    segmenter.segment(da, parameters={'traversal_paths': ['c']})

    # the first detected person spans the whole image, so segmenting the chunk produces 3 detections
    assert len(da[0].chunks[0].chunks) == 3
    assert all(
        label in ['person', 'cell phone']
        for label in
        map(itemgetter('label'), da[0].chunks[0].chunks.get_attributes('tags'))
    )
