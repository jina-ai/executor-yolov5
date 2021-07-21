__copyright__ = "Copyright (c) 2020-2021 Jina AI Limited. All rights reserved."
__license__ = "Apache-2.0"

from jina import Flow, Document, DocumentArray


def test_flow_from_yml(build_da):
    da = build_da()
    with Flow.load_config('tests/integration/flow.yml') as f:
        resp = f.post(on='test', inputs=da, return_results=True)

    assert resp is not None


def test_chunks_exists(build_da):
    da = build_da()
    with Flow.load_config('tests/integration/flow.yml') as f:
        responses = f.post(on='segment', inputs=da, return_results=True)

    for doc in responses[0].docs:
        assert len(doc.chunks) > 0
        for chunk in doc.chunks:
            assert chunk.blob.ndim == 3
            assert chunk.tags.get("label")
            assert chunk.tags.get("conf")
