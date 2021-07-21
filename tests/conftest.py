import glob

import cv2
import pytest
from jina import DocumentArray, Document


@pytest.fixture(scope='package')
def build_da():
    def _build_da():
        return DocumentArray([
            Document(blob=cv2.imread(path), tags={'filename': path.split('/')[-1]})
            for path in glob.glob('tests/data/img/*.jpg')
        ])

    return _build_da
