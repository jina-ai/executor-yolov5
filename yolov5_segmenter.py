__copyright__ = "Copyright (c) 2020-2021 Jina AI Limited. All rights reserved."
__license__ = "Apache-2.0"

import torch

from typing import Optional, Tuple, Dict, Iterable, List

from jina import Executor, DocumentArray, requests, Document
from jina_commons.batching import get_docs_batch_generator
from jina.logging.logger import JinaLogger

class YoloV5Segmenter(Executor):
    """
    Segment the image into bounding boxes and set labels

    :param model: the yolov5 model to use
    """

    def __init__(self,
                 model_name_or_path: str = 'yolov5s',
                 default_batch_size: int = 32,
                 default_traversal_paths: Tuple = ('r',),
                 device: str = 'cuda',
                 size: int = None,
                 augment: bool = False,
                 conf_thres: float = 0.25,
                 iou_thres: float = 0.45,
                 classes: Optional[List[int]] = None,
                 agnostic_nms: bool = False,
                 max_det: bool = 1000,
                 *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.model_name_or_path = model_name_or_path
        self.default_batch_size = default_batch_size
        self.default_traversal_paths = default_traversal_paths
        self.logger = JinaLogger(self.__class__.__name__)
        self.default_size = size
        self.default_augment = augment

        if device not in ['cpu', 'cuda']:
            self.logger.error('Torch device not supported. Must be cpu or cuda!')
            raise RuntimeError('Torch device not supported. Must be cpu or cuda!')
        if device == 'cuda' and not torch.cuda.is_available():
            self.logger.warning(
                'You tried to use GPU but torch did not detect your'
                'GPU correctly. Defaulting to CPU. Check your CUDA installation!'
            )
            device = 'cpu'
        self.device = torch.device(device)
        self.model = self._attempt_load(self.model_name_or_path)

        self.names = self.model.module.names if hasattr(self.model, 'module') else self.model.names

        self.conf_thres = conf_thres
        self.iou_thres = iou_thres
        self.classes = classes
        self.agnostic_nms = agnostic_nms
        self.max_det = max_det

    @requests
    def segment(self, docs: Optional[DocumentArray], parameters: Dict, **kwargs):
        """
        Segment all docs into bounding boxes and set labels
        :param docs: documents sent to the segmenter. The docs must have `blob`.
        :param parameters: dictionary to define the `traversal_paths` and the `batch_size`. For example,
        `parameters={'traversal_paths': ['r'], 'batch_size': 10}` will override the `self.default_traversal_paths` and
        `self.default_batch_size`.
        """

        if docs:
            document_batches_generator = get_docs_batch_generator(
                docs,
                traversal_path=parameters.get('traversal_paths', self.default_traversal_paths),
                batch_size=parameters.get('batch_size', self.default_batch_size),
                needs_attr='blob'
            )
            self._segment_docs(document_batches_generator, parameters=parameters)

    def _segment_docs(self, document_batches_generator: Iterable, parameters: Dict):
        with torch.no_grad():
            for document_batch in document_batches_generator:
                images = [d.blob for d in document_batch]
                predictions = self.model(
                    images,
                    # size=parameters.get("size", self.default_size),
                    # augment=parameters.get("augment", self.default_augment)
                ).pred

                for doc, prediction in zip(document_batch, predictions):
                    for det in prediction:
                        *xyxy, conf, cls = det
                        c = int(cls)
                        crop = doc.blob[int(xyxy[1]):int(xyxy[3]), int(xyxy[0]):int(xyxy[2]), :]
                        doc.chunks.append(Document(
                                blob=crop,
                                tags={"label": self.names[c], "conf": float(conf)}
                            ))

    def _attempt_load(self, model_name_or_path):
        return torch.hub.load("ultralytics/yolov5", model_name_or_path, 'yolov5s.pt', device=self.device)
