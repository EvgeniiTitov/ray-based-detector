# type: ignore
import os
import typing as t

import cv2
import numpy as np

from detectors.yolov4.abstract_detector import Detector
from helpers import LoggerMixin


class DetectionModel(Detector, LoggerMixin):
    WEIGHTS = os.path.join(
        os.getcwd(), "detectors", "dependencies", "{}.weights"
    )
    CONFIG = os.path.join(os.getcwd(), "detectors", "dependencies", "{}.cfg")
    CLASSES = os.path.join(
        os.getcwd(), "detectors", "dependencies", "{}_classes.txt"
    )

    def __init__(
        self,
        model_name: str,
        conf: float = 0.15,
        nms: float = 0.3,
        image_size: int = 608,
    ) -> None:
        if not 0.0 <= conf <= 1.0 or not 0.0 <= nms <= 1.0:
            raise Exception(
                "Incorrect threshold(s) provided." "Expected: (0, 1)"
            )
        self._model_name = model_name
        self._model = self._init_model(model_name)
        self._nms = nms
        self._conf = conf
        self._image_size = image_size
        self._classes = self._read_model_classes()
        self.logger.info(f"{model_name} successfully initialized")

    def _init_model(self, model_name: str) -> cv2.dnn_DetectionModel:
        net = cv2.dnn.readNet(
            DetectionModel.CONFIG.format(model_name),
            DetectionModel.WEIGHTS.format(model_name),
        )
        net.setPreferableBackend(cv2.dnn.DNN_BACKEND_OPENCV)
        model = cv2.dnn_DetectionModel(net)
        model.setInputParams(size=(608, 608), scale=1 / 255, swapRB=True)
        return model

    def _read_model_classes(self) -> t.List[str]:
        with open(DetectionModel.CLASSES.format(self._model_name), "r") as f:
            classes = f.read().splitlines()
        return classes

    def predict(self, image: np.ndarray) -> t.Tuple[list, list, list]:
        return self._model.detect(image, self._conf, self._nms)
