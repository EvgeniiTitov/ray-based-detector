import os
import typing as t
from abc import ABC
from abc import abstractmethod

import cv2
import numpy as np

from .logger_mixin import LoggerMixin


class ResultProcessor(ABC):
    @abstractmethod
    def draw_boxes(self, image: np.ndarray, detections: t.List[list]) -> None:
        ...

    @abstractmethod
    def save_on_disk(self, image_name: str, image: np.ndarray) -> None:
        ...


class TheResultProcessor(ResultProcessor, LoggerMixin):
    def __init__(self, save_path: str) -> None:
        self._save_path = save_path
        if not os.path.exists(save_path):
            self._create_save_dir(save_path)
        self.logger.info("Result processor initialized")

    def _create_save_dir(self, folder: str) -> None:
        try:
            os.mkdir(folder)
        except Exception as e:
            self.logger.error(
                f"Failed to create the destination folder." f"Error: {e}"
            )
            raise e

    def draw_boxes(self, image: np.ndarray, detections: t.List[list]) -> None:
        for detection in detections:
            left, top, right, bot, conf, cls = detection
            cv2.rectangle(image, (left, top), (right, bot), (0, 255, 0), 2)
            cv2.putText(
                image,
                f"{cls}_{conf: .3f}",
                (left, top + 10),
                cv2.FONT_HERSHEY_SIMPLEX,
                2,
                (0, 0, 0),
                2,
            )

    def save_on_disk(self, image_name: str, image: np.ndarray) -> None:
        try:
            cv2.imwrite(os.path.join(self._save_path, image_name), image)
        except Exception as e:
            self.logger.error(
                f"Failed while saving image {image_name} "
                f"on disk.Error: {e}"
            )
            raise e
        self.logger.info(f"Saved image {image_name} to {self._save_path}")
