import logging
import typing as t

import numpy as np
import ray
from abstract_model import Model
from ray.util.queue import Queue


Pred = t.Tuple[int, t.List[t.Any]]


@ray.remote
def process_image_tile(image: np.ndarray, quadrant: int, model: Model) -> Pred:
    if quadrant not in (1, 2, 3, 4):
        raise Exception("Incorrect quadrant value provided. Expected: 1-4")
    if not isinstance(model, Model) and not hasattr(model, "predict"):
        raise Exception("Incorrect model provided. Use AbsractModel interface")

    h, w, c = image.shape
    if quadrant == 1:
        tile: np.ndarray = image[0 : h // 2, w // 2 :, :]
    elif quadrant == 2:
        tile = image[0 : h // 2, 0 : w // 2, :]
    elif quadrant == 3:
        tile = image[h // 2 :, 0 : w // 2, :]
    else:
        tile = image[h // 2 :, w // 2 :, :]

    preds = model.predict(tile)
    return quadrant, preds


@ray.remote
class ObjectDetectorActor:
    def __init__(
        self,
        model: Model,
        queue_in: Queue,
        queue_out: Queue,
        logger: logging.Logger,
    ) -> None:
        self._model = model
        self._queue_in = queue_in
        self._queue_out = queue_out
        self._logger = logger
        self._logger.info("ObjectDetectorActor initialized")

    def run(self):
        while True:
            image_name, image = self._queue_in.get()
            self._logger.info(f"Got image {image_name} for object detection")
            futures = [
                process_image_tile.remote(image, i, self._model)
                for i in (1, 2, 3, 4)
            ]
            results = ray.get(futures)
            self._logger.info(f"Detections for image {image_name}: {results}")
            # TODO: Make sure you send the image_ref here
            self._queue_out.put((image_name, image, results))
