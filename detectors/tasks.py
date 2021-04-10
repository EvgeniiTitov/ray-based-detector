import typing as t

import numpy as np
import ray
from detectors.yolov4.abstract_model import Model


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
