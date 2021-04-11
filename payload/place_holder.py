import time
import typing as t

import numpy as np


def process_people_class(i: int, image: np.ndarray) -> t.Dict[int, t.Any]:
    image.fill(255)
    time.sleep(3)
    return {i: image.shape}


def process_car_class(i: int, image: np.ndarray) -> t.Dict[int, t.Any]:
    time.sleep(3)
    image.fill(0)
    return {i: image.shape}
