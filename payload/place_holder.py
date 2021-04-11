import time
import typing as t

import numpy as np


"""
Here goes any computationally intensive image processing tasks - say
determining inclination angle of a detected concrete pole, which involves
quite a bit of loops and operations.

Tasks need to be pickable as they will be executed in separate processes
as Ray tasks

Images stored in the object store are immutable, said that any modifications
will need to happen on copies.
"""

TDetection = t.TypeVar("TDetection", int, float, str)


def process_people_class(
    i: int, image: np.ndarray, person_detection: t.List[TDetection]
) -> t.Dict[int, t.Any]:
    # TODO: Slice image
    print(person_detection)
    image.fill(255)
    time.sleep(3)
    return {i: image.shape}


def process_car_class(
    i: int, image: np.ndarray, car_detection: t.List[TDetection]
) -> t.Dict[int, t.Any]:
    print(car_detection)
    time.sleep(3)
    image.fill(0)
    return {i: image.shape}
