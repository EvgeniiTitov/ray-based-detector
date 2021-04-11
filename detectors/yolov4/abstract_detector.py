import typing as t
from abc import ABC
from abc import abstractmethod

import numpy as np

TNum = t.TypeVar("TNum", int, float)

Detections = t.List[t.List[t.Sequence[TNum]]]


class Detector(ABC):
    @abstractmethod
    def predict(self, batch: t.List[np.ndarray]) -> Detections:
        ...
