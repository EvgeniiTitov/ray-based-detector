import typing as t
from abc import ABC
from abc import abstractmethod

import numpy as np


class Model(ABC):
    @abstractmethod
    def predict(self, batch: np.ndarray) -> t.List[t.List[t.Any]]:
        ...
