from abc import ABC, abstractmethod
import typing as t
import numpy as np


class Model(ABC):

    @abstractmethod
    def predict(self, batch: np.ndarray) -> t.List[t.List[t.Any]]:
        ...
