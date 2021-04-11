import math
import threading
import typing as t

import numpy as np
import ray
from ray.util.queue import Queue

from config import Config
from detectors.yolov4.abstract_detector import Model
from helpers import LoggerMixin


class NetRunnerThread(threading.Thread, LoggerMixin):
    def __init__(
        self, queue_in: Queue, queue_out: Queue, model: Model, *args, **kwargs
    ) -> None:
        super().__init__(*args, **kwargs)
        self._queue_in = queue_in
        self._queue_out = queue_out
        self._model = model
        self._n_tiles = Config.TILES_PER_IMAGE
        if self._n_tiles % 2:
            msg = "Number of files must be even number!"
            self.logger.exception(msg)
            raise Exception(msg)
        self._batch_size = Config.BATCH_SIZE
        self.logger.info("NetRunner thread initialized")

    def run(self) -> None:
        while True:
            res = self._queue_in.get()
            if "KILL" in res:
                self.logger.info("Net runner thread killed")
                self._queue_out.put("KILL")
                break
            image_name, image_ref = res
            try:
                image: np.ndarray = ray.get(image_ref)
            except Exception as e:
                self.logger.error(
                    f"Failed to extract image {image_name} from "
                    f"the object store. Error: {e}."
                )
                raise Exception
            # Cut image into tiles
            tile_pairs = self._split_image_into_tiles(image)
            # Split into batches
            if len(tile_pairs) < self._batch_size:
                batches = [tile_pairs]
            else:
                batches = [
                    tile_pairs[i : i + self._batch_size]
                    for i in range(0, len(tile_pairs), self._batch_size)
                ]
            # Run inference for each batch, rescale detections coordinates
            # relative to the image of the original size
            scaled_detections = []
            for batch in batches:
                relative_coords = [e[0] for e in batch]
                tiles = [e[1] for e in batch]
                detections = self._model.predict(tiles)
                for preds, tile, rel_coord in zip(
                    detections, tiles, relative_coords
                ):
                    for pred in preds:
                        left, top, right, bot, conf, _, cls = pred
                        abs_top, abs_bot, abs_left, abs_right = rel_coord
                        scaled_detections.append(
                            [
                                left + abs_left,
                                top + abs_top,
                                right + abs_left,
                                bot + abs_top,
                                conf,
                                cls,
                            ]
                        )
            self._queue_out.put((image_name, image_ref, scaled_detections))
            self.logger.info(
                f"NetRunner thread sent results for "
                f"image {image_name} to ResultProcessor. "
                f"Detections: {scaled_detections}"
            )

    def _split_image_into_tiles(
        self, image: np.ndarray
    ) -> t.Sequence[t.Tuple[t.Tuple[t.Any, ...], np.ndarray]]:
        """Imitates a grid on the image returning tiles (slices)
        of the image. Slices are views of the original image, not copies"""
        h, w, c = image.shape
        tile_height = (
            math.ceil(h / (self._n_tiles // 2 - 1))
            if self._n_tiles > 4
            else math.ceil(h / (self._n_tiles // 2))
        )
        tile_width = math.ceil(w / (self._n_tiles // 2))
        tiles = []  # type: ignore
        for i in range(0, h, tile_height):
            for j in range(0, w, tile_width):
                tiles.append(
                    (
                        (i, i + tile_height, j, j + tile_width),
                        image[i : i + tile_height, j : j + tile_width, :],
                    )
                )
        return tiles
