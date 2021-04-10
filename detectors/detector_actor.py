from ray.util.queue import Queue
import ray

from detectors.yolov4.yolov4_cv2based import DetectionModel
from helpers import LoggerMixin


@ray.remote
class ObjectDetectorActor(LoggerMixin):
    # TODO: Consider creating 9 actors and sending then an image ref and a
    #       quadrant to process
    def __init__(
        self,
        model_name: str,
        queue_in: Queue,
        queue_out: Queue,
    ) -> None:
        self._model = ObjectDetectorActor._init_model(model_name)
        self._queue_in = queue_in
        self._queue_out = queue_out
        self._logger.info("ObjectDetectorActor initialized")

    @staticmethod
    def _init_model(model_name: str) -> DetectionModel:
        return DetectionModel(model_name)

    def process_quadrant(self, image, quadrant):
        # Slice image
        # Inference
        # Get new coordinates
        # Run tesseract
        pass