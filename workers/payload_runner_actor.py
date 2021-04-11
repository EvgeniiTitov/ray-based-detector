import typing as t

import ray
from ray.util.queue import Queue

from helpers import LoggerMixin


@ray.remote
def run_payload_function(func: t.Callable):
    return func()


@ray.remote
class PayloadRunnerActor(LoggerMixin):
    def __init__(
        self,
        queue_in: Queue,
        queue_out: Queue,
        payload: t.Dict[str, t.Callable],
    ) -> None:
        self._queue_in = queue_in
        self._queue_out = queue_out
        self._payload = payload
        self.logger.info("PayloadRunner actor initialized")

    def run(self) -> None:
        # Get image ref and detections
        # Start creating tasks for each detection calling one of the payload
        # functions

        # Each payload function to process an image section will be called
        # as a separate process
        while True:
            res = self._queue_in.get()
            if "KILL" in res:
                self.logger.info("PayloadRunner actor killed")
                self._queue_out.put("KILL")
                break
            image_name, image_ref, detections = res
            try:
                image = ray.get(image_ref)
            except Exception as e:
                self.logger.error(
                    f"Failed to extract image {image_name} from the object"
                    f"store. Error: {e}"
                )
                raise Exception
            classes_to_process = set(self._payload.keys())
            classes_detected = set([d[-1] for d in detections])
            intersection = list(
                classes_detected.intersection(classes_to_process)
            )
            if not len(intersection):
                self.logger.warning(
                    "No payload functions to run. There's no ∩ between"
                    "classes_to_process and classes_detected"
                )
            else:
                pass
                # Depending on an object class, spawn an appropriate function
                # for it using run_payload_function task
            print(image.shape)
            # Put results into the queue to results processor
