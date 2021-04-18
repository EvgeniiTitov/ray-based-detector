import threading

import ray
from ray.util.queue import Queue

from helpers import FailedObjectStorePull
from helpers import LoggerMixin
from helpers import ResultProcessor
from helpers import SlackMixin


class ResultWriterThread(threading.Thread, LoggerMixin, SlackMixin):
    def __init__(
        self, result_writer: ResultProcessor, queue_in: Queue, *args, **kwargs
    ) -> None:
        super().__init__(*args, **kwargs)
        SlackMixin.__init__(self)
        self._result_processor = result_writer
        self._queue_in = queue_in
        self.logger.info("ResultWriter thread initialized")

    def run(self) -> None:
        while True:
            res = self._queue_in.get()
            if "KILL" in res:
                self.logger.info("ResultWriter thread killed")
                break
            image_name, image_ref, detections = res
            try:
                image = ray.get(image_ref)
            except FailedObjectStorePull as e:
                self.logger.error(
                    f"Failed to extract image {image_name} from "
                    f"the object store. Error: {e}"
                )
                raise Exception
            detections = [e[0] for e in detections]
            payload_results = [e[1] for e in detections]  # noqa
            self._result_processor.draw_boxes(image, detections)
            self._result_processor.save_on_disk(image_name, image)

            # TODO: Find out if I can manually pop image off the object store
            # TODO: Payload results could be saved to json, xml or whatever
            self._log_msg(f"Image {image_name} processed")

    def _log_msg(self, msg: str) -> None:
        self.logger.info(msg)
        self.slack_msg(msg)
