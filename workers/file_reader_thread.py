import os
import threading

import cv2
import ray
from ray.util.queue import Queue

from helpers import LoggerMixin


class FileReader(threading.Thread, LoggerMixin):
    def __init__(
        self, queue_in: Queue, queue_out: Queue, *args, **kwargs
    ) -> None:
        super().__init__(*args, **kwargs)
        self.queue_in = queue_in
        self.queue_out = queue_out
        self.logger.info("FileReader thread initialized")

    def run(self) -> None:
        while True:
            res = self.queue_in.get()
            if res == "KILL":
                self.logger.info("FileReader thread killed")
                self.queue_out.put("KILL")
                break

            filepath = res
            if not os.path.exists(filepath):
                self.logger.error(f"Failed to locate filepath: {filepath}")
                continue
            image = cv2.imread(filepath)
            if image is None:
                self.logger.exception(f"Failed to open image: {filepath}")

            try:
                image_ref = ray.put(image)
            except Exception as e:
                self.logger.exception(
                    f"Failed to move image {filepath} to "
                    f"object store. Error: {e}"
                )
                continue
            image_name = os.path.basename(filepath)
            self.queue_out.put((image_name, image_ref))