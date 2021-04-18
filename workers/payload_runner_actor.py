import typing as t

import ray
from ray.util.queue import Queue

from helpers import LoggerMixin


TDetection = t.TypeVar("TDetection", int, float, str)


@ray.remote
def run_payload_function(func: t.Callable) -> t.Any:
    """Payload function is to be pickable as it will be run as a Ray task
    executed in a different process
    """
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
        for cls_name, func in payload.items():  # type: ignore
            if not callable(func):
                self.logger.error(
                    f"Incorrect payload provided for the class "
                    f"{cls_name}. Must be callable!"
                )
                raise Exception("Payload not callable")
            self.logger.info(
                f"Class {cls_name} - registered payload "
                f"function {func.__name__}"
            )
        self._registered_classes = set(self._payload.keys())
        self.logger.info("PayloadRunner actor initialized")
        self.run()

    def run(self) -> None:
        # Start creating tasks for each detection calling one of the payload
        # functions
        while True:
            res = self._queue_in.get()
            if "KILL" in res:
                self.logger.info("PayloadRunner actor killed")
                self._queue_out.put("KILL")
                break
            image_name, image_ref, detections = res
            classes_detected = set([d[-1] for d in detections])
            classes_to_process = list(
                classes_detected.intersection(self._registered_classes)
            )
            output = []  # type: ignore
            if not len(classes_to_process):
                self.logger.warning(
                    f"No payload functions registered to process classes "
                    f"{' '.join([str(e) for e in classes_detected])}"
                )
                self._queue_out.put(
                    (image_name, image_ref, [(d, None) for d in detections])
                )
            else:
                # For each cls, for each object launch a job remembering which
                # object got which job
                jobs, futures = [], []
                i = 0
                for cls in classes_to_process:
                    func: t.Optional[t.Callable] = self._payload.get(cls)
                    objects_to_process = [
                        e for e in detections if e[-1] == cls
                    ]
                    for object_to_process in objects_to_process:
                        future = run_payload_function.remote(
                            lambda: func(
                                image_ref, object_to_process
                            )  # type: ignore
                        )
                        # Jobs index corresponds to future's index in the list
                        jobs.append((i, object_to_process))
                        futures.append(future)
                        i += 1
                        self.logger.debug(
                            f"For object {i}: {object_to_process} got a future"
                            f" {future}"
                        )
                assert len(futures) == len(jobs), "len(jobs) != len(futures)"
                payload_results = ray.get(futures)
                self.logger.info("All future results obtained")
                for payload_result, job in zip(payload_results, jobs):
                    output.append((job[-1], payload_result))
                self._queue_out.put((image_name, image_ref, output))
