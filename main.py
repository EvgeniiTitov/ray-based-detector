import argparse
import os
import typing as t

import ray
from ray.util.queue import Queue

from config import Config
from detectors import YOLOv4
from helpers import LoggerMixin
from helpers import SlackMixin
from helpers import TheResultProcessor
from helpers import timer
from workers import FileReaderThread
from workers import NetRunnerThread
from workers import ResultWriterThread


class Detector(LoggerMixin, SlackMixin):
    def __init__(
        self,
        source: str,
        dest: str,
        batch_size: int,
        tiles: int,
        webhook: str,
        gpu: bool,
    ) -> None:
        SlackMixin.__init__(self, webhook)
        self._source = source
        self._dest = dest
        self._batch_size = batch_size
        self._n_tiles = tiles
        self._threads = []

        self._q_to_file_reader = Queue()
        self._q_freader_to_detector = Queue(maxsize=Config.Q_READER_NET_RUNNER)
        self._q_detector_fwriter = Queue(maxsize=Config.Q_RUNNER_WRITER)
        self.logger.info("Queues initialized")

        self._file_reader_thread = FileReaderThread(
            queue_in=self._q_to_file_reader,
            queue_out=self._q_freader_to_detector,
        )
        self._model = YOLOv4("yolov4", device="gpu" if gpu else "cpu")
        self._net_runner_thread = NetRunnerThread(
            queue_in=self._q_freader_to_detector,
            queue_out=self._q_detector_fwriter,
            model=self._model,
        )
        self._result_processor = TheResultProcessor(dest)
        self._result_processor_thread = ResultWriterThread(
            result_writer=self._result_processor,
            queue_in=self._q_detector_fwriter,
        )
        self._threads.append(self._file_reader_thread)
        self._threads.append(self._net_runner_thread)  # type: ignore
        self._threads.append(self._result_processor_thread)  # type: ignore
        self._start()
        self.logger.info("Detector started")

    def process_images(self):
        for image_path in self._get_images_to_process():
            self._q_to_file_reader.put(image_path)
            self.logger.info(
                f"Image {os.path.basename(image_path)} " f"sent to file reader"
            )

    def _get_images_to_process(self) -> t.Generator:
        for item in os.listdir(self._source):
            if any(item.endswith(ext.lower()) for ext in Config.ALLOWED_EXTS):
                yield os.path.join(self._source, item)
            else:
                self.logger.warning(
                    f"Cannot process file: {item}. Unsupported extension"
                )

    def _start(self) -> None:
        for thread in self._threads:
            thread.start()

    def stop(self) -> None:
        self._q_to_file_reader.put("KILL")
        for thread in self._threads:
            thread.join()
        self.logger.info("Detected stopped")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--source", type=str, default="source")
    parser.add_argument("--output", type=str, default="output")
    parser.add_argument("--batch_size", type=int, default=Config.BATCH_SIZE)
    parser.add_argument("--n_tiles", type=int, default=Config.TILES_PER_IMAGE)
    parser.add_argument("--webhook", type=str, default=Config.SLACK_HOOK)
    parser.add_argument("--gpu", action="store_true")
    return parser.parse_args()


def validate_args(args: argparse.Namespace) -> None:
    if not os.path.exists(args.source):
        raise Exception("Failed to locate the source of data")
    if not os.path.exists(args.output):
        try:
            os.mkdir(args.output)
        except Exception as e:
            print(f"Failed to create the output folder. Error: {e}")
            raise e


@timer
def main() -> int:
    args = parse_args()
    validate_args(args)
    d = Detector(
        source=args.source,
        dest=args.output,
        batch_size=args.batch_size,
        tiles=args.n_tiles,
        webhook=args.webhook,
        gpu=args.gpu,
    )
    d.process_images()
    d.stop()
    return 0


if __name__ == "__main__":
    ray.init(num_cpus=Config.RAY_CPUS)
    try:
        main()
    except Exception as e:
        print(f"Failed. Exception: {e}")
    ray.shutdown()
