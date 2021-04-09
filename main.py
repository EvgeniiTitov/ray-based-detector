import argparse
import logging
import os
import typing as t

import cv2
import ray
from ray.util.queue import Queue

from detectors import DetectionModel
from detectors import ObjectDetectorActor


logging.basicConfig(
    level=logging.INFO,
    format="%(lineno)d in %(filename)s at %(asctime)s: %(message)s",
)
logger = logging.getLogger(__name__)


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--source", type=str, default="source")
    parser.add_argument("--output", type=str, default="output")
    return parser.parse_args()


def validate_args(args: argparse.Namespace) -> None:
    if not os.path.exists(args.source):
        raise Exception("Failed to locate the source of data")
    if not os.path.exists(args.output):
        try:
            os.mkdir(args.output)
        except Exception as e:
            logger.exception(f"Faild to create the output folder. Error: {e}")
            raise e


def get_images_to_process(source: str = "source") -> t.Generator:
    for item in os.listdir(source):
        if any(item.endswith(e.lower()) for e in (".jpg", ".jpeg", ".png")):
            yield os.path.join(source, item)
        else:
            logger.info(f"Cannot process file: {item}. Unsupported extension")


def main() -> int:
    args = parse_args()
    validate_args(args)

    model = DetectionModel("objdet")

    queue_to_objdef = Queue(maxsize=5)
    queue_objdet_to_txtdet = Queue(maxsize=5)
    detector = ObjectDetectorActor.remote(  # type: ignore
        model, queue_to_objdef, queue_objdet_to_txtdet, logger
    )
    detector.run.remote()

    for image_path in get_images_to_process(args.source):
        image = cv2.imread(image_path)
        if image is None:
            logger.error(f"Failed to open image: {image_path}")
            continue
        image_ref = ray.put(image)
        queue_to_objdef.put((os.path.basename(image_path), image_ref))

    while not queue_objdet_to_txtdet.empty():
        image_name, image, results = queue_objdet_to_txtdet.get()
        logger.info(
            f"Got results for image: {image_name}. " f"Detections: {results}"
        )

    return 0


if __name__ == "__main__":
    ray.init()
    try:
        main()
    except Exception as e:
        print(f"Failed. Exception: {e}")
    ray.shutdown()
