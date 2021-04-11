import os
import sys
from typing import List
from typing import Tuple

import cv2
import torch
import torch.nn.functional as F
import torchvision.transforms as transforms

from .abstract_detector import Detector
from .tool.darknet2pytorch import Darknet
from .tool.utils import *

sys.path.append("..")
from config import Config
from helpers import LoggerMixin


class YOLOv4(Detector, LoggerMixin):
    conf_thresh = Config.DETECTOR_CONF
    NMS_thresh = Config.DETECTOR_NMS

    def __init__(self, model_name: str, device: str = "gpu") -> None:
        folder = os.path.join(os.path.dirname(__file__))
        config_path = os.path.join(folder, "dependencies", model_name + ".cfg")
        weights_path = os.path.join(
            folder, "dependencies", model_name + ".weights"
        )
        classes_path = os.path.join(
            folder, "dependencies", model_name + ".txt"
        )
        self.logger.info(
            f"%s's dependencies loaded from: '%s'",
            model_name,
            os.path.join(os.path.dirname(__file__), "dependencies"),
        )
        if device != "cpu":
            self.device = torch.device(
                "cuda" if torch.cuda.is_available() else "cpu"
            )
        else:
            self.device = torch.device("cpu")

        # Initialize the model
        try:
            self.model: Darknet = Darknet(config_path)
        except Exception as e:
            self.logger.debug(
                f"Failed to initialize %s. Error: '%s'", model_name, e
            )
            raise e

        # Load model's weights
        try:
            # logger_ml.info("Loading weights version: '%s'", WEIGHTS_VER)
            self.model.load_weights(weights_path)
        except Exception as e:
            self.logger.debug(
                f"Failed to load %s's weights. Error: '%s'", model_name, e
            )
            raise e

        # Load classes and move model to device and prepare for inference
        self.classes = YOLOv4.read_classes_txt(classes_path)
        self.model.to(self.device).eval()
        self.logger.info(f"%s loaded to device: '%s'", model_name, self.device)

    def predict(self, images: List[np.ndarray]) -> List[list]:
        """
        Receives a batch of images, preprocesses them, runs through the net,
        postprocesses detections and returns predictions for each image in the
        batch
        """
        # Preprocess data: resize, normalization, batch etc
        images_ = self.preprocess_image_v1(images)
        # images_ = self.preprocess_image_v2(images)
        images_ = torch.autograd.Variable(images_)

        # Run data through the net
        with torch.no_grad():
            output = self.model(images_)

        # Postprocess data: NMS, thresholding
        boxes = self.postprocess_detections(output)
        boxes = self.rescale_boxes(boxes, images)

        return boxes

    def preprocess_image_v1(self, images: List[np.ndarray]) -> torch.Tensor:
        """
        Preprocesses image(s) in accordance with the preprocessing steps taken
        during model training and collects them in batch
        """
        # Preprocess images
        processed_images = list()
        for image in images:
            image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
            image = cv2.resize(image, (self.model.width, self.model.height))
            image = torch.from_numpy(image.transpose(2, 0, 1)).float()
            image = image.div(255).unsqueeze(0)
            processed_images.append(image)

        # Collect images in a batch
        try:
            batch_images = torch.cat(processed_images)
        except Exception as e:
            self.logger.debug(
                f"Failed to concat images into a batch tensor. Error: '%s'", e
            )
            raise e

        # Move batch to the same device on which the model's sitting
        batch_images = batch_images.to(device=self.device)

        return batch_images

    def preprocess_image_v2(self, images: List[np.ndarray]) -> torch.Tensor:
        """
        Another preprocessing approach that resizes image keeping its aspect
        ratio using padding
        """
        processed_images = list()
        for image in images:
            image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
            image = YOLOv4.preprocess_image(image, self.model.height)
            processed_images.append(image)

        # Collect images in a batch
        try:
            batch_images = torch.cat(processed_images)
        except Exception as e:
            self.logger.debug(
                f"Failed to concat images into a batch tensor. Error: '%s'", e
            )
            raise e

        # Move batch to the same device on which the model's sitting
        batch_images = batch_images.to(device=self.device)

        return batch_images

    def postprocess_detections(self, predictions: list) -> List[list]:
        # [batch, num, 1, 4]
        box_array = predictions[0]
        # [batch, num, num_classes]
        confs = predictions[1]
        if type(box_array).__name__ != "ndarray":
            box_array = box_array.cpu().detach().numpy()
            confs = confs.cpu().detach().numpy()

        num_classes = confs.shape[2]
        # [batch, num, 4]
        box_array = box_array[:, :, 0]
        # [batch, num, num_classes] --> [batch, num]
        max_conf = np.max(confs, axis=2)
        max_id = np.argmax(confs, axis=2)
        bboxes_batch = []
        for i in range(box_array.shape[0]):
            argwhere = max_conf[i] > self.conf_thresh
            l_box_array = box_array[i, argwhere, :]
            l_max_conf = max_conf[i, argwhere]
            l_max_id = max_id[i, argwhere]
            bboxes = []
            # nms for each class
            for j in range(num_classes):
                cls_argwhere = l_max_id == j
                ll_box_array = l_box_array[cls_argwhere, :]
                ll_max_conf = l_max_conf[cls_argwhere]
                ll_max_id = l_max_id[cls_argwhere]
                keep = nms_cpu(ll_box_array, ll_max_conf, self.NMS_thresh)
                if keep.size > 0:
                    ll_box_array = ll_box_array[keep, :]
                    ll_max_conf = ll_max_conf[keep]
                    ll_max_id = ll_max_id[keep]
                    for k in range(ll_box_array.shape[0]):
                        bboxes.append(
                            [
                                ll_box_array[k, 0],
                                ll_box_array[k, 1],
                                ll_box_array[k, 2],
                                ll_box_array[k, 3],
                                ll_max_conf[k],
                                ll_max_conf[k],
                                ll_max_id[k],
                            ]
                        )

            bboxes_batch.append(bboxes)

        return bboxes_batch

    def rescale_boxes(
        self, boxes_for_batch: list, original_images: List[np.ndarray]
    ) -> List[list]:
        """
        Converts and rescales boxes from standard net output format to:
        left, top, right, bot, obj, conf, index
        """
        if len(boxes_for_batch) != len(original_images):
            raise Exception("Nb of images != nb of detections made by the net")
        boxes_batch_rescaled = []
        for boxes, image in zip(boxes_for_batch, original_images):
            boxes_rescaled = list()
            orig_h, orig_w = image.shape[:2]

            # #The amount of padding that was added
            # pad_x = max(orig_h - orig_w, 0) * (cur_dim / max(orig_h, orig_w))
            # pad_y = max(orig_w - orig_h, 0) * (cur_dim / max(orig_h, orig_w))
            # # Image height and width after padding is removed
            # unpad_h = cur_dim - pad_y
            # unpad_w = cur_dim - pad_x

            for box in boxes:
                new_left = int(box[0] * orig_w)
                new_left = 2 if new_left <= 0 else new_left
                new_top = int(box[1] * orig_h)
                new_top = 2 if new_top <= 0 else new_top
                new_right = int(box[2] * orig_w)
                new_right = orig_w - 2 if new_right >= orig_w else new_right
                new_bot = int(box[3] * orig_h)
                new_bot = orig_h - 2 if new_bot >= orig_h else new_bot
                if new_left > new_right or new_top > new_bot:
                    self.logger.debug(
                        "Wrong BB coordinates. Expected: left < right, "
                        "actual: %d < %d;   top < bot, actual: %d < %d",
                        new_left,
                        new_right,
                        new_top,
                        new_bot,
                    )
                # new_left = int((box[0] - pad_x // 2) * (orig_w / unpad_w))
                # new_left = 2 if new_left == 0 else new_left
                # new_top = int((box[1] - pad_y // 2) * (orig_h / unpad_h))
                # new_top = 2 if new_top == 0 else new_top
                # new_right = int((box[2] - pad_x // 2) * (orig_w / unpad_w))
                # new_bot = int((box[3] - pad_y // 2) * (orig_h / unpad_h))
                obj_score = round(box[4], 4)
                conf = round(box[5], 4)
                i = self.classes[box[6]]
                boxes_rescaled.append(
                    [new_left, new_top, new_right, new_bot, obj_score, conf, i]
                )
            boxes_batch_rescaled.append(boxes_rescaled)

        return boxes_batch_rescaled

    @staticmethod
    def pad_to_square(
        img: torch.Tensor, pad_value: int
    ) -> Tuple[torch.Tensor, tuple]:
        """
        Pads image with pad_value colour so that it becomes a square h = w
        """
        c, h, w = img.shape
        dim_diff = np.abs(h - w)
        # (upper / left) padding and (lower / right) padding
        pad1, pad2 = dim_diff // 2, dim_diff - dim_diff // 2
        # Determine padding
        pad = (0, 0, pad1, pad2) if h <= w else (pad1, pad2, 0, 0)
        # Add padding
        img = F.pad(img, pad, "constant", value=pad_value)

        return img, pad

    @staticmethod
    def resize(image: torch.Tensor, size: int) -> torch.Tensor:
        image = F.interpolate(image.unsqueeze(0), size=size, mode="nearest")
        return image

    @staticmethod
    def preprocess_image(
        image: np.ndarray, image_size: int = 608
    ) -> torch.Tensor:
        image = transforms.ToTensor()(image)
        # Padd the image so that it is a square
        image, _ = YOLOv4.pad_to_square(image, 0)
        # Resize the image to the expected net's resolution
        image = YOLOv4.resize(image, image_size)

        return image

    @staticmethod
    def read_classes_txt(filepath: str) -> List[str]:
        with open(filepath, "r") as f:
            classes = f.read().splitlines()
        return classes
