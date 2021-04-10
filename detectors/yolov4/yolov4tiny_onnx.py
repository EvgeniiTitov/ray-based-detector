# type: ignore
import os
from typing import List
from typing import Sequence

import cv2
import numpy as np
import onnxruntime as rt


class YOLOv4TinyONNX(AbstractModel):
    _path_to_dependencies = os.path.join(
        os.path.dirname(__file__), "model_instances"
    )
    _conf_thresh = 0.75
    _NMS_thresh = 0.3
    _file_names = "retail_small"

    def __init__(self, model_name: str, conf: float = None, nms: float = None):
        super().__init__(model_name)
        if conf:
            YOLOv4TinyONNX._conf_thresh = conf
        if nms:
            YOLOv4TinyONNX._NMS_thresh = nms
        _path_to_dependencies = os.path.join(
            self._path_to_dependencies, model_name
        )
        logger_ml.info(
            f"ONNX-v4 %s model's dependencies loaded from: '%s'",
            model_name,
            _path_to_dependencies,
        )
        try:
            p = os.path.join(_path_to_dependencies, f"{self._file_names}.onnx")
            self._sess = rt.InferenceSession(p)
            self._input_name = self._sess.get_inputs()[0].name
            self._input_shape = tuple(self._sess.get_inputs()[0].shape)
            self._output_name = self._sess.get_outputs()[0].name
        except Exception as e:
            logger_ml.debug("Failed to init ONNX-v4 session. Error: %s", e)
            raise e

        # Load classes
        self.classes = YOLOv4TinyONNX.load_class_names(
            os.path.join(_path_to_dependencies, f"{self._file_names}.txt")
        )
        logger_ml.debug(f"ONNX-v4 %s model initialized", model_name)

    def predict(self, image: np.ndarray) -> Sequence[Sequence]:
        """
        Receives an images, preprocesses it, runs through the net,
        postprocesses detections and returns boxes
        """
        original_image = image.copy()
        image = self._preprocess_image(image)
        assert image.shape == self._input_shape
        output = self._sess.run(None, {self._input_name: image})
        res = self._postprocess_results(output, original_image)
        return res

    def _preprocess_image(self, image: np.ndarray) -> np.ndarray:
        image_height = self._sess.get_inputs()[0].shape[2]
        image_width = self._sess.get_inputs()[0].shape[3]
        resized = cv2.resize(
            image, (image_width, image_height), interpolation=cv2.INTER_LINEAR
        )
        img = cv2.cvtColor(resized, cv2.COLOR_BGR2RGB)
        img = np.transpose(img, (2, 0, 1)).astype(np.float32)
        img = np.expand_dims(img, axis=0)  # batch 1
        img /= 255.0
        return img

    def _postprocess_results(
        self, output: Sequence, image: np.ndarray
    ) -> Sequence:
        """
        Performs result postprocessing by applying NMS and conf  thresholding
        """
        img_h, img_w = image.shape[:2]
        # [batch, num, 1, 4]
        box_array = output[0]
        # [batch, num, num_classes]
        confs = output[1]
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
            argwhere = max_conf[i] > self._conf_thresh
            l_box_array = box_array[i, argwhere, :]
            l_max_conf = max_conf[i, argwhere]
            l_max_id = max_id[i, argwhere]
            bboxes = []
            # nms for each class
            for j in range(num_classes):
                cls_argwhere = l_max_id == j
                ll_box_arr = l_box_array[cls_argwhere, :]
                ll_max_conf = l_max_conf[cls_argwhere]
                ll_max_id = l_max_id[cls_argwhere]
                keep = YOLOv4TinyONNX.nms_cpu(
                    ll_box_arr, ll_max_conf, self._NMS_thresh
                )
                if keep.size > 0:
                    ll_box_arr = ll_box_arr[keep, :]
                    ll_max_conf = ll_max_conf[keep]
                    ll_max_id = ll_max_id[keep]
                    for k in range(ll_box_arr.shape[0]):
                        bboxes.append(
                            [
                                int(
                                    (
                                        ll_box_arr[k, 0]
                                        if ll_box_arr[k, 0] > 0
                                        else 0
                                    )
                                    * img_w
                                ),
                                int(
                                    (
                                        ll_box_arr[k, 1]
                                        if ll_box_arr[k, 1] > 0
                                        else 0
                                    )
                                    * img_h
                                ),
                                int((ll_box_arr[k, 2]) * img_w),
                                int((ll_box_arr[k, 3]) * img_h),
                                ll_max_conf[k],
                                ll_max_conf[k],
                                self.classes[ll_max_id[k]],
                            ]
                        )
            bboxes_batch.append(bboxes)
        return bboxes_batch

    @staticmethod
    def nms_cpu(boxes, confs, nms_thresh=0.5, min_mode=False):
        x1 = boxes[:, 0]
        y1 = boxes[:, 1]
        x2 = boxes[:, 2]
        y2 = boxes[:, 3]
        areas = (x2 - x1) * (y2 - y1)
        order = confs.argsort()[::-1]
        keep = []
        while order.size > 0:
            idx_self = order[0]
            idx_other = order[1:]
            keep.append(idx_self)
            xx1 = np.maximum(x1[idx_self], x1[idx_other])
            yy1 = np.maximum(y1[idx_self], y1[idx_other])
            xx2 = np.minimum(x2[idx_self], x2[idx_other])
            yy2 = np.minimum(y2[idx_self], y2[idx_other])
            w = np.maximum(0.0, xx2 - xx1)
            h = np.maximum(0.0, yy2 - yy1)
            inter = w * h
            if min_mode:
                over = inter / np.minimum(areas[order[0]], areas[order[1:]])
            else:
                over = inter / (areas[order[0]] + areas[order[1:]] - inter)
            inds = np.where(over <= nms_thresh)[0]
            order = order[inds + 1]
        return np.array(keep)

    @staticmethod
    def load_class_names(namesfile: str) -> List[str]:
        class_names = []
        with open(namesfile, "r") as fp:
            lines = fp.readlines()
        for line in lines:
            line = line.rstrip()
            class_names.append(line)
        return class_names
