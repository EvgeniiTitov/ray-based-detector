import logging
import os
import typing as t

import cv2
import numpy as np

from detectors.abstract_model import Model


class DetectionModel(Model):
    WEIGHTS = os.path.join(
        os.getcwd(), "detectors", "dependencies", "{}.weights"
    )
    CONFIG = os.path.join(os.getcwd(), "detectors", "dependencies", "{}.cfg")
    CLASSES = os.path.join(
        os.getcwd(), "detectors", "dependencies", "{}_classes"
    )

    def __init__(
        self,
        model_name: str,
        conf: float = 0.5,
        nms: float = 0.4,
        image_size: int = 416,
    ) -> None:
        if not 0.0 <= conf <= 1.0 or not 0.0 <= nms <= 1.0:
            raise Exception(
                "Incorrect threshold(s) provided." "Expected: (0, 1)"
            )
        self._model_name = model_name
        self._model = DetectionModel._init_model(model_name)
        self._nms = nms
        self._conf = conf
        self._image_size = image_size
        self._layers = self._get_output_layers()
        self._classes = self._read_model_classes()
        logging.info(f"Model {model_name} initialized")

    @staticmethod
    def _init_model(model_name: str) -> cv2.dnn_DetectionModel:
        net = cv2.dnn.readNetFromDarknet(
            DetectionModel.CONFIG.format(model_name),
            DetectionModel.WEIGHTS.format(model_name),
        )
        net.setPreferableBackend(cv2.dnn.DNN_BACKEND_OPENCV)
        return net

    def _read_model_classes(self) -> t.List[str]:
        with open(DetectionModel.CLASSES.format(self._model_name), "r") as f:
            classes = f.read().splitlines()
        return classes

    def _get_output_layers(self) -> t.List[str]:
        layers = self._model.getLayerNames()
        return [
            layers[i[0] - 1] for i in self._model.getUnconnectedOutLayers()
        ]

    def predict(self, image: np.ndarray):
        blob = cv2.dnn.blobFromImage(
            image,
            1 / 255.0,
            (self._image_size, self._image_size),
            swapRB=True,
            crop=False,
        )
        self._model.setInput(blob)
        outputs = self._model.forward(self._layers)
        return self._postprocess_output(image, outputs)

    def _postprocess_output(
        self, image: np.ndarray, outputs: t.List[list]
    ) -> t.List[list]:
        outputs = np.vstack(outputs)
        h, w = image.shape[:2]
        boxes, confidences, class_ids = [], [], []
        for output in outputs:
            scores = output[5:]
            class_id = np.argmax(scores)
            confidence = scores[class_id]
            if confidence > self._conf:
                x, y, w, h = output[:4] * np.array([w, h, w, h])
                p0 = int(x - w // 2), int(y - h // 2)
                # p1 = int(x + w // 2), int(y + h // 2)
                boxes.append([*p0, int(w), int(h)])
                confidences.append(float(confidence))
                class_ids.append(class_id)
        indices = cv2.dnn.NMSBoxes(boxes, confidences, self._conf, self._nms)
        detections = []
        if len(indices):
            for i in indices.flatten():
                left = boxes[i][0] if boxes[i][0] > 0 else 2
                top = boxes[i][1] if boxes[i][1] > 0 else 2
                right, bot = (boxes[i][2] + left, boxes[i][3] + top)
                class_ = self._classes[class_ids[i]]
                confidence = confidences[i]
                detections.append([left, top, right, bot, confidence, class_])
        return detections


if __name__ == "__main__":
    import cv2
    import os

    model = DetectionModel("objdet")
    print("Model initialized")

    for item in os.listdir("source"):
        image = cv2.imread(os.path.join("source", item))
        if image is None:
            print("Failed to read image:", image)
        detections = model.predict(image)
        if len(detections):
            for detection in detections:
                left, top, right, bot, conf, cls = detection
                cv2.rectangle(
                    image,
                    (left, top),
                    (right, bot),
                    (0, 255, 0),
                    3,
                    cv2.FONT_HERSHEY_PLAIN,
                )
                cv2.putText(
                    image,
                    f"{cls}_{conf: .4f}",
                    (left, top + 15),
                    cv2.FONT_HERSHEY_PLAIN,
                    2,
                    (0, 0, 0),
                    2,
                )
        cv2.imshow("", image)
        cv2.waitKey(0)
