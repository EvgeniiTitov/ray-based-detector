import typing as t

from helpers import PayloadNotCallable
from payload import process_car_class
from payload import process_people_class


class Config:
    DETECTED_CLASSES = ["person"]
    DETECTOR_NMS = 0.5
    DETECTOR_CONF = 0.6
    BATCH_SIZE = 2
    TILES_PER_IMAGE = 4

    ALLOWED_EXTS = (".jpg", ".jpeg", ".png")

    SLACK_HOOK = (
        r"https://hooks.slack.com/services/TJQEQPNLC/B01GG8HMX5Y"
        r"/IrguHEnfxgjIyk7E9OmxREnA"
    )

    Q_READER_NET_RUNNER = 3
    Q_NET_RUNNER_PAYLOAD_RUNNER = 3
    Q_PAYLOAD_RUNNER_WRITER = 3

    RAY_CPUS = 6

    # Register payload functions for each detected class
    _PAYLOAD = {"person": process_people_class, "car": process_car_class}

    @staticmethod
    def get_payload() -> t.Dict[str, t.Callable]:
        if Config.validate_payload():
            return Config._PAYLOAD  # type: ignore
        else:
            raise PayloadNotCallable

    @staticmethod
    def validate_payload() -> bool:
        for cls, func in Config._PAYLOAD.items():  # type: ignore
            if not callable(func):
                return False
        return True
