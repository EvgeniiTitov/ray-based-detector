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
    Q_RUNNER_WRITER = 3

    RAY_CPUS = 6
