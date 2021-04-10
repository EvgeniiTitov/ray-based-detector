class Config:
    DETECTOR_NMS = 0.5
    DETECTOR_CONF = 0.6
    ALLOWED_EXTS = (".jpg", ".jpeg", ".png")
    BATCH_SIZE = 5
    TILES_PER_IMAGE = 6

    SLACK_HOOK = r"kek"

    Q_READER_NET_RUNNER = 3
    Q_RUNNER_WRITER = 3
