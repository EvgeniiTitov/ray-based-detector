class SlackMixin:
    def __init__(self, webhook: str) -> None:
        self._webhook = webhook

    def send_msg(self, message: str) -> None:
        ...
