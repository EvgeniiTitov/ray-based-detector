import json

import requests

from config import Config


class SlackMixin:
    def __init__(
        self,
        webhook_url: str = Config.SLACK_HOOK,
        project_name: str = "ray-based-detector",
    ) -> None:
        self._webhook = webhook_url
        self._project_name = project_name

    def slack_msg(self, msg: str) -> None:
        slack_data = {"text": f"{self._project_name} | {msg}"}
        response = requests.post(
            url=self._webhook,
            data=json.dumps(slack_data),
            headers={"Content-Type": "application/json"},
        )
        if response.status_code != 200:
            raise ValueError(
                f"Request to the slack server is unsuccessful."
                f"Error: {response.status_code}, "
                f"Response: {response.text}"
            )
