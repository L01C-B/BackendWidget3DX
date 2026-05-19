import os
import requests


DESSIA_SERVICE_URL = os.getenv(
    "DESSIA_SERVICE_URL",
    "http://127.0.0.1:8001"
)


def run_dessia_analysis(tool_name: str, arguments: dict) -> dict:
    response = requests.post(
        f"{DESSIA_SERVICE_URL}/analyze",
        json={
            "tool_name": tool_name,
            "arguments": arguments,

        },
        timeout=30,
    )

    response.raise_for_status()

    data = response.json()
    return data["result"]