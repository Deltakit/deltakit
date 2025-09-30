# (c) Copyright Riverlane 2020-2025.
from dataclasses import dataclass

@dataclass
class FakeResponse:
    """Fake response to support network-related tests"""

    status_code: int = 200
    status: str = "SUBMITTED"
    text: str = "BODY text"

    @property
    def ok(self):
        return self.status_code < 400

    def json(self):
        return {
            "status": self.status,
            "type": "simulate",
            "request_id": "some_id"
        }
