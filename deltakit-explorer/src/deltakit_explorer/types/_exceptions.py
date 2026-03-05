# (c) Copyright Riverlane 2020-2025.
"""Exception classes for the project"""

from __future__ import annotations


class ServerException(Exception):  # pragma: nocover
    """Exception, which happened on a server side."""

    message: str
    """Error message from server"""

    def __init__(self, message: str):
        self.message = message
