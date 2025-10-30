# (c) Copyright Riverlane 2020-2025.
"""This module contains an implementation of a data string wrapper,
which allows to send arbitrary data inside a request, without creating files.
"""
from __future__ import annotations

from pathlib import Path


class DataString:
    """Data string is a normal string which encodes arbitrary binary data (e.g.
    Stim circuit file, or measurements data).
    These strings may be used instead of file names in any calls to QEC
    Explorer client. The server recognises these strings by detecting a prefix,
    and treats them as data.

    `DataString` class provides methods to generate such string from files,
    strings and byte strings, as well as methods to parse them when the server responds
    with a data string.

    To generate a data string from a file::

        dstring_object = DataString.from_file(path_to_file)
        dstring = str(dstring_object)  # generate a data string
        len(dstring_object.data)  # raw data is stored in `DataString.data` field

    To parse a data string::

        dstring_object = DataString.from_data_string(datastring)

    To generate a data string from a variable::

        bytes_content = b"1,1,0,1,0\\n1,1,1,0,1\\n"
        csv_measurements = DataString(bytes_content)
        b8_measurements = qec_client.convert_format(
            input_file=csv_measurements,
            input_format="csv",
            output_format="b8",
        )
        b8_measurements.to_file("data.b8")

    To convert the content of a DataString object to a string::

        string = data_string.to_string()

    Data strings result in additional processing on client and server side,
    they introduce redundancy in data, and they also increase network traffic
    between client and server nodes.
    """

    empty: str = "duck://"
    """Empty data string"""

    def __init__(self, data: bytes | str = b"") -> None:
        """Construct an object from the data.

        Args:
            data (bytes | str, optional):
                bytes data to be encode, or a string, interpreted as
                UTF-8. Defaults to b"".
        """
        if isinstance(data, bytes):
            self._data = data
        elif isinstance(data, str):
            self._data = data.encode("utf-8")

    @staticmethod
    def from_file(filename: str | Path) -> DataString:
        """Create a data string from a file name.

        Args:
            filename (str | Path): file with data.

        Returns:
            DataString: object representing the data.
        """
        with open(filename, "rb") as file:
            return DataString(file.read())

    @staticmethod
    def _hex_string_to_bytes(hex_string: str) -> bytes:
        return bytes.fromhex(hex_string)

    @staticmethod
    def is_data_string(data_string: str) -> bool:
        """Check that the string corresponds to
        a data string syntax.

        Args:
            data_string (str): string to check.

        Returns:
            bool: True if the string is a good data string.
        """
        if len(data_string) % 2 == 0:
            return False
        if not data_string.startswith(DataString.empty):
            return False
        return not set(data_string[7:]) - set("0123456789abcdef")

    @classmethod
    def from_data_string(cls, data_string: str) -> DataString:
        """Parse the data string into a bytes object.

        Args:
            data_string (str): string representation.

        Raises:
            ValueError:
                if the string is not a valid data string,
                error is raised.

        Returns:
            DataString: data string object.
        """
        if not cls.is_data_string(data_string):
            msg = (
                "String is not a valid Deltakit Explorer data string."
                " Expected that string starts with `duck://` and only "
                "contains characters `0123456789abcdef`"
            )
            raise ValueError(
                msg
            )
        return DataString(cls._hex_string_to_bytes(data_string[7:]))

    @property
    def data(self) -> bytes:
        """Read-only accessor to the data."""
        return self._data

    def to_file(self, path: str | Path) -> None:
        """Dump data to a file.

        Args:
            path (str | Path): file to save the data.
        """
        with Path.open(Path(path), "wb") as file:
            file.write(self._data)

    def to_string(self, encoding: str = "utf-8") -> str:
        """Converts DataString object to string.
        Args:
            encoding (str): The encoding with which to decode the bytes.
            Defaults to UTF-8.
        """
        return self._data.decode(encoding)

    def __str__(self) -> str:
        return self.empty + self._data.hex()

    def __eq__(self, other: object) -> bool:
        if isinstance(other, DataString):
            return self._data == other.data
        return False

    def __hash__(self) -> int:
        return self._data.__hash__()

    def __repr__(self) -> str:
        return f"<DataString, data length={len(self._data)}>"
