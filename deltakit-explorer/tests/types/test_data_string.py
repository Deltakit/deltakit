# (c) Copyright Riverlane 2020-2025.
from __future__ import annotations

from tempfile import TemporaryDirectory

import pytest

from deltakit_explorer.types import DataString


class TestDataString:

    def test_string_to_bytes(self):
        dstring = DataString.from_data_string("duck://0a0d24")  # \r\n$
        assert dstring.data == b"\n\r$"

    @pytest.mark.parametrize(
            ("input_data", "encoding"),
            [
                ("", "utf-8"),
                ("aaa", "utf-8"),
                ("!$%", "utf-8"),
                ("", "ascii"),
                ("aaa", "ascii"),
                ("!$%", "ascii")
            ]
    )
    def test_to_string_succeeds(self, input_data, encoding):
        dstring = DataString(input_data.encode(encoding))
        assert dstring.to_string(encoding) == input_data

    @pytest.mark.parametrize(
            "input_bytes",
            [
                [b"\xc0"],
                ["ö".encode()],
            ]
    )
    def test_to_string_fails(self, input_bytes):
        dstring = DataString(input_bytes)
        with pytest.raises(Exception):
            dstring.to_string()

    def test_from_datastring(self):
        with pytest.raises(ValueError):
            DataString.from_data_string("not a datastring")

    def test_empty_data_string(self):
        ds = DataString()
        assert ds.data == b""
        assert str(ds) == "duck://"

    def test_get_uri_from_bytes(self):
        ds = DataString(data=b"0123")
        assert str(ds) == "duck://30313233"

    def test_filename_string_read_write(self):
        ds = DataString(b"12345\n$")
        with TemporaryDirectory() as tempdir:
            file = tempdir + "/testfile.txt"
            ds.to_file(file)
            ds2 = DataString.from_file(file)
            assert ds == ds2

    def test_eq_for_datastrings(self):
        assert DataString(b"123") == DataString(b"123")
        assert DataString(b"1234") != DataString(b"123")

    def test_eq_for_other_types(self):
        assert DataString(b"123") != b"123"
        assert DataString(b"123") != str(DataString(b"123"))

    def test_hash_function(self):
        assert hash(DataString(b"1234")) == hash(DataString(b"1234"))
        assert hash(DataString(b"1234")) != hash(DataString(b"1234555"))

    def test_is_data_string(self):
        assert DataString.is_data_string("duck://")
        assert DataString.is_data_string("duck://ff")
        assert DataString.is_data_string("duck://ff00")

    def test_is_not_data_string(self):
        assert not DataString.is_data_string("ducks://ff00")
        assert not DataString.is_data_string("duck://ff001")
        assert not DataString.is_data_string("duck://ff001r")
        assert not DataString.is_data_string("duckie!")
        assert not DataString.is_data_string("")

    def test_empty_str(self):
        assert DataString.empty == "duck://"
