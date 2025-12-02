# (c) Copyright Riverlane 2020-2025.
from __future__ import annotations

import logging
import os
from tempfile import NamedTemporaryFile

import pytest
from deltakit_explorer import Client
from deltakit_explorer._api import _auth
from deltakit_explorer._utils._logging import LOG_FILENAME, Logging
from deltakit_explorer._utils._utils import get_log_directory
from deltakit_explorer.types import QECExperimentDefinition
import contextlib


class TestLogging:

    def setup_class(self):
        """Redirect logging to a temporary file"""

        self.logfile = NamedTemporaryFile("w+")  # noqa: SIM115
        handler = logging.StreamHandler(self.logfile)
        handler.setFormatter(
            logging.Formatter(
                "[%(asctime)s.%(msecs)03d][%(module)s][%(levelname)s] %(message)s"
            )
        )
        Logging.logger.addHandler(handler)
        Logging.set_log_level(logging.INFO)

    def teardown_class(self):
        self.logfile.close()
        del self.logfile

    def test_info_and_generate_uid_logs(self):
        uid = Logging.info_and_generate_uid({"a": 1, "b": 2})
        self.logfile.seek(0)
        # take the recent log line
        logline = self.logfile.readlines()[-1]
        assert "{'a': '1', 'b': '2'}" in logline
        assert "test_info_and_generate_uid_logs" in logline
        assert "[INFO]" in logline
        assert uid in logline
        assert len(uid) == 32

    def test_info_logs(self):
        Logging.info({"a": 1, "b": 2}, "apple")
        self.logfile.seek(0)
        # take the recent log line
        logline = self.logfile.readlines()[-1]
        assert "{'a': 1, 'b': 2}" in logline
        assert "test_info_logs" in logline
        assert "[INFO]" in logline
        assert "[apple]" in logline

    def test_warn_logs(self):
        Logging.warn("warning", "apple")
        self.logfile.seek(0)
        logline = self.logfile.readlines()[-1]
        assert "warning" in logline
        assert "test_warn_logs" in logline
        assert "[WARNING]" in logline
        assert "[apple]" in logline

    def test_error_logs(self):
        Logging.error(Exception("error"), "apple")
        self.logfile.seek(0)
        logline = self.logfile.readlines()[-1]
        assert "error" in logline
        assert "test_error_logs" in logline
        assert "[ERROR]" in logline
        assert "[apple]" in logline

    def test_first_info_and_warn_logs(self):
        uid = Logging.info_and_generate_uid({"a": 1, "b": 2})
        Logging.warn("warning", uid)
        self.logfile.seek(0)
        logline = self.logfile.readlines()[-1]
        assert "warning" in logline
        assert "test_first_info_and_warn_logs" in logline
        assert "WARN" in logline
        assert uid in logline

    def test_info_logs_indirectly(self, mocker):
        mocker.patch("deltakit_explorer._utils._utils.APP_NAME", "deltakit-testplorer")
        _auth.set_token("123")
        client = Client("base")
        with pytest.raises(Exception):
            client.generate_circuit(
                QECExperimentDefinition.get_repetition_z_quantum_memory(-1, -1))
        self.logfile.seek(0)
        loglines = self.logfile.readlines()
        assert "generate_circuit" in loglines[-2]
        assert "INFO" in loglines[-2]
        assert "{'experiment_definition': " in loglines[-2]
        assert "generate_circuit" in loglines[-1]
        assert "ERROR" in loglines[-1]

    @pytest.fixture
    def log_path(self, tmp_path):
        log_format = "[%(asctime)s.%(msecs)03d][%(name)s][%(levelname)s] %(message)s"
        tmp_log_path = tmp_path / "test.log"
        logging.basicConfig(
            filename=tmp_log_path, format=log_format, level=logging.INFO
        )
        logger = logging.getLogger("deltakit-explorer")
        logger.setLevel(logging.INFO)
        file_handler = logging.FileHandler(tmp_log_path)
        file_handler.setFormatter(logging.Formatter(log_format))
        logger.addHandler(file_handler)
        return tmp_log_path

    @pytest.mark.parametrize(
        ("method", "ref_text"),
        [
            (
                Logging.info,
                "[deltakit-explorer][INFO] test_log_method_correct_format_and_content:"
                " test | reqid=[test]\n",
            ),
            (
                Logging.warn,
                "[deltakit-explorer][WARNING] test_log_method_correct_format_and_content:"
                " test | reqid=[test]\n",
            ),
            (
                Logging.error,
                "[deltakit-explorer][ERROR] test_log_method_correct_format_and_content:"
                " test | reqid=[test]\n",
            ),
        ],
    )
    def test_log_method_correct_format_and_content(self, log_path, method, ref_text):
        method("test", "test")
        with open(log_path) as logfile:
            text = logfile.read()
        get_lsb_indices = [i for i, x in enumerate(text) if x == "["]
        string_start = get_lsb_indices[1]  # ignore the date+time
        assert text[string_start:] == ref_text

    def test_info_and_generate_uid_format_and_content(self, log_path):
        uid = Logging.info_and_generate_uid({"test": "test"})
        with open(log_path) as logfile:
            text = logfile.read()
        get_lsb_indices = [i for i, x in enumerate(text) if x == "["]
        string_start = get_lsb_indices[1]  # ignore the date+time
        expected_string = (
            "[deltakit-explorer][INFO] "
            "test_info_and_generate_uid_format_and_content({'test': 'test'});"
            f" reqid=[{uid}]\n"
        )
        assert text[string_start:] == expected_string

    def test_client_log_rotation(self):
        long_text = "1234567890abcdefghijklmnopq" * 10000
        for i in range(200):
            Logging.info(long_text, f"fake_uid_{i}")
        files = [
            get_log_directory() / LOG_FILENAME,
            *[get_log_directory() / f"{LOG_FILENAME}.{i}" for i in range(1, 4)]
        ]

        # cleanup
        for file in files:
            assert os.path.getsize(file) <= 256 * 1024 * 1024
            with contextlib.suppress(PermissionError):
                os.remove(file)

    def test_string_shortening(self, log_path):
        local1 = "12345" * 1000
        local2 = "asdfbdx" * 100
        args = { "local1": local1, "local2": local2, "log_path": log_path }
        Logging.info_and_generate_uid(args)
        with open(log_path) as logfile:
            text = logfile.read()
        assert len(text) <= (
            # cap of 63 per variable value + quotes
            len(args) * (63 + 2)
            # 4 for quotes and commas + variable names
            + sum(len(k) + 4 for k in args)
            # reqid 16 bytes in hex
            + 32
            # surrounding
            + len("[2025-02-12 14:54:30,397.397][logging][INFO]"
                  "     api:logging.py:76 test_string_shortening({}); reqid=[]")
        )


    def test_set_log_to_console_off(self):
        Logging.set_log_to_console(False)
        handlers = [
            handler
            for handler in Logging.logger.handlers
            if type(handler)
            is logging.StreamHandler
        ]
        assert len(handlers) == 0


    def test_set_log_to_console_on(self):
        Logging.set_log_to_console(True)
        handlers = [
            handler
            for handler in Logging.logger.handlers
            if type(handler)
            is logging.StreamHandler
        ]
        assert len(handlers) == 1
        Logging.set_log_to_console(False)


    def test_set_log_to_console_on_off(self):
        Logging.set_log_to_console(True)
        handlers = [
            handler
            for handler in Logging.logger.handlers
            if type(handler)
            is logging.StreamHandler
        ]
        assert len(handlers) == 1
        Logging.set_log_to_console(False)
        handlers = [
            handler
            for handler in Logging.logger.handlers
            if type(handler)
            is logging.StreamHandler
        ]
        assert len(handlers) == 0

    def test_set_log_to_console_off_on(self):
        Logging.set_log_to_console(False)
        handlers = [
            handler
            for handler in Logging.logger.handlers
            if type(handler)
            is logging.StreamHandler
        ]
        assert len(handlers) == 0
        Logging.set_log_to_console(True)
        handlers = [
            handler
            for handler in Logging.logger.handlers
            if type(handler)
            is logging.StreamHandler
        ]
        assert len(handlers) == 1
        Logging.set_log_to_console(False)
