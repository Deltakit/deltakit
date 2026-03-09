# (c) Copyright Riverlane 2020-2025.
"""
The `Logging` class is a service class which enables keeping track of the calls
made and errors which occurred. In the log file, `./client.log`, there will be a
trace of all calls and exceptions that occurred during your sessions. Pay attention to
the `reqid=[...]` fields in this file. Each request to the server is accompanied by
a unique string, which may be used to facilitate incident investigation.
"""

from __future__ import annotations

import inspect
import logging
import logging.handlers
import uuid
from typing import Any

from deltakit_explorer._utils._utils import get_log_directory

LOG_FILENAME = get_log_directory() / "client.log"
LOG_FORMAT = "[%(asctime)s.%(msecs)03d][%(name)s][%(levelname)s] %(message)s"


class Logging:
    """This class enables client-side logging, generates unique request identifiers
    and proxies events from the calling environment (e.g. JupyterHub or gql library).
    """

    # inherit logging values
    CRITICAL = logging.CRITICAL
    FATAL = logging.FATAL
    ERROR = logging.ERROR
    WARNING = logging.WARNING
    INFO = logging.INFO
    DEBUG = logging.DEBUG
    NOTSET = logging.NOTSET

    # switch off requests logging, as it logs full data strings
    logging.getLogger("requests").disabled = True
    logging.getLogger("urllib3").disabled = True
    # logs in a file will be searchable
    # with cat client.log | grep "\[deltakit-explorer\]"
    logger = logging.getLogger("deltakit-explorer")
    logger.setLevel(logging.WARNING)
    file_handler = logging.handlers.RotatingFileHandler(
        LOG_FILENAME, maxBytes=10 * 1024 * 1024, backupCount=3
    )
    file_handler.setFormatter(logging.Formatter(LOG_FORMAT))
    # log to file
    logger.addHandler(file_handler)
    # log to stderr stream by default
    stream_handler = logging.StreamHandler()
    stream_handler.setFormatter(logging.Formatter(LOG_FORMAT))
    logger.addHandler(stream_handler)

    @staticmethod
    def info_and_generate_uid(args: dict[str, Any]) -> str:
        """This method is used to log the API method invocation,
        and to generate a unique string which represents the
        call in the log files on both client and server sides.

        Args:
            args (Dict[str, Any]):
                any arguments, assumed to be a result
                of `locals()` call inside the method. Will be saved
                as is, with `self` value removed.

        Returns:
            str: unique string, identifying function call.
        """
        # get a calling function name
        name = inspect.stack()[1].function
        # generate a unique ID
        uid = uuid.uuid4().hex
        args.pop("self", None)

        newargs = {}
        for key, value in args.items():
            # long string or any other object representation
            string_value = str(value)
            if len(string_value) >= 64:
                string_value = string_value[:30] + "..." + string_value[-30:]
            newargs[key] = string_value
        Logging.logger.info("%s(%s); reqid=[%s]", name, newargs, uid)
        return uid

    @staticmethod
    def info(message: str, uid: str):
        """This method is used to log events during the API method
        invocation. This method should be used after `info_and_generate_uid()`,
        and requires a `uid` as an argument::

            uid = Logging.info_and_generate_uid(locals())
            ...
            Logging.info("Your message", uid)

        Args:
            message (str): message
            uid (str): unique string, result of `info_and_generate_uid()`
        """
        name = inspect.stack()[1].function
        Logging.logger.info("%s: %s | reqid=[%s]", name, message, uid)

    @staticmethod
    def warn(message: str, uid: str):
        """This method is used to log warnings during the API method
        invocation. This method should be used after `info_and_generate_uid()`,
        and requires a `uid` as an argument::

            uid = Logging.info_and_generate_uid(locals())
            ...
            Logging.warn("Your message", uid)

        Args:
            message (str): warning message
            uid (str): unique string, result of `info_and_generate_uid()`
        """
        name = inspect.stack()[1].function
        Logging.logger.warning("%s: %s | reqid=[%s]", name, message, uid)

    @staticmethod
    def error(ex: Exception, uid: str):
        """Use this call for `Exception` logging.
        `uid` obtained from `info_and_generate_uid()` is required::

            uid = Logging.info_and_generate_uid(locals())
            ...
            try:
                # raises here
            except Exception as e:
                Logging.error(e, uid)

        Args:
            ex (Exception): exception object from `except Exception as ex:`.
            uid (str): unique string, result of `info_and_generate_uid()`
        """
        name = inspect.stack()[1].function
        Logging.logger.error("%s: %s | reqid=[%s]", name, ex, uid)

    @classmethod
    def set_log_to_console(cls, log_to_console: bool):
        """Switches console logging on and off."""
        stream_handlers = [
            handler
            for handler in cls.logger.handlers
            # check exactly console logger
            if type(handler)  # pylint: disable=unidiomatic-typecheck
            is logging.StreamHandler
        ]
        if log_to_console:
            if len(stream_handlers) == 0:
                cls.logger.addHandler(cls.stream_handler)
        else:
            # remove all stream handlers
            for handler in stream_handlers:
                cls.logger.removeHandler(handler)

    @classmethod
    def set_log_level(cls, loglevel: int):
        """
        Sets logging level, use `logging.INFO`,
        `logging.DEBUG`, ... values.
        """
        cls.logger.setLevel(loglevel)
