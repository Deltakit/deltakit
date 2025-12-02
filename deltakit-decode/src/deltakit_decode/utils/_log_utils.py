# (c) Copyright Riverlane 2020-2025.
import logging

# Global dictionary with all QEC loggers and handlers
QECLoggers: dict[str, logging.Logger] = {}
QECHandlers: dict[str, logging.Handler] = {}


def make_logger(lvl: int, name: str) -> logging.Logger:
    """Produce a logger with sensible properties.

    Parameters
    ----------
    lvl : int
        Logging level.
    name : str
        Logger name.

    Examples
    --------
    Only messages above the defined level will be logged:

    .. code-block:: python

        log = make_logger(logging.WARNING, "Test")
        log.error("Error example")
        # Test [ERROR]: Error example
        log.warning("Warning example")
        # Test [WARNING]: Warning example
        log.info("Info example")
        log.debug("Debug example")
    """
    if name in QECLoggers:
        QECLoggers[name].setLevel(lvl)
        QECHandlers[name].setLevel(lvl)
        return QECLoggers[name]

    log = logging.getLogger(name)
    log.setLevel(lvl)
    if name in QECHandlers:
        # log already has handler, we just need to set the level
        QECHandlers[name].setLevel(lvl)
    else:
        ch = logging.StreamHandler()
        ch.setLevel(lvl)

        fmt = logging.Formatter("%(name)s [%(levelname)s]: %(message)s")

        ch.setFormatter(fmt)
        log.addHandler(ch)
        QECHandlers[name] = ch

    QECLoggers[name] = log
    return log
