"""A logging unilty file"""
import logging
import sys
import io
import warnings
import inspect
import traceback


DEFAULT_FORMAT = "%(asctime)s %(levelname)s %(message)s"


def get_logger(name: str = None, level: int = logging.CRITICAL):
    logger = logging.getLogger(name=name)
    logger.setLevel(level)
    return logger


def create_handler(logger: logging.Logger, stream: io.TextIOWrapper = sys.stdout, format_string: str = DEFAULT_FORMAT):
    handler = logging.StreamHandler(stream)
    handler.setLevel(logger.level)
    formatter = logging.Formatter(format_string)
    handler.setFormatter(formatter)
    return handler


def setup_logging(
    name: str = None,
    level: int = logging.CRITICAL,
    stream: io.TextIOWrapper = sys.stdout,
    format_string: str = DEFAULT_FORMAT,
):
    logger = get_logger(name=name, level=level)
    handler = create_handler(logger, stream=stream, format_string=format_string)
    logger.addHandler(handler)

    return logger, handler


def get_slug(obj: object):
    name = obj.__class__.__name__
    ref = hex(id(object))
    return "{0} object at {1}".format(name, ref)


def get_source(level=1):
    frame = inspect.stack()
    if type(frame) is list:
        frame = frame[level]

    file = frame[1]
    line = str(frame[2])
    func = frame[3]

    return file, line, func


def add_info(msg, level=3):
    source = get_source(level=level)
    if isinstance(msg, Exception):
        msg = f"{msg.__class__.__name__}: {str(msg)}\n" + "".join(traceback.format_tb(msg.__traceback__))
    msg = f"{source[0]}.{source[2]}.{source[1]}: {msg}"
    return msg


def debug(msg, *args, **kwargs):
    return logging.debug(add_info(msg), *args, **kwargs)


def info(msg, *args, **kwargs):
    return logging.info(add_info(msg), *args, **kwargs)


def warning(msg, *args, **kwargs):
    return logging.warning(add_info(msg), *args, **kwargs)


def warn(msg, *args, **kwargs):
    return warnings.warn(add_info(msg), *args, **kwargs)


def error(msg, *args, **kwargs):
    return logging.error(add_info(msg), *args, **kwargs)
