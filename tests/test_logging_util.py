# Test with PyTest

from coast._utils import logging_util
import logging


def test_get_logger():
    logger = logging_util.get_logger(name=None)
    assert isinstance(logger, logging.RootLogger)
    assert logger.name == "root"

    logger = logging_util.get_logger(name="TestLogger")
    assert isinstance(logger, logging.Logger)
    assert logger.name == "TestLogger"


def test_create_handler():
    logger = logging_util.get_logger()
    handler = logging_util.create_handler(logger)
    assert isinstance(handler, logging.StreamHandler)


def test_get_slug():
    obj = object()
    address = hex(id(object))  # Address in memory for CPython, should still be unique in any implementation
    slug = logging_util.get_slug(obj)
    assert slug == f"{obj.__class__.__name__} object at {address}"


def test_get_source():
    file, line, func = logging_util.get_source()
    assert file == __file__
    assert line == "31"  # This must ALWAYS be the line that "logging_util.get_source()" is on or the test will FAIL!
    assert func == "test_get_source"  # This must ALWAYS be the name of the current function, or the test will FAIL!


def test_add_info():
    error_text = "Hello World! I am an error!"
    error = ValueError(error_text)
    assert error_text in logging_util.add_info(error)
