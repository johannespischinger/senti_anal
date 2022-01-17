import logging
from pathlib import Path
from typing import Union


def get_project_root() -> Path:
    """
    return Path to the project directory, top folder of opensentiment
    """
    return Path(__file__).parent.parent.resolve()


def get_logger_default(name: Union[None, str]) -> logging.Logger:
    """
    configure logger

    args:
        name
    returns:
        logger
    """
    if name is None:
        name = __name__
    logger = logging.getLogger(name)

    # configure logger below
    log_fmt = "%(asctime)s - %(name)s - %(levelname)s - %(message)s"
    fmt = logging.Formatter(fmt=log_fmt)
    ch = logging.StreamHandler()
    ch.setFormatter(fmt)
    logger.addHandler(ch)
    # return modified logger

    return logger
