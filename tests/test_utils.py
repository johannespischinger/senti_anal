from webbrowser import get
from opensentiment.utils import get_logger_default, get_project_root
import os
import logging


def test_project_root():
    pr = get_project_root()
    assert os.path.exists(pr), f"does get_project_root() point to the pr? {pr}"
    assert os.path.exists(
        os.path.join(pr, "opensentiment")
    ), f"does get_project_root {pr}/opensentiment exist?"
    assert os.path.exists(
        os.path.join(pr, "tests")
    ), f"does get_project_root {pr}/tests exist?"


def test_logger():
    assert isinstance(get_logger_default("testloggers"), logging.Logger)
