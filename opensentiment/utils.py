from pathlib import Path


def get_project_root() -> Path:
    """return Path to the project directory, top folder of opensentiment"""
    return Path(__file__).parent.parent
