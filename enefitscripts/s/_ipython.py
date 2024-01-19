import os
from pathlib import Path


def run_as_dev(name):
    if not name == "__main__":
        return False
    if not _run_from_ipython():
        return False
    if "OUTSIDE" in os.environ:
        return False
    return True


def _run_from_ipython():
    try:
        __IPYTHON__
        return True
    except NameError:
        return False


def get_parent_dir():
    if _run_from_ipython():
        return Path().resolve()
    else:
        return Path(__file__).resolve().parent


def setup_autoreload():
    if _run_from_ipython():
        get_ipython().run_line_magic("load_ext", "autoreload")
        get_ipython().run_line_magic("autoreload", "2")
