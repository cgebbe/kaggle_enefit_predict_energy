import socket
from pathlib import Path


def is_local():
    return socket.gethostname() == "cgebbe"


def get_input_dirpath():
    s = {
        True: "/mnt/sda1/projects/git/competitions/20240118_kaggle_enefit/gitignore/input_data",
        False: "/kaggle/input/predict-energy-behavior-of-prosumers",
    }[is_local()]
    p = Path(s)
    assert p.exists(), p
    return p
