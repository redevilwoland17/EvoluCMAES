"""Locate the bundled data files (results/ and libraries/) from any working directory."""
import glob
import os

ROOT = os.path.abspath(os.path.join(os.path.dirname(os.path.abspath(__file__)), '..'))


def data(name):
    """Return the path of a bundled file (or glob pattern), searching results/ and then libraries/."""
    for sub in ('results', 'libraries'):
        path = os.path.join(ROOT, sub, name)
        if os.path.exists(path) or glob.glob(path):     # a file, or a glob pattern with matches
            return path
    return name
