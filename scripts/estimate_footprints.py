""" Script for estimating building footprints.

Usage:
    ``python estimate_footprints.py <path_to_config>``
"""

import yaml
import sys
import os

sys.path.append(os.path.abspath('.'))
sys.path.append(os.path.abspath('..'))
from functions.footprints.prediction import run_footprints

if __name__ == "__main__":
    with open(sys.argv[1]) as file:
        params = yaml.safe_load(file)

    run_footprints(params)
