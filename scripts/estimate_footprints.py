import yaml
import sys
import os
import numpy as np
from shutil import copyfile
sys.path.append(os.path.abspath('.'))
from functions.footprints.prediction import run_footprints

params = {}
with open(sys.argv[1]) as file:
    params = yaml.safe_load(file)

run_footprints(params)