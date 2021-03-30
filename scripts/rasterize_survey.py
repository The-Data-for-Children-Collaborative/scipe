import yaml
import sys
import os
import numpy as np
from shutil import copyfile
sys.path.append(os.path.abspath('.'))
from functions.survey import rasterize_survey

params = {}
with open(sys.argv[1]) as file:
    params = yaml.safe_load(file)

rasterize_survey(params)
    
src = sys.argv[1]
dst = os.path.join(params['out_dir'],os.path.basename(src))
copyfile(src,dst) # copy params to experiment dir 