import yaml
import sys
import os
from shutil import copyfile

sys.path.append(os.path.abspath('.'))
sys.path.append(os.path.abspath('..'))
from functions.population.survey import rasterize_survey

if __name__ == "__main__":
    with open(sys.argv[1]) as file:
        params = yaml.safe_load(file)

    rasterize_survey(params)

    src = sys.argv[1]
    dst = os.path.join(params['out_dir'], os.path.basename(src))
    copyfile(src, dst)  # copy params to experiment dir
