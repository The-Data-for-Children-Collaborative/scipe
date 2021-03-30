# Population Estimation Project

## Scripts
There are several scripts included with the project. They are intended to be run in the following order:
1. /scripts/rasterize_survey.py <path/to/config> - rasterize survey data to geoTiff. This script is dependent on survey format, and is designed for SpaceSUR population survey from Mozambique.
2. /scripts/estimate_footprints.py <path/to/config> - estimate probability maps using building footprint models defined in the config file.
3. /scripts/run_pipeline.py <path/to/config> - run preprocessing, build dataset, predict population, and output results of experiments according to the config file.

## Config
Each script takes a separate YAML config file as input, examples of each can be found in /scripts/config/

## Examples
Executing run_pipeline.py yields the following results for the example config file (/scripts/pipeline.yaml):
[Add some results]

## Data
Data required to run the pipeline can be downloaded from [TBC].