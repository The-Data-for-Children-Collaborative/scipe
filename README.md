# SCIPE

Research code for _Sustainable Census Independent Population Estimation (SCIPE)_ - a pipeline for predicting population maps using microcensus data and satellite imagery.

SCIPE is introduced in the paper ['Census-Independent Population Estimation using Representation Learning' [Scientific Reports (2022)]](https://www.nature.com/articles/s41598-022-08935-1).

## Scripts
There are several scripts included with the project. They are intended to be run in the following order:
1. [`/scripts/rasterize_survey.py`](./scripts/rasterize_survey.py) `<path/to/config>` - rasterize survey data to geoTiff. This script is dependent on survey format, and is designed for SpaceSUR population survey from Mozambique.
2. [`/scripts/estimate_footprints.py`](./scripts/estimate_footprints.py) `<path/to/config>` - estimate probability maps using building footprint models defined in the config file.
3. [`/scripts/run_pipeline.py`](./scripts/run_pipeline.py) `<path/to/config>` - run preprocessing, build dataset, predict population, and output results of experiments according to the config file.
4. [`/scripts/split_imagery.py`](./scripts/split_imagery.py) `<path/to/imagery>` `<path/to/survey>` `<out/path>` - split large raster files contained within `<path/to/imagery>` into grid defined by geoTiff at `<path/to/survey>` 

## Config
Each script takes a separate YAML config file as input, examples of each can be found in [`/scripts/config/`](./scripts/config/).

## Models
Pretrained population models are available from the authors upon request.

## Documentation
Documentation is available at [`/docs/_build/html/index.html`](./docs/_build/html/index.html)

## Examples
Executing `run_pipeline.py` yields the following results for the example [`pipeline.yaml`](/scripts/config/pipeline.yaml) with outliers removed:

<img src="/experiments/barlow/outliers_removed/prediction_error.png" alt="Predicted vs. observed values"  height="400">
<img src="/experiments/barlow/outliers_removed/rf_importance.png" alt="Feature importance"  height="400">
