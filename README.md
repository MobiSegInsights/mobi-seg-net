# Mobility network embeddings reveal socio-spatial segregation
This repository explores how spatial embeddings derived from mobility networks reveal socio-spatial segregation, 
and investigates how the built environment influences those patterns. 
It is part of the [parent project](https://github.com/MobiSegInsights).

## Dependencies
Python (version 3.11) code and R (version 4.0.2) code were used to analyse and visualize the data. 
The stays have been detected via infostop (version 0.1.11) and pyspark (version 3.5.1).

## Data
The data supporting the findings of this study were purchased from [PickWell](https://www.pickwell.co/) and 
are subject to restrictions due to licensing 
and privacy considerations under the European General Data Protection Regulation. 
Consequently, these data are not publicly available. 
Venue locations and categories were obtained from [OpenStreetMap](https://download.geofabrik.de/europe.html).

## Scripts
The repo contains the scripts (`src/`), libraries (`lib/`) for conducting the data processing, analysis, and visualisation.
The original input data are stored under `dbs/` locally and intermediate results are stored in a local database.
The produced figures are stored under `figures/`.

Under `src/`, the scripts are stored by their functionality, with the first number indicating the
order of running the script.
This is because some later analysis may depend on earlier steps.

- `src/data_etl/` do data extraction and preprocessing.
- `src/feature_eng/` compute metrics of mobility and segregation.
- `src/data_exp/` explore the data, produce descriptive statistics, and conduct statistical analysis.
- `src/simulations/` include the two counterfactual simulations, random mixing scenario, and group exposure analysis.
- `src/visualization/` produce figures inserted in the manuscript.