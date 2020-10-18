# SIR-input-scripts
Scripts to feed the MATLAB code for SIR Model. This code is integrating part of the software resources published in the paper __<link to the paper/DOI here>__.

### Description:

Both scripts `adjust_temporal_serie_4_br_states.py` and `adjust_temporal_serie_4_br_cities.py` fit the parameters for each city/state individually for the given input files (which obey the wcota repository csv structure).

In order to execute the code run on paper, one should rename files inside `input` folder so they have no date before the file extension, i.e., `cities.csv` and `states.csv`. In other hand, the user can edit the first argument to the call `observed_data` at line 12 of both scripts to match the desired input file.

Each script outputs the parameters, the fitted time series csv and a graph of fitted vs. target curve in a svg file. Everything goes into the `output` folder and is organized based on the city/state to which it relates to. For ex., all Salvador outputs will be written into the following path `output/BA/Salvador/`, as Salvador is a city from Bahia (BA). Bahia therefore will have its outputs written into `output/BA/`, as it's a Federative State, which contains cities.

### Requirements:

All codes were tested under Python 3.8.2 environment with the latest following packages offered by the conda package manager at the time of development: numpy, scipy, matplotlib, seaborn and pandas

In order to execute this repository codes, one must install the aforementioned packages to your working environment. For instance, `python -m pip install numpy scipy matplotlib seaborn pandas`.

### Execution:

In order to execute, the user must first install all `Requirements` and rename the input files (or edit the line 12 in scripts) and then execute the python scripts inside the root folder. A common execution command line would be: `python adjust_temporal_serie_4_br_states.py` or `python adjust_temporal_serie_4_br_cities.py`.

### Paper data:

The input tables are in the `input` folder. Each table has its data untill the written date in its filename. `cities` or `states` at the beginning of the filename determines what data the file contains. Have a look inside them.

The output data in saved in the `output` folder. We've already put the generated data for the paper into this repository. One should walk through the directory-tree to understand how it's structured and how the fitting worked. It's pretty easy.
