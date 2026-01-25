[![DOI](https://zenodo.org/badge/DOI/10.5281/zenodo.18244944.svg)](https://doi.org/10.5281/zenodo.18244944)

## Introduction

This repository contains the figures and codes confirmed for the master thesis 
> **Estimation of Geomagnetic Secular Variation by Machine Learning with Extended Kalman Filter**
> 
> by Sho SATO,  
submitted to the Graduate School of Science, Kyoto University,  
in fulfillment of the requirements for a Master’s degree,  
April 2026.

The code is designed to estimate future geomagnetic secular variation by training on time series of geomagnetic field data and Earth's Length-Of-Day data using machine learning techniques.

For more details on the methodology, please refer to the paper (to appear in *Earth, Planets and Space*):

> **Recurrent neural network trained with the extended Kalman filter to forecast the geomagnetic secular variation for IGRF-14**


## Files

This repository consists of the following four directories:

1. `code`
2. `data`
3. `output`
4. `visualization`


### code

This directory contains python code files used to process data and train models using machine learning. The hidden node size of RNN is set to $D_\mathbf{h} = 34$.

* `EKFtrainedRNN.py`: A module that defines the RNN model class trained with the extended Kalman filter (EKF).
* `geomagRNNpred.py`: A module that defines functions for geomagnetic field prediction using the EKF-RNN model.

* `2023_1214_SARIMAX(p,d,0)_heatmap.ipynb`: A script that generates heatmaps of SARIMAX(p,d,0) model performance for various combinations of parameters (p,d) (Chapter 2).
* `2023_1217_VAR_p1_d2.ipynb`: A script that generates geomagnetic forecast with VAR(1) model trained with SA snapshots (Chapter 2).

* `2024_0912_processMCM2024.ipynb`: A script that computes differences in the provided magnetic field data and converts them to `.csv` and `.npy` formats

* `2025_1017_yBnLODnLDT_h34_s0-32.py`: 
  A script for performing exhaustive grid search of 
  
  * Order of derivative $d$ in the range of $0 \leq d \leq 4$
  * Initial states $\mathbf{w}_0^s$ in the range of $00000 \leq s \leq 11111$, 
  
  Training is performed using 
  * MCM-2024

* `2025_1105_processLODdata.ipynb`: A script that computes moving-average filtered LOD data and variances.

* `2025_1108_yByLODnLDT_h34_s0-32.py`: 
  A script for performing exhaustive grid search of 
  
  * Order of derivative $d$ in the range of $0 \leq d \leq 4$
  * Initial states $\mathbf{w}_0^s$ in the range of $00000 \leq s \leq 11111$, 
  
  Training is performed using 
  * MCM-2024
  * LOD data

* `2025_1205_yBnLODyLDT_h34_s0-32.py`: 
  A script for performing exhaustive grid search of 
  
  * Order of derivative $d$ in the range of $0 \leq d \leq 4$
  * Initial states $\mathbf{w}_0^s$ in the range of $00000 \leq s \leq 11111$, 
  
  Training is performed using 
  * MCM-2024
  * First time derivative of LOD data

### data

This directory contains training data used for machine learning.

1. **geomagnetic field snapshots (gauss coefficients derived from MCM-2024 model)** provided by the *Institut de Physique du Globe de Paris* (IPGP) in France.
2. **Observed Length-Of-Day (LOD) data** provided by the *International Earth Rotation and Reference Systems Service* (IERS).

The data are used as training inputs for the machine learning models.

* `raw/`: Raw data as originally provided by IPGP and IERS
* `processed/`: Preprocessed data, where magnetic field differences have been computed and saved in `.csv` and `.npy` formats


### output

This directory contains the output results of the RNN models.


### visualization

This directory contains jupyter notebooks for visualizing the results presented in the manuscript:

- `visualization/2024_0514_ARparams_demo.ipynb`: Notebook demonstrating the AR model parameters used in Subsection 2.1.3.

- `visualization/2024_0912_displayMCM2024.ipynb`: Notebook for visualizing MCM-2024 model data.

- `visualization/2025_1105_processLODdata.ipynb`: Notebook for processing LOD data (used in `code/2025_1108_yByLODnLDT_h34_s0-32.py` and `code/2025_1205_yBnLODyLDT_h34_s0-32.py`).

- `visualization/2025_1111_vizMCM-RNN_derivative.ipynb`: Notebook for visualizing the results of RNN trained with MCM-2024 model with different orders of derivatives (for Chapter 3). 
  - Other notebooks for visualizing the results of RNN trained with MCM-2024 model with different initial states are available on the following repository: 
    > Sato, S., Lesur, V., Nakano, S., Minami, T., Matsushima, M., & Toh, H. (2025). IGRF-14 Japanese Candidate Model. Zenodo. <https://doi.org/10.5281/zenodo.15726524> 
  
- `visualization/2025_1112_viz2015SApulse_analysis.ipynb`: Notebooks for visualizing the results of RNN trained with MCM-2024 model, RNN trained with 2015 SA data, and RNN trained with GAP data, respectively (for Chapter 4).
  - `visualization/2025_1114_viz2015MCM_Br.ipynb`
  - `visualization/2025_1113_viz2015RNN_Br.ipynb`
  - `visualization/2025_1113_viz2015GAP_Br.ipynb`
- `visualization/2025_1209_vizMCM_LOD_LDT-RNN.ipynb`: Notebook for visualizing the results of RNN trained with MCM-2024 + LOD data + first time derivative of LOD data (for Chapter 4 and 5).

## Closing Remarks

### Execution Environment

The code in this repository can be executed using the Python environment described in [environment.yml](environment.yml).

However, the notebook `visualization/2025_11xx_viz2015XXX_Br.ipynb` requires a different virtual environment, as described in the [IAGA tutorial](https://github.com/IAGA-VMOD/IGRF14eval/blob/main/README.md#local-development):

[https://github.com/IAGA-VMOD/IGRF14eval/blob/main/environment-base.yml](https://github.com/IAGA-VMOD/IGRF14eval/blob/main/environment-base.yml)


### License

Code in this repository is licensed under MIT, while data and documentation are licensed under CC BY 4.0. Refer to the repository LICENSE files for details.