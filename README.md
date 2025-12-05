# EKF-RNN_B_LOD_LDT

## Contributers

- Sho SATO, *Graduate School of Science, Kyoto University*


## Introduction

This repository contains the figures and codes confirmed for the master thesis 
> **Leveraging Length-of-day Data In Recurrent Neural Networks For Predicting Geomagnetic Secular Acceleration**
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
3. `log`
4. `output`
5. `visualization`


### code

This directory contains python code files used to process data and train models using machine learning. The hidden node size of RNN is set to $D_\mathbf{h} = 34$.

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


### log
This directory contains log files generated during the execution of the python scripts in the `code` directory. These logs provide information about the training process.


### output

This directory contains the output results of the RNN models.


### visualization

This directory contains jupyter notebooks for visualizing the results presented in the manuscript:


* `xxx.ipynb`: Generates **Figure 3**
* `xxx.ipynb`: Generates **Table 2**, **Figures 11 and 12**
* `xxx.ipynb`: Generates **Figure 10**
* `xxx.ipynb`: Generates **Table 1**, **Figures 5, 6, 7, and 9**
* `xxx.ipynb`: Generates **Table 1** and **Figure 8**


## Closing Remarks

### Execution Environment

The code in this repository can be executed using the Python environment described in [environment.yml](environment.yml).

However, the notebook `visualization/2025_12xx_SV_r_magnetomap.ipynb` requires a different virtual environment, as described in the [IAGA tutorial](https://github.com/IAGA-VMOD/IGRF14eval/blob/main/README.md#local-development):

[https://github.com/IAGA-VMOD/IGRF14eval/blob/main/environment-base.yml](https://github.com/IAGA-VMOD/IGRF14eval/blob/main/environment-base.yml)


### License

Code in this repository is licensed under MIT, while data and documentation are licensed under CC BY 4.0. Refer to the repository LICENSE files for details.