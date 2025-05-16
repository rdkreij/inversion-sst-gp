# inversion-sst-gp

Physics-informed Gaussian process inversion of sea surface temperature to predict submesoscale near-surface ocean currents, based on the paper by Rick J. B. de Kreij et al.

We present a novel method to estimate fine-scale ocean surface currents using satellite sea surface temperature (SST) data. Our approach uses Gaussian process (GP) regression guided by the tracer transport equation, providing not only current predictions but also associated uncertainty. The model effectively handles noisy and incomplete SST data (e.g., due to cloud cover).

This repository contains all code, data, and notebooks to reproduce the experiments and results presented in the paper, including synthetic observing system simulation experiments (OSSEs) and real-world satellite SST applications.

---

## Repository structure

- The **data/** folder contains NetCDF files with SST, altimeter currents, and synthetic datasets with noise or clouds for OSSEs.

- The **notebooks/** folder has Jupyter notebooks demonstrating:  
  1. OSSE individual case studies (noisy data, cloud coverage, etc.)  
  2. OSSE evaluation metrics and model validation  
  3. Satellite SST data inversion over the eastern Indian Ocean  

- The **outputs/** folder stores CSV files with precomputed model results for each experiment scenario.

- The **scripts/** folder includes Python and shell scripts for optimizing model parameters, running simulations, and merging outputs.

- The **src/inversion_sst_gp/** directory contains the core Python modules implementing GP regression, metrics, particle tracking, and utilities.

- LICENSE, README.md, pyproject.toml, and poetry.lock manage licensing and environment setup.

---

## Basic usage

1. Clone this repository and install dependencies (Python 3.10 required):

   ```bash
   git clone https://github.com/yourusername/inversion-sst-gp.git
   cd inversion-sst-gp
   poetry install
   poetry shell
