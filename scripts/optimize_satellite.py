import sys
import os

import numpy as np
import xarray as xr

from inversion_sst_gp import gp_regression, utils

# Parse command-line arguments
time_str = sys.argv[1]
id = sys.argv[2]

# Set model hyperparameters
initial_params = {
    "sigma_u": 9e-2,
    "l_u": 3e4,
    "tau_u": 1e-2,
    "sigma_S": 3e-7,
    "l_S": 2e4,
    "tau_S": 2e-7,
    "sigma_tau": 1e-2,
}
penalty_params = {
    "l_u": [3e4, 0.5e4],
    "sigma_u": [9e-2, 2e-2],
    "tau_u": [1e-2, 0.1e-2],
    "l_S": [2e4, 2e4],
    "sigma_S": [3e-7, 5e-6],
    "tau_S": [2e-7, 5e-6],
}
bounds_params = {
    "sigma_u": [1e-10, 10],
    "l_u": [1, 1e6],
    "tau_u": [1e-10, 1],
    "sigma_S": [1e-8, 1e-3],
    "l_S": [1, 1e6],
    "tau_S": [1e-15, 1e-3],
    "sigma_tau": [1e-15, 1],
}
prop_sat = {
    "initial_params": initial_params,
    "const_params": {},
    "penalty_params": penalty_params,
    "share_len": True,
    "share_tau": True,
    "share_sigma": True,
    "solve_log": True,
    "bounds_params": bounds_params,
}

# Load Himawari data
ds = xr.open_dataset("data/himawari.nc").sel(time=np.datetime64(time_str))
time_step = ds.time_step.item()
lon, lat, To, dTdto = (
    ds[var].values for var in ("lon", "lat", "T", "dTdt")
)
tstep = ds.time_step.item()
lonc, latc, X, Y, LON, LAT = utils.calculate_grid_properties(lon, lat)
dTds1o, dTds2o = utils.finite_difference_2d(X, Y, To)

# Metric functions
def run_gprm_optim(time_str, dTds1, dTds2, dTdt, X, Y, tstep, prop, callback="on"):
    mask = np.ones_like(X, dtype=bool)
    gprm = gp_regression.GPRegressionJoint(dTds1, dTds2, dTdt, tstep, X, Y, mask)
    est_params = gprm.estimate_params(**prop, callback=callback)
    return {
        "step": time_str,
        "est_params": est_params,
    }

# Run model
results = run_gprm_optim(
    time_str, dTds1o, dTds2o, dTdto, X, Y, tstep, prop_sat
)
print(results)

# Save results
os.makedirs('intermediate', exist_ok=True)
utils.save_json(results, f"intermediate/satellite_{id}.json")
