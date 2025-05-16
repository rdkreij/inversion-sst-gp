import sys
import os

import numpy as np
import xarray as xr

from inversion_sst_gp import gp_regression, metrics, other_methods, utils

# Parse command-line arguments
step = float(sys.argv[1])
model_type = sys.argv[2]
test_type = sys.argv[3]
id = sys.argv[4]

# Load base dataset and compute grid properties
time_base_str = "2014-02-19T18:00:00"

# Set model hyperparameters
if model_type == "gprm":
    theta = utils.extract_params(
        "outputs/num_model_estimated_t.csv", "time", time_base_str, type="num_est"
    )
    prop_osse = {
        "initial_params": {"sigma_tau": 0.01},
        "const_params": theta,
        "penalty_params": {},
        "share_len": False,
        "share_tau": False,
        "share_sigma": False,
        "solve_log": True,
        "bounds_params": {},
    }

elif model_type == "gprm_e":
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
    prop_osse = {
        "initial_params": initial_params,
        "const_params": {},
        "penalty_params": penalty_params,
        "share_len": True,
        "share_tau": True,
        "share_sigma": True,
        "solve_log": True,
        "bounds_params": bounds_params,
    }

elif model_type == "optimum":
    time_base = np.datetime64(time_base_str)
    ds_base = xr.open_dataset("data/suntans_1h.nc").sel(time=time_base)

    lon, lat, To, dTdto, u, v, S = (
        ds_base[var].values for var in ("lon", "lat", "T", "dTdt", "u", "v", "S")
    )
    _, _, X, Y, _, _ = utils.calculate_grid_properties(lon, lat)

    sigma_u, l_u, tau_u = gp_regression.estimate_params_process(
        u, X, Y, 1e-1, 4e4, 1e-3
    )
    sigma_v, l_v, tau_v = gp_regression.estimate_params_process(
        v, X, Y, 1e-1, 4e4, 1e-3
    )
    sigma_S, l_S, tau_S = gp_regression.estimate_params_process(
        S, X, Y, 3e-7, 3e4, 2e-7
    )

    theta = {
        "sigma_u": sigma_u,
        "l_u": l_u,
        "tau_u": tau_u,
        "sigma_v": sigma_v,
        "l_v": l_v,
        "tau_v": tau_v,
        "sigma_S": sigma_S,
        "l_S": l_S,
        "tau_S": tau_S,
    }

    prop_osse = {
        "initial_params": {"sigma_tau": 0.01},
        "const_params": theta,
        "penalty_params": {},
        "share_len": False,
        "share_tau": False,
        "share_sigma": False,
        "solve_log": True,
        "bounds_params": {},
    }

# Load test dataset
if test_type == "noise":
    ds = xr.open_dataset("data/suntans_measurement_error.nc").sel(noise=step)
elif (test_type == "time_24h") or (test_type == "time_1h"):
    ds = xr.open_dataset("data/suntans_24h.nc").sel(
        time=time_base + np.timedelta64(step, "s")
    )
elif test_type == "time_1h":
    ds = xr.open_dataset("data/suntans_1h.nc").sel(
        time=time_base + np.timedelta64(step, "s")
    )
elif test_type == "cloud_sparse":
    ds = xr.open_dataset("data/suntans_sparse_cloud.nc").sel(coverage_sparse=step)
elif test_type == "cloud_dense":
    ds = xr.open_dataset("data/suntans_dense_cloud.nc").sel(coverage_dense=step)
else:
    raise ValueError(f"Unknown test_type: {test_type}")

tstep = ds.time_step.item()
lon, lat, To, dTdto, u, v, S = (
    ds[var].values for var in ("lon", "lat", "T", "dTdt", "u", "v", "S")
)
lonc, latc, X, Y, LON, LAT = utils.calculate_grid_properties(lon, lat)
dTds1o, dTds2o = utils.finite_difference_2d(X, Y, To)

# Metric functions
def run_gos_optim(step, dTds1, dTds2, dTdt, u, v):
    gos = other_methods.GlobalOptimalSolution(dTds1, dTds2, dTdt)
    n = gos.optimize_n(u, v)
    ugos, vgos, _ = gos.compute(n)
    rmse = metrics.overview(u, v, ugos, vgos)["RMSE"]
    return {"step": step, "n": n, "RMSE": rmse}


def run_gprm_optim(step, dTds1, dTds2, dTdt, u, v, X, Y, tstep, prop, callback="on"):
    mask = np.ones_like(X, dtype=bool)
    gprm = gp_regression.GPRegressionJoint(dTds1, dTds2, dTdt, tstep, X, Y, mask)
    est_params = gprm.estimate_params(**prop, callback=callback)
    mu_u, mu_v, _, std_u, std_v, _, _ = gprm.predict(est_params)
    results = metrics.overview(u, v, mu_u, mu_v, std_u, std_v)
    return {
        "step": step,
        "est_params": est_params,
        "RMSE": results["RMSE"],
        "crps_norm": results["crps_norm"],
        "coverage90": results["coverage90"],
    }

# Run model
if model_type == "gos":
    results = run_gos_optim(step, dTds1o, dTds2o, dTdto, u, v)
else:
    results = run_gprm_optim(
        step, dTds1o, dTds2o, dTdto, u, v, X, Y, tstep, prop_osse
    )
print(results)

# Save results
os.makedirs('intermediate', exist_ok=True)
utils.save_json(results, f"intermediate/osse_{model_type}_{test_type}_{id}.json")
