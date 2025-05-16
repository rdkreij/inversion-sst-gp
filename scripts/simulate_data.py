import xarray as xr
import numpy as np

from inversion_sst_gp import utils, simulate_obs

# Load dataset
time_stamp = np.datetime64("2014-02-19T18:00:00")
ds = xr.open_dataset("data/suntans_1h.nc").sel(time=time_stamp)
tstep = ds.time_step.item()
lon, lat = ds.lon.values, ds.lat.values
T, dTdt, u, v, S = (ds[var].values for var in ("T", "dTdt", "u", "v", "S"))

lonc, latc, X, Y, LON, LAT = utils.calculate_grid_properties(lon, lat)
Ny, Nx = len(lat), len(lon)

# Configuration for test types
test_configs = {
    "measurement_error": {
        "val_range": np.arange(0, 0.016, 0.001),
        "param_name": "sigma_tau",
        "dataset_name": "suntans_measurement_error",
    },
    "sparse_cloud": {
        "val_range": np.linspace(0, 0.75, 26),
        "param_name": "coverage_sparse",
        "dataset_name": "suntans_sparse_cloud",
    },
    "dense_cloud": {
        "val_range": np.linspace(0, 0.75, 26),
        "param_name": "coverage_dense",
        "dataset_name": "suntans_dense_cloud",
    },
}

for test_type, config in test_configs.items():
    val_range = config["val_range"]
    param_name = config["param_name"]
    dataset_name = config["dataset_name"]
    N = len(val_range)

    # Allocate arrays
    Toc = np.empty((N, Ny, Nx))
    dTdtoc = np.empty_like(Toc)
    dTds1oc = np.empty_like(Toc)
    dTds2oc = np.empty_like(Toc)
    uc = np.tile(u, (N, 1, 1))
    vc = np.tile(v, (N, 1, 1))
    Sc = np.tile(S, (N, 1, 1))
    maskcc = np.empty_like(Toc)
    
    np.random.seed(0)  # For reproducibility

    for i, val in enumerate(val_range):
        # kwargs = {param_name: val}
        # Toc[i], dTds1oc[i], dTds2oc[i], dTdtoc[i], maskcc[i] = (
        #     simulate_obs.simulate_tracer_observations_from_model_data(
        #         T, dTdt, tstep, X, Y, **kwargs
        #     )
        # )
        if test_type == "measurement_error":
            Toc[i], dTds1oc[i], dTds2oc[i], dTdtoc[i], maskcc[i] =simulate_obs.ModifyData(T, dTdt, tstep, X, Y).noise(val).convert_to_input()
        elif test_type == "sparse_cloud":
            Toc[i], dTds1oc[i], dTds2oc[i], dTdtoc[i], maskcc[i] =simulate_obs.ModifyData(T, dTdt, tstep, X, Y).sparse_cloud(val).convert_to_input()
        elif test_type == "dense_cloud":    
            Toc[i], dTds1oc[i], dTds2oc[i], dTdtoc[i], maskcc[i] =simulate_obs.ModifyData(T, dTdt, tstep, X, Y).circ_cloud(val).convert_to_input()


    # Create dataset
    out_ds = xr.Dataset(
        {
            "T": ([param_name, "lat", "lon"], Toc),
            "dTdt": ([param_name, "lat", "lon"], dTdtoc),
            "u": ([param_name, "lat", "lon"], uc),
            "v": ([param_name, "lat", "lon"], vc),
            "S": ([param_name, "lat", "lon"], Sc),
        },
        coords={
            param_name: val_range,
            "time": time_stamp,
            "time_step": tstep,
            "lat": lat,
            "lon": lon,
        },
    )

    # Save to file if needed
    out_ds.to_netcdf(f"data/{dataset_name}.nc")
