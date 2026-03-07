import xarray as xr
import numpy as np
from matplotlib import rc
from inversion_sst_gp import (
    utils,
    other_methods,
    metrics,
    gp_regression,
    simulate_from_gp,
)

# Matplotlib configuration
rc("font", family="serif", serif=["Computer Modern"])
rc("text", usetex=True)
rc("text.latex", preamble=r"\usepackage{amsmath}")

# Plotting parameters
lonlims = (115, 118)
latlims = (-15.5, -12.5)


# Helper functions
def load_and_prepare_dataset(path, selection_key=None, selection_val=None):
    print(f"Loading dataset: {path}")
    ds = xr.open_dataset(path)
    if selection_key and selection_val is not None:
        ds = ds.sel({selection_key: selection_val})
        print(f"Selecting dataset with {selection_key}={selection_val}")
    time_step = ds.time_step.item()
    lon, lat, To, dTdto, u, v, S, eta = (
        ds[var].values for var in ("lon", "lat", "T", "dTdt", "u", "v", "S", "eta")
    )
    lonc, latc, X, Y, LON, LAT = utils.calculate_grid_properties(lon, lat)
    dTds1o, dTds2o = utils.finite_difference_2d(X, Y, To)

    print("Dataset loaded and preprocessed")
    return {
        "ds": ds,
        "time_step": time_step,
        "lon": lon,
        "lat": lat,
        "To": To,
        "dTdto": dTdto,
        "u": u,
        "v": v,
        "S": S,
        "eta": eta,
        "lonc": lonc,
        "latc": latc,
        "X": X,
        "Y": Y,
        "LON": LON,
        "LAT": LAT,
        "dTds1o": dTds1o,
        "dTds2o": dTds2o,
    }


def run_gp_regression(
    dTds1o,
    dTds2o,
    dTdto,
    params_path,
    param_key,
    param_val,
    param_type,
    X,
    Y,
    time_step,
):
    print(
        f"Extracting parameters from {params_path} for {param_key}={param_val} ({param_type})"
    )
    params = utils.extract_params(params_path, param_key, param_val, type=param_type)
    print("Calculating GP regression predictions")
    muustar, muvstar, muSstar, stdustar, stdvstar, stdSstar, Kxstar_vel, Kxstar = (
        gp_regression.calculate_prediction_gpregression(
            dTds1o, dTds2o, dTdto, params, X, Y, time_step, return_Kxstar=True
        )
    )
    return (
        muustar,
        muvstar,
        muSstar,
        stdustar,
        stdvstar,
        stdSstar,
        Kxstar_vel,
        Kxstar,
    )


def run_global_optimal_solution(
    dTds1o, dTds2o, dTdto, params_path, param_key, param_val, param_type, u, v
):
    print(
        f"Extracting GOS parameters from {params_path} for {param_key}={param_val} ({param_type})"
    )
    params = utils.extract_params(params_path, param_key, param_val, type=param_type)
    n_gos = int(params["n"])
    print(f"Calculating Global Optimal Solution with n={n_gos}")
    ugos, vgos, Sgos = other_methods.calculate_prediction_gos(
        dTds1o, dTds2o, dTdto, n_gos
    )
    print("Calculating GOS metrics")
    metrics_gos = metrics.overview(u, v, ugos, vgos, print_bool=False)
    return ugos, vgos, Sgos, metrics_gos


def compute_time_series(time_step_label, add_samples):
    print(f"Computing time series for time step: {time_step_label}")

    time_base_str: str = "2014-02-19T18:00:00"
    time_base = np.datetime64(time_base_str)
    if time_step_label == "1h":
        time_base_str: str = "2014-02-19T18:00:00"
        time_base = np.datetime64(time_base_str)
        time_step_series = list(range(0, 172801, 3600))
        print("Every hour from 0 to 48 hours")
    elif time_step_label == "24h":
        time_step_series = list(range(0, 8640001, 86400))
        print("Every day from 0 to 100 days")
    else:
        raise ValueError(
            f"Invalid time_step_label: {time_step_label}. Must be '1h' or '24h'."
        )

    print(f"Generating samples for time steps: {add_samples}")

    n_samples = 100

    for idx, time_step in enumerate(time_step_series):
        time = time_base + np.timedelta64(time_step, "s")
        print(
            f"\nProcessing time step {idx + 1}/{len(time_step_series)}: {time} (time_step={time_step} seconds)"
        )

        data = load_and_prepare_dataset(
            f"1_preproc_data/proc_data/suntans_{time_step_label}.nc",
            "time",
            time,
        )

        muustar, muvstar, muSstar, _, _, stdSstar, Kxstar_vel, Kxstar = (
            run_gp_regression(
                data["dTds1o"],
                data["dTds2o"],
                data["dTdto"],
                f"2_covariance_parameter_estimation/outputs/time_{time_step_label}_gp_obs_t.csv",
                "time_sec",
                time_step,
                "gp",
                data["X"],
                data["Y"],
                data["time_step"],
            )
        )

        ugos, vgos, Sgos, _ = run_global_optimal_solution(
            data["dTds1o"],
            data["dTds2o"],
            data["dTdto"],
            f"2_covariance_parameter_estimation/outputs/time_{time_step_label}_gos_t.csv",
            "time_sec",
            time_step,
            "gos",
            data["u"],
            data["v"],
        )

        # make ds of all results that go in plot_and_save_predictions
        ds = xr.Dataset(
            {
                "To": (("time", "lat", "lon"), np.expand_dims(data["To"], 0)),
                "dTds1o": (("time", "lat", "lon"), np.expand_dims(data["dTds1o"], 0)),
                "dTds2o": (("time", "lat", "lon"), np.expand_dims(data["dTds2o"], 0)),
                "dTdto": (("time", "lat", "lon"), np.expand_dims(data["dTdto"], 0)),
                "muSstar": (("time", "lat", "lon"), np.expand_dims(muSstar, 0)),
                "Kxstar_vel": (
                    ("time", "lat", "lon", "n1", "n2"),
                    np.expand_dims(Kxstar_vel, 0),
                ),
                "stdSstar": (("time", "lat", "lon"), np.expand_dims(stdSstar, 0)),
                "muustar": (("time", "lat", "lon"), np.expand_dims(muustar, 0)),
                "muvstar": (("time", "lat", "lon"), np.expand_dims(muvstar, 0)),
                "u": (("time", "lat", "lon"), np.expand_dims(data["u"], 0)),
                "v": (("time", "lat", "lon"), np.expand_dims(data["v"], 0)),
                "S": (("time", "lat", "lon"), np.expand_dims(data["S"], 0)),
                "eta": (("time", "lat", "lon"), np.expand_dims(data["eta"], 0)),
                "ugos": (("time", "lat", "lon"), np.expand_dims(ugos, 0)),
                "vgos": (("time", "lat", "lon"), np.expand_dims(vgos, 0)),
                "Sgos": (("time", "lat", "lon"), np.expand_dims(Sgos, 0)),
            },
            coords={
                "LON": (("lat", "lon"), data["LON"]),
                "LAT": (("lat", "lon"), data["LAT"]),
                "time": (("time",), [time]),
            },
            attrs={
                "lonlims": lonlims,
                "latlims": latlims,
            },
        )

        if add_samples:
            print(f"Simulating {n_samples} velocity samples from GP")
            u_samples, v_samples = simulate_from_gp.simulate_velocity_samples(
                muustar, muvstar, Kxstar, n_samples=n_samples
            )

            ds["u_samples"] = (
                ("time", "sample", "lat", "lon"),
                np.expand_dims(u_samples, 0),
            )
            ds["v_samples"] = (
                ("time", "sample", "lat", "lon"),
                np.expand_dims(v_samples, 0),
            )

        ds.to_netcdf(
            f"3_observing_system_simulation_experiment/intermediate/osse_time_{time_step_label}_{idx}.nc"
        )

    print("Time series computation completed and results saved")

    pass


if __name__ == "__main__":
    print("--- computing time series for OSSE evaluation metrics ---")
    compute_time_series("1h", add_samples=True)
    compute_time_series("24h", add_samples=False)
    print("--- Time series computation completed ---")
