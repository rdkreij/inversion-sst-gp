import matplotlib.pyplot as plt
import numpy as np
import xarray as xr
from inversion_sst_gp import plot_helper, utils
from matplotlib import rc

# Matplotlib configuration
rc("font", family="serif", serif=["Computer Modern"])
rc("text", usetex=True)
rc("text.latex", preamble=r"\usepackage{amsmath}")

# Plotting parameters
lonlims = (115, 118)
latlims = (-15.5, -12.5)


# Helper functions
def compute_mean_geostrophic_velocity_fields(ds_all):
    nt = len(ds_all.time)
    nx = len(ds_all.lon)
    ny = len(ds_all.lat)

    ug = np.empty((nt, ny, nx))
    vg = np.empty((nt, ny, nx))

    _, _, X, Y, _, LAT = utils.calculate_grid_properties(
        ds_all.LON.values[0, :], ds_all.LAT.values[:, 0]
    )

    for i in range(nt):
        eat_i = ds_all.eta.isel(time=i).values
        ug_i, vg_i = utils.calculate_geostrophic_velocity_fields(X, Y, LAT, eat_i)
        ug[i, :, :] = ug_i
        vg[i, :, :] = vg_i

    ug_mean = np.mean(ug, axis=0)
    vg_mean = np.mean(vg, axis=0)
    return ug_mean, vg_mean


def load_and_plot_spatial_time_series(time_step_label):
    print(f"Selected time step: {time_step_label}")

    # Load dataset
    print("Loading time series dataset")
    ds_all = xr.open_mfdataset(
        f"3_observing_system_simulation_experiment/intermediate/osse_time_{time_step_label}_*.nc",
        combine="by_coords",
    )

    # Compute mean geostrophic velocity fields over time
    print("Compute geostrophic velocity fields and statistics")
    ug_mean, vg_mean = compute_mean_geostrophic_velocity_fields(ds_all)

    # Compute std u and v over time
    print("Computing standard deviation of velocity fields over time")
    u_std = ds_all.u.std(dim="time").values
    v_std = ds_all.v.std(dim="time").values

    # Compute mean u_sample and v_sample over time and samples
    print("Computing mean of GP samples over time and samples")
    u_sample_mean = ds_all.u_samples.mean(dim=["time", "sample"]).values
    v_sample_mean = ds_all.v_samples.mean(dim=["time", "sample"]).values

    # Compute std u_sample and v_sample over time and samples
    print("Computing standard deviation of GP samples over time and samples")
    u_sample_std = ds_all.u_samples.std(dim=["time", "sample"]).values
    v_sample_std = ds_all.v_samples.std(dim=["time", "sample"]).values

    # Plot time series for point
    print("Plotting spatial overview")
    fig, _ = plot_helper.plot_spatial_time_series(
        ds_all.LON.values,
        ds_all.LAT.values,
        ug_mean,
        vg_mean,
        u_sample_mean,
        v_sample_mean,
        u_sample_std,
        v_sample_std,
        u_std,
        v_std,
        lonlims,
        latlims,
        plimspeed=None,
        plimvar=(0, 0.05),
    )
    fig.savefig(
        f"3_observing_system_simulation_experiment/outputs/osse_spatial_time_step_{time_step_label}.png",
        bbox_inches="tight",
        dpi=300,
    )
    plt.close(fig)
    pass


# Main processing function
def main():
    print("--- Generating time series spatial overview figure for OSSE ---")
    load_and_plot_spatial_time_series("1h")
    print("\nTime series spatial overview figure for OSSE generated")


if __name__ == "__main__":
    main()
