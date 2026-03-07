import xarray as xr
from matplotlib import rc

from inversion_sst_gp import plot_helper

# Matplotlib configuration
rc("font", family="serif", serif=["Computer Modern"])
rc("text", usetex=True)
rc("text.latex", preamble=r"\usepackage{amsmath}")

# Helper functions
def load_and_plot_point_time_series(time_step_label, lon_idx, lat_idx):
    print(f"Selected time step: {time_step_label} at point (lon_idx={lon_idx}, lat_idx={lat_idx})")
    
    # Load dataset
    print("Loading time series dataset")
    ds_all = xr.open_mfdataset(f"3_observing_system_simulation_experiment/intermediate/osse_time_{time_step_label}_*.nc", combine="by_coords")
    ds_point = ds_all.isel(lon=lon_idx, lat=lat_idx)
    
    # Print point location
    lon_point = ds_all.LON.values[lat_idx, lon_idx]
    lat_point = ds_all.LAT.values[lat_idx, lon_idx]
    print(f"Point location: lon={lon_point:.1f}, lat={lat_point:.1f}")

    # Plot time series for point
    print("Plotting time series for selected point")
    fig, _ = plot_helper.plot_time_series_point(ds_point)
    fig.savefig(f"3_observing_system_simulation_experiment/outputs/osse_point_time_step_{time_step_label}.png", bbox_inches="tight", dpi=300)
    pass
    
if __name__ == "__main__":
    print('--- Generating point time series figure for OSSE ---')
    load_and_plot_point_time_series("1h", lon_idx=25, lat_idx=25)
    print("--- Point time series figure generated and saved ---")