import xarray as xr
import numpy as np
import imageio.v2 as imageio
from pathlib import Path
import matplotlib
matplotlib.use("Agg")  # Prevent Tkinter backend

from matplotlib import rc
import matplotlib.pyplot as plt

from inversion_sst_gp import plot_helper

# Matplotlib configuration
rc("font", family="serif", serif=["Computer Modern"])
rc("text", usetex=True)
rc("text.latex", preamble=r"\usepackage{amsmath}")

# Helper functions
def plot_instance_of_series(
    ds_all,
    time,
    idx,
    time_step_label,
    base_time,
    pscale,
    nx,
    ny,
    plimTgrad=None,
    plimspeed=None,
    plimstdS=None,
    plimdTdt=None,
    plimT=None,
):
    # Extract variables into a dictionary
    data = {
        "LON": ds_all.LON.values,
        "LAT": ds_all.LAT.values,
        "To": ds_all.To.sel(time=time).values,
        "dTds1o": ds_all.dTds1o.sel(time=time).values,
        "dTds2o": ds_all.dTds2o.sel(time=time).values,
        "dTdto": ds_all.dTdto.sel(time=time).values,
        "muSstar": ds_all.muSstar.sel(time=time).values,
        "Kxstar_vel": ds_all.Kxstar_vel.sel(time=time).values,
        "stdSstar": ds_all.stdSstar.sel(time=time).values,
        "muustar": ds_all.muustar.sel(time=time).values,
        "muvstar": ds_all.muvstar.sel(time=time).values,
        "lonlims": ds_all.attrs["lonlims"],
        "latlims": ds_all.attrs["latlims"],
        "u": ds_all.u.sel(time=time).values,
        "v": ds_all.v.sel(time=time).values,
        "S": ds_all.S.sel(time=time).values,
        "ugos": ds_all.ugos.sel(time=time).values,
        "vgos": ds_all.vgos.sel(time=time).values,
        "Sgos": ds_all.Sgos.sel(time=time).values,
    }

    print(f"Plotting idx: {idx}, time: {np.datetime_as_string(time, unit='s')}")

    fig, ax = plot_helper.plot_predictions_osse(
        **data,
        pscale=pscale,
        nx=nx,
        ny=ny,
        plimTgrad=plimTgrad,
        plimspeed=plimspeed,
        plimstdS=plimstdS,
        plimdTdt=plimdTdt,
        plimT=plimT,
        return_fig=True,
    )
    text_str = f"{np.datetime_as_string(time, unit='s')}"
    time_diff = time - base_time
    if time_step_label == "1h":
        time_diff_hours = time_diff / np.timedelta64(1, "h")
        text_str += "\n" + f"$t=${time_diff_hours:.0f} h"
    elif time_step_label == "24h":
        time_diff_days = time_diff / np.timedelta64(1, "D")
        text_str += "\n" + f"$t=${time_diff_days:.0f} d"
    ax[0, 3].text(
        0.5,
        0.5,
        text_str,
        transform=ax[0, 3].transAxes,
        ha="center",
        va="center",
        fontsize=10,
        multialignment='left',
    )

    fig.savefig(
        f"3_observing_system_simulation_experiment/intermediate/osse_time_{time_step_label}_{idx}.png",
        bbox_inches="tight",
        dpi=200,
    )
    plt.close(fig)
    pass


def images_to_video(time_step_label,fps=2):
    png_dir = Path("3_observing_system_simulation_experiment/intermediate")
    output = f"3_observing_system_simulation_experiment/outputs/osse_time_step_{time_step_label}.mp4"

    frames = png_dir.glob(f"osse_time_{time_step_label}_*.png")
    
    # Get idx from filename and sort by idx
    frames = sorted(frames, key=lambda x: int(x.stem.split("_")[-1]))
    
    print(f"Found {len(frames)} frames for video creation")
    with imageio.get_writer(output, fps=fps) as writer:
        for frame in frames:
            writer.append_data(imageio.imread(frame))
    print(f"Video saved to {output}")
            

if __name__ == "__main__":
    print("--- Running time series to video script ---")
    time_step_label = "24h"

    ds_all = xr.open_mfdataset(
        f"3_observing_system_simulation_experiment/intermediate/osse_time_{time_step_label}_*.nc",
        combine="by_coords",
    )

    plot_config = {
        "pscale": 4,
        "nx": 17,
        "ny": 17,
        "plimTgrad": [0, 5e-5],
        "plimspeed": [0, 0.6],
        "plimstdS": [0, 2e-7],
        "plimdTdt": None,
        "plimT": None,
    }
    base_time = ds_all.time.values[0]

    for idx, time in enumerate(ds_all.time.values):
        plot_instance_of_series(
            ds_all, time, idx, time_step_label, base_time, **plot_config
        )
    print("Finished plotting time series\n")
        
    print("Creating video from images")
    images_to_video(time_step_label)
    print("Finished creating video")
    
    print("--- Finished time series to video script ---")