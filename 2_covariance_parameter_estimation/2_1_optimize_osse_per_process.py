import xarray as xr
import numpy as np
import pandas as pd

from inversion_sst_gp import (
    utils,
    gp_regression,
)

# Load dataset
time_str = "2014-02-19T18:00:00"
ds = xr.open_dataset("data/suntans_1h.nc").sel(time=np.datetime64(time_str))
time_step = ds.time_step.item()
lon, lat, To, dTdto, u, v, S = (
    ds[var].values for var in ("lon", "lat", "T", "dTdt", "u", "v", "S")
)
lonc, latc, X, Y, LON, LAT = utils.calculate_grid_properties(lon, lat)

# GPRM model - optimization of theta
sigma_u, l_u, tau_u = gp_regression.estimate_params_process(u, X, Y,1e-1,4e4,1e-3)
sigma_v, l_v, tau_v = gp_regression.estimate_params_process(v, X, Y,1e-1,4e4,1e-3)
sigma_S, l_S, tau_S = gp_regression.estimate_params_process(S, X, Y,3e-7,3e4,2e-7)

# collect hyperparameters
theta = {'sigma_u':sigma_u, 'l_u':l_u, 'tau_u':tau_u,
         'sigma_v':sigma_v, 'l_v':l_v, 'tau_v':tau_v,
         'sigma_S':sigma_S, 'l_S':l_S, 'tau_S':tau_S,
}

# Add time field
data = theta.copy()
data['time'] = time_str

# Save to CSV
df = pd.DataFrame([data])
df.to_csv('outputs/num_model_estimated_t.csv', index=False)
