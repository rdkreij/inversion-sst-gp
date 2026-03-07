import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import xarray as xr
from inversion_sst_gp import (
    gp_regression,
    metrics,
    other_methods,
    plot_helper,
    simulate_from_gp,
    spectral_analysis,
    utils,
)
from matplotlib import rc
from matplotlib.ticker import FuncFormatter, LogLocator

# Matplotlib configuration
rc("font", family="serif", serif=["Computer Modern"])
rc("text", usetex=True)
rc("text.latex", preamble=r"\usepackage{amsmath}")

# Plotting parameters
LON_LIMITS = (115, 118)
LAT_LIMITS = (-15.5, -12.5)


# Helper functions
def load_and_prepare_dataset(path, selection_key=None, selection_val=None):
    print(f"Loading dataset: {path}")
    ds = xr.open_dataset(path)
    if selection_key and selection_val is not None:
        ds = ds.sel({selection_key: selection_val})
        print(f"Selecting dataset with {selection_key}={selection_val}")
    time_step = ds.time_step.item()
    lon, lat, To, dTdto, u, v, S = (
        ds[var].values for var in ("lon", "lat", "T", "dTdt", "u", "v", "S")
    )
    lonc, latc, X, Y, LON, LAT = utils.calculate_grid_properties(lon, lat)
    dTds1o, dTds2o = utils.finite_difference_2d(X, Y, To)
    plot_helper.visualize_data(
        LON, LAT, To, dTdto, dTds1o, dTds2o, lonlims=LON_LIMITS, latlims=LAT_LIMITS
    )
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
        "lonc": lonc,
        "latc": latc,
        "X": X,
        "Y": Y,
        "LON": LON,
        "LAT": LAT,
        "dTds1o": dTds1o,
        "dTds2o": dTds2o,
    }


def run_gp_regression_and_metrics(
    dTds1o,
    dTds2o,
    dTdto,
    params,
    X,
    Y,
    time_step,
    u,
    v,
):
    print("Calculating GP regression predictions")
    muustar, muvstar, muSstar, stdustar, stdvstar, stdSstar, Kxstar_vel, Kxstar = (
        gp_regression.calculate_prediction_gpregression(
            dTds1o, dTds2o, dTdto, params, X, Y, time_step, return_Kxstar=True
        )
    )
    print("Calculating GP metrics")
    metrics_gp = metrics.overview(
        u, v, muustar, muvstar, stdustar, stdvstar, print_bool=False
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
        metrics_gp,
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


def compute_theoretical_psd(
    kmin,
    kmax,
    sigma,
    l,
    tau,
    spacing,
    log_bins=False,
    nbins=200,
    nu=3 / 2,
    n=2,
):
    if log_bins:
        k = np.logspace(
            np.log10(kmin),
            np.log10(kmax),
            nbins,
        )
    else:
        k = np.linspace(
            kmin,
            kmax,
            nbins,
        )

    S = spectral_analysis.calculate_theoretical_psd_matern(
        k, sigma=sigma, l=l, nu=nu, n=n, spacing=spacing, tau=tau
    )
    return k, S


def simulate_samples(Kxstar, muustar, muvstar, muSstar, parameter, n_samples):
    if parameter == "u" or parameter == "v":
        u_samples, v_samples = simulate_from_gp.simulate_velocity_samples(
            muustar, muvstar, Kxstar, n_samples=n_samples
        )
        if parameter == "u":
            data_samples = u_samples
        else:
            data_samples = v_samples

    elif parameter == "S":
        S_samples = simulate_from_gp.simulate_source_samples(
            muSstar, Kxstar, n_samples=n_samples
        )
        data_samples = S_samples
    else:
        raise ValueError("Invalid parameter choice. Must be 'u', 'v', or 'S'.")
    return data_samples


def compute_psd_dict_all(
    data_model,
    data_gos,
    data_samples,
    spacing,
    nbins,
    hann,
    log_bins,
    sigma,
    l,
    tau,
    kmin_theory=None,
    kmax_theory=None,
    nbins_theory=500,
    nu=3 / 2,
):
    psd_dict = {}

    # Single-field PSDs (k identical for all)
    for key, field in zip(
        ["model", "gos"],
        [data_model, data_gos],
    ):
        psd_dict[key] = spectral_analysis.isotropic_psd_2d(
            field,
            spacing,
            nbins=nbins,
            hann=hann,
            log_bins=log_bins,
        )

    # Samples PSDs
    (
        k_s,
        psd_mean,
        psd_1sigma_lower,
        psd_1sigma_upper,
        psd_2sigma_lower,
        psd_2sigma_upper,
    ) = spectral_analysis.samples_psd_stats(
        data_samples,
        spacing,
        nbins,
        hann,
        log_bins,
    )
    psd_dict["samples_mean"] = (k_s, psd_mean)
    psd_dict["samples_1sigma_lower"] = (k_s, psd_1sigma_lower)
    psd_dict["samples_1sigma_upper"] = (k_s, psd_1sigma_upper)
    psd_dict["samples_2sigma_lower"] = (k_s, psd_2sigma_lower)
    psd_dict["samples_2sigma_upper"] = (k_s, psd_2sigma_upper)

    # Compute theoretical PSD based on model parameters
    # Use the k range from the model PSD
    k_model, psd_model = psd_dict["model"]
    kmin_model = k_model[~np.isnan(psd_model)].min()
    kmax_model = k_model[~np.isnan(psd_model)].max()

    if kmin_theory is None:
        kmin_theory = kmin_model
    if kmax_theory is None:
        kmax_theory = kmax_model

    k_theory_full, psd_theory_full = compute_theoretical_psd(
        kmin=kmin_theory,
        kmax=kmax_theory,
        sigma=sigma,
        l=l,
        tau=tau,
        log_bins=True,
        spacing=spacing,
        nbins=nbins_theory,
        nu=nu,
    )

    k_theory_no_noise, psd_theory_no_noise = compute_theoretical_psd(
        kmin=kmin_theory,
        kmax=kmax_theory,
        sigma=sigma,
        l=l,
        tau=0,
        log_bins=True,
        spacing=spacing,
        nbins=nbins_theory,
        nu=nu,
    )

    psd_dict["theory"] = (k_theory_full, psd_theory_full)
    psd_dict["theory_no_noise"] = (k_theory_no_noise, psd_theory_no_noise)
    return psd_dict, kmin_model, kmax_model


def plot_psd_comparison(
    psd_dict_num,
    psd_dict_obs,
    kmin_num,
    kmax_num,
    kmin_obs,
    kmax_obs,
    parameter,
    spacing,
    filename,
):
    fig, axs = plt.subplots(
        1, 2, figsize=(10, 3.5), sharey=True, gridspec_kw={"wspace": 0}
    )

    for i, ax in enumerate(axs):
        if i == 0:
            psd_dict = psd_dict_num
            kmin = kmin_num
            kmax = kmax_num
        else:
            psd_dict = psd_dict_obs
            kmin = kmin_obs
            kmax = kmax_obs

        plot_helper.plot_psd_overview(ax, psd_dict, kmin, kmax, parameter)

        if i == 0:
            if parameter == "u":
                ax.set_ylabel(r"PSD $u$ (m$^2\,$s$^{-2}\,$cpm$^{-1}$)")
            elif parameter == "v":
                ax.set_ylabel(r"PSD $v$ (m$^2\,$s$^{-2}\,$cpm$^{-1}$)")
            elif parameter == "S":
                ax.set_ylabel(r"PSD $S$ (K$^2\,$s$^{-2}\,$cpm$^{-1}$)")

        if parameter == "u" or parameter == "v":
            ax.set_ylim(1e2, 9e7)
            reference_points = utils.calculate_exponential_line(
                k_0=4e-5, f_0=3e5, k_1=1e-4, slope=-5
            )
            ax.plot(*reference_points, "--", color="k", lw=1)[0].set_dashes([5, 5])
            ax.text(5e-5, 0.2e6, r"$k^{-5}$", fontsize=10, color="k")
        elif parameter == "S":
            ax.set_ylim(1e-9, 2e-4)
            reference_points = utils.calculate_exponential_line(
                k_0=1.5e-5, f_0=2e-4, k_1=1e-4, slope=-5
            )
            ax.plot(*reference_points, "--", color="k", lw=1)[0].set_dashes([5, 5])
            ax.text(3.5e-5, 1.3e-5, r"$k^{-5}$", fontsize=10, color="k")

        # vertical line at 1/spacing and nyquist wavenumber
        ax.axvline(0.5 / spacing, color="k", lw=1, ls=":")
        ylims = ax.get_ylim()
        ax.text(
            0.5 / spacing * 0.95,
            10 ** (0.92 * np.log10(ylims[1] / ylims[0])) * ylims[0],
            r"$k_{Ny}$",
            fontsize=10,
            color="k",
            ha="right",
        )

        if i == 0:
            ax.legend(frameon=False, fontsize=8)

        xlims = ax.get_xlim()
        ax.set_xlim(xlims[0], 0.6 / spacing)

        # Major ticks at powers of 10
        ax.xaxis.set_major_locator(LogLocator(base=10))

        # Minor ticks at 2–9 in each decade
        ax.xaxis.set_minor_locator(LogLocator(base=10, subs=np.arange(2, 10) * 0.1))

        # Label only 2 and 5 in each decade
        def minor_log_formatter(val, pos):
            if val <= 0:
                return ""
            exponent = np.floor(np.log10(val))
            mantissa = val / 10**exponent

            if np.isclose(mantissa, 2) or np.isclose(mantissa, 5):
                return r"$%g$$\times$$10^{%d}$" % (mantissa, exponent)
            return ""

        ax.xaxis.set_minor_formatter(FuncFormatter(minor_log_formatter))

        # Make sure minor labels are shown
        ax.tick_params(axis="x", which="minor", labelsize=8)

        ax.annotate(
            "GP numerical-model\nestimated parameters"
            if i == 0
            else "GP observation-\nestimated parameters",
            xy=(0.4, 0.96),
            xycoords="axes fraction",
            va="top",
            ha="left",
            fontsize=12,
        )
        ax.annotate(
            "a)\n" if i == 0 else "b)\n",
            xy=(0.39, 0.97),
            xycoords="axes fraction",
            va="top",
            ha="right",
            fontsize=12,
        )

        # Create twin axis on top
        ax_top = ax.secondary_xaxis("top")
        ax_top.set_xlabel(r"$1/k$ (km)")
        xticks_top = [1e-5]
        xticks_top_labels = [r"$100$"]
        ax_top.set_xticks(xticks_top)
        ax_top.set_xticklabels(xticks_top_labels)

        # same for minor ticks
        ticks_top_minor = [5e-6, 2e-5, 5e-5]
        ticks_top_minor_labels = [r"$200$", r"$50$", r"$20$"]
        ax_top.set_xticks(ticks_top_minor, minor=True)
        ax_top.set_xticklabels(ticks_top_minor_labels, minor=True)
        ax_top.tick_params(axis="x", which="major", length=0)
        ax_top.tick_params(axis="x", which="minor", length=0)
    fig.savefig(filename, bbox_inches="tight", dpi=300)


def plot_and_save_predictions(
    LON,
    LAT,
    To,
    dTds1o,
    dTds2o,
    dTdto,
    muSstar,
    Kxstar_vel,
    stdSstar,
    muustar,
    muvstar,
    LON_LIMITS,
    LAT_LIMITS,
    u=None,
    v=None,
    S=None,
    ugos=None,
    vgos=None,
    Sgos=None,
    filename=None,
):
    print(f"Plotting predictions and saving to {filename}")
    fig, ax = plot_helper.plot_predictions_osse(
        LON,
        LAT,
        To,
        dTds1o,
        dTds2o,
        dTdto,
        muSstar,
        Kxstar_vel,
        stdSstar,
        muustar,
        muvstar,
        LON_LIMITS,
        LAT_LIMITS,
        u=u,
        v=v,
        S=S,
        ugos=ugos,
        vgos=vgos,
        Sgos=Sgos,
        pscale=4,
        nx=17,
        ny=17,
        plimdTdt=[-2.4e-6, -0.4e-6],
        return_fig=True,
    )
    fig.savefig(filename, bbox_inches="tight", dpi=300)


def store_transect_data(lon, To, v, muvstar, stdvstar):
    print("Storing transect data")
    return {
        "lon": lon,
        "maskc": np.isnan(To[25, :]),
        "v": v[25, :],
        "muvstar": muvstar[25, :],
        "stdvstar": stdvstar[25, :],
    }


def experiment_fully_observed():
    print("\n--- Running Fully Observed Noiseless Experiment ---")
    data = load_and_prepare_dataset(
        "1_preproc_data/proc_data/suntans_1h.nc",
        "time",
        np.datetime64("2014-02-19T18:00:00"),
    )

    params_path_obs = "2_covariance_parameter_estimation/outputs/noise_gp_obs_t.csv"
    params_path_num = "2_covariance_parameter_estimation/outputs/noise_gp_num_t.csv"

    param_key = "noise_sd"
    param_val = 0
    param_type = "gp"
    print(
        f"\nExtracting obs-parameters from {params_path_obs} for {param_key}={param_val} ({param_type})"
    )
    params_obs = utils.extract_params(
        params_path_obs, param_key, param_val, type=param_type
    )

    print("Compute predictions with GP obs-parameters")
    (
        muustar,
        muvstar,
        muSstar,
        _,
        stdvstar,
        stdSstar,
        Kxstar_vel,
        Kxstar,
        metrics_gp,
    ) = run_gp_regression_and_metrics(
        data["dTds1o"],
        data["dTds2o"],
        data["dTdto"],
        params_obs,
        data["X"],
        data["Y"],
        data["time_step"],
        data["u"],
        data["v"],
    )

    print(
        f"\nExtracting num-parameters from {params_path_num} for {param_key}={param_val} ({param_type})"
    )
    params_num = utils.extract_params(
        params_path_num, param_key, param_val, type=param_type
    )
    print("Compute predictions with GP num-parameters")
    (
        muustar_num,
        muvstar_num,
        muSstar_num,
        _,
        _,
        _,
        _,
        Kxstar_num,
        _,
    ) = run_gp_regression_and_metrics(
        data["dTds1o"],
        data["dTds2o"],
        data["dTdto"],
        params_num,
        data["X"],
        data["Y"],
        data["time_step"],
        data["u"],
        data["v"],
    )

    print("\nCompute predictions with GOS")
    ugos, vgos, Sgos, metrics_gos = run_global_optimal_solution(
        data["dTds1o"],
        data["dTds2o"],
        data["dTdto"],
        "2_covariance_parameter_estimation/outputs/noise_gos_t.csv",
        "noise_sd",
        0,
        "gos",
        data["u"],
        data["v"],
    )

    print("\nPlot and save predictions")
    plot_and_save_predictions(
        data["LON"],
        data["LAT"],
        data["To"],
        data["dTds1o"],
        data["dTds2o"],
        data["dTdto"],
        muSstar,
        Kxstar_vel,
        stdSstar,
        muustar,
        muvstar,
        LON_LIMITS,
        LAT_LIMITS,
        u=data["u"],
        v=data["v"],
        S=data["S"],
        ugos=ugos,
        vgos=vgos,
        Sgos=Sgos,
        filename="3_observing_system_simulation_experiment/outputs/osse_instance_fully_observed.png",
    )
    transect = store_transect_data(
        data["lon"], data["To"], data["v"], muvstar, stdvstar
    )

    print("\n*** Start fully observed PSD analysis ***")

    # Config for PSD computation
    parameter = "u"
    spacing = 6600  # meters
    hann = True
    log_bins = False
    nbins = 60
    n_samples = 100
    print(
        f"PSD config: parameter={parameter}, spacing={spacing}, hann={hann}, log_bins={log_bins}, nbins={nbins}, n_samples={n_samples}"
    )

    # Choose the corresponding GOS data for the parameter of interest
    data_gos = vgos if parameter == "v" else ugos if parameter == "u" else Sgos

    print(
        "Simulating velocity samples from GP posterior with both obs-parameters and num-parameters"
    )

    # Simulate samples for both sets of parameters
    print("Simulating samples with GP num-parameters")
    data_samples_num = simulate_samples(
        Kxstar_num, muustar_num, muvstar_num, muSstar_num, parameter, n_samples
    )

    print("Computing PSDs for numerical-parameter samples")
    psd_dict_num, kmin_num, kmax_num = compute_psd_dict_all(
        data_model=data[parameter],
        data_gos=data_gos,
        data_samples=data_samples_num,
        spacing=spacing,
        nbins=nbins,
        hann=hann,
        log_bins=log_bins,
        sigma=params_num[f"sigma_{parameter}"],
        l=params_num[f"l_{parameter}"],
        tau=params_num[f"tau_{parameter}"],
    )

    print("Simulating samples with GP obs-parameters")
    data_samples_obs = simulate_samples(
        Kxstar, muustar, muvstar, muSstar, parameter, n_samples
    )

    print("Computing PSDs for observation-parameter samples")
    psd_dict_obs, kmin_obs, kmax_obs = compute_psd_dict_all(
        data_model=data[parameter],
        data_gos=data_gos,
        data_samples=data_samples_obs,
        spacing=spacing,
        nbins=nbins,
        hann=hann,
        log_bins=log_bins,
        sigma=params_obs[f"sigma_{parameter}"],
        l=params_obs[f"l_{parameter}"],
        tau=params_obs[f"tau_{parameter}"],
    )

    print("Plotting PSD comparison and saving figure")
    plot_psd_comparison(
        psd_dict_num,
        psd_dict_obs,
        kmin_num,
        kmax_num,
        kmin_obs,
        kmax_obs,
        parameter,
        spacing,
        filename="3_observing_system_simulation_experiment/outputs/osse_instance_fully_observed_psd.png",
    )

    return transect, metrics_gp, metrics_gos


def experiment_measurement_error(noise=0.005):
    print(f"\n--- Running Measurement Error Experiment (noise={noise}) ---")
    data = load_and_prepare_dataset(
        "1_preproc_data/proc_data/suntans_measurement_error.nc", "sigma_tau", noise
    )

    params_path = "2_covariance_parameter_estimation/outputs/noise_gp_obs_t.csv"
    param_key = "noise_sd"
    param_val = noise
    param_type = "gp"
    print(
        f"Extracting parameters from {params_path} for {param_key}={param_val} ({param_type})"
    )
    params = utils.extract_params(params_path, param_key, param_val, type=param_type)

    (
        muustar,
        muvstar,
        muSstar,
        _,
        stdvstar,
        stdSstar,
        Kxstar_vel,
        _,
        metrics_gp,
    ) = run_gp_regression_and_metrics(
        data["dTds1o"],
        data["dTds2o"],
        data["dTdto"],
        params,
        data["X"],
        data["Y"],
        data["time_step"],
        data["u"],
        data["v"],
    )
    ugos, vgos, Sgos, metrics_gos = run_global_optimal_solution(
        data["dTds1o"],
        data["dTds2o"],
        data["dTdto"],
        "2_covariance_parameter_estimation/outputs/noise_gos_t.csv",
        "noise_sd",
        noise,
        "gos",
        data["u"],
        data["v"],
    )
    plot_and_save_predictions(
        data["LON"],
        data["LAT"],
        data["To"],
        data["dTds1o"],
        data["dTds2o"],
        data["dTdto"],
        muSstar,
        Kxstar_vel,
        stdSstar,
        muustar,
        muvstar,
        LON_LIMITS,
        LAT_LIMITS,
        u=data["u"],
        v=data["v"],
        S=data["S"],
        ugos=ugos,
        vgos=vgos,
        Sgos=Sgos,
        filename="3_observing_system_simulation_experiment/outputs/osse_instance_noise.png",
    )
    transect = store_transect_data(
        data["lon"], data["To"], data["v"], muvstar, stdvstar
    )
    return transect, metrics_gp, metrics_gos


def experiment_dense_cloud(coverage_dense=0.3):
    print(f"\n--- Running Dense Cloud Experiment (coverage_dense={coverage_dense}) ---")
    data = load_and_prepare_dataset(
        "1_preproc_data/proc_data/suntans_dense_cloud.nc",
        "coverage_dense",
        coverage_dense,
    )
    params_path = "2_covariance_parameter_estimation/outputs/cloud_dense_gp_obs_t.csv"
    param_key = "coverage_dense"
    param_val = coverage_dense
    param_type = "gp"
    print(
        f"Extracting parameters from {params_path} for {param_key}={param_val} ({param_type})"
    )
    params = utils.extract_params(params_path, param_key, param_val, type=param_type)
    (
        muustar,
        muvstar,
        muSstar,
        _,
        stdvstar,
        stdSstar,
        Kxstar_vel,
        _,
        metrics_gp,
    ) = run_gp_regression_and_metrics(
        data["dTds1o"],
        data["dTds2o"],
        data["dTdto"],
        params,
        data["X"],
        data["Y"],
        data["time_step"],
        data["u"],
        data["v"],
    )
    plot_and_save_predictions(
        data["LON"],
        data["LAT"],
        data["To"],
        data["dTds1o"],
        data["dTds2o"],
        data["dTdto"],
        muSstar,
        Kxstar_vel,
        stdSstar,
        muustar,
        muvstar,
        LON_LIMITS,
        LAT_LIMITS,
        u=data["u"],
        v=data["v"],
        S=data["S"],
        filename="3_observing_system_simulation_experiment/outputs/osse_instance_dense_cloud.png",
    )
    transect = store_transect_data(
        data["lon"], data["To"], data["v"], muvstar, stdvstar
    )
    return transect, metrics_gp


def experiment_sparse_cloud(coverage_sparse=0.3):
    print(
        f"\n--- Running Sparse Cloud Experiment (coverage_sparse={coverage_sparse}) ---"
    )
    data = load_and_prepare_dataset(
        "1_preproc_data/proc_data/suntans_sparse_cloud.nc",
        "coverage_sparse",
        coverage_sparse,
    )
    params_path = "2_covariance_parameter_estimation/outputs/cloud_sparse_gp_obs_t.csv"
    param_key = "coverage_sparse"
    param_val = coverage_sparse
    param_type = "gp"
    print(
        f"Extracting parameters from {params_path} for {param_key}={param_val} ({param_type})"
    )
    params = utils.extract_params(params_path, param_key, param_val, type=param_type)

    (
        muustar,
        muvstar,
        muSstar,
        _,
        stdvstar,
        stdSstar,
        Kxstar_vel,
        _,
        metrics_gp,
    ) = run_gp_regression_and_metrics(
        data["dTds1o"],
        data["dTds2o"],
        data["dTdto"],
        params,
        data["X"],
        data["Y"],
        data["time_step"],
        data["u"],
        data["v"],
    )
    plot_and_save_predictions(
        data["LON"],
        data["LAT"],
        data["To"],
        data["dTds1o"],
        data["dTds2o"],
        data["dTdto"],
        muSstar,
        Kxstar_vel,
        stdSstar,
        muustar,
        muvstar,
        LON_LIMITS,
        LAT_LIMITS,
        u=data["u"],
        v=data["v"],
        S=data["S"],
        filename="3_observing_system_simulation_experiment/outputs/osse_instance_sparse_cloud.png",
    )
    transect = store_transect_data(
        data["lon"], data["To"], data["v"], muvstar, stdvstar
    )
    return transect, metrics_gp


def plot_and_save_transects(
    transect_fully_observed,
    transect_measurement_error,
    transect_dense_cloud,
    transect_sparse_cloud,
):
    print("\n--- Plotting transects overview ---")
    fig, ax = plot_helper.plot_transects(
        transect_fully_observed,
        transect_measurement_error,
        transect_dense_cloud,
        transect_sparse_cloud,
        LON_LIMITS,
        [-0.35, 0.25],
        return_fig=True,
    )
    file_name = "3_observing_system_simulation_experiment/outputs/osse_instance_overview_transect.png"
    print(f"Saving transects overview to {file_name}")
    fig.savefig(
        file_name,
        bbox_inches="tight",
        dpi=300,
    )


def make_metric_overview_gp(
    metrics_gp_fully_observed,
    metrics_gos_fully_observed,
    metrics_gp_measurement_error,
    metrics_gos_measurement_error,
    metrics_gp_dense_cloud,
    metrics_gp_sparse_cloud,
):
    print("\n--- Creating metric overview ---")
    overview = []
    overview.append(
        ["Experiment", "GP RMSE (m/s)", "GP coverage90 (%)", "GOS RMSE (m/s)"]
    )
    overview.append(
        [
            "Fully observed",
            "{:.4e}".format(metrics_gp_fully_observed["RMSE"]),
            "{:.4f}".format(metrics_gp_fully_observed["coverage90"] * 100),
            "{:.4e}".format(metrics_gos_fully_observed["RMSE"]),
        ]
    )
    overview.append(
        [
            "Measurement error",
            "{:.4e}".format(metrics_gp_measurement_error["RMSE"]),
            "{:.4f}".format(metrics_gp_measurement_error["coverage90"] * 100),
            "{:.4e}".format(metrics_gos_measurement_error["RMSE"]),
        ]
    )
    overview.append(
        [
            "Dense cloud",
            "{:.4e}".format(metrics_gp_dense_cloud["RMSE"]),
            "{:.4f}".format(metrics_gp_dense_cloud["coverage90"] * 100),
            "-",
        ]
    )
    overview.append(
        [
            "Sparse cloud",
            "{:.4e}".format(metrics_gp_sparse_cloud["RMSE"]),
            "{:.4f}".format(metrics_gp_sparse_cloud["coverage90"] * 100),
            "-",
        ]
    )

    df_overview = pd.DataFrame(overview[1:], columns=overview[0])
    file_name = "3_observing_system_simulation_experiment/outputs/osse_instance_overview_metrics.csv"
    print(f"Saving metric overview to {file_name}")
    df_overview.to_csv(file_name, index=False)


# Main processing function
def main():
    print("--- Running individual cases for OSSE ---")
    transect_fully_observed, metrics_gp_fully_observed, metrics_gos_fully_observed = (
        experiment_fully_observed()
    )
    (
        transect_measurement_error,
        metrics_gp_measurement_error,
        metrics_gos_measurement_error,
    ) = experiment_measurement_error(noise=0.005)
    transect_dense_cloud, metrics_gp_dense_cloud = experiment_dense_cloud(
        coverage_dense=0.3
    )
    transect_sparse_cloud, metrics_gp_sparse_cloud = experiment_sparse_cloud(
        coverage_sparse=0.3
    )

    plot_and_save_transects(
        transect_fully_observed,
        transect_measurement_error,
        transect_dense_cloud,
        transect_sparse_cloud,
    )

    make_metric_overview_gp(
        metrics_gp_fully_observed,
        metrics_gos_fully_observed,
        metrics_gp_measurement_error,
        metrics_gos_measurement_error,
        metrics_gp_dense_cloud,
        metrics_gp_sparse_cloud,
    )

    print("\nAll experiments completed and figures saved.")


if __name__ == "__main__":
    main()
