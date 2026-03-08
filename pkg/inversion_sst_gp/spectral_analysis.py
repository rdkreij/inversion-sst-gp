import numpy as np
from scipy.special import gamma as gamma_func


def isotropic_psd_2d(x, spacing, nbins=50, hann=True, log_bins=False):
    """
    x : 2D array (ny, nx)
    spacing : grid spacing in meters (assumed equal in x & y)
    nbins : number of radial wavenumber bins
    hann : whether to apply a Hanning window before FFT
    log_bins : whether to use logarithmic bins for wavenumbers
    """
    ny, nx = x.shape

    # Remove mean
    x = x - np.mean(x)

    if hann:
        wy = np.hanning(ny)
        wx = np.hanning(nx)
        window = wy[:, None] * wx[None, :]
        x = x * window

        U = (window**2).sum() / (nx * ny)
    else:
        U = 1.0

    # 2D FFT
    fft2 = np.fft.fft2(x)
    # psd2 = np.abs(fft2) ** 2 * spacing**2 / (nx * ny)
    psd2 = (np.abs(fft2) ** 2) * spacing**2 / (nx * ny * U)

    # Wavenumbers
    kx = np.fft.fftfreq(nx, d=spacing)
    ky = np.fft.fftfreq(ny, d=spacing)
    kx, ky = np.meshgrid(kx, ky)
    k = np.sqrt(kx**2 + ky**2)

    # Flatten
    k = k.ravel()
    psd2 = psd2.ravel()

    kmax = 0.5 / spacing

    # Keep only isotropically valid region
    valid = (k > 0) & (k <= kmax)
    k = k[valid]
    psd2 = psd2[valid]

    # Radial bins
    if log_bins:
        k_bins = np.logspace(np.log10(k.min()), np.log10(kmax), nbins + 1)
        k_center = np.sqrt(k_bins[:-1] * k_bins[1:])
    else:
        k_bins = np.linspace(k.min(), kmax, nbins + 1)
        k_center = 0.5 * (k_bins[:-1] + k_bins[1:])

    psd_iso = np.zeros(nbins)

    for i in range(nbins):
        bin_mask = (k >= k_bins[i]) & (k < k_bins[i + 1])
        if np.any(bin_mask):
            psd_iso[i] = psd2[bin_mask].mean()
        else:
            psd_iso[i] = np.nan

    return k_center, psd_iso


def multi_isotropic_psd_2d(
    x, spacing, nbins=50, hann=True, return_all=False, log_bins=False
):
    """
    x : 3D array (nt, ny, nx)
    spacing : grid spacing in meters (assumed equal in x & y)
    nbins : number of radial wavenumber bins
    """
    nt = x.shape[0]

    k_center = np.array([])
    psd_iso_all = []

    for t in range(nt):
        k_center, psd_iso = isotropic_psd_2d(
            x[t, :, :], spacing, nbins, hann=hann, log_bins=log_bins
        )
        psd_iso_all.append(psd_iso)

    if return_all:
        return k_center, np.array(psd_iso_all)
    else:
        psd_iso_mean = np.nanmean(psd_iso_all, axis=0)

        return k_center, psd_iso_mean


def samples_psd_stats(
    data_samples, spacing, nbins, hann, log_bins=False, calc_in_log_space=True
):
    # compute mean and std of PSD across samples
    k, psd_samples = multi_isotropic_psd_2d(
        data_samples,
        spacing,
        nbins=nbins,
        hann=hann,
        return_all=True,
        log_bins=log_bins,
    )
    # allocate nan array
    psd_mean = np.full_like(k, np.nan)
    psd_2sigma_lower = np.full_like(k, np.nan)
    psd_2sigma_upper = np.full_like(k, np.nan)
    psd_1sigma_lower = np.full_like(k, np.nan)
    psd_1sigma_upper = np.full_like(k, np.nan)

    mask = ~np.isnan(psd_samples).any(axis=0)

    if calc_in_log_space:
        psd_samples_log = np.log(psd_samples[:, mask])
        psd_mean_log = np.nanmean(psd_samples_log, axis=0)
        psd_mean[mask] = np.exp(psd_mean_log)

        psd_sd_log = np.nanstd(psd_samples_log, axis=0)
        psd_2sigma_lower[mask] = np.exp(psd_mean_log - 2 * psd_sd_log)
        psd_2sigma_upper[mask] = np.exp(psd_mean_log + 2 * psd_sd_log)
        psd_1sigma_lower[mask] = np.exp(psd_mean_log - psd_sd_log)
        psd_1sigma_upper[mask] = np.exp(psd_mean_log + psd_sd_log)

    else:
        psd_mean[mask] = np.nanmean(psd_samples[:, mask], axis=0)
        psd_sd = np.nanstd(psd_samples[:, mask], axis=0)

        psd_2sigma_lower[mask] = psd_mean[mask] - 2 * psd_sd
        psd_2sigma_upper[mask] = psd_mean[mask] + 2 * psd_sd
        psd_1sigma_lower[mask] = psd_mean[mask] - psd_sd
        psd_1sigma_upper[mask] = psd_mean[mask] + psd_sd
    return (
        k,
        psd_mean,
        psd_1sigma_lower,
        psd_1sigma_upper,
        psd_2sigma_lower,
        psd_2sigma_upper,
    )


def calculate_theoretical_psd_matern_3_2(k, sigma=1.0, l=1.0, gamma=0.0):
    # power spectral density of the Matérn covariance with nu = 3/2
    prefactor = sigma**2 * 18 * np.pi * np.sqrt(3) / (l ** (3))
    return prefactor * (3 / l**2 + 4 * np.pi**2 * k**2) ** (-(5 / 2)) + gamma**2


def calculate_theoretical_psd_matern(f, sigma, l, nu, n, spacing, gamma=0.0):
    """
    Calculates the Power Spectral Density S(f) for a Matérn kernel.

    Parameters:
    f     : frequency (scalar or np.array)
    sigma : amplitude parameter
    l     : practical range/length scale parameter
    nu    : smoothness parameter
    n     : dimensionality of the space
    spacing : grid spacing (used to convert frequency to wavenumber)
    gamma   : noise variance (default 0.0)
    """
    # Numerator part of the constant fraction
    numerator = (
        (sigma**2) * (2**n) * (np.pi ** (n / 2)) * gamma_func(nu + n / 2) * ((2 * nu) ** nu)
    )

    # Denominator part of the constant fraction
    denominator = gamma_func(nu) * (l ** (2 * nu))

    # The frequency-dependent term
    freq_term = ((2 * nu) / (l**2) + 4 * (np.pi**2) * (f**2)) ** (-(nu + n / 2))

    return (numerator / denominator) * freq_term + gamma**2 * spacing**n
