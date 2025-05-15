import numpy as np
from src.inversion_sst_gp.utils import finite_difference_2d


def simulate_tracer_observations_from_model_data(
    T, dTdt, tstep, X, Y, sigma_tau=None, coverage_sparse=None, coverage_dense=None
):
    T = T.copy()
    dTdt = dTdt.copy()

    N2, N1 = T.shape

    # Mask out any point where T or dTdt is NaN
    invalid_mask = np.isnan(T) | np.isnan(dTdt)
    T[invalid_mask] = np.nan
    dTdt[invalid_mask] = np.nan

    # Add Gaussian noise if specified
    if sigma_tau is not None:
        T += np.random.normal(0, sigma_tau, size=T.shape)
        dTdt += np.random.normal(0, (np.sqrt(0.5) / tstep) * sigma_tau, size=T.shape)

    # Simulate sparse coverage
    if coverage_sparse is not None:
        mask_sparse = np.random.rand(N2, N1) <= coverage_sparse
        T[mask_sparse] = np.nan
        dTdt[mask_sparse] = np.nan

    # Simulate dense coverage cloud at center
    if coverage_dense is not None:
        total_points = N1 * N2
        num_covered = int(total_points * coverage_dense)

        g2, g1 = np.ogrid[:N2, :N1]
        center2, center1 = N2 // 2, N1 // 2
        distances = np.sqrt((g1 - center1) ** 2 + (g2 - center2) ** 2)
        radius_threshold = np.sort(distances.ravel())[num_covered]
        mask_dense = distances <= radius_threshold

        T[mask_dense] = np.nan
        dTdt[mask_dense] = np.nan

    # Compute spatial gradients and final mask
    dTds1, dTds2 = finite_difference_2d(X, Y, T)
    maskc = np.isnan(dTds1) | np.isnan(dTds2) | np.isnan(dTdt)

    return T, dTds1, dTds2, dTdt, maskc
