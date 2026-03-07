import numpy as np


def simulate_velocity_samples(muustar, muvstar, Kxstar, n_samples=100):
    # simulate velocity samples from the GP posterior using the mean and covariance

    # reshape mean and covariance to match the combined (u,v) vector
    ny, nx = muustar.shape
    n = ny * nx
    K = Kxstar[: 2 * n, : 2 * n]
    mean = np.hstack([muustar.ravel(), muvstar.ravel()])

    # Cholesky factor
    L = np.linalg.cholesky(K)

    # standard normal samples
    z = np.random.randn(n_samples, K.shape[0])

    # transform
    samples = mean + z @ L.T

    # reshape back to (n_samples, ny, nx)
    u_samples = samples[:, :n].reshape(n_samples, ny, nx)
    v_samples = samples[:, n : 2 * n].reshape(n_samples, ny, nx)
    return u_samples, v_samples


def simulate_source_samples(muSstar, Kxstar, n_samples=100):
    # simulate source samples from the GP posterior using the mean and covariance

    n = muSstar.size
    K = Kxstar[2 * n :, 2 * n :]
    mean = muSstar.ravel()

    # Cholesky factor
    L = np.linalg.cholesky(K)

    # standard normal samples
    z = np.random.randn(n_samples, K.shape[0])

    # transform
    samples = mean + z @ L.T

    # reshape back to (n_samples, ny, nx)
    S_samples = samples.reshape(n_samples, *muSstar.shape)
    return S_samples
