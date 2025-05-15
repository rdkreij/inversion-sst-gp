from scipy.linalg.lapack import dpotrf, dpotri
import numpy as np
from numpy.linalg import multi_dot as mdot
from scipy.linalg import block_diag
from scipy.optimize import minimize, shgo
from scipy.spatial import distance_matrix

from src.inversion_sst_gp.utils import map_val


class GPRegression(object):
    # Gaussian Process Regression Model

    def __init__(
        self, dTds1o, dTds2o, dTdto, tstep, X, Y, maskp, degreeu=2, degreev=2, degreeS=2
    ):        
        self.dTds1o = dTds1o
        self.dTds2o = dTds2o
        self.dTdto = dTdto
        self.maskp = maskp
        self.tstep = tstep

        # mask
        self.masko = np.logical_not(
            np.isnan(self.dTds1o) | np.isnan(self.dTds2o) | np.isnan(self.dTdto)
        )

        # dimensions
        self.m = np.sum(self.masko)
        self.n = np.sum(self.maskp)

        # grid
        s = np.stack([X, Y],2)
        sp = s[maskp]
        phiu = phi(sp[:, 0], sp[:, 1], degreeu)
        phiv = phi(sp[:, 0], sp[:, 1], degreev)
        phiS = phi(sp[:, 0], sp[:, 1], degreeS)
        self.phix = block_diag(phiu, phiv, phiS)
        self.d = distance_matrix(sp, sp)
        self.match_mask = self.d == 0

        # construct HA matrix
        self.HA = self.construct_HA()

        # data vector
        self.z = self.dTdto[self.masko]

    def construct_HA(self):
        # construct HA matrix
        maskp_flat = self.maskp.flatten()
        masko_flat = self.masko.flatten()
        dTds1o_flat = self.dTds1o.flatten()
        dTds2o_flat = self.dTds2o.flatten()
        HA = np.zeros([self.m, 3 * self.n])  # allocate
        idxp = np.where(maskp_flat)[0]  # indices of prediction locations

        j = 0  # iteration index over observation
        for i, idx in enumerate(idxp):  # loop over prediction locations
            if masko_flat[idx]:  # when prediction location is a observation location
                # set HA components
                HA[j, i] = -dTds1o_flat[idx]
                HA[j, i + self.n] = -dTds2o_flat[idx]
                HA[j, i + 2 * self.n] = 1
                j += 1  # next observation
        return HA

    def construct_Kx(self, params):
        # construct Kx
        Ku = kernel_matern_3_2_var(
            self.d, self.match_mask, params["sigma_u"], params["l_u"], params["tau_u"]
        )
        Kv = kernel_matern_3_2_var(
            self.d, self.match_mask, params["sigma_v"], params["l_v"], params["tau_v"]
        )
        KS = kernel_matern_3_2_var(
            self.d, self.match_mask, params["sigma_S"], params["l_S"], params["tau_S"]
        )
        return block_diag(Ku, Kv, KS)

    def construct_Kz(self, params, HA, Kx):
        # construct Kz
        Ktildetau = params["sigma_tau"] ** 2 / (2 * self.tstep**2) * np.eye(self.m)
        return mdot([HA, Kx, HA.T]) + Ktildetau

    @staticmethod
    def calculate_rlml_z(
        params_val,
        params_key,
        gprm,
        const_params,
        penalty_params,
        share_len,
        share_sigma,
        share_tau,
        solve_log,
    ):
        # calculate restricted log marginal likelihood of z

        # couple variable
        z = gprm.z
        phix = gprm.phix
        HA = gprm.HA

        # make dictionary of parameters
        if solve_log:
            params_val_i = np.exp(params_val)
        else:
            params_val_i = params_val
        params = dict(zip(params_key, params_val_i)) | const_params
        if share_len:
            params["l_v"] = params["l_u"]
        if share_sigma:
            params["sigma_v"] = params["sigma_u"]
        if share_tau:
            params["tau_v"] = params["tau_u"]

        # covariance matrix
        Kx = gprm.construct_Kx(params)  # construct Kx
        Kz = gprm.construct_Kz(params, HA, Kx)  # construct Kz
        Lz = chol(Kz)
        Qz = chol2inv(Lz)

        # universal kriging
        betacov = cholinv(mdot([phix.T, HA.T, Qz, HA, phix]))
        beta = mdot([betacov, phix.T, HA.T, Qz, z])
        mux = mdot([phix, beta])

        # penalty
        penalty = 0
        for param_name in penalty_params:
            mu, sigma = penalty_params[param_name]
            penalty += penalty_gauss_centre(params[param_name], mu, sigma)

        # compute rlml (withouth constant)
        return (
            -1 / 2 * mdot([(z - mdot([HA, mux])).T, Qz, z - mdot([HA, mux])])
            - np.sum(np.log(np.diag(Lz)))
            - 1 / 2 * np.log(np.linalg.det(mdot([phix.T, HA.T, Qz, HA, phix])))
            + penalty
        )

    @staticmethod
    def calculate_negative_rlml_z(
        params_val,
        params_key,
        gprm,
        const_params,
        penalty_params,
        share_len,
        share_sigma,
        share_tau,
        solve_log,
        callback,
    ):
        # calculate negative restricted log marginal likelihood of z

        lml = GPRegression.calculate_rlml_z(
            params_val,
            params_key,
            gprm,
            const_params,
            penalty_params,
            share_len,
            share_sigma,
            share_tau,
            solve_log,
        )

        if callback != "off":
            if solve_log:
                pval = np.append(np.exp(params_val), lml)  # values to print
            else:
                pval = np.append(params_val, lml)  # values to print

            pstring = "   ".join(
                ["{:<11.5}".format(xi) for xi in pval]
            )  # string to print

            if callback == "compact":
                print(pstring, end="\r")
            elif callback == "on":
                print(pstring)  # print

        return -lml

    def estimate_params(
        self,
        initial_params,
        const_params,
        penalty_params={},
        bounds_params={},
        share_len=False,
        share_sigma=False,
        share_tau=False,
        solve_log=True,
        shgo_bool=False,
        callback="off",
    ):
        if shgo_bool & (len(bounds_params) == 0):
            print("shgo requires bounds")

        # optimising rlml
        params_key = list(initial_params.keys())
        args = (
            params_key,
            self,
            const_params,
            penalty_params,
            share_len,
            share_sigma,
            share_tau,
            solve_log,
            callback,
        )

        if callback != "off":  # when callback is requested
            pkey = np.append(params_key, "lml")  # header keys
            pkey_comb = "   ".join(
                ["{0: <11}".format(xi) for xi in pkey]
            )  # combine keys into single line
            print(pkey_comb)  # create header

        if solve_log:
            initial_params_val = np.log(np.array(list(initial_params.values())))
        else:
            initial_params_val = np.array(list(initial_params.values()))

        if len(bounds_params) > 0:
            bounds = []
            for param_name in params_key:
                if param_name in bounds_params:
                    boundl, boundu = bounds_params[param_name]
                    if boundl is not None:
                        if solve_log:
                            boundl = np.log(boundl)
                    if boundu is not None:
                        if solve_log:
                            boundu = np.log(boundu)
                    bounds += [[boundl, boundu]]
                else:
                    bounds += [[None, None]]

            if shgo_bool:
                result = shgo(
                    GPRegression.calculate_negative_rlml_z, bounds, args=args, n=1e3
                )
            else:
                result = minimize(
                    GPRegression.calculate_negative_rlml_z,
                    initial_params_val,
                    args=args,
                    bounds=bounds,
                )
        else:
            result = minimize(
                GPRegression.calculate_negative_rlml_z, initial_params_val, args=args
            )

        if solve_log:
            params_val = list(np.exp(result.x))
        else:
            params_val = list(result.x)

        params = dict(zip(params_key, params_val)) | const_params
        if share_len:
            params["l_v"] = params["l_u"]
        if share_sigma:
            params["sigma_v"] = params["sigma_u"]
        if share_tau:
            params["tau_v"] = params["tau_u"]
        return params

    def format_output(self, mux, Kx):
        # format output

        # standard deviation
        stdx = np.sqrt(np.diag(Kx))

        # outputs
        muu_flat = mux[: self.n]
        muv_flat = mux[self.n : 2 * self.n]
        muS_flat = mux[2 * self.n : 3 * self.n]
        stdu_flat = stdx[: self.n]
        stdv_flat = stdx[self.n : 2 * self.n]
        stdS_flat = stdx[2 * self.n : 3 * self.n]

        # covariance velocity
        Kx_uu = np.diag(Kx[: self.n, : self.n])
        Kx_uv = np.diag(Kx[: self.n, self.n : 2 * self.n])
        Kx_vv = np.diag(Kx[self.n : 2 * self.n, self.n : 2 * self.n])
        Kx_vel_flat = np.stack([[Kx_uu, Kx_uv], [Kx_uv, Kx_vv]])  # stack components
        Kx_vel_flat = np.swapaxes(Kx_vel_flat, 0, 2)  # switch axes order

        # convert to 2d
        muu = map_val(muu_flat, self.maskp)
        muv = map_val(muv_flat, self.maskp)
        muS = map_val(muS_flat, self.maskp)
        stdu = map_val(stdu_flat, self.maskp)
        stdv = map_val(stdv_flat, self.maskp)
        stdS = map_val(stdS_flat, self.maskp)
        Kx_vel = map_val(Kx_vel_flat, self.maskp)
        return muu, muv, muS, stdu, stdv, stdS, Kx_vel

    def predict(self, params, return_prior=False, return_Kxstar=False):
        # predict using GPRegression

        Kx = self.construct_Kx(params)  # construct Kx
        Kz = self.construct_Kz(params, self.HA, Kx)  # construct Kz
        Lz = chol(Kz)
        Qz = chol2inv(Lz)

        # universal kriging
        betacov = cholinv(mdot([self.phix.T, self.HA.T, Qz, self.HA, self.phix]))

        # prediction covariance
        kappa_term = self.phix.T - mdot([self.phix.T, self.HA.T, Qz, self.HA, Kx])
        kappa = mdot([kappa_term.T, betacov, kappa_term])
        Kxstar = Kx - mdot([Kx, self.HA.T, Qz, self.HA, Kx]) + kappa

        # prediction mean
        beta = mdot([betacov, self.phix.T, self.HA.T, Qz, self.z])
        mux = mdot([self.phix, beta])
        muxstar = mux - mdot([Kx, self.HA.T, Qz, mdot([self.HA, mux]) - self.z])

        # format output
        muustar, muvstar, muSstar, stdustar, stdvstar, stdSstar, Kxstar_vel = (
            self.format_output(muxstar, Kxstar)
        )

        output = [muustar, muvstar, muSstar, stdustar, stdvstar, stdSstar, Kxstar_vel]
        if return_prior:  # return prior and posterior
            muu, muv, muS, stdu, stdv, stdS, Kx_vel = self.format_output(mux, Kx)
            output += [muu, muv, muS, stdu, stdv, stdS, Kx_vel]
        if return_Kxstar:
            output += [Kxstar]
        return tuple(output)


def calculate_prediction_gpregression(
    dTds1, dTds2, dTdt, params, X, Y, tstep, maskp = None, degreeu=2, degreev=2, degreeS=2, return_Kxstar=False
):
    if maskp is None:
        maskp = np.ones_like(dTds1, dtype=bool)
        
    # GP regression
    gprm = GPRegression(
        dTds1,
        dTds2,
        dTdt,
        tstep,
        X, 
        Y,
        maskp,
        degreeu=degreeu,
        degreev=degreev,
        degreeS=degreeS,
    )
    
    if not return_Kxstar:
        muustar, muvstar, muSstar, stdustar, stdvstar, stdSstar, Kxstar_vel = gprm.predict(
            params, return_prior=False, return_Kxstar=False
        )
        return muustar, muvstar, muSstar, stdustar, stdvstar, stdSstar, Kxstar_vel
    else:
        muustar, muvstar, muSstar, stdustar, stdvstar, stdSstar, Kxstar_vel, Kxstar = gprm.predict(
            params, return_prior=False, return_Kxstar=True
        )
        return muustar, muvstar, muSstar, stdustar, stdvstar, stdSstar, Kxstar_vel, Kxstar

def chol(M):
    # Cholesky decomposition
    return dpotrf(M, 1)[0]


def chol2inv(L):
    # Calculate inversion using Cholesky decomposition
    inv = dpotri(L, 1)[0]
    inv += np.tril(inv, k=-1).T
    return inv


def cholinv(M):
    # Directly calculate inversion through Cholesky decomposition
    L = chol(M)
    return chol2inv(L)


def kernel_matern_3_2_var(d, match_mask, sigma, ls, tau):
    # Matern covariance function nu = 3/2 (p=1) including the additional variance τ**2
    matern = sigma**2 * (1 + np.sqrt(3) * d / ls) * np.exp(-np.sqrt(3) * d / ls)
    var = tau**2 * match_mask
    return matern + var


def penalty_gauss_centre(theta, mu, sigma):
    # for each parameter theta, add to the objective you want to maximise
    return -np.log(sigma) - 0.5 * (theta - mu) ** 2 / sigma**2


def linear_mean(s, M, R1, R2):
    # linear mean
    s1 = s[:, 0]
    s2 = s[:, 1]
    return M + R1 * s1 + R2 * s2


def phi(s1, s2, degree):
    # compute phi
    N = len(s1)
    constant = [np.ones(N)]
    component1 = [(s1**i) for i in np.arange(1, degree)]
    component2 = [(s2**i) for i in np.arange(1, degree)]
    return np.stack(constant + component1 + component2).T
