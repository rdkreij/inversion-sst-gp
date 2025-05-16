# import numpy as np

# from inversion_sst_gp.utils import finite_difference_2d


# def simulate_tracer_observations_from_model_data(
#     T, dTdt, tstep, X, Y, sigma_tau=None, coverage_sparse=None, coverage_dense=None
# ):
#     T = T.copy()
#     dTdt = dTdt.copy()

#     N2, N1 = T.shape

#     # Mask out any point where T or dTdt is NaN
#     invalid_mask = np.isnan(T) | np.isnan(dTdt)
#     T[invalid_mask] = np.nan
#     dTdt[invalid_mask] = np.nan

#     # Add Gaussian noise if specified
#     if sigma_tau is not None:
#         T += np.random.normal(0, sigma_tau, size=T.shape)
#         dTdt += np.random.normal(0, (np.sqrt(0.5) / tstep) * sigma_tau, size=T.shape)

#     # Simulate sparse coverage
#     if coverage_sparse is not None:
#         mask_sparse = np.random.rand(N2, N1) <= coverage_sparse
#         T[mask_sparse] = np.nan
#         dTdt[mask_sparse] = np.nan

#     # Simulate dense coverage cloud at center
#     if coverage_dense is not None:
#         total_points = N1 * N2
#         num_covered = int(total_points * coverage_dense)

#         g2, g1 = np.ogrid[:N2, :N1]
#         center2, center1 = N2 // 2, N1 // 2
#         distances = np.sqrt((g1 - center1) ** 2 + (g2 - center2) ** 2)
#         radius_threshold = np.sort(distances.ravel())[num_covered]
#         mask_dense = distances <= radius_threshold

#         T[mask_dense] = np.nan
#         dTdt[mask_dense] = np.nan

#     # Compute spatial gradients and final mask
#     dTds1, dTds2 = finite_difference_2d(X, Y, T)
#     maskc = np.isnan(dTds1) | np.isnan(dTds2) | np.isnan(dTdt)

#     return T, dTds1, dTds2, dTdt, maskc

import copy
import numpy as np
from inversion_sst_gp.utils import finite_difference_2d

class ModifyData(object):
    # modify T and dTdt

    def __init__(self, T, dTdt, tstep, X, Y):
        self.T = copy.deepcopy(T)
        self.dTdt = copy.deepcopy(dTdt)
        self.tstep = tstep
        self.s = np.stack([X, Y],2)
        self.N2, self.N1 = np.shape(self.T)

        # only keep points if they have both a T and dTdt value 
        masko_inv = np.isnan(self.T) | np.isnan(self.dTdt)
        self.masko = np.logical_not(masko_inv)
        self.T[masko_inv] = np.nan
        self.dTdt[masko_inv] = np.nan

    def noise(self, sigma_tau):
        # add noise
        self.T += np.random.normal(0,sigma_tau,size=(self.N2,self.N1))
        self.dTdt += np.random.normal(0,np.sqrt(.5)/self.tstep*sigma_tau,size=(self.N2,self.N1))
        return self
        
    def sparse_cloud(self, coverage):
        # generate sparse cloud
        rand_num = np.random.rand(self.N2,self.N1)
        maskc = rand_num <= coverage
        self.T[maskc] = np.nan
        self.dTdt[maskc] = np.nan
        return self

    def circ_cloud(self, coverage):
        # generate circular cloud at centeer
        N = self.N1*self.N2 # total number of points
        Nc = int(N*coverage) # covered pixels
        g2, g1 = np.ogrid[:self.N2, :self.N1] # create grid
        center1, center2 = self.N1//2, self.N2//2 # centre points
        dis = np.sqrt((g1 - center1)**2 + (g2 - center2)**2) # distance from center
        sort_dis = np.sort(dis.flatten()) # sort distances
        radius = sort_dis[Nc] # get radius
        maskc = dis <= radius # cloud mask
        self.T[maskc] = np.nan
        self.dTdt[maskc] = np.nan
        return self

    def convert_to_input(self):
        # convert to input
        dTds1, dTds2 = finite_difference_2d(self.s[:,:,0],self.s[:,:,1],self.T)
        maskc = np.isnan(dTds1) | np.isnan(dTds2) | np.isnan(self.dTdt) 
        return self.T, dTds1, dTds2, self.dTdt, maskc