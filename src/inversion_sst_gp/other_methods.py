import numpy as np
from scipy.interpolate import griddata
from scipy.linalg import lstsq

from src.inversion_sst_gp.utils import map_val


def RMSE(u0, v0, up, vp):
    # compute Root Mean Square Error (RMSE) for velocity vectors
    return np.sqrt(0.5 * np.nanmean((u0 - up)**2 + (v0 - vp)**2))

class GlobalOptimalSolution:
    def __init__(self, dTds1, dTds2, dTdt):
        # initializing the Chen inversion method
        self.dTds1 = dTds1
        self.dTds2 = dTds2
        self.dTdt = dTdt
        self.N2, self.N1 = np.shape(self.dTdt)
        self.N = self.N2*self.N1

    def subgrid(self, n, corner=0):
        # number of knots
        Nn1 = 1 + (self.N1-1)//n # number of knots in x direction
        Nn2 = 1 + (self.N2-1)//n # number of knots in y direction
        Nn  = Nn1*Nn2 # total number of knots

        # number pixels remaining after crop
        N1c = 1 + n*(Nn1-1) # number of pixels in x direction after cropping
        N2c = 1 + n*(Nn2-1) # number of pixels in y direction after cropping
        Nc = N1c*N2c

        # create subgrid from corner
        if corner == 0:
            dis1 = 0
            dis2 = 0
        elif corner == 1:
            dis1 = self.N1-N1c
            dis2 = 0
        elif corner == 2:
            dis1 = self.N1-N1c
            dis2 = self.N2-N2c
        elif corner == 3:
            dis1 = 0
            dis2 = self.N2-N2c

        # crop mask
        crop_mask = np.zeros([self.N2,self.N1]).astype(bool)
        crop_mask[dis2:(N2c+dis2),dis1:(N1c+dis1)] = True

        # knot mask
        knot_mask = np.zeros([self.N2,self.N1]).astype(bool)
        knot_mask[dis2:(N2c+dis2):n,dis1:(N1c+dis1):n] = True
        return Nn1, Nn2, Nn, N1c, N2c, Nc, crop_mask, knot_mask
        
    def solve_at_knots(self, n, corner=0):
        # ivert heat equation in 2D according to Chen et al. (2008)

        # create subgrid and crop data
        Nn1, Nn2, Nn, N1c, N2c, Nc, crop_mask, knot_mask = self.subgrid(n, corner)
        dTds1f = self.dTds1[crop_mask]
        dTds2f = self.dTds2[crop_mask]
        dTdtf = self.dTdt[crop_mask]

        # start building linear system
        b = dTdtf # set b data vector

        # allocate L matrix values
        Lu = np.zeros([Nn,Nc]) # L for longitudinal velocity u
        Lv = np.zeros([Nn,Nc]) # L for lateral velocity v
        LS = np.zeros([Nn,Nc]) # L for source term S

        for idx1 in range(N1c):
            for idx2 in  range(N2c):
                i, j = idx1%n, idx2%n # index within function block 2d
                ii, jj = idx1//n, idx2//n # knot index 2d
                knot0 = ii + jj*Nn1 # origin knot index 
                point = idx1 + idx2*N1c # point index

                if (i==0) & (j==0): # at knot
                    list_knot = np.array([knot0])
                    list_weight = np.array([1])
                elif (i==0) & (j!=0): # at side 
                    list_knot = knot0 + np.array([0, Nn1])
                    list_weight = np.array([n-j,j])/n
                elif (i!=0) & (j==0): # at side
                    list_knot = knot0 + np.array([0, 1])
                    list_weight = np.array([n-i,i])/n
                elif (i!=0) & (j!=0): # within
                    list_knot = knot0 + np.array([0, 1, Nn1, Nn1+1])
                    list_weight = np.array([(n-i)*(n-j), i*(n-j), (n-i)*j, i*j])/n**2
                for k,knot in enumerate(list_knot):
                    Lu[knot,point] = -dTds1f[point]*list_weight[k]
                    Lv[knot,point] = -dTds2f[point]*list_weight[k]
                    LS[knot,point] = list_weight[k]
        L = np.vstack([Lu,Lv,LS]) # combine L matrices
        a = lstsq(L.T,b)[0] # solve linear system using least squares

        # decompose
        uknot = a[:Nn]
        vknot = a[Nn:2*Nn]
        Sknot = a[2*Nn:3*Nn]
        return uknot, vknot, Sknot, knot_mask
    
    def compute(self, n, corner='all'):
        # compute GOS
        if corner == 'all': # initiate from all corners
            for corner_i in [0,1,3,2]:
                uknot_i, vknot_i, Sknot_i, knot_mask_i = self.solve_at_knots(n, corner_i)
                ugos_i = map_val(uknot_i, knot_mask_i)
                vgos_i = map_val(vknot_i, knot_mask_i)
                Sgos_i = map_val(Sknot_i, knot_mask_i)
                if corner_i == 0: # main corner
                    ugos = ugos_i
                    vgos = vgos_i
                    Sgos = Sgos_i
                elif corner_i == 1: # side
                    ugos[:,-1] = ugos_i[:,-1]
                    vgos[:,-1] = vgos_i[:,-1]
                    Sgos[:,-1] = Sgos_i[:,-1]
                elif corner_i == 2: # single point
                    ugos[-1,-1] = ugos_i[-1,-1]
                    vgos[-1,-1] = vgos_i[-1,-1]
                    Sgos[-1,-1] = Sgos_i[-1,-1]
                elif corner_i == 3: # side
                    ugos[-1,:] = ugos_i[-1,:]
                    vgos[-1,:] = vgos_i[-1,:]
                    Sgos[-1,:] = Sgos_i[-1,:]
        else: # initiate from single corner
            uknot, vknot, Sknot, knot_mask = self.solve_at_knots(n, corner)
            ugos = map_val(uknot, knot_mask)
            vgos = map_val(vknot, knot_mask)
            Sgos = map_val(Sknot, knot_mask)

        # interpolate solution
        ugos = self.interpolate(ugos)
        vgos = self.interpolate(vgos)
        Sgos = self.interpolate(Sgos)
        return ugos, vgos, Sgos
        
    def interpolate(self, field, method='linear'):
        # interpolate field 
        d1, d2 = np.meshgrid(range(self.N1),range(self.N2)) # distances
        data_mask = np.logical_not(np.isnan(field)) # data mask
        points = (d1[data_mask],d2[data_mask])
        values = field[data_mask]
        return griddata(points, values, (d1, d2), method=method)

    def optimize_n(self, u0, v0, nmin=None, nmax=None):
        # optimization of n given reference
        if nmin is None:
            nmin = 3
        if nmax is None:
            nmax = min(self.N1//2,self.N2//2)
        n_range = np.arange(nmin, nmax + 1)
        list_RMSE = np.empty(len(n_range))
        for i,n in enumerate(n_range):
            ugos, vgos, Sgos = self.compute(n)
            list_RMSE[i] = RMSE(u0, v0, ugos, vgos)
        return n_range[np.argmin(list_RMSE)]
    
def calculate_prediction_gos(
    dTds1, dTds2, dTdt, n
):
    gos = GlobalOptimalSolution(dTds1, dTds2, dTdt)
    ugos, vgos, Sgos = gos.compute(n)
    return ugos, vgos, Sgos