import numpy as np
from src.inversion_sst_gp.gp_regression import chol
from geographiclib.geodesic import Geodesic
from scipy.interpolate import LinearNDInterpolator

def simulate_particle_flow_field(X,Y,u,v,lon0,lat0,lonc,latc,tstep,N):

    # flatten data
    Xf = X.flatten()
    Yf = Y.flatten()
    uf = u.flatten()
    vf = v.flatten()

    # define the interpolation function
    interp_u = LinearNDInterpolator(list(zip(Xf, Yf)), uf)
    interp_v = LinearNDInterpolator(list(zip(Xf, Yf)), vf)

    # allocate
    lons = np.empty([N])
    lats = np.empty([N])

    x0 = np.sign(lon0-lonc)*Geodesic.WGS84.Inverse(lat0,lon0,lat0,lonc)['s12']
    y0 = np.sign(lat0-latc)*Geodesic.WGS84.Inverse(lat0,lon0,latc,lon0)['s12']

    lons[0], lats[0] = (lon0, lat0)
    xp, yp = (x0, y0)

    for i in range(1,N):
        up = interp_u(xp,yp).item()
        vp = interp_v(xp,yp).item()

        # update locations
        xp += up*tstep
        yp += vp*tstep

        geo_dict = Geodesic.WGS84.Direct(latc,lonc,np.rad2deg(np.arctan2(xp,yp)),np.sqrt(xp**2+yp**2))

        lons[i] = geo_dict['lon2']
        lats[i] = geo_dict['lat2']

    return lons, lats

def simulate_from_kernel(K, n_samples=1, mu = None):
    n_elem = len(K)
    L = chol(K)
    
    samples = np.empty([n_samples,n_elem])
    
    if mu is None:
        mu = 0
    
    for i in range(n_samples):
        samples[i,:] = np.dot(L,np.random.normal(0,1,n_elem)) + mu
    
    return samples
    
def simulate_flow_fields(Kxstar, muustar, muvstar,Ny,Nx, n_sample):
    
    samples = simulate_from_kernel(Kxstar, n_sample)
    N = Ny*Nx

    us = np.empty([n_sample,Ny,Nx])
    vs = np.empty([n_sample,Ny,Nx])

    for i in range(n_sample):
        us[i,:,:] = samples[i,:N].reshape([Ny,Nx]) + muustar
        vs[i,:,:] = samples[i,N:2*N].reshape([Ny,Nx]) + muvstar
        
    return us, vs

def simulate_particle_multi_flow_fields(X,Y, us, vs,lon0,lat0,lonc,latc,tstep,Nstep):

    n_sample = us.shape[0]
    lons = np.empty([n_sample,Nstep])
    lats = np.empty([n_sample,Nstep])

    for i in range(n_sample):
        lons[i,:], lats[i,:] = simulate_particle_flow_field(X,Y, us[i,:,:], vs[i,:,:],lon0,lat0,lonc,latc,tstep,Nstep)
    return lons, lats

