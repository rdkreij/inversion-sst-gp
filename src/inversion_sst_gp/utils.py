import numpy as np
import pandas as pd
from geographiclib.geodesic import Geodesic

def find_centre(lon, lat):
    latlims = [np.min(lat), np.max(lat)]
    lonlims = [np.min(lon), np.max(lon)]
    latc = np.mean(latlims)
    lonc = np.mean(lonlims)
    return latc, lonc

def geo_to_car(lon, lat, lonc, latc):
    # convert geographic coordinate to cartesian coordinates with the centre as origin        
    Nx  = len(lon)
    Ny  = len(lat)
    X = np.empty([Ny,Nx])
    Y = np.empty([Ny,Nx])
    for i in range(Nx):
        for j in range(Ny): 
            X[j,i] = np.sign(lon[i]-lonc)*Geodesic.WGS84.Inverse(lat[j],lon[i],lat[j],lonc)['s12']
            Y[j,i] = np.sign(lat[j]-latc)*Geodesic.WGS84.Inverse(lat[j],lon[i],latc,lon[i])['s12']
    return X,Y

def calculate_grid_properties(lon, lat):
    latc, lonc = find_centre(lon, lat)
    X, Y = geo_to_car(lon, lat, lonc, latc)
    LON, LAT = np.meshgrid(lon, lat)
    return lonc, latc, X, Y, LON, LAT

def finite_difference_1d(s, x):
    inan = np.isnan(x) 
    if np.any(inan): 
        inanR = np.hstack([inan[1:],True]) 
        inanL = np.hstack([True,inan[:-1]]) 

        sp = np.roll(s,-1) 
        xp = np.roll(x,-1) 
        sm = np.roll(s,1) 
        xm = np.roll(x,1) 

        iforward = (~inan) & (~inanR) & inanL
        icentral = (~inanL) & (~inanR)
        ibackward = (~inan) & inanR & (~inanL)

        dxds = np.empty(len(x)) 
        dxds.fill(np.nan)
        dxds[iforward] = (xp[iforward]-x[iforward])/(sp[iforward]-s[iforward]) 
        dxds[icentral] = (xp[icentral]-xm[icentral])/(sp[icentral]-sm[icentral])
        dxds[ibackward] = (x[ibackward]-xm[ibackward])/(s[ibackward]-sm[ibackward]) 
    else: 
        forward_diff = (x[1]-x[0])/(s[1]-s[0]) 
        central_diff = (x[2:]-x[:-2])/(s[2:]-s[:-2])
        backward_diff = (x[-1]-x[-2])/(s[-1]-s[-2]) 

        dxds = np.hstack([forward_diff,central_diff,backward_diff])
    return dxds   

def finite_difference_2d(s1_2d, s2_2d, x_2d):
    N2,N1 = x_2d.shape
    dxds1 = np.stack([finite_difference_1d(s1_2d[i,:],x_2d[i,:]) for i in range(N2)])
    dxds2 = np.stack([finite_difference_1d(s2_2d[:,i],x_2d[:,i]) for i in range(N1)]).T
    return dxds1,dxds2

def calculate_vorticity(s1_2d,s2_2d,u,v):
    _, dudy = finite_difference_2d(s1_2d,s2_2d,u)
    dvdx, _ = finite_difference_2d(s1_2d,s2_2d,v)
    return dvdx - dudy

def calculate_coriolis_parameter(latitude_deg):
    omega = 7.2921e-5
    latitude_rad = np.deg2rad(latitude_deg)
    return 2 * omega * np.sin(latitude_rad)

def calculate_dynamic_rossby_number(s1_2d,s2_2d,u,v,latitude_deg):
    vorticity = calculate_vorticity(s1_2d,s2_2d,u,v)
    coriolis_parameter = calculate_coriolis_parameter(latitude_deg)
    return vorticity/coriolis_parameter

def map_val(x,mask):
    # map value given mask
    xf = np.full(list(np.shape(mask)) + list(x.shape[1:]),np.nan) # allocate
    xf[mask] = x
    return xf

def map_mask(maski,mask):
    # map maski given mask
    maskf = np.full(len(mask),False) # allocate
    maskf[mask] = maski
    return maskf

def extract_params(filename, name, value, type):
    if type == "gp":
        keys=["sigma_u","l_u","tau_u","l_v","tau_v","sigma_v","sigma_S","l_S","tau_S","sigma_tau"]
    elif type == "gos":
        keys=["n"]
    df = pd.read_csv(filename)
    row = df[df[name] == value].iloc[0]
    return {k: row[k] for k in keys} if keys else row.to_dict()