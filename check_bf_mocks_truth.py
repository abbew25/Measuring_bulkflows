import numpy as np 
import scipy as sp 
import pandas as pd 
import matplotlib.pyplot as plt 
from scipy.integrate import quad 
from astropy.coordinates import SkyCoord 
import astropy.units as u 
from scipy.stats import chi2
c = 2.99792458e5 # km/s

Om = 0.3
q0 = -0.55 
j0 = 1.0
H0 = 67.32  # km/s/Mpc

# function used to integrate to compute proper distance from Friedmann Equation
def E_z_inverse(z):
    '''
    Compute the inverse of the E(z) function (from the first Friedmann Equation).
    '''
    return 1.0/(np.sqrt((Om*(1.0+z)**3)+(1.0-Om)))

# function that computes the proper distance as a function of redshift (dimensionless)
def rz(red):
    '''
    Calculates the proper radial distance to an object at redshift z for the given cosmological model.
    '''
    try:
        d_com = (c/H0)*quad(E_z_inverse, 0.0, red, epsabs = 5e-5)[0]
        return d_com
    except:
        distances = np.zeros(len(red))  
        for i, z in enumerate(red):
            distances[i] = (c/H0)*quad(E_z_inverse, 0.0, z, epsabs = 5e-5)[0]
        return distances 
    
# get zmod 
def zed_mod(z):
    
    zmod = z*( 1.0 + 0.5*(1.0 - q0)*z - (1.0/6.0)*(j0 - q0 - 3.0*(pow(q0, 2)) + 1.0 )*(pow(z, 2)) )

    return zmod 

# right ascension, declination, observed spectroscopic redshift, observed log-distance ratio, 
# observed log-distance ratio uncertainty/error, n(r) 
# (mean number density of objects (per Mpc * 1e6) in the survey at the point the galaxy lies at). 
# The last 3 columns have the real vx,vy and vz velocities.
data_mock = pd.read_csv("example_surveymock.dat", names=["RA", "Dec", "z_obs", "eta_obs",
    "eta_obs_err", "n1e6", "vx", "vy", "vz"], delim_whitespace=True)


simple_av_bf_x = np.mean(data_mock["vx"].to_numpy())
simple_av_bf_y = np.mean(data_mock["vy"].to_numpy())
simple_av_bf_z = np.mean(data_mock["vz"].to_numpy())

print(simple_av_bf_x, simple_av_bf_y, simple_av_bf_z)


distances_objs = rz(data_mock["z_obs"].to_numpy())

gaussian_weight_av_bf_x = np.average(data_mock["vx"].to_numpy(), weights=np.exp(-1.0*(distances_objs**2)/(2.0*70.0*70.0)))
gaussian_weight_av_bf_y = np.average(data_mock["vy"].to_numpy(), weights=np.exp(-1.0*(distances_objs**2)/(2.0*70.0*70.0)))
gaussian_weight_av_bf_z = np.average(data_mock["vz"].to_numpy(), weights=np.exp(-1.0*(distances_objs**2)/(2.0*70.0*70.0)))


print(gaussian_weight_av_bf_x, gaussian_weight_av_bf_y, gaussian_weight_av_bf_z)


zmod_objs = zed_mod(data_mock["z_obs"].to_numpy())
err_vels = (c*zmod_objs/(1.0+zmod_objs))*data_mock['eta_obs'].to_numpy()*np.log(10.0)
err_vels_tot = 300.0**2 + err_vels

err_weighted_av_bf_x = np.average(data_mock["vx"].to_numpy(), weights=1.0/err_vels_tot)
err_weighted_av_bf_y = np.average(data_mock["vy"].to_numpy(), weights=1.0/err_vels_tot)
err_weighted_av_bf_z = np.average(data_mock["vz"].to_numpy(), weights=1.0/err_vels_tot)

print(err_weighted_av_bf_x, err_weighted_av_bf_y, err_weighted_av_bf_z)

