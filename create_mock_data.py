import numpy as np 
import scipy as sp 
import pandas as pd 
import matplotlib.pyplot as plt 
from scipy.integrate import quad 
from scipy.interpolate import CubicSpline
from astropy.coordinates import SkyCoord 
import astropy.units as u 
from scipy.stats import chi2
c = 2.99792458e5 # km/s

Om = 0.3
q0 = -0.55 
j0 = 1.0
H0 = 67.32  # km/s/Mpc

# set the bulk flow to be a random vector 
Bx = 200.0
By = -300.0
Bz = 50.0 

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
    

# create 5000 random points in a sphere of radius 500 Mpc 

xpos, ypos, zpos = [], [], []

while len(xpos) < 5000:
    
    xposrand = np.random.random() * 500.0 - np.random.random() * 500.0 
    yposrand = np.random.random() * 500.0 - np.random.random() * 500.0 
    zposrand = np.random.random() * 500.0 - np.random.random() * 500.0 
    rposrand = np.sqrt(xposrand**2 + yposrand**2 + zposrand**2)
    if rposrand <= 500.0 and rposrand >= 10:
        xpos.append(xposrand)
        ypos.append(yposrand)
        zpos.append(zposrand)
        
xpos = np.array(xpos)
ypos = np.array(ypos)
zpos = np.array(zpos)

rpos = np.sqrt(xpos**2 + ypos**2 + zpos**2)

# compute RA and DEC 

RA = np.arctan2(ypos,xpos)
RA = np.mod(RA, 2.0*np.pi)
DEC = np.arcsin(zpos/rpos)

# to go back from RA , DEC to x/y/z 
# xpos = rpos * np.cos(RA)*np.cos(DEC)
# ypos = rpos * np.sin(RA)*np.cos(DEC)
# zpos = rpos * np.sin(DEC)

# use an interpolator to get z_recession from rpos (=D(z_rec))

zarrinterp = np.linspace(0.0, 0.5, 200) 
dzinterp = rz(zarrinterp)
interpolator = CubicSpline(dzinterp, zarrinterp)

z_rec = interpolator(rpos)

# generate a random component of motion - this does not give any linear theory correlations between
# the galaxies, which requires one to actually compute the PV covariance to do this correctly 

rand_motion_x = np.random.normal(loc=0.0, scale=200.0, size=5000)
rand_motion_y = np.random.normal(loc=0.0, scale=200.0, size=5000)
rand_motion_z = np.random.normal(loc=0.0, scale=200.0, size=5000)

pv_galaxy_true = (Bx + rand_motion_x) * np.cos(RA)*np.cos(DEC) + (By + rand_motion_y) * np.sin(RA) * np.cos(DEC) + (Bz + rand_motion_z) * np.sin(DEC)

# now get z_peculiar 

z_pec = np.sqrt( (1.0 + pv_galaxy_true/c) / (1.0 - pv_galaxy_true / c )) - 1.0 

z_obs = (1.0 + z_pec) * (1.0 + z_rec) - 1.0 

eta_true = np.log10(rz(z_obs)/rpos)

eta_obs = np.random.normal(loc=eta_true, scale=0.05)

# we have a uniform distribution so the number density is roughly the same everywhere. But we can estimate it anyway

from scipy.spatial import cKDTree
pos = np.vstack([xpos, ypos, zpos]).reshape(5000,3)
print(pos.shape)
tree = cKDTree(pos)             
d, _ = tree.query(pos, k=6)      
density1e6  = 5 / ((4/3)*np.pi*d[:,-1]**3) * 1.0e6


data = {
    "RA": RA * 180.0 / np.pi,
    "Dec": DEC * 180.0 / np.pi, 
    "z_obs": z_obs,
    "eta_obs": eta_obs,
    "eta_obs_err": 0.05, 
    "n1e6": density1e6,
    "vx": Bx + rand_motion_x, 
    "vy": By + rand_motion_y, 
    "vz": Bz + rand_motion_z
}

data = pd.DataFrame(data)
# data.to_csv('example_surveymock.dat', index=False, sep=' ', header=False)