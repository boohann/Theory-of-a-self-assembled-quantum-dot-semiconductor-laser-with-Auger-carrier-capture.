##########################################################
##### Program to simulate laser-rate equations for #######
###### quantum dot laser under feedback in Python ########
############### Niall Boohan 2019 ########################
##########################################################
##########################################################


### Theory and equations sourced from:
# Title: Theory of a self-assembled quantum-dot semiconductor laser with Auger carrier capture: Quantum efficiency and nonlinear gain
# Authors: Uskov, A. V.; Boucher, Y.; Le Bihan, J.; McInerney, J.
# DOI: 10.1063/1.122185
# URL: https://www.researchgate.net/publication/8524953 Sensitivity

from scipy.integrate import ode
from pylab import *
import matplotlib.pyplot as plt
import numpy as np
import scipy as sci

### Run progress bar ###
import time
import progressbar

bar = progressbar.ProgressBar(max_value=progressbar.UnknownLength)
for i in range(20):
    time.sleep(0.1)
    bar.update(i)


### Simualtion Outputs ###
N       =  []      # y[0] Carrier concentration
roRes   =  []      # y[1] Recombination function
ro      =  []      # y[2] Dot population
S       =  []      # y[3] Photon concentration

### Simualtion input  parameters ###
I = 25                                          # Pump current (mA)
w = 3                                           # Cavity width (um)
L = 375                                         # Cavity length (um)
J = (I/1e3)/((w/1e6)*(L/1e6))                   # Pump current density (Am^-2)
q = 1.602176634e-19                             # Coulomb unit charge (C)  
l = 3                                           # Quantity of QD layers
t_0 = 1e-9                                      # Wetting layer lifetime (s)
C =  2e-20                                      # Auger recombination factor (m^4s)
N_d = 2e15                                      # 2D density of dots (m^-2)
c = 2.99792458e8                                # SOL (ms^-1)
n_rfr = 3.4                                     # Refractive index
v_g = c/n_rfr                                   # Group velocity
B_tran = 0#1e-4                                   # Rate constant of Auger transport (m^2s^-1)
mu = q*3e-10                                    # Dipole moment of optical transition (Cm)
hbar = 6.582119569e-16                          # Reduced Plank's constant(eVs)
omg_0 = 1/hbar                                  # Angular frequency of central frequency (rad s^-1)
Ep_0 = (55.26349406*(q**2))/(1e9*1e-15)         # Permittivity of free space (C^2eV^-1m^-1)
r_star = 0.56                                   # Parameter of inhomogeneousity
T_2 = 1e-13                                     # Dephasing time (s) 
gam_2 = 1/T_2                                   # Homogenous broadening (s^-1) 
t_D0 = 1e-9                                     # Carrier lifetime QD (s)
t_ph = 3.2e-12                                  # Photon lifetime (s)
Gam = 0.06                                      # Confinement factor
dnm = 10                                        # Layer thickness (nm)
d = dnm/1e9                                     # Layer thickness (m)
dcm = dnm/1e7

### Initial conditions ###
N_0 = 1e10                                 # Initial carrier density (cm^-2)
roRes_0 = 0.1                              # Initial occupancy rate resonant dots          
ro_0 = 0.1                                 # Initial occupancy inhomegeneous dots 
S_0 = 1e17                                 # Initial photon concentration (m^-3)



### Calculate carrier photon cross-section ###
sigRes = (np.pi*(mu**2)*omg_0)/(2*hbar*c*n_rfr*Ep_0*gam_2)

### Define equations to be solved
def Laser_rates(t, y, p):
    
    dy = zeros([4])
    dy[0] = J/(q*l) - y[0]/t_0 - C*(y[0]**2)*2*N_d*(1-y[2])                                           # N
    dy[1] = -v_g*sigRes*(2*y[1]-1)*y[3] - y[1]/t_D0 + C*(y[0]**2)*(1-y[1]) - B_tran*(y[1]-y[2])       # roRes
    dy[2] = -v_g*sigRes*r_star*(2*y[1]-1)*y[3] - y[2]/t_D0 + C*(y[0]**2)*(1-y[2])                     # ro
    dy[3] = -y[3]/t_ph + Gam*v_g*((2*N_d*sigRes*r_star*(2*y[1]-1))/d)*y[3]                            # S
    return dy
    

### Time and initial conditions ###  
t0 = 0; tEnd = 1e-8; dt = 1e-14                         # Time constraints
y0 = [N_0*1e4, roRes_0, ro_0, S_0]                      # Initial conditions
Y=[]; T=[]                                              # Create empty lists

### Parameters for odes ###
p = [J, q, l, t_0, C, N_d, v_g, sigRes, t_D0, B_tran, r_star, t_D0, t_ph, Gam]

# Setup integrator with desired parameters
r = ode(Laser_rates).set_integrator('lsoda', method = 'bdf')
#r = ode(Laser_rates).set_integrator('dopri5', nsteps = 1e6)
r.set_f_params(p).set_initial_value(y0, t0)

### Simualtion check ###
while r.successful() and r.t+dt < tEnd:
    r.integrate(r.t + dt)
    Y.append(r.y)                           # Makes a list of 1d arrays
    T.append(r.t)

### Format output ###
Y     = array(Y)                                # Convert from list to 2d array
N     = Y[:, 0]
roRes = Y[:, 1] 
ro    = Y[:, 2] 
S     = Y[:, 3]

### Plotting ###
f, axarr = plt.subplots(3, sharex=True) # Two subplots, the axes array is 1-d
axarr[0].plot(T, N/1e4, 'G')
axarr[0].set_ylabel("Carrier Conc (cm^-2)")
#axarr[0].set_yscale("log")
axarr[0].set_title('Laser-Rate Simulation')
axarr[1].plot(T, S/1e6, 'B')
axarr[1].set_ylabel("Photon Conc (cm^-3)")
#axarr[1].set_yscale("log")
axarr[2].plot(T, ro, 'K')
axarr[2].set_ylabel("ro")
axarr[2].set_xlabel("Time (s)")
plt.savefig('CalcOut.png')
plt.show()

