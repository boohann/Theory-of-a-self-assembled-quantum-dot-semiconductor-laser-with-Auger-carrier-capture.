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


### Import modules ###
import matplotlib.pyplot as plt
import numpy as np
from scipy.integrate import ode

### Script mode setting ###
Mode = 1              # Mode = 0 for dynamic curves and 1 for LI curves

### Simualtion Outputs [Initialising globally] ###
N          =  []      # Carrier concentration
rhoRes     =  []      # Recombination function
rho        =  []      # Dot population
S          =  []      # Photon concentration
P          =  []      # Power output
t_dwl      =  []      # Relaxation time between QD and wetting layer 
t_leff     =  []      # Relaxation time between QDs    
e_shb      =  []      # Spectral hole burning
e_dwl      =  []      # Dot wetting layer hole burning
nu_r       =  []      # Relaxation oscillation frequency
Gam_r      =  []      # Relaxation oscillation decay rate
g_diff     =  []      # Differential Gain

### Extra arrays for steady-state LI outputs ###
N_end       =  []       # Takes final steady-state N value
rhoRes_end  =  []       # Takes final steady-state rho resonant value
rho_end     =  []       # Takes final steady-state rho value
S_end       =  []       # Takes final steady-state S value to conver to power for LI

# NEX = 0  Simualtion iterator

### Simualtion input  parameters ###

# Current/current density 
I      = 20                                     # Single current sweep
iI     = np.linspace(1, 10, 10)                 # LI sweep
ITR    = np.linspace(0, 9, 10)
w      = 3                                      # Cavity width (um)
L      = 375                                    # Cavity length (um)
dnm    = 10                                     # Layer thickness (nm)
d      = dnm/1e9                                # Layer thickness (m)
l      = 3                                      # Quantity of QD layers
Gam    = 0.06                                   # Confinement factor
A      = (w/1e6)*(L/1e6)                        # Diode area (m^2)
V      = (A*d*l)/Gam                            # Active region volume (m^-3)
J      = (I/1e3)/A                              # Pump current density single sweep (Am^-2)
iJ     = [(k/1e3)/A for k in iI]                # Pump current density LI sweep (Am^-2) 

# Define physical constants
c      = 2.99792458e8                           # SOL (ms^-1)
q      = 1.602176634e-19                        # Coulomb unit charge (C)  
h      = 6.62607004e-34                         # Plank's contant (Js)
hbar   = 6.582119569e-16                        # Reduced Plank's constant(eVs)
Ep_0   = (55.26349406*(q**2))/(1e9*1e-15)       # Permittivity of free space (C^2eV^-1m^-1)

# Lifetimes
t_D0   = 1e-9                                   # Carrier lifetime QD (s)
t_ph   = 3.2e-12                                # Photon lifetime (s)
t_0    = 1e-9                                   # Wetting layer lifetime (s)

# Device material parameters
C      = 2e-20#0.5e-20                          # Auger recombination factor (m^4s)
B_tran = 1e-4#0.5e-4                            # Rate constant of Auger transport (m^2s^-1)
N_d    = 2e15                                   # 2D density of dots (m^-2)
n_rfr  = 4.3                                    # Refractive index
v_g    = c/n_rfr                                # Group velocity (ms^-1)
mu     = q*3e-10                                # Dipole moment of optical transition (Cm)
omg_0  = 1/hbar                                 # Angular frequency of central frequency (rad s^-1)
r_star = 0.56                                   # Parameter of inhomogeneousity
T_2    = 1e-13                                  # Dephasing time (s) 
gam_2  = 1/T_2                                  # Homogenous broadening (s^-1) 
WL     = 1300                                   # WL (nm)
f      = c/(WL/1e9)                             # Frequency (Hz)

### Initial conditions ###
N_0      = 1e9                                  # Initial carrier density 2D (cm^-2)
roRes_0  = 0.8                                  # Initial occupancy rate resonant dots          
ro_0     = 0.8                                  # Initial occupancy inhomegeneous dots 
S_0      = 1e3                                  # Initial photon concentration (m^-3)

### Presimulation calculations ###
sigRes = (np.pi*(mu**2)*omg_0)/(2*hbar*c*n_rfr*Ep_0*gam_2)      # Carrier photon cross-section
g_diff = (2*sigRes*r_star)/(1 + (1/(N_d*np.sqrt(C*t_0))))       # g_diff
    

### Function to structure data ###
def DefCallSolv(k, NEX):

    ### Define equations to be solved ###
    def Laser_rates(t, y, p):        
        dy     = np.zeros([4])
        dy[0]  = k/(q*l) - y[0]/t_0 - C*(y[0]**2)*2*N_d*(1-y[2])                                                   # N
        dy[1]  = -v_g*sigRes*(2*y[1]-1)*y[3] - y[1]/t_D0 + C*(y[0]**2)*(1-y[1]) - B_tran*(y[1]-y[2])               # roRes
        dy[2]  = -v_g*sigRes*r_star*(2*y[1]-1)*y[3] - y[2]/t_D0 + C*(y[0]**2)*(1-y[2])                             # ro
        dy[3]  = -y[3]/t_ph + Gam*v_g*((2*N_d*sigRes*r_star*(2*y[1]-1))/d)*y[3]                                    # S  
        return dy

    ### Input data structures ###
    Y=[]; T=[]                                                                              # Create empty lists
    p = [k, q, l, t_0, C, N_d, v_g, sigRes, B_tran, r_star, t_D0, t_ph, Gam]                # Parameters for odes
    
    # Time and initial conditions   
    t0 = 0; tEnd = 5e-9; dt = 1e-14       # Time constraints
    y0 = [N_0, roRes_0, ro_0, S_0]        # Initial conditions
    
    # Function to setup integrator with desired parameters
    r = ode(Laser_rates).set_integrator('lsoda', method = 'bdf')
    r.set_f_params(p).set_initial_value(y0, t0)

    ### Function to run simualtion & check ###
    while r.successful() and r.t+dt < tEnd:
        r.integrate(r.t + dt)
        Y.append(r.y)                                   # Makes a list of 1d arrays
        T.append(r.t)
    
    ### Format output ###
    Y       = np.array(Y)                               # Convert from list to 2d array
    N       = Y[:, 0] 
    rhoRes  = Y[:, 1] 
    rho     = Y[:, 2] 
    S       = Y[:, 3]
    
    ### Take last value for steady-state condition ###
    S_end.append(S[-1:])
    N_end.append(N[-1:])
    rho_end.append(rho[-1:])
    rhoRes_end.append(rhoRes[-1:]) 
    
    ### Save multiple outputs to data file ###
    #header = "T N rhoRes rho S"
    #filename = "data_%s.dat" % str(NEX)
    #np.savetxt(filename, np.column_stack((T, Y)), header=header)
    
    ### Save single outputs to data file ###
    #header = "T N rhoRes rho S"
    #np.savetxt('data2.dat', np.column_stack((T, Y)), header=header)
    #np.savetxt('data.dat', t_leff)


### Function for plotting dynamic time based curves ###
def PlotDynm():   
       
    ### Function for post solver steady-state LI calculations ###    
    P       = h*f*((S*V)/t_ph)*1e3                                                           # Power output (mW)
    QE      = P/I                                                                            # Convert for quantum efficiency 
    t_dwl   = 1/(1/t_0 + C*(N**2) + 4*C*N*N_d*(1-rho))                                       # t_dwl
    t_leff  = 1/(1/t_0 + C*(N**2) + B_tran*N)                                                # t_leff        
    e_shb   = 2*v_g*sigRes*(1-r_star)*t_leff                                                 # e_shb
    e_dwl   = 2*v_g*sigRes*r_star*t_dwl                                                      # e_dwl
    e       = e_shb + e_dwl                                                                  # Add the two parameters
    nu_r    = (1/(2*np.pi))*np.sqrt(((v_g*g_diff*S)/t_ph) - (((e)**2)*(S**2))/(4*(t_ph**2))) # nu_r
    Gam_r   = (1/(2*np.pi))*((e*S)/(2*t_ph))                                                 # Gam_r (Hz)

    ### Rescale values for plotting ###
    Gam_rPlot  = Gam_r/1e9                                                                   # Gam_r for plotting (GHz)
    nu_rPlot   = nu_r/1e9                                                                    # nu_r for plotting
    t_leffPlot = t_leff/1e-12                                                                # Convert time for plotting (ps)
    t_dwlPlot  = t_dwl/1e-12                                                                 # Convert time for plotting (ps)
    
    # Plotting S and N 
    fA, axarrA = plt.subplots(2, sharex=True)           # Two subplots, the axes array is 1-d
    axarrA[0].plot(T, N/1e4, 'G')
    axarrA[0].set_ylabel("N $(cm^{-2})$")
    axarrA[0].set_title('B=1e20($m^2s^{-1}$)')
    #axarrA[1].plot(T, t_leff/1e-12, 'B')
    axarrA[1].plot(T, S/1e6, 'B')
    axarrA[1].set_ylabel("Time (ps)")
    plt.savefig('CalcOut_S_N.png')

    # Plotting ro and roRes 
    fB, axarrB = plt.subplots(2, sharex=True)           # Two subplots, the axes array is 1-d
    axarrB[0].plot(T, rho, 'G')
    axarrB[0].set_ylabel("$rho$")
    axarrB[0].set_title('B=1e20($m^2s^{-1}$)')
    axarrB[1].plot(T, rhoRes, 'B')
    axarrB[1].set_ylabel("$rho_{Res}$")
    plt.savefig('CalcOut_ro_roRes.png')
    plt.show()
    return;


### Plot in steady-state mode ###
def PlotSS():
         
    ### Function for post solver steady-state LI calculations ###    
    P          = [h*f*((i*V)/t_ph)*1e3 for i in S_end]                                                                       # Power output (mW)
    QE         = [i/j for i,j in zip(P, iI)]                                                                                  # Convert for quantum efficiency 
    t_dwl      = [1/(1/t_0 + C*(i**2) + 4*C*i*N_d*(1-j)) for i,j in zip(N_end, rho_end)]                                     # t_dwl
    t_leff     = [1/(1/t_0 + C*(i**2) + B_tran*i) for i in N_end]                                                            # t_leff        
    e_shb      = [2*v_g*sigRes*(1-r_star)*i for i in t_leff]                                                                 # e_shb
    e_dwl      = [2*v_g*sigRes*r_star*i for i in t_dwl]                                                                      # e_dwl
    e          = [i+j for i,j in zip(e_shb, e_dwl)]                                                                          # Add the two parameters
    nu_r       = [(1/(2*np.pi))*np.sqrt(((v_g*g_diff*i)/t_ph) - (((j)**2)*(i**2))/(4*(t_ph**2))) for i,j in zip(S_end, e)]   # nu_r
    Gam_r      = [(1/(2*np.pi))*((i*j)/(2*t_ph)) for i,j in zip(e, S_end)]                                                   # Gam_r (Hz)

    ### Rescale values for plotting ###
    Gam_rPlot  = [i/1e9 for i in Gam_r]                                                                                      # Gam_r for plotting (GHz)
    nu_rPlot   = [i/1e9 for i in nu_r]                                                                                       # nu_r for plotting
    t_leffPlot = [i/1e-12 for i in t_leff]                                                                                   # Convert time for plotting (ps)
    t_dwlPlot  = [i/1e-12 for i in t_dwl]                                                                                    # Convert time for plotting (ps)
    
    ### Plotting two parameters on one plot ###
    fig, ax1 = plt.subplots()
    ax2 = ax1.twinx()
    ax1.plot(iI, nu_rPlot, 'g-')
    ax2.plot(iI, P, 'b-')
    ax1.set_xlabel('Current (mA)')
    ax1.set_ylabel('Relaxation Freq (GHz)', color='g')
    ax2.set_ylabel('Power (mW)', color='b')
    plt.show()

    ### Function for plotting single steady-state LI curves ###
    #plt.plot(iI, t_leffPlot, 'B')
    #plt.plot(i, P, 'G')
    #plt.legend()
    #plt.xlim(0, 25)
    #plt.ylim(0, 1e-9)
    #plt.title('C=1e-22')
    #plt.ylabel('Relaxation Time (ps)')
    #plt.xlabel('Current (mA)')
    #plt.show()
    return; 


### Dynamic mode ###
if(Mode == 0):
    DefCallSolv(J)
    PlotDynm()


### Steady-state mode ###
if(Mode == 1):
    NEX = 0
    for x1, x2 in zip(iJ, iI):
        DefCallSolv(x1, x2) 
    PlotSS()

