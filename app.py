import streamlit as st
import numpy as np
import matplotlib.pyplot as plt

# ---------- Functions to generate curves ----------
#############################
# Pile Geometry and Loading #
#############################

# Calculation of forces and moments:

# Rotor thrust:
h_w = 30                                    # [m] water mean level
h_h = 150                                   # [m] hub level from water mean level
rho_a = 1.2                                 # [kg/m3] volumetric mass of air at 20 degrees Celsius
D = 240                                     # [m] diameter of the rotor
A_R = np.pi * (D/2)**2                      # [m2] swept area by the rotor
C_T = 0.5                                   # [-] thrust coefficient
U_hh = 5 - 0.05*(-h_w-h_h+30)                # [m/s] wind speed at the hub height
F_rot = 1/2 * rho_a * A_R * C_T * U_hh**2   # [N] Rotor thrust force
M_rot = F_rot * (h_h+h_w)                    # [N.m] Overturning moment by the rotor
# print("The rotor thrust force is {0:.2f} MN and the associated overturning moment at the mudline is {1:.0f} MN.m".format(F_rot/10**6,M_rot/10**6))

# Tower drag:
C_D = 0.4           # [-] Drag coefficient
def lin_Drag_tow(z):
    dF_tow = 1/2 * rho_a * C_D * (107/10+7/300*z) * (5-0.05*(z+30))**2  # [N/m] linear drag force on the tower
    dM_tow = dF_tow * abs(z)                                            # [N.m/m] overtuning moment for the linear drag force on the tower
    return (dF_tow,dM_tow)
F_tow = 0 
M_tow = 0
for i in range(-180,-29):
    dF_tow,dM_tow = lin_Drag_tow(i)
    F_tow += dF_tow
    M_tow += dM_tow
# print("The tower drag force is {0:.1f} kN and the associated overturning moment at the mudline is {1:.2f} MN.m".format(F_tow/10**3,M_tow/10**6))

# Force from the waves
rho_w = 1030        # [kg/m3] approximation of sea water volumetric mass
H_m = 10            # [m] characteristic wave height
T = 15              # [s] wave period
omega = 2*np.pi/T   # [rad/s] amgular frequency of the wave
lam = 200           # [m] wavelength
k = 2*np.pi/lam     # [rad/m] angular wave number
C_m = 2             # [-] inertia coefficient
def lin_FwI(z,D,t):
    COSI1 = np.cosh(-k*z) / np.sinh(k*h_w) * np.sin(omega*t)
    wdot = - H_m/2 * omega**2 * COSI1
    A = np.pi*D**2 / 4
    dF_wI = C_m * rho_w * A * wdot
    dM_wI = dF_wI * abs(z)
    return (dF_wI,dM_wI)
def lin_FwD(z,D,t):
    COSI2 = np.cosh(-k*z) / np.sinh(k*h_w) * np.cos(omega*t)
    w = H_m/2 * omega * COSI2
    dF_wD = 1/2 * rho_w * D * C_D * w*np.abs(w)
    dM_wD = dF_wD * abs(z)
    return (dF_wD,dM_wD)
def FwI(h,D,t):
    F_wI = 0 
    M_wI = 0
    nb = 50
    for i in np.linspace(-h+h/nb,0-h/nb,nb):
        dF,dM = lin_FwI(i,D,t)
        F_wI += dF * (h/nb)
        M_wI += dM * (h/nb)
    return (F_wI,M_wI)
def FwD(h,D,t):
    F_wD = 0 
    M_wD = 0
    nb = 50
    for i in np.linspace(-h+h/nb,0-h/nb,nb):
        dF,dM = lin_FwD(i,D,t)
        F_wD += dF * (h/nb)
        M_wD += dM * (h/nb)
    return (F_wD,M_wD)
#plt.plot(np.linspace(0,30,100),[1.35*(FwD(30,12,i)[1]+FwI(30,12,i)[1])/10**6 for i in np.linspace(0,30,100)])


def lin_FM_wav_max(z,D):
    dF_w_max = 1/8 * rho_w * H_m * D * omega**2 * np.cosh(-k*z) / np.sinh(k*h_w) \
        * (C_D * H_m * np.cosh(-k*z) / np.sinh(k*h_w) + C_m * np.pi * D)
    dM_w_max = dF_w_max * abs(z)
    return (dF_w_max,dM_w_max)
def tot_FM_wave(D):
    F_wav = 0 
    M_wav = 0
    for i in range(-30,1):
        dF_wav,dM_wav = lin_FM_wav_max(i,D)
        F_wav += dF_wav
        M_wav += dM_wav
    return(F_wav,M_wav)
# def tot_FM_wave(D):
#     F_wav = 0 
#     M_wav = 0
#     nb = 50
#     h=30
#     c=1
#     for i in np.linspace(-h+h/nb,0-h/nb,nb):
#         # print('Depth:',i)
#         # print('number:',c)
#         c+=1
#         dF_wav,dM_wav = lin_FM_wav_max(i,D)
#         F_wav += dF_wav * (h/nb)
#         M_wav += dM_wav * (h/nb)
#     return(F_wav,M_wav)
# F_wav10,M_wav10 = tot_FM_wave(10)
# print("For D=10m, the max. force born from the wave is {0:.1f} MN and the associated overturning moment at the mudline is {1:.2f} MN.m".format(F_wav10/10**6,M_wav10/10**6))

# Total environmental loading:
C_s = 1.35 # [-] coefficient of safety for environmental loading
def tot_FM(D):
    F_wav,M_wav = tot_FM_wave(D)
    F = C_s * ( F_rot + F_tow + F_wav )
    M = C_s * ( M_rot + M_tow + M_wav)
    return (F,M)    
# F10,M10 = tot_FM(10)
# print("For D=10m, the design horizontal force is {0:.1f} MN and the associated overturning moment at the mudline is {1:.2f} MN.m".format(F10/10**6,M10/10**6))

########################
# P-Y Curve Definition #
########################

def API_sand(y,z,D,phideg,gamma):
    if z == 0:
        return 10**(-3)
    A = max(3-0.8*z/D,0.9)
    k = 22 * 10**6 # N/m3
    phi = phideg/180*np.pi
    alpha = phi/2
    beta = np.pi/4 + alpha
    K0 = 0.4
    Ka = (1-np.sin(phi)) / (1+np.sin(phi))
    C1 = (np.tan(beta))**2 * np.tan(alpha) / np.tan(beta-phi) \
        + K0 * (np.tan(phi)*np.sin(beta)/(np.cos(alpha)*np.tan(beta-phi)) + np.tan(beta)*(np.tan(phi)*np.sin(beta)-np.tan(alpha))) 
    # print('C1',C1)
    C2 = np.tan(beta)/np.tan(beta-phi) - Ka
    # print('C2',C2)
    C3 = Ka * ( (np.tan(beta)**8-1) + K0*np.tan(phi)*np.tan(beta)**4 )
    # print('C3',C3)
    
    pu = min ( (C1*z+C2*D)*gamma*z , C3*D*gamma*z)
    # print('pu in MN',pu/10**6)
    p = A * pu * np.tanh( k*z/(A*pu) * y )
    return p

# plt.plot([x/1000 for x in range(500)],[API_sand(x/1000,5,10,35,11000) for x in range(500)])

#######################
# P-Y analysis scheme #
#######################

def py_analysis(L=30.0, D=10.0, N_el=10, N_it=10, plot = 0, solv = 1, t_fix = 0 ):
    """
    Models a laterally loaded pile using the p-y method.
    Solving EI*d4y/dz4 + ky = 0 using the finite difference method.

    Inputs:
    L           Embedded length of the pile (m)
    D           Outer diameter of pile (m)
    t           Wall thickness of pile (m)
    V           Horizontal force applied to the pile  (N)
    M           Moment at pile head (N-m)
    N_el        Number of elements (50 by default)
    N_it        Number of iterations

    Output:
    ------
    y           - Lateral displacement at each node, length = n + 5, (n+1) real nodes and 4 imaginary nodes
    z           - Vector of node locations along pile
    """
    E=210e9                                     # Elastic modulus of pile material (Pa)
    # Pile geometry
    if t_fix == 0:
        t = min( (6.35/1000 + D/100), 0.09 )    # [m] wall thickness
    else: t = float(t_fix)
    I       = np.pi/4 * ((D/2)**4 - (D/2-t)**4) # Second moment of area
    EI      = E * I
    h       = L/N_el                            # Element size
    N_od    = (N_el+1)+4                        # (n+1) Real + 4 Imaginary nodes
    W = np.pi*D*t*(L+h_w)*7.85                  # [t] Weight of the monopile
    # Loads
    H,M0 = tot_FM(D)
    # print("The design horizontal force is {0:.1f} MN and the associated overturning moment at the mudline is {1:.2f} MN.m".format(H/10**6,M0/10**6))
    # Soil data
    phideg = 35
    gamma = 11000

    # Array for displacements at nodes, including imaginary nodes.
    y = np.ones(N_od)*(0.001*D)  # Arbitrary uniform initial value

    # Initialize and assemble array/list of p-y curves at each real node
    z = np.zeros(N_od)
    k_secant = np.zeros(N_od)

    for i in [0, 1]:        # Top two imaginary nodes
        z[i] = (i-2)*h
        k_secant[i] = 0.0

    for i in range(2, N_el+3):  # Real nodes
        z[i] = (i-2)*h
        k_secant[i] = API_sand(y[i],z[i],D,phideg,gamma)/y[i]

    for i in [N_el+3, N_el+4]:   # Bottom two imaginary nodes
        z[i] = (i-2)*h
        k_secant[i] = 0.0
    if solv == 1:
        func = solver_1
    for j in range(N_it):
        y = func( N_od, L, h, EI, H, M0, k_secant)
        for i in range(2, N_el+3):
            k_secant[i] = API_sand(y[i],z[i],D,phideg,gamma)/y[i]
    M = np.zeros(N_od-4)
    V = np.zeros(N_od-4)
    sig = np.zeros(N_od-4)
    for i in range(2,N_od-2):
        M[i-2] = EI * (y[i-1]-2*y[i]+y[i+1]) / h**2                 # [N.m] bending moment 
        V[i-2] = EI/2 * (-y[i-2]+2*y[i-1]-2*y[i+1]+y[i+2]) / h**3   # [N] shear
        sig[i-2] = M[i-2] * (D/2) / I                               # [Pa] maximum stress
    kh = 11.93*10**6
    La=100
    N_ela = 250
    N_oda = N_ela+1
    yt,Vt,Mt,le =theo_curves(La,D,N_ela,H,M0,EI,kh)
    return y[2:-2], z[2:-2], M, V, sig, W, yt, Vt, Mt, le

def theo_curves(L,D,N_el,H,M,EI,kh):
    lamb = (kh*D/(4*EI))**0.25
    le = 1/ lamb
    print('Lambda:',lamb)
    print('le:',le)
    # print('EI:',EI)
    N_od = N_el+1   # Careful, no ghost nodes here
    yt = np.zeros(N_od)
    Vt = np.zeros(N_od)
    Mt = np.zeros(N_od)
    for i in range(N_od):
        z = i/N_od*L
        yt[i] = -1/(2*EI*lamb**2) * np.exp(-lamb*z) * ( M*np.sin(lamb*z) - (H/lamb+M)*np.cos(lamb*z) )
        Mt[i] = np.exp(-lamb*z) * ( M*np.cos(lamb*z) + (H/lamb+M)  *np.sin(lamb*z) )
        Vt[i] = np.exp(-lamb*z) * ( H*np.cos(lamb*z) - (2*lamb*M+H)*np.sin(lamb*z) )
    return (yt,Vt,Mt,le)

##########
# Solver #
##########

def solver_1(N_od, L, h, EI, H, M, k_sec):
    '''Solves the finite difference equations from 'py_analysis'. This function should be run iteratively because the p-y curves are non-linear.
    Inputs:
    -----
    N_od    - Total number of nodes
    h       - Element size
    EI      - Flexural rigidity of pile
    V       - Shear at pile head
    M       - Moment at pile head/tip
    k_sec   - Secant stiffness from p-y curves

    Output:
    ------
    y       - updated lateral displacement at each node
    '''
    from scipy import linalg

    # Initialize and assemble matrix
    K = np.zeros((N_od, N_od))

    # (n+1) finite difference equations for (n+1) real nodes
    for i in range(0, N_od-4):
        K[i, i] = 1.0
        K[i, i+1] = -4.0
        K[i, i+2] = 6.0 + k_sec[i+2]*h**4/EI
        K[i, i+3] = -4.0
        K[i, i+4] = 1.0
    # Initialize vector b
    b = np.zeros(N_od)
    # Boundary conditions
    # Moment at pile head
    K[-4, 1:4] = [1.0,-2.0,1.0]
    b[-4] = M*h**2
    # Shear at pile head
    K[-3, 0:5] = [-1.0,2.0,0.0,-2.0,1.0]
    b[-3] = 2*H*h**3
    # Moment at pile tip
    K[-2, N_od-4:N_od-1] = [1.0,-2.0,1.0]
    b[-2] = 0
    # Shear at pile tip
    K[-1, N_od-5:N_od] = [-1.0,2.0,0.0,-2.0,1.0]
    b[-1] = 0

    y = linalg.solve(EI*K, b)

    return y





# ---------- Streamlit App ----------
st.title("Monopile monotonic response")

# Inputs
L = st.number_input("Embedment depth L [m]", min_value=1.0, max_value=100.0, value=20.0, step=1.0)
D = st.number_input("Pile diameter D [m]", min_value=0.5, max_value=20.0, value=5.0, step=0.1)

if st.button("Calculate"):

    y, z, M, V, sig, W,yt,Vt,Mt,le = py_analysis(L, D, N_el=10, N_it=10, plot = 0, solv = 1, t_fix = 0 )
    H,M0 = tot_FM(D)
    # Plots
    kh = 11.93*10**6
    La=100
    N_ela = 250
    N_oda = N_ela+1
    z_plot = 40
    E=210e9                                     # Elastic modulus of pile material (Pa)
    # Pile geometry
    t = min( (6.35/1000 + D/100), 0.09 )    # [m] wall thickness
    I       = np.pi/4 * ((D/2)**4 - (D/2-t)**4) # Second moment of area
    EI      = E * I
    fig, axes = plt.subplots(4,1,figsize=(10,40))
    fig.suptitle('For D={0:.2f} m, L={1:.2f} m and t={2:.0f} mm, we get y(0)/D={3:.3f}, a bending stress safety factor {5:.2f}, and Steel amount {4:.0f} t\n'\
            .format(D,L,t*1000,y[2]/D,W,355*10**6/(max(np.abs(M))*D/2/I))+\
        'The theoretical profile is pictured in dashed black for kh={0:.1f} MN'\
            .format(kh/10**6) )
    plt.subplots_adjust(bottom=0.1,top=0.9,left=0.05,right=0.98,wspace=0.25, hspace=0.3)
    axes[0].plot(y,z)
    axes[0].plot(yt,-np.linspace(0,La,N_oda),color='k',linestyle='--')
    axes[0].plot([D/10,D/10],[z_plot,0], color='r',linestyle=':')
    axes[0].set_title('Lateral displacement [m]')
    axes[1].plot(M/10**6,z)
    axes[1].plot(Mt/10**6,-np.linspace(0,La,N_oda),color='k',linestyle='--')
    axes[1].plot([M0/10**6,M0/10**6],[z_plot,0],color='k',linestyle=':')
    axes[1].set_title('Bending moment [MN.m]')
    axes[2].plot(V/10**6,z)
    axes[2].plot(Vt/10**6,-np.linspace(0,La,N_oda),color='k',linestyle='--')
    axes[2].plot([H/10**6,H/10**6],[z_plot,0],color='k',linestyle=':')
    AA = np.pi*D*t
    taum = 2*AA*355/3**0.5/np.pi
    axes[2].plot([taum,taum],[z_plot,0],color='r',linestyle=':')
    axes[2].plot([-taum,-taum],[z_plot,0],color='r',linestyle=':')
    axes[2].set_title('Shear force [MN]')
    axes[3].plot(sig/10**6,z)
    axes[3].plot(M*D/2/I/2/10**6+((M*D/2/I/2)**2+(V/(2*np.pi*D/2*t))**2)**0.5/10**6,z,color='g',linestyle='--')
    axes[3].plot([355,355],[z_plot,0],color='r',linestyle=':')
    axes[3].set_title('Bending stress [MPa]')
    axes[1].set_xlim([-0.15*D, 0.15*D])
    for i in range(4):
            axes[i].set_ylim([z_plot, 0])
            axes[i].grid(linestyle=':',linewidth=1)
    axes[1].set_ylabel('Depth [m]')

    st.pyplot(fig)






