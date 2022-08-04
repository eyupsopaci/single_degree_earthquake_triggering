#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Jul  2 11:13:02 2022

@author: esopaci
contact eyup.sopaci@metu.edu.tr
"""
import numpy as np
import numba
# import timeit


# STATE EVOLUTION FORMULAS [dieterich (slowness or ageing), ruina (slip), 
# perrin (modified slowness) and nagata et. al (2012)]
@numba.njit()
def state_evol_d(y,b,dc,c,v0, dtau, alpha, sigma_rate):
    """
    State evolution for slawness law proposed by Dieterich (1979).
    
    J. H. Dieterich, “Modeling of rock friction 1. Experimental results and constitu-
    tive equations,” Journal of Geophysical Research: Solid Earth, vol. 84, no. B5,
    pp. 2161–2168, 1979.
    
    Parameters
    ----------
    y : numpy array
        intial values : [velocity, state, normal_stress, elastic_stress].
    b : float
        state evolution effect parameter .
    dc : float
        critical slip distance.
    c : float
        shear-stress state coupling parameter.
    v0 : float
        reference velocity 1.0E-06.
    dtau : float
        stress rate [only applied if the state evolution law is Nagata et.al. (2012)].
    alpha : float
        normal stress-state coupling.
    sigma_rate : float
        effective normal stress rate.

    Returns
    -------
    float
        state evolution rate.

    """
    return b*v0/dc * np.exp(-y[1]/b) - b/dc*y[0] - alpha * sigma_rate / y[2]
    
@numba.njit()
def state_evol_r(y,b,dc,c,v0,dtau, alpha, sigma_rate):
    """
    State evolution for slawness law proposed by Ruina (1983).
    
    A. Ruina, “Slip instability and state variable friction laws.,” Journal of Geophys-
    ical Research, vol. 88, no. B12, pp. 10359–10370, 1983.
    
    Parameters
    ----------
    y : numpy array
        intial values : [velocity, state, normal_stress, elastic_stress].
    b : float
        state evolution effect parameter .
    dc : float
        critical slip distance.
    c : float
        shear-stress state coupling parameter.
    v0 : float
        reference velocity 1.0E-06.
    dtau : float
        stress rate [only applied if the state evolution law is Nagata et.al. (2012)].
    alpha : float
        normal stress-state coupling.
    sigma_rate : float
        effective normal stress rate.

    Returns
    -------
    float
        state evolution rate.

    """
    return -y[0]/dc*(y[1]+b*np.log(y[0]/v0)) - alpha * sigma_rate / y[2]

@numba.njit()
def state_evol_p(y,b,dc,c,v0, dtau, alpha, sigma_rate):
    """
    State evolution for slawness law proposed by Perrin et.al. (1995).
    
    G. Perrin, J. R. Rice, and G. Zheng, “Self-healing slip pulse on a frictional sur-
    face,” Journal of the Mechanics and Physics of Solids, vol. 43, no. 9, pp. 1461–
    1495, 1995.
    
    Parameters
    ----------
    y : numpy array
        intial values : [velocity, state, normal_stress, elastic_stress].
    b : float
        state evolution effect parameter .
    dc : float
        critical slip distance.
    c : float
        shear-stress state coupling parameter.
    v0 : float
        reference velocity 1.0E-06.
    dtau : float
        stress rate [only applied if the state evolution law is Nagata et.al. (2012)].
    alpha : float
        normal stress-state coupling.
    sigma_rate : float
        effective normal stress rate.

    Returns
    -------
    float
        state evolution rate.

    """
    return 0.5*b*v0/dc * np.exp(-y[1]/b) - 0.5*b/dc/v0*y[0]**2*np.exp(y[1]/b) - alpha * sigma_rate / y[2]

@numba.njit()
def state_evol_n(y,b,dc,c,v0,dtau, alpha, sigma_rate):
    """
    State evolution for slawness law proposed by Nagata et.al. (2012).
    
    K. Nagata, M. Nakatani, and S. Yoshida, “A revised rate-and state-dependent
    friction law obtained by constraining constitutive and evolution laws separately
    with laboratory data,” Journal of Geophysical Research: Solid Earth, vol. 117,
    no. B2, 2012.
    
    Parameters
    ----------
    y : numpy array
        intial values : [velocity, state, normal_stress, elastic_stress].
    b : float
        state evolution effect parameter .
    dc : float
        critical slip distance.
    c : float
        shear-stress state coupling parameter.
    v0 : float
        reference velocity 1.0E-06.
    dtau : float
        stress rate [only applied if the state evolution law is Nagata et.al. (2012)].
    alpha : float
        normal stress-state coupling.
    sigma_rate : float
        effective normal stress rate.

    Returns
    -------
    float
        state evolution rate.

    """    
    return b*v0/dc * np.exp(-y[1]/b) - b*y[0]/dc - c/y[2]*dtau - alpha * sigma_rate / y[2] 
######################################################################3

# Rate-and-State Friction
@numba.njit()
def friction(y, a, f0, v0):
    """
    Rate-and-State friction

    Parameters
    ----------
    y : numpy array
        intial values : [velocity, state, normal_stress, elastic_stress].
    a : float
        direct velocity effect parameter.
    f0 : float
        reference friction at reference velocity v0.
    v0 : float
        reference velocity.

    Returns
    -------
    friction value of the interface

    """
    return( f0 + y[1] + a* np.log(y[0]/v0))
############################################

# Analytical relation for Computing Static triggering (gaussian shape)
@numba.njit
def stattic_trig(t, tp, CFF):
    """
    This fucntion is an approximation for static triggering rate. 
    The outcome is a gaussian shape, which generate a smooth step-like shape 
    within 6 seconds

    Parameters
    ----------
    t : float
        time [seconds].
    tp : float
        onsettime [seconds].
    CFF : float
        amplitude of static stress [Pa].

    Returns
    -------
    float
        static stress rate at time t.

    """
    return CFF / np.sqrt(2.*np.pi) * np.exp(-0.5 * (t - tp - 3)**2)

##################################################################3

# # Quasi-static & full-dynamic approximation
# @numba.njit
# def fun(t, y, K, v0, Vpl, m, a, b, dc, c, f0, alpha, phi, St, Xt):
#     """
#     THis function provides explicity ODE for quasi-static & full-dynamic
#     simulation strategy.

#     Parameters
#     ----------
#     t : float
#         time [second].
#     y : numpy array
#         intial values : [velocity, state, normal_stress, elastic_stress].
#     K : float
#         elastic spring constant Gamma*G/L, where G shear modulus, Length of
#         asperity, Gamma is the constant depndeing on the asperity shape
#     v0 : float
#         reference velocity.
#     Vpl : float
#         average slip rate on the fault.
#     m : float
#         mass per unit area.
#     a : float
#         direct velocity effect parameter.
#     b : float
#         state evolution effect parameter.
#     dc : float
#         critical slip distance.
#     c : float
#         shear stres - state coupling parameter.
#     f0 : float
#         reference friction at reference velocity v0.
#     alpha : float
#         normal stress-state coupling.
#     phi : float
#         reverse dipping angle.
#     St : float
#         static triggering at time t in [Pa].
#     Xt : float
#         dynamic triggering at time t in [m/s].

#     Returns
#     -------
#     numpy array
#         [slip_rate, state_rate, normal_stress_rate, elastic_stress_rate ].

#     """
    
#     # if state law is Nagata et al. (2012) scale the parameters as follows
#     # aminb =a-b
#     # a = a * (c + 1)
#     # b = a - aminb
#     # dc = dc / (c + 1)
#     # alpha = alpha * (c + 1)
    
#     A = y[2] * a 

#     taudot = K * (Vpl + Xt - y[0])
#     sigmadot = taudot * np.tan(phi)
    
#     # change the state evolution by adding _d, _r, _p or _n for dieterich, ruina, perrin or nagata
#     thetadot = state_evol_d(y,b,dc,c,v0, taudot, alpha, sigmadot)

#     # quasi-static if slip velocity is lower than a critical value    
#     if y[0] < 1e-2:  
#         y[0] = v0 * np.exp((y[3] + St - f0 * y[2] - y[1] * y[2]) / A)
#         vdot = ( taudot - y[2] * thetadot ) / ( A / y[0])        
#     else:
#         # full-dynamic
#         vdot = (y[3] - friction(y, a, f0, v0)*y[2]) / m  
        
#     return np.array([vdot, thetadot, sigmadot, taudot])
##############################################################################

# quasi-dynamic approximation
@numba.njit
def fun( t, y,K, v0, Vpl, nu, a, b, dc, c, f0, alpha, phi, St, Xt ):
    """
    This function provides explicity ODE for quasi-dynamic
    simulation strategy.

    Parameters
    ----------
    t : float
        time [second].
    y : numpy array
        intial values : [velocity, state, normal_stress, elastic_stress].
    K : float
        elastic spring constant Gamma*G/L, where G shear modulus, Length of
        asperity, Gamma is the constant depndeing on the asperity shape
    v0 : float
        reference velocity.
    Vpl : float
        average slip rate on the fault.
    nu : float
        damping term G/vs. G:shear modulus, vs:shear velocity
    a : float
        direct velocity effect parameter.
    b : float
        state evolution effect parameter.
    dc : float
        critical slip distance.
    c : float
        shear stres - state coupling parameter.
    f0 : float
        reference friction at reference velocity v0.
    alpha : float
        normal stress-state coupling.
    phi : float
        reverse dipping angle.
    St : float
        static triggering at time t in [Pa].
    Xt : float
        dynamic triggering at time t in [m/s].

    Returns
    -------
    numpy array
        [slip_rate, state_rate, normal_stress_rate, elastic_stress_rate ].

    """
    # # if state law is Nagata scale the parameters as follows
    # aminb =a-b
    # a = a * (c + 1)
    # b = a - aminb
    # dc = dc / (c + 1)
    # alpha = alpha * (c + 1)
    
    stress_rate = K * (Vpl + Xt - y[0]) + St
    sigma_rate = stress_rate * np.tan(phi)
    state_rate = state_evol_d(y,b,dc,c,v0,stress_rate, alpha, sigma_rate)
    vdot = ( stress_rate - y[2] * state_rate ) / ( a * y[2] / y[0] + nu )
    
    return np.array([vdot, state_rate, sigma_rate, stress_rate])


# Runge-Kutta Cash-Karp solver for quasi-static & full-dynamic or quasi-dynamic approximations
@numba.njit
def rkck(x, y, h, K, v0, Vpl, m, a, b, dc, c, f0, alpha, phi, strategy = 0, St=0, Xt=0, tol=1.0e-6, xmin=1e-20, xmax = 1e7):
    """
    This is the adaptive step Runge Kutta solver using coefficients by 
    Cash & Karp (1990)
    J. R. Cash and A. H. Karp, “A Variable Order Runge-Kutta Method for Initial
    Value Problems with Rapidly Varying Right-Hand Sides,” ACM Transactions
    on Mathematical Software (TOMS), vol. 16, no. 3, pp. 201–222, 1990
    Parameters
    ----------
    x : TYPE
        DESCRIPTION.
    y : TYPE
        DESCRIPTION.
    h : TYPE
        DESCRIPTION.
    K : TYPE
        DESCRIPTION.
    v0 : TYPE
        DESCRIPTION.
    Vpl : TYPE
        DESCRIPTION.
    m : TYPE
        DESCRIPTION.
    a : TYPE
        DESCRIPTION.
    b : TYPE
        DESCRIPTION.
    dc : TYPE
        DESCRIPTION.
    c : TYPE
        DESCRIPTION.
    f0 : TYPE
        DESCRIPTION.
    alpha : TYPE
        DESCRIPTION.
    phi : TYPE
        DESCRIPTION.
    strategy : TYPE, optional
        DESCRIPTION. The default is 0.
    St : TYPE, optional
        DESCRIPTION. The default is 0.
    Xt : TYPE, optional
        DESCRIPTION. The default is 0.
    tol : TYPE, optional
        DESCRIPTION. The default is 1.0e-6.
    xmin : TYPE, optional
        DESCRIPTION. The default is 1e-20.
    xmax : TYPE, optional
        DESCRIPTION. The default is 1e7.

    Returns
    -------
    xn : TYPE
        next xn+h
    yn : TYPE
        next values. yn+dyn/dxn*h
    h : TYPE
        step.
    err : TYPE
        numericsal error.

    """
    # NOTE: for quasi-static & full dynamic m = (T/2pi)^2*K
    # for quasi-dynamic approximation simply ignore the m notation and consider m as nu=G/2Vs inertia effect
    err = 2 * tol
    istop=5
    ii = 0
    while (err > tol):
        k1 = h*fun(x,y,
                               K, v0, Vpl, m, a, b, dc, c, f0, alpha, phi, St, Xt)
        k2 = h*fun(x+(1/5)*h,y+((1/5)*k1),
                                K, v0, Vpl, m, a, b, dc, c, f0, alpha, phi, St, Xt)
        k3 = h*fun(x+(3/10)*h,y+((3/40)*k1)+((9/40)*k2),
                                K, v0, Vpl, m, a, b, dc, c, f0, alpha, phi, St, Xt)
        k4 = h*fun(x+(3/5)*h,y+((3/10)*k1)-((9/10)*k2)+((6/5)*k3),
                                K, v0, Vpl, m, a, b, dc, c, f0, alpha, phi, St, Xt)
        k5 = h*fun(x+(1/1)*h,y-((11/54)*k1)+((5/2)*k2)-((70/27)*k3)+((35/27)*k4),
                                K, v0, Vpl, m, a, b, dc, c, f0, alpha, phi, St, Xt)
        k6 = h*fun(x+(7/8)*h,y+((1631/55296)*k1)+((175/512)*k2)+((575/13824)*k3)+((44275/110592)*k4)+((253/4096)*k5),
                                K, v0, Vpl, m, a, b, dc, c, f0, alpha, phi, St, Xt)
        dy4 = ((37/378)*k1)+((250/621)*k3)+((125/594)*k4)+((512/1771)*k6)
        dy5 = ((2825/27648)*k1)+((18575/48384)*k3)+((13525/55296)*k4)+((277/14336)*k5)+((1/4)*k6)
        err = 1e-2*tol+max(np.abs(dy4-dy5))
        h = max(min(0.95 * h * (tol/err)**(1/5), xmax), xmin)
        if ii>=istop:
            break
        ii+=1
        
        # h = 0.8 * h * (tol*h/err)**(1/4)
    xn = x + h
    yn = y + dy4
    return xn, yn, h, err
