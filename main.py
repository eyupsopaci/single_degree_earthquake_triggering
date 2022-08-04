#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Aug  1 17:49:40 2022

@author: esopaci
contact eyup.sopaci@metu.edu.tr
"""

from src import *
from process import *
import numpy as np
import pandas as pd
import obspy
import multiprocessing
import os
from itertools import repeat    

def main():
    
    pars = dict({       
    "tyr": float(365 * 3600 * 24), # time conversion
    #fault paramters
    "a": 0.001,
    "b": 0.006,
    "c": 20,
    "dc": 0.001,
    "V0": 1e-6,
    "f0":0.6,
    "G":30.0E9,
    "sigma": 100e6,
    "L": 5.0E3,
    "phi": 0.0 *np.pi/180,
    "alpha": 0.5, 
    "cs":3.0E3,
    "T": 5,
    "tol": 1e-6
    })
    
    pars["K"] = pars["G"]/pars["L"]
    pars["m"] = np.power(0.5*pars["T"]/np.pi,2)*pars["K"]
    pars["nu"] = 0.5*pars["G"]/pars["cs"]
    pars["Vpl"]= 20.0e-3/pars["tyr"]
    
    
    # simulation paramters
    t0 = 0.0 
    dt = 1e-6
    tf = 2e10
    y0 = np.array([pars["Vpl"]*0.99, 0.1, pars["sigma"], pars["sigma"]*pars["f0"]])
    
    sol_u = unperturbed(t0, y0, tf, dt, pars)
    
    data = {}
    cols = ["v", "theta", "sigma", "tau","err", "dt", "t"]
    for i in range(len(cols)):
        data["%s"%cols[i]] = sol_u[:,i]
    dfu = pd.DataFrame(data)    
    dfu1 = dfu.loc[dfu.v >1, "v"]
    dyn_index = dfu1.loc[np.insert(np.diff(dfu1.index.values), 0, 2) != 1].index.values
    
    sind = dyn_index[-2]-1000
    ti = dfu.t[sind]
    dti = dfu.dt[sind]
    yi = np.array([dfu.v[sind],
                    dfu.theta[sind],
                    dfu.sigma[sind],
                    dfu.tau[sind],
                    ])
    tfi = dfu.t[dyn_index[-1]]
     
    st = obspy.read("data.mseed").select(channel="E")
    st.detrend()
    st[0].data = st[0].data - st[0].data.mean()  
    st.filter("bandpass", freqmin=0.05, freqmax=5)
    st.decimate(10)
    st[0].data[-20:]=0; st[0].data[:20]=0
    
    

    tbs = [x**1.1 for x in np.arange(0.5,32)]

    
    for stx, CFF in [(0,1e5), (2,1e5), (5,0.), (5,1e5), (10,0.), (10,1e5)]:

        with multiprocessing.Pool() as p:
            result = p.starmap(triggered, 
                               zip(repeat(ti), 
                                   repeat(yi), 
                                   tbs, 
                                   repeat(tfi), 
                                   repeat(dti), 
                                   repeat(pars), 
                                   repeat(st), 
                                   repeat(CFF), 
                                   repeat(stx),
                                   )
                               )
            print(result)
        

if __name__ == '__main__':
    main()

