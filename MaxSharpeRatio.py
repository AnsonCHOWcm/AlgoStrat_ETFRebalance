#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Jul 19 16:31:02 2021

@author: ccm
"""

import numpy as np
from scipy.optimize import minimize

# MSR

def Sharpe_Ratio_function(weights, ret):
 mu = ret.mean()
 cov = ret.cov()
 numator = np.dot(weights , mu - 0.016/12)
 demonmator = (np.dot(np.dot(weights,cov),weights))**(1/2)
 return (-1 * numator / demonmator)

def Maximum_Sharpe_Ratio(data, long = 1):
 cov = data.cov()
 n = cov.shape[0]
 weights = np.ones(n) /n
 cons = ({'type': 'ineq', 'fun': lambda x: 1 - sum(x)}) 
 bnds = [(0 ,1) for i in weights]
 if long == 1:
  res = minimize(Sharpe_Ratio_function, x0 = weights, args = (data), method = 'SLSQP', constraints = cons,
  bounds = bnds, tol = 1e-30)
 else:
  res = minimize(Sharpe_Ratio_function, x0 = weights, args = (data), method = 'SLSQP', constraints = cons, tol = 1e-30)
 return res.x
    