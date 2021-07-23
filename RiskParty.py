#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Jul 19 16:26:26 2021

@author: ccm
"""
import numpy as np
from scipy.optimize import minimize

# Risk Parity

def risk_parity_function(weights , cov):
    sigma = (np.dot(np.dot(weights , cov),weights.T))**(1/2)
    n = len(weights)
    temp = np.dot(cov,weights.T) / sigma
    MC = np.multiply(weights,temp.T)
    
    return sum((sigma/n - MC)**(2))

def risk_parity(data):
    cov = data.cov()
    n = cov.shape[0]
    weights = np.ones(n)/n
    cons = ({'type' : 'ineq' , 'fun' :  lambda x : 1-sum(x)})
    bnds = [(0,1) for i in weights]
    res = minimize(risk_parity_function , x0=weights , args = (cov) , method = 'SLSQP' , constraints = cons , bounds = bnds , tol = 1e-30)
    return(res.x)
