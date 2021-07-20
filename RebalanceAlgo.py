#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Jul 19 16:35:12 2021

@author: ccm
"""
import numpy as np
import pandas as pd
import RiskParty
import MaxSharpeRatio

## Read Data 

VOO_price_df = pd.read_csv('VOO.csv' , index_col = 0)
QQQ_price_df = pd.read_csv('QQQ.csv' , index_col = 0)
DIA_price_df = pd.read_csv('DIA.csv' , index_col = 0)

## Compute the Ret 

VOO_ret_df = VOO_price_df.pct_change().dropna()
QQQ_ret_df = QQQ_price_df.pct_change().dropna()
DIA_ret_df = DIA_price_df.pct_change().dropna()

## Set the Hyperparameter

look_back_window = 20

## Data Preprocess

def DataSetGeneration (data , window):
    
    N = len(data)
    Prev_Info = []
    Current_Info = []
    
    for i in range(N - window):
        X = data.iloc[i : i+window]
        y = data.iloc[i+window]
        Prev_Info.append(X)
        Current_Info.append(y)
        
    return Prev_Info , Current_Info

ETF_ret_df = pd.concat([VOO_ret_df , QQQ_ret_df , DIA_ret_df]  , axis = 1)

ETF_ret_df.columns = ['VOO' , 'QQQ' , 'DIA']

look_back_ret , coming_ret = DataSetGeneration(ETF_ret_df, look_back_window)

## Algo

portfolio_weight = []
ret = []

N = len(look_back_ret)

for i in range(N):
    
    w = MaxSharpeRatio.Maximum_Sharpe_Ratio(look_back_ret[i])
    
    portfolio_weight.append(w)
    
    ret.append(np.dot(w , coming_ret[i]))
    
cum_ret = np.cumprod(1 + ret)

perfomance_df = pd.dataFrame(cum_ret)

performance_df.plot()







