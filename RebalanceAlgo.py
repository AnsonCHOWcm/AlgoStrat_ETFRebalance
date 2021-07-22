#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Jul 19 16:35:12 2021

@author: ccm
"""
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import RiskParty
import MaxSharpeRatio
import Performance

## Read Data 

VOO_price_df = pd.read_csv('VOO.csv' , index_col = 0)
QQQ_price_df = pd.read_csv('QQQ.csv' , index_col = 0)
DIA_price_df = pd.read_csv('DIA.csv' , index_col = 0)

## Compute the Ret 

VOO_ret_df = VOO_price_df.pct_change().dropna()
QQQ_ret_df = QQQ_price_df.pct_change().dropna()
DIA_ret_df = DIA_price_df.pct_change().dropna()

ETF_ret_df = pd.concat([VOO_ret_df , QQQ_ret_df , DIA_ret_df]  , axis = 1)

ETF_ret_df.columns = ['VOO' , 'QQQ' , 'DIA']

## Define a function for data processing

    
def DataSetGeneration (data , window , frequency):
    
    N = len(data)
    Prev_Info = []
    Current_Info = []
    
    for i in range(0,N - window , frequency):
        X = data.iloc[i : i+window]
        y = np.cumprod(1+np.array(data.iloc[i+window : i+window+frequency]) , axis = 0)[-1] -1
        Prev_Info.append(X)
        Current_Info.append(y)
        
    return Prev_Info , Current_Info

## Seting the window set

windows = [20,30 , 40]

freq = 20

## Define the list for storing the performance

train_perfromance = []

test_perfromance = []

AnnualizedSharpe = []

CalmarRatio = []

MaxDrawDown = []

for j in range(len(windows)):

    ## Set the Hyperparameter
    
    look_back_window = windows[j]
    slip = 0.0002
    test_size = 0.5
    
    ## Prepare the Data Set
    
    look_back_ret , coming_ret = DataSetGeneration(ETF_ret_df, look_back_window , freq)
    
    ## Algo
    
    portfolio_weight = []
    ret = []
    
    N = len(look_back_ret)
    
    for i in range(N):
        
        w = MaxSharpeRatio.Maximum_Sharpe_Ratio(look_back_ret[i])
        
        portfolio_weight.append(w)
        
        ret.append(np.dot(w , coming_ret[i]))
        
    ## Backtest Result
    
    ret_arr = np.array(ret)
    
    length = len(ret_arr)
    
    train_ret_arr = ret_arr[:int(length * test_size)]
    
    test_ret_arr = ret_arr[int(length * test_size) : ]
        
    train_cum_ret = np.cumprod(1 + train_ret_arr - slip)
    
    test_cum_ret = np.cumprod(1 + test_ret_arr - slip)
    
    train_perfromance.append(train_cum_ret)
    
    test_perfromance.append(test_cum_ret)
    
    train_mea = Performance.Measures(train_cum_ret)
    
    test_mea = Performance.Measures(test_cum_ret)
    
    AnnualizedSharpe.append([train_mea.Annualized_Sharpe() , test_mea.Annualized_Sharpe()])
    CalmarRatio.append([train_mea.CalmarRatio() , test_mea.CalmarRatio() ])
    MaxDrawDown.append([train_mea.MaxDrawDown() , test_mea.MaxDrawDown()])
    
    
## Exporting the Graphs

name = ["lookback_period:20" , "lookback_period:30" , "lookback_period:40" ]

## Prepare the benchmark

branchmark_ret = VOO_ret_df['adjclose']

branchmark_train_perfromance = []

branchmark_test_perfromance = []

Branchmark_AnnualizedSharpe = []

Branchmark_CalmarRatio = []

Branchmark_MaxDrawDown = []

for h in range(len(windows)):
    
    time_length = len(train_perfromance[h])
    
    window = windows[h]
    
    _ ,branchmark_ret_adj = DataSetGeneration(branchmark_ret, window , freq)
    
    train_cum_ret = np.cumprod(1 + np.array(branchmark_ret_adj[:time_length]))
    
    test_cum_ret = np.cumprod(1 + np.array(branchmark_ret_adj[time_length:]))
    
    branchmark_train_perfromance.append(train_cum_ret)
    
    branchmark_test_perfromance.append(test_cum_ret)
    
    train_mea = Performance.Measures(train_cum_ret)
    
    test_mea = Performance.Measures(test_cum_ret)
    
    Branchmark_AnnualizedSharpe.append([train_mea.Annualized_Sharpe() , test_mea.Annualized_Sharpe()])
    Branchmark_CalmarRatio.append([train_mea.CalmarRatio() , test_mea.CalmarRatio() ])
    Branchmark_MaxDrawDown.append([train_mea.MaxDrawDown() , test_mea.MaxDrawDown()])
    
for k in range(len(name)):
    
    time_length = len(train_perfromance[k])
    
    plt.plot( range(time_length) ,train_perfromance[k] , label = 'Portfolio')
    plt.plot( range(time_length) ,branchmark_train_perfromance[k] , label = 'Benchmark')
    plt.legend()
    plt.savefig(name[k])
    plt.show()
    
for k in range(len(name)):
    
    time_length = len(test_perfromance[k])
    
    plt.plot( range(time_length) ,test_perfromance[k] , label = 'Portfolio')
    plt.plot( range(time_length) ,branchmark_test_perfromance[k] , label = 'Benchmark')
    plt.legend()
    plt.savefig(name[k])
    plt.show()
    
## Export the Measure Results

AnnualizedSharpe_df = pd.DataFrame(AnnualizedSharpe)
CalmarRatio_df = pd.DataFrame(CalmarRatio)
MaxDrawDown_df = pd.DataFrame(MaxDrawDown)

AnnualizedSharpe_df.index = name
AnnualizedSharpe_df.columns = ['Train' , 'Test']

CalmarRatio_df.index = name
CalmarRatio_df.columns = ['Train' , 'Test']

MaxDrawDown_df.index = name
MaxDrawDown_df.columns = ['Train' , 'Test']


Branchmark_AnnualizedSharpe_df = pd.DataFrame(Branchmark_AnnualizedSharpe)
Branchmark_CalmarRatio_df = pd.DataFrame(Branchmark_CalmarRatio)
Branchmark_MaxDrawDown_df = pd.DataFrame(Branchmark_MaxDrawDown)

Branchmark_AnnualizedSharpe_df.index = name
Branchmark_AnnualizedSharpe_df.columns = ['Train' , 'Test']

Branchmark_CalmarRatio_df.index = name
Branchmark_CalmarRatio_df.columns = ['Train' , 'Test']

Branchmark_MaxDrawDown_df.index = name
Branchmark_MaxDrawDown_df.columns = ['Train' , 'Test']

AnnualizedSharpe_df.to_csv('Sharpe.csv')
CalmarRatio_df.to_csv('Calmar.csv')
MaxDrawDown_df.to_csv('MaxDrawDown.csv')

Branchmark_AnnualizedSharpe_df.to_csv('Benrchmark_Sharpe.csv')
Branchmark_CalmarRatio_df.to_csv('Branchmark_Calmar.csv')
Branchmark_MaxDrawDown_df.to_csv('Branchmark_MaxDrawDown.csv')










