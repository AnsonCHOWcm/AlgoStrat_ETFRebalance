#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Jul 18 23:08:31 2021

@author: ccm
"""
import pandas as pd
from yahoofinancials import YahooFinancials


## Download the Data from Yahoo Finance

yahoo_financials = YahooFinancials('DIA')

data = yahoo_financials.get_historical_price_data(start_date='2016-01-01', 
                                                  end_date='2020-12-31', 
                                                  time_interval='daily')

data_df = pd.DataFrame(data['DIA']['prices'])
data_df = data_df.drop('date',axis =1).set_index('formatted_date')

AdjustClose_df = data_df['adjclose']

## Output the csv

AdjustClose_df.to_csv('DIA.csv')

