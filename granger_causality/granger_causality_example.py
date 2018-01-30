import os
import sys
import csv
import math
import json
import datetime
import random
import requests

from collections import OrderedDict

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import statsmodels
from statsmodels import tsa
from statsmodels.tsa import stattools
import scipy.stats as ss
from scipy import signal


def generate_random_walk_vector(length):
    """
    Generate a random walk vector of a particular length
    """
    data = [0]
    for j in range(length-1):
        step_x = random.randint(0,1)
        val = 0.0
        if step_x == 1:
            val = data[j] + 0.3 + 0.05*np.random.normal()
            if val > 1.0:
                val = 1.0
            else:
                val = data[j] - 0.3 + 0.05*np.random.normal()
                if val < -1.0:
                    val = -1.0
        data.append(val)
    return np.array(data)

def create_combined_random_vector(assessment_file):
    """
    Create a new combined vector for ingestion into
    the granger causality function
    Using a random walk for closing, as opposed to real data
    """

    comb_df = pd.read_csv(assessment_file)
    days = []
    for i in range(len(comb_df.values)):
	days.append(i)
    trend = comb_df['trend'].values
    close = generate_random_walk_vector(len(trend))

    # Resample and normalize to same size   
    trend = ss.zscore(trend)
    close = ss.zscore(signal.resample(np.diff(close), len(comb_df.values)))
    
    return (trend, close, days)


def create_combined_vector(assessment_file):
    """
    Create a new combined vector for ingestion into 
    the granger causality function
    """
    
    comb_df = pd.read_csv(assessment_file)
    days = []
    for i in range(len(comb_df.values)):
        days.append(i)
    trend = comb_df['trend'].values
    close = comb_df['adjusted_close'].values

    # Resample and normalize to same size
    trend = ss.zscore(trend)
    close = ss.zscore(signal.resample(np.diff(close), len(comb_df.values)))

    return (trend, close, days)


list_of_test_files = [
    'data/ADA+cardano-ada.csv',
    'data/BCH+bitcoin cash-bch.csv',
    'data/BTC+bitcoin-btc.csv',
    'data/DASH+dash-digitalcash.csv',
    'data/EOS+eos.csv',
    'data/ETH+ethereum-eth.csv',
    'data/LTC+litecoin-ltc.csv',
    'data/NEO+neo.csv',
    'data/TRX+tron-trx-tronix.csv',
    'data/XEM+nem-xem.csv',
    'data/XLM+stellar-xlm.csv',
    'data/XMR+monero-xmr.csv',
    'data/XRB+raiblocks-xrb.csv',
    'data/XRP+ripple-xrp.csv',
    'data/ZEC+zcash-zec.csv'    
]

fmt_output = []

no_causality  = []
yes_causality = []

strong_p_value = 0.05
super_strong_p_value = 0.001

total_lag = [] # p-value less than strong_p_value
best_lag  = [] # p-value less than super_strong_p_value 

for filename in list_of_test_files:
    
    (trend, close, days) = create_combined_vector(filename)
    asset_search = filename.split("+")[1].replace(".csv", "").replace("-", "|")
    trend_search = filename.split("+")[0]

    combined_vector = []
    for i in range(len(trend)):
        combined_vector.append((trend[i], close[i]))

    maxlag = int(len(combined_vector) * 0.2)
    if maxlag >= 60:
        maxlag = 60


    # Test whether the time series in the second column Granger causes
    # the time series in the first column
    
    # The Null hypothesis for grangercausalitytests is that the time series in the   
    # second column, x2, does NOT Granger cause the time series in the first column,
    # x1. Grange causality means that past values of x2 have a statistically significant
    # effect on the current value of x1, taking past values of x1 into account as
    # regressors. We reject the null hypothesis that x2 does not Granger cause x1
    # if the pvalues are below a desired size of the test. 
    try:
        gc = stattools.grangercausalitytests(combined_vector,
                                             maxlag,
                                             addconst=True ,
                                             verbose=False)
    except Exception as e:
        print(e)
        gc = []

    lag_pvalue = {}
    lag_numbers = []
    for lag in gc:
        
        fp = (lag, gc[lag][0]['ssr_ftest'][0], gc[lag][0]['ssr_ftest'][1])

        if gc[lag][0]['ssr_ftest'][1] < strong_p_value and lag > 1:
            lag_numbers.append(lag)
            total_lag.append(lag)
            if gc[lag][0]['ssr_ftest'][1] < super_strong_p_value:
                best_lag.append(lag)


    if len(lag_numbers):
        yes_causality.append((asset_search, trend_search, len(combined_vector)))
        print("Successfully found instances of 'causation' for %s and %s" %
              (asset_search, trend_search))
    else:
        no_causality.append((asset_search, trend_search, len(combined_vector)))
        print("Did not find instances of 'causation' for %s and %s" %
              (asset_search, trend_search))

    for lag in lag_numbers:

        try:
            corr = np.corrcoef(trend[lag:], close[:-lag])[0][1]
            fmt_output.append((asset_search, trend_search,
                               lag, gc[lag][0]['params_ftest'][1], corr))
        except Exception as e:
            print(e)
            print(lag)


"""
Big formatted output section
"""
output_file     = open("data/output.csv", 'wb')
output          = csv.writer(output_file, delimiter=',')
row = ["asset", "trend", "lag", "p-value", "corr"]
print(row)
output.writerow(row) # header
for row in fmt_output:
    print("%s, %s, %3d, %7.5f, %5.3f" % row)
    output.writerow(row)

count_causality_same_as_asset = 0
count = 0

print("\n===========================================================\n")

print("\n-------------------------------------------------------\n")
print("** - represents that the phrases match the asset name\n\n")
print("No Causility\n")
for row in no_causality:
    print("%s and %s showed NO causality, count: %d" % row)
    count += 1
print("\n-------------------------------------------------------\n")
print("Causality\n")
for row in yes_causality:
    print("%s and %s showed causality, count: %d" % row)
    count += 1

print("\n~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~\n")
print("Causality Count: %d" % (len(yes_causality)))
print("No Causality Count: %d\n" % (len(no_causality)))

np_total_lag = np.array(total_lag)

print('std', np.std(np_total_lag))
print('avg', np.average(np_total_lag))

if count == 0:
    count = 99999999999999

print("Percent showing causality: %3.2f\n" %
      (float(len(yes_causality)) / float(count)))

print("Percent showing causality, same as asset: %3.2f\n" %
      (float(count_causality_same_as_asset) / float(count)))

lag_dict = {x:total_lag.count(x) for x in total_lag}
lag = OrderedDict(sorted(lag_dict.items()))

best_lag_dict = {x:best_lag.count(x) for x in best_lag}
best_lag = OrderedDict(sorted(best_lag_dict.items()))

y = []
x = []

for i in lag:
    x.append(i)
    y.append(lag[i])

x_best = []
y_best = []
for i in best_lag:
    x_best.append(i)
    y_best.append(best_lag[i])

print("Strong p-value: %d" % np.sum(y))
print("Super strong p-value: %d" % np.sum(y_best))

"""
Generate graph
"""
width = 1/1.5
plt.bar(x, y, width, color="blue")
plt.bar(x_best, y_best, width, color="red")
plt.title("Correlations Prior to Event (in Days)")
plt.xlabel("Days Prior to an Event")
plt.ylabel("Number of Correlations")
plt.show()
