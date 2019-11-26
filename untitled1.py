t#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri May 24 22:02:48 2019

@author: kostya
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import statsmodels.tsa.api as smt
import statsmodels.api as sm
import scipy.stats as scs
from statsmodels.tsa.arima_model import ARIMA
from statsmodels.tsa.statespace.sarimax import SARIMAX
from statsmodels.tsa.arima_process import arma_generate_sample
from sklearn.model_selection import TimeSeriesSplit
from statsmodels.tsa.stattools import adfuller
import warnings
from pmdarima.arima import auto_arima
from statsmodels.tsa.arima_model import ARMAResults

def tsplot(y, lags=None, figsize=(10, 8), style='bmh'):
    if not isinstance(y, pd.Series):
        y = pd.Series(y)
    with plt.style.context(style):    
        fig = plt.figure(figsize=figsize)
        #mpl.rcParams['font.family'] = 'Ubuntu Mono'
        layout = (3, 2)
        ts_ax = plt.subplot2grid(layout, (0, 0), colspan=2)
        acf_ax = plt.subplot2grid(layout, (1, 0))
        pacf_ax = plt.subplot2grid(layout, (1, 1))
        #plt.gca().set_ylim(ymin=-50)
        qq_ax = plt.subplot2grid(layout, (2, 0))
        pp_ax = plt.subplot2grid(layout, (2, 1))
        
        y.plot(ax=ts_ax)
        ts_ax.set_title('Time Series Analysis Plots')
        smt.graphics.plot_acf(y, lags=lags, ax=acf_ax, alpha=0.05)
        smt.graphics.plot_pacf(y, lags=lags, ax=pacf_ax, alpha=0.05)
        sm.qqplot(y, line='s', ax=qq_ax)
        qq_ax.set_title('QQ Plot')        
        scs.probplot(y, sparams=(y.mean(), y.std()), plot=pp_ax)

        plt.tight_layout()
    return 


def auto_arima_grid(x, ar, i, ma):
    min_aic = float("inf")
    #catch_warnings context manager would only suppress warnings for this function
    with warnings.catch_warnings():
        warnings.filterwarnings("ignore")
        for p in range(ar+1):
            for d in range(i+1):
                for q in range(ma+1):
                    try:
                        #disp parameter suppress printing convergence information
                        aic = ARIMA(x, (p,d,q)).fit(disp=False).aic
                        if aic < min_aic:
                            min_aic = aic
                            best_params = (p, d, q)
                        print("ARIMA(%s, %s, %s), aic: %s" % (p, d, q, aic))
                    except Exception as error:
                        print("ARIMA(%s, %s, %s): " % (p, d, q), error)
                        continue
        print("Best model:")
        print("ARIMA(%s, %s, %s), aic: %s" % (best_params[0], 
                                              best_params[1], 
                                              best_params[2], 
                                              min_aic))



x = np.zeros(1000)
e = np.random.normal(size=1000)
for t in range(1000):
    if t <= 0:
        x[t] = 0.1
    else:
        x[t] = 0.9*x[t-1] + e[t]
        
tsplot(x, lags=30)
auto_arima_grid(x, 2, 2, 2)

tsplot(e**2, lags=30)


aa = auto_arima(x, suppress_warnings=True, error_action="ignore")
print(aa)
aa.aic()
aa.fit(x).summary()
help(auto_arima)




from sklearn.metrics import mean_squared_error
from sklearn.model_selection import TimeSeriesSplit


#AR(1) model is (Y_t-c) = phi*(Y_{t-1} - c) + e_t
prm = ARIMA(x, (1,0,0)).fit(disp=False).params
prm
ARIMA(x, (1,0,0)).fit(disp=False).predict()[999]
x[998]
ARIMA(x, (1,0,0)).fit(disp=False).predict(start=999)
x[998]
ARIMA(x, (1,0,0)).fit(disp=False).predict(start=1000)
x[999]
#start parameter works a little weird
ARIMA(x, (1,0,0)).fit(disp=False).predict(start=1000)
ARIMA(x, (1,0,0)).fit(disp=False).predict(start=1001)
ARIMA(x, (1,0,0)).fit(disp=False).predict(start=1000, end=1001)
ARIMA(x, (1,0,0)).fit(disp=False).predict(start=1002)
ARIMA(x, (1,0,0)).fit(disp=False).predict(start=1000, end=1002)

#fit function changes parameters even if supplied with start_params
x2 = np.append(x, e[:100])
model = ARIMA(x2, (1,0,0)).fit(start_params=prm, disp=False)
prm
model.params

#Use SARIMAX with filter to keep the same parameters
#SARIMAX forecasts is easier to calculate than ARIMA: 
# Y_t = phi*Y_{t-1} + c + e_t
#ARIMA uses trend parameter by default but SARIMAX doesn't so it should be
#provided so they can be compared
model = SARIMAX(x, order=(1,0,0), trend='c').fit(disp=False)
model.params

#Ljung-Box null hypothesis is no autocorrelation
#Null hypothesis is homoscedesticity
model.summary()

#same predictions
model.predict()[999]
ARIMA(x, (1,0,0)).fit(disp=False).predict()[999]
#different predictions cause fit function changed parameters after provided with new data x2
#fit function is like predicting in sample - parameters are calculated on x2
#filter allows to use parameters calculated on x and predict out of sample using x2 as new incoming data
#fit function with start="outside of x2" would produce predictions based on previously generated predictions
SARIMAX(x2, order=(1,0,0), trend='c').filter(model.params).predict(start=1000, end=1000)
SARIMAX(x2, order=(1,0,0), trend='c').fit(start_params=model.params, disp=False).predict(start=1000, end=1000)
SARIMAX(x2, order=(1,0,0), trend='c').fit(start_params=model.params, disp=False).fittedvalues[1000]
ARIMA(x2, (1,0,0)).fit(start_params=prm, disp=False).predict(start=1000, end=1000)
x2[999]
model.params





tscv = TimeSeriesSplit(n_splits=4)
for train, test in tscv.split(x):
    print(len(train), len(test))
model = SARIMAX(x[train], order=(1,0,0)).fit(disp=False)
predictions = SARIMAX(x, order=(1,0,0)).filter(model.params).predict(start=test[0], end=test[-1])




def cv(x, p, d, q):
    tscv = TimeSeriesSplit(n_splits=4)
    mse = [] 
    for train, valid in tscv.split(x):
        model = SARIMAX(x[train], order=(p,d,q), trend='c').fit()
        predictions = SARIMAX(x, order=(p,d,q), trend='c').filter(model.params).predict(start=valid[0], end=valid[-1])
        actuals = x[valid]
        mse.append(mean_squared_error(actuals, predictions))
    return np.mean(np.array(mse)), np.std(np.array(mse))
cv(x, 1, 0, 0)
cv(x, 2, 0, 1)


def auto_arima_grid_cv(x, ar, i, ma):
    min_mse = float("inf")
    #catch_warnings context manager would only suppress warnings for this function
    with warnings.catch_warnings():
        warnings.filterwarnings("ignore")
        for p in range(ar+1):
            for d in range(i+1):
                for q in range(ma+1):
                    try:
                        result = cv(x, p, d, q)
                        if result[0] < min_mse:
                            min_mse = result[0]
                            best_params = (p, d, q)
                        print("ARIMA(%s, %s, %s), mse: %s, std: %s" % 
                              (p, d, q, result[0], result[1]))
                    except Exception as error:
                        print("ARIMA(%s, %s, %s): " % (p, d, q), error)
                        continue
        print("Best model:")
        print("ARIMA(%s, %s, %s), mse: %s" % (best_params[0], 
                                              best_params[1], 
                                              best_params[2], 
                                              min_mse))

auto_arima_grid_cv(x, 2, 2, 2)






import pandas as pd
import datetime
dt = pd.read_csv('dataset.csv')
#dt.groupby(['SystemCodeNumber', 'Capacity']).count()
#dt.groupby(['SystemCodeNumber']).Capacity.nunique()
#parkinglots = dt.SystemCodeNumber.unique()
#for p in parkinglots:
#    dt[['Occupancy', 'LastUpdated2']][dt['SystemCodeNumber']==p].plot(title=p)
#dt.head()
#for c in dt:
#    print(type(dt[c][1]))
#dt['LastUpdated2'] = pd.to_datetime(dt['LastUpdated'])
#dt['LastUpdated2'] = dt['LastUpdated2'].dt.date

#dt[['Occupancy', 'LastUpdated']][dt['SystemCodeNumber']==parkinglots[0]].plot(title=parkinglots[0])
#c = dt[dt['SystemCodeNumber']==parkinglots[0]].groupby('LastUpdated2').count()
#dt[dt['SystemCodeNumber']==parkinglots[0]].head(50)
#c[c['SystemCodeNumber']!=18]
#dt['LastUpdated'][(dt['SystemCodeNumber']==parkinglots[0]) & 
#                  (dt['LastUpdated2']==datetime.date(2016,12,15))]
#c = dt[dt['SystemCodeNumber']=='BHMEURBRD01'].groupby('LastUpdated2').count()
#c[c['SystemCodeNumber']!=18]

#dt[dt['SystemCodeNumber']==parkinglots[0]].groupby('LastUpdated2').count()


#x = dt['Occupancy'][dt['SystemCodeNumber']=='BHMEURBRD01'].reset_index(drop=True)
#tsplot(x, lags=400)
#x_sdiff = x - x.shift(18)
#x_sdiff = x.diff(periods=18)
#tsplot(x_sdiff[18:], lags=100)
#x_sdiff2 = x_sdiff - x_sdiff.shift(1)
#x_sdiff2 = x_sdiff.diff(periods=19)
#tsplot(x_sdiff2[19:800], lags=20)
#x_sdiff3 = x_sdiff2 - x_sdiff2.shift(1)
#tsplot(x_sdiff3[20:], lags=20)
#adfuller(x_sdiff3[20:])






x = dt[['Occupancy','LastUpdated']][dt['SystemCodeNumber']=='BHMEURBRD01'].copy()
x.index = pd.to_datetime(x.LastUpdated)
#x.index = pd.to_datetime(x.index)
#x[70:120].plot()
#plt.xticks(rotation=45)
#x['LastUpdated2'] = pd.to_datetime(x.index)
#x.info()
x.LastUpdated = pd.to_datetime(x.LastUpdated)
x['LastUpdated_date'] = x.LastUpdated.dt.date
#x['LastUpdated_date'] = pd.to_datetime(x.LastUpdated_date)
x['LastUpdated_time'] = x.LastUpdated.dt.time
#x['LastUpdated_time'] = x['LastUpdated_time'].astype('float64')
#x.index = x.index.astype('object')



#g = x.groupby('LastUpdated_date')

#for i in g:
#    print(i, g.LastUpdated_time.min())
    

#g.LastUpdated_time.max().max()
#smr = g.LastUpdated_time.agg([min, max, 'count'])
#smr.sort_values(by='max').head()
#smr.loc[datetime.date(2016,10,30)]

#x.loc[x.LastUpdated_date=='2016-10-30']

from datetime import datetime, timedelta

def date_round30(date):
    rounding = (date.minute + 15)//30 * 30
    date = date + timedelta(minutes=rounding - date.minute)
    return date.replace(second=0, microsecond=0)

x['LastUpdated_round'] = x.LastUpdated.apply(date_round30)
x.groupby('LastUpdated_round').Occupancy.count().sort_values(ascending=False).head(10)
x.groupby('LastUpdated_round').Occupancy.nunique().sort_values(ascending=False).head(10)

# x2 is x without duplicates by 'LastUpdated_round'
x2 = x.drop_duplicates(subset=('LastUpdated_round', 'Occupancy'))
x2.groupby('LastUpdated_round').Occupancy.count().sort_values(ascending=False).head(10)
x2.index = x2.LastUpdated_round


#plt.figure(figsize=(12,8))
#plt.plot(x2.Occupancy)
#plt.figure(figsize=(12,8))
#plt.plot(x.Occupancy)

#dct = {}
#for i, j in g:
#    j.index = j.index.time
#    dct[i] = j

#keys = list(dct.keys())


#for i in range(0,len(keys)+1,7):
#    fig, ax = plt.subplots(1,7,sharey=True, figsize=(12,3))    
#    for j in range(7):
#        ax[j].plot(dct[keys[i+j]].Occupancy, label=len(dct[keys[i+j]].Occupancy))
#        ax[j].legend()
#        ax[j].set_title(keys[i+j])




#x.info()
#list(pd.Series(x.LastUpdated_date.unique()).apply(lambda x: x.weekday()))
#apply(lambda x: x.weekday())
#x.LastUpdated[0].weekday()

#dct[datetime.date(2016,11,22)]

#x['Occupancy2'] = x.Occupancy.interpolate(method='time')
#plt.plot(x.index[:100], x.Occupancy[:100])



# creating new index x3 that would have all days and 18 observations for each day
pd.date_range(start='2018-01-01 08:00:00', periods=18, freq='30min')
pd.date_range(datetime(2018,1,1,8,0,0), periods=18, freq='30min')

for i in range((x2.LastUpdated_date.max() - x2.LastUpdated_date.min()).days + 1):
    if i == 0:
        d = pd.date_range(datetime(2016,10,4,8,0,0), periods=18, freq='30min')
    else:
        d = d.append(pd.date_range(datetime(2016,10,4,8,0,0)+timedelta(days=i), 
                           periods=18, freq='30min'))

x3 = pd.DataFrame(index=d)
x4 = x3.merge(x2, how='left', left_index=True, right_index=True)
x4['LastUpdated_round'] = x4.index
x4['LastUpdated_date'] = x4.LastUpdated_round.dt.date
x4.Occupancy.plot()
#plt.plot(x4.Occupancy[400:1000])
#plt.plot(x4.Occupancy.loc[:datetime(2016,10,30,8,0)])
#plt.plot(x4.Occupancy.loc[datetime(2016,11,1,8,0):])



x4[x4.Occupancy.isna()].groupby('LastUpdated_date')['LastUpdated_round'].count()
x4['Occupancy2'] = x4.Occupancy.shift(126)
x4.Occupancy2 = np.where(x4.Occupancy.isna(), x4.Occupancy2, x4.Occupancy)
x4.Occupancy2 = np.where(x4.Occupancy2.isna(), x4.Occupancy2.shift(126), x4.Occupancy2)
#plt.plot(x4.Occupancy.loc[:datetime(2016,10,30,8,0)])
#plt.plot(x4.Occupancy2.loc[:datetime(2016,10,30,8,0)])
#plt.plot(x4.Occupancy.loc[datetime(2016,11,1,8,0):])
#plt.plot(x4.Occupancy2.loc[datetime(2016,11,1,8,0):])
x4.Occupancy2.plot()
x4[x4.Occupancy2.isna()].groupby('LastUpdated_date')['LastUpdated_round'].count()




#x4.index[x4.Occupancy.isna()]
#x4[x4.Occupancy.isna()].groupby('LastUpdated_date')['LastUpdated_round'].count()
#x4.LastUpdated_date.min()
#x4.LastUpdated_date.max()



#min(list(x.groupby('LastUpdated2')['LastUpdated_time'].min()))
#max(list(x.groupby('LastUpdated2')['LastUpdated_time'].min()))
#min(list(x.groupby('LastUpdated2')['LastUpdated_time'].max()))
#max(list(x.groupby('LastUpdated2')['LastUpdated_time'].max()))

#aa = auto_arima(np.array(x4.Occupancy2[:1100]), 
#                start_p=1, start_q=1, 
#                max_P=8, max_Q=8,
#                m=18, suppress_warnings=True, error_action="ignore")
#aa.summary()
#plt.plot(np.array(aa.predict(n_periods=500)))
#x4['Predict'] = x4.Occupancy2
#x4['Predict'][1100:] = np.nan
#x4['Predict'][1100:1200] = aa.predict(n_periods=100)
#plt.plot(x4.Predict.iloc[:1200], color='orange')
#plt.plot(x4.Occupancy2.iloc[:1100], color='blue')

#plt.xticks(rotation=45)
#plt.plot(x4.Occupancy2.iloc[400:420])
#x4[400:420]
#plt.xticks(rotation=45)
#plt.plot(x4.Occupancy2.iloc[:500])
#plt.xticks(rotation=45)
#plt.plot(x4.Occupancy2.iloc[300:500])
#plt.xticks(rotation=45)



tsplot(np.array(x4.Occupancy2), lags=400)
tsplot(np.array(x4.Occupancy2), lags=50)

#ACF 1x126, PCF 3x126, 1x1, maybe 1x19
x4w = x4.Occupancy2 - x4.Occupancy2.shift(126)
tsplot(np.array(x4w.iloc[127:]), lags=400)
tsplot(np.array(x4w.iloc[127:]), lags=50)

#ACF 1x126, PCF 2 or 3x126, 1x1, maybe 1x19
x4w2 = x4w - x4w.shift(126)
tsplot(np.array(x4w2.iloc[255:]), lags=400)
tsplot(np.array(x4w2.iloc[255:]), lags=270)
tsplot(np.array(x4w2.iloc[255:]), lags=40)

#overdifferenced
x4wd = x4w - x4w.shift(18)
tsplot(np.array(x4wd.iloc[145:]), lags=270)
tsplot(np.array(x4wd.iloc[145:]), lags=100)

#ACF 1x126 maybe 18x1, PCF 3x126 maybe 18x1, could be overdifferenced
x4w1 = x4w - x4w.shift(1)
tsplot(np.array(x4w1.iloc[128:]), lags=400)
tsplot(np.array(x4w1.iloc[128:]), lags=40)

#ACF 1x126 maybe 18x1, PCF 2x126 maybe 18x1, could be overdifferenced
x4w2_1 = x4w2 - x4w2.shift(1)
tsplot(np.array(x4w2_1.iloc[256:]), lags=300)
tsplot(np.array(x4w2_1.iloc[256:]), lags=40)







def auto_arima_grid(x, ar, i, ma):
    results = []
    errors = []
    min_aic = float("inf")
    #catch_warnings context manager would only suppress warnings for this function
    with warnings.catch_warnings():
        warnings.filterwarnings("ignore")
        for p in range(ar+1):
            for d in range(i+1):
                for q in range(ma+1):
                    try:
                        #disp parameter suppress printing convergence information
                        aic = ARIMA(x, (p,d,q)).fit(disp=False).aic
                        results.append([(p, d, q), aic])
                        #print("ARIMA(%s, %s, %s), aic: %s" % (p, d, q, aic))
                    except Exception as error:
                        errors.append([(p, d, q), error])
                        #print("ARIMA(%s, %s, %s): " % (p, d, q), error)
                        continue
                    
        #print("Best model:")
        #print("ARIMA(%s, %s, %s), aic: %s" % (best_params[0], 
        #                                      best_params[1], 
        #                                      best_params[2], 
        #                                      min_aic))
        results.sort(key=lambda x: x[1])
        return results, errors
    
        
a1 = auto_arima_grid(np.array(x4.Occupancy2), 21, 2, 4)

results = a1[0]
for i in range(len(results)):
    if np.isnan(results[i][1]):
        results[i][1] = 999
results = sorted(a1[0], key=lambda x: x[1])


a2 = auto_arima_grid(np.array(x4.Occupancy2), 4, 2, 21)

results = a2[0]
for i in range(len(results)):
    if np.isnan(results[i][1]):
        results[i][1] = 999
results = sorted(a2[0], key=lambda x: x[1])




#aa = auto_arima(np.array(x4.Occupancy2[:1100]), 
#                start_p=1, start_q=1, 
#                max_p=19, max_q=19,
#                suppress_warnings=True, error_action="ignore", max_order=50)
#aa.summary()
#SARIMAX(x4.Occupancy2[:1100], order=(19,0,1)).fit(simple_differencing=True, disp=False).summary()

#x4wd_1 = x4wd - x4wd.shift(1)
#tsplot(np.array(x4wd_1.iloc[146:]), lags=280)

#x4d = x4.Occupancy2 - x4.Occupancy2.shift(18)
#tsplot(np.array(x4d.iloc[18:]), lags=100)
#x4d_1 = x4d - x4d.shift(1)
#tsplot(np.array(x4d_1.iloc[19:]), lags=200)
#x4d_2 = x4d_1 - x4d_1.shift(1)
#tsplot(np.array(x4d_2.iloc[20:]), lags=160)
#x4d2 = x4d - x4d.shift(18)
#tsplot(np.array(x4d2.iloc[36:]), lags=1000)


#x4_1 = x4.Occupancy2 - x4.Occupancy2.shift(1)
#tsplot(np.array(x4_1.iloc[1:]), lags=50)

#x4_2 = x4_1 - x4_1.shift(1)
#tsplot(np.array(x4_2.iloc[2:]), lags=50)


#s1 = SARIMAX(x4.Occupancy2[:1100], order=(0,1,0), seasonal_order=(0,1,2,126)).fit(simple_differencing=True, disp=False)
#s2 = SARIMAX(x4.Occupancy2[:1100], order=(3,1,0), seasonal_order=(1,1,1,126)).fit(simple_differencing=True, disp=False)
#s1.summary()
#s2.summary()
#plt.plot(np.array(s1.predict(start=1100, end=3000)))
#plt.plot(np.array(s2.predict(start=1100, end=2000)))
#tsplot(np.array(s1.resid), lags=300)
#plt.plot(np.array(s1.fittedvalues))
#plt.plot(np.concatenate((np.array(s1.fittedvalues), 
#                np.array(s1.predict(start=1100, end=1500)))))
#plt.plot(s1.predict(start=1100, end=1386))
#plt.plot(np.array(x4.Occupancy2[1100:1386]))

#s2 = SARIMAX(x4.Occupancy2[:1100], order=(0,2,1), seasonal_order=(7,1,0,18)).fit(simple_differencing=True, disp=False)
#plt.plot(np.array(s2.predict(start=1100, end=1500)))
#plt.plot(np.concatenate((np.array(s2.fittedvalues), 
#                np.array(s2.predict(start=1100, end=1500)))))





from statsmodels.tsa.seasonal import seasonal_decompose
decomposition = seasonal_decompose(np.array(x4.Occupancy2), freq=126)  
decomposition.plot()  





#fit results are similar to filter as sarima with filter are not provided with "new incoming data"
#so it uses previously generated predictions for each new prediction the same thing that fit does





exog = pd.DataFrame(index=np.arange(len(x4.index)))
exog['sin1'] = np.sin(2*np.pi*1*exog.index/126)
exog['cos1'] = np.cos(2*np.pi*1*exog.index/126)
exog['sin2'] = np.sin(2*np.pi*2*exog.index/126)
exog['cos2'] = np.cos(2*np.pi*2*exog.index/126)
exog['sin3'] = np.sin(2*np.pi*3*exog.index/126)
exog['cos3'] = np.cos(2*np.pi*3*exog.index/126)
exog['sin4'] = np.sin(2*np.pi*4*exog.index/126)
exog['cos4'] = np.cos(2*np.pi*4*exog.index/126)
exog['sin5'] = np.sin(2*np.pi*5*exog.index/126)
exog['cos5'] = np.cos(2*np.pi*5*exog.index/126)
exog['sin6'] = np.sin(2*np.pi*6*exog.index/126)
exog['cos6'] = np.cos(2*np.pi*6*exog.index/126)
exog['sin7'] = np.sin(2*np.pi*7*exog.index/126)
exog['cos7'] = np.cos(2*np.pi*7*exog.index/126)

model = SARIMAX(np.array(x4.Occupancy2[:1100]), exog=exog[:1100], order=(1,0,1), seasoanl_order=(7,0,1,18)).fit(simple_differencing=True, disp=False)
model.predict(start=1100, end=1385, exog=exog[1100:1386]).plot()






from arch import arch_model

import sys
sys.path




#Garch
z = np.zeros(1000)
for t in range(1000):
    if t <= 0:
        z[t] = 0.1
    else:
        z[t] = 0.9*(t-1000)/1000*z[t-1] + e[t]
        
tsplot(z, lags=30)
auto_arima(z, 2, 2, 2)
adfuller(z)

sm.stats.diagnostic.het_breuschpagan(z, sm.add_constant(np.arange(1000)))
sm.stats.diagnostic.het_white(z, sm.add_constant(np.arange(1000)))
sm.stats.diagnostic.het_goldfeldquandt(z, sm.add_constant(np.arange(1000)))


#Dickey-Fuller test
for i in range(50):
    print(adfuller(x, maxlag=i, autolag=None)[0:3])
    






