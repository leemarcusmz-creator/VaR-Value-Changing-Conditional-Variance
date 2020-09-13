import pandas as pd 
import numpy as np 
import scipy.stats as stats
import matplotlib.pyplot as plt
import matplotlib.lines
from datetime import datetime as dt

euro_data = pd.read_csv('euro.csv')
euro_data.index = pd.to_datetime(euro_data.date,format="%Y-%m-%d").dt.date
#retVec = euro_data['close'].values 

#Question 1a
plt.figure(figsize=(17,5))
plt.plot(euro_data['close'],'b-',lw=.5) 
plt.grid(linestyle='--', linewidth='0.2', color='blue') 
plt.xlabel('Date')
plt.ylabel('Euro_Close')
plt.title('Euro Closing Value')

print('--- Question 1.a) ---')
plt.show()

#Question 1b
#Create log return
euro_data['lagClose'] = euro_data.close.shift(1)

# delete first entry (no lag)
euro_data = euro_data[1:]
euro_data['logdiff']= (np.log(euro_data['close'])- np.log(euro_data['lagClose']))

#Creating a pure numpy vector of log returns
euro_data_logretvec = euro_data['logdiff'].values
euro_data_logdiff_mean = np.mean(euro_data_logretvec)
euro_data_logdiff_std = np.std(euro_data_logretvec)

plt.figure(figsize = (17,5))
plt.plot(euro_data['logdiff'], 'b-', lw = 0.5)
plt.grid(linestyle = '--', linewidth='0.2', color = 'blue')
plt.title('Euro Exchange Rate Closing Values')
plt.xlabel('Dates')
plt.ylabel('Closing Values')

print('--- Question 1.b) ---')
plt.show()

#Question 1c
euro_data['ret']=euro_data['close']/euro_data['lagClose']-1
euro_data_retvec = euro_data['ret'].values

euro_data_retvec_mean = np.mean(euro_data_retvec)
euro_data_retvec_std = np.std(euro_data_retvec)

RStar = stats.norm.ppf(.025,loc=euro_data_retvec_mean,scale=euro_data_retvec_std)
q1c_VaR = -(100)*RStar
#p = .025 using log returns 
RStar_L = stats.norm.ppf(.025,loc=euro_data_logdiff_mean,scale=euro_data_logdiff_std)
q1c_VaR_log = -(100)*RStar_L


# T = len(retVec)
# p = 0.025

# retMC = np.random.normal(loc=retMean, scale=retStd, size=T)

# port1Day = (1+retMC)*100.
# PStarMC = np.percentile(port1Day,100.*p)
# VaRMC = 100.-PStarMC
# RStar = stats.norm.ppf(p,loc=retMean,scale=retStd)
# q1c_VaR = -RStar*100.

print('--- Problem 1.c ---')
print('One Day VAR (p=0.025): ', q1c_VaR)
print('One Day Log VAR(p=0.025): ', q1c_VaR_log)

#Question 1d
euro_data_2 = pd.read_csv('euro_copy.csv')
euro_data_2.index = pd.to_datetime(euro_data_2.date,format="%Y-%m-%d").dt.date
#remove the time
euro_data_2['lagClose'] = euro_data_2.close.shift(1)
# delete first entry (no lag)
euro_data_2 = euro_data_2[1:] 
euro_data_2['ret']=euro_data_2['close']/euro_data_2['lagClose']-1 
euro_data_2_retvec = euro_data_2['ret'].values
#Calculating mean and stdev. of return vector
euro_data_2_retvec_mean = np.mean(euro_data_2_retvec)
euro_data_2_retvec_std = np.std(euro_data_2_retvec)
#First I square the returns
euro_data_2['retsq'] = (euro_data_2['ret'] - euro_data_2_retvec_mean)**2. 
#euro_data2_retvecsq = euro_data2['retsq'].values
rollwindow2 = euro_data_2.rolling(window=50,win_type="boxcar") 
rollmean2 = rollwindow2.mean()
rollwindow3 = euro_data_2.rolling(window=50).std

plt.figure(figsize=(17,5))
plt.plot(np.sqrt(rollmean2['retsq'])*np.sqrt(250.),'b-',lw=.5)
plt.grid(linestyle='--', linewidth='0.2', color='blue')
plt.xlabel('Date')
plt.ylabel('Standard Deviation')
plt.title('Rolling Standard Deviation')

print('--- Question 1.d) ---')
plt.show()

#Question 1e
# euro_data_3 = pd.read_csv('euro_copy_copy.csv')
# euro_data_3.index = pd.to_datetime(euro_data_3.date,format="%Y-%m-%d").dt.date

# euro_data_3['lagClose'] = euro_data_3.close.shift(1)
# # delete first entry (no lag)
# euro_data_3 = euro_data_3[1:] 
# euro_data_3['ret']=euro_data_3['close']/euro_data_3['lagClose']-1 
# euro_data_3_retvec = euro_data_3['ret'].values
# #Calculating mean and stdev. of return vector
# euro_data_3_retvec_mean = np.mean(euro_data_3_retvec)
# euro_data_3retvec_std = np.std(euro_data_3_retvec)
# #First I square the returns
# euro_data_3['retsq'] = (euro_data_3['ret'] - euro_data_3_retvec_mean)**2. 
# #euro_data2_retvecsq = euro_data2['retsq'].values
# rollwindow3 = euro_data_3.rolling(window=50,win_type="boxcar") 

# rollwindow4 = euro_data_3.rolling(window=50,win_type="boxcar") 
# rollmean4 = rollwindow4.mean()

rollmean2['standardized_ret']=rollmean2['ret']/rollmean2['retsq']
kurtosis_standardardized_series = rollmean2['standardized_ret'].kurt() 
kurtosis_orig_series = euro_data_2['ret'].kurt()

rollmean8 = rollmean2
rollmean8 = rollmean8.dropna()

kurtosis_standardardized_series_jarque_bera = stats.jarque_bera(rollmean8['standardized_ret'])
kurtosis_orig_series_jarque_bera = stats.jarque_bera(euro_data_2['ret'])

print('--- Question 1.e) ---')
print('Kurtosis of standardized series: ', kurtosis_standardardized_series)
print('Kurtosis of original 1 day return: ', kurtosis_orig_series)

print('P-value of standardized series: ', kurtosis_standardardized_series_jarque_bera)
print('P-value of original 1 day return: ', kurtosis_orig_series_jarque_bera)

# #Question 1f
# rollwindow_f = euro_data_2.rolling(window=1,win_type="boxcar") 
# rollmean_f = rollwindow_f.mean()
# rollwindow_f = euro_data_2.rolling(window=1).std

# RStar = stats.norm.ppf(.025,loc=rollmean_f,scale=rollwindow_f)
# q1f_VaR = -(100)*RStar

# print('--- Question 1.f) ---')
# print('VaR: ', q1f_VaR)

euro_data_2['ret']=euro_data_2['close']/euro_data_2['lagClose']-1
euro_data_2_retvec = euro_data_2['ret'].values

euro_data_2_retvec_mean = np.mean(euro_data_2_retvec)
euro_data_2_retvec_std = np.std(euro_data_2_retvec)

RStar = stats.norm.ppf(.025,loc=euro_data_retvec_mean,scale=euro_data_retvec_std)
q1f_VaR = -(100)*RStar



rollwindow6 = euro_data_2.rolling(window=1,win_type="boxcar") 
rollmean6 = rollwindow6.mean()
rollwindow6 = euro_data_2.rolling(window=1).std

plt.figure(figsize=(17,5))
plt.plot(np.sqrt(rollmean6['retsq'])*np.sqrt(250.),'b-',lw=.5)
plt.grid(linestyle='--', linewidth='0.2', color='blue')
plt.xlabel('Date')
plt.ylabel('Standard Deviation')
plt.title('Rolling Standard Deviation')

print('--- Question 1.f) ---')
print('Rolling 1 Day VaR: ', q1f_VaR)
plt.show()

#Question 1g
print('--- Question 1.g) ---')
print('Given that VaR uses the close in prices at time t, the exact 1 day horizon it refers to is the peak within the dataset.')


#Question 1h
mn = np.mean(euro_data_2.ret)

p = 0.025
lemda = 0.94
# generate squared returns series
euro_data_2['retsq']  = (euro_data_2.ret-mn)**2.
# exponetially weighted moving averages
euro_data_2_EWM =  euro_data_2.ewm(alpha=(1.-lemda),adjust=False).mean()
# Rolling moving average for comparisons
# dowDayEWM =  dowDay.rolling(window=25,min_periods=1).mean()
# critical, Rstar values (for realized and squared returns)
normcrit = stats.norm.ppf(p)
euro_data_2_EWM['rstar']   = mn + normcrit*np.sqrt(euro_data_2['retsq'])
# euro_data_2_EWM['rstarRV'] = mn + normcrit*(euro_data_2_EWM['dayvol'])

# generate VaR exceptions
# remember need to use volatility with lag (shift(1))
etemp = euro_data_2['ret'] < euro_data_2_EWM['rstar'].shift(1)
exceptions = pd.DataFrame(etemp,columns=['retsq'])
# exceptions['RV'] = euro_data_2['ret'] < euro_data_2_EWM['rstarRV'].shift(1)
# rolling exceptions window
exceptionsRoll = exceptions.rolling(window=1250,min_periods=1).mean()
plt.plot(exceptionsRoll)
plt.legend(['based on retsq','based on RV'])
plt.xlabel('Year')
plt.ylabel('Exceptions')
plt.grid()

print('--- Question 1.h) ---')
plt.show()

#Question 1i
plt.figure(figsize=(17,5))
plt.scatter(x=exceptionsRoll, y=np.sqrt(rollmean6['retsq'])*np.sqrt(250.), s=2)
plt.grid(linestyle='--', linewidth='0.2', color='blue')
plt.xlabel('Moving Average VaR')
plt.ylabel('Exponential Moving Average')
plt.title('Combined Graph')

print('--- Question 1.i) ---')
plt.show()

#Question 1j
rollwindow7 = euro_data_2.rolling(window=250,win_type="boxcar") 
rollmean7 = rollwindow7.mean()
rollwindow7 = euro_data_2.rolling(window=250).std

plt.figure(figsize=(17,5))
plt.plot(np.sqrt(rollmean7['retsq'])*np.sqrt(250.),'b-',lw=.5)
plt.grid(linestyle='--', linewidth='0.2', color='blue')
plt.xlabel('Date')
plt.ylabel('Standard Deviation')
plt.title('Rolling Standard Deviation')

print('--- Question 1.j) ---')
print('Rolling 1 Day VaR: ', q1f_VaR)
plt.show()
