#!/usr/bin/env python
# coding: utf-8

# # Importing libraries

import numpy as np
from numpy import newaxis
import matplotlib.pyplot as plt
import pandas as pd
import streamlit as sl
from sklearn.preprocessing import MinMaxScaler
from tensorflow.keras import Sequential
from tensorflow.keras.layers import Dense, LSTM, Dropout
import streamlit.components.v1 as components
from PIL import Image


# # Loading and splitting the dataset
img = Image.open("ENIGMA.png")
sl.image(img)

sl.write('''
    # Introduction

    #### What Is Diversification? 
    Diversification is a risk management strategy that mixes a wide variety of investments within a portfolio. A diversified portfolio contains a mix of distinct asset types and investment vehicles in an attempt at limiting exposure to any single asset or risk. The rationale behind this technique is that a portfolio constructed of different kinds of assets will, on average, yield higher long-term returns and lower the risk of any individual holding or security.

''')

col1,col2 = sl.columns(2)

col1.header("Past Performance")
img2 = Image.open("chartcomb.jpeg")
col1.image(img2, use_column_width=True)


col2.header("How we chose our data")
col2.write('''
we have choosen 15 stocks 3 from different sectors ranging automobile, banking, metals, medical, and technology.

we plan to optimize by diversyfying our portfolio so as to decrease the risk and that we do by choosing the maximum returns calculated by lstm model and which are highly correlated to their competitors.
lets understand this by taking a hypothetical example.
I have to invest money in 2 stocks A and B, both tech giants and i find stock A gives 20 percent higher returns compared to B. But i need to reduce my risk because they are highly correlated so if one stock drops there is a higher probability that the other stock drops too, therefore i choose to invest in A and another stock which is less likely correlated to A to reduce my risk.
''')




sl.write(''' # Predicting Stock prices''')

col3,col4 = sl.columns(2)
col3.header('LSTM Output')
col3.write('''
we have choosen 15 stocks 3 from different sectors ranging automobile, banking, metals, medical, and technology.

we plan to optimize by diversyfying our portfolio so as to decrease the risk and that we do by choosing the maximum returns calculated by lstm model and which are highly correlated to their competitors.
lets understand this by taking a hypothetical example.
I have to invest money in 2 stocks A and B, both tech giants and i find stock A gives 20 percent higher returns compared to B. But i need to reduce my risk because they are highly correlated so if one stock drops there is a higher probability that the other stock drops too, therefore i choose to invest in A and another stock which is less likely correlated to A to reduce my risk.
''')

col4.header("")
img3 = Image.open("lstmoutput.jpeg")
col4.image(img3, use_column_width=True)

#################

col5,col6 = sl.columns((1,2))
col5.header('Comparing different models')
col5.write('''
We are using LSTM to predict the stock prices as it outperforms both, Arima and seq2seq models with the least Mean Absolute Error.
''')

col6.header("")
img4 = Image.open("lstmcompare.png")
col6.image(img4, use_column_width=True)



#Loading the data
data = pd.read_excel(r'Combined_Stocks.xlsx', date_parser = True)
data=data.dropna()
# Taking data from 2000 to 2020 as training set
data_training = data[data['Date']<'01-01-2020'].copy()
#0-3702

#test data
data_test = data[data['Date']>='01-01-2020'].copy()

# Data preprocessing of training data
training_data = data_training.drop(['Date'], axis=1)
testing_data = data_test.drop(['Date'], axis=1)






from keras.callbacks import EarlyStopping
earlyStop=EarlyStopping(monitor="val_loss",verbose=2,mode='min',patience=3)


# # Using LSTM to predict stock prices


# for i in range(0,len(training_data.columns)):
#     stock = training_data[training_data.columns[i]]
#     stocks_train = stock.to_frame()
#     stock_test = testing_data[testing_data.columns[i]]
#     stocks_test = stock_test.to_frame()
    
#     scaler = MinMaxScaler()
#     stocks = scaler.fit_transform(stocks_train)
    
        
#     X_train = []
#     y_train = []
#     for j in range(200, stocks.shape[0]):
#         X_train.append(stocks[j-200:j])  
#         y_train.append(stocks[j,0])
    
#     X_train, y_train = np.array(X_train), np.array(y_train)

#     past_200_days = stocks_train.tail(200)
#     df = past_200_days.append(stocks_test, ignore_index=True)
    
#     inputs = scaler.transform(df)

#     X_test = []
#     y_test = []
    
#     for j in range(200, inputs.shape[0]):
#         X_test.append(inputs[j-200:j])
#         y_test.append(inputs[j,0])
    
#     X_test, y_test  = np.array(X_test), np.array(y_test)
    
#     model = Sequential()
#     model.add(LSTM(units = 60, activation = 'relu', return_sequences = True, input_shape = (X_train.shape[1],1)))
#     model.add(Dropout(0.2))

#     model.add(LSTM(units = 60, activation = 'relu', return_sequences = True))
#     model.add(Dropout(0.2))

#     model.add(LSTM(units = 80, activation = 'relu', return_sequences = True))
#     model.add(Dropout(0.2))

#     model.add(LSTM(units = 120, activation = 'relu'))
#     model.add(Dropout(0.2))

#     model.add(Dense(units = 1))
#     #model.summary()

#     model.compile(optimizer='adam', loss = 'mse', metrics = ['accuracy'])
    
#     history = model.fit(X_train, y_train, validation_data = (X_test, y_test), epochs=1, batch_size=32, verbose=2, callbacks=[earlyStop])

    
#     y_pred = model.predict(X_test)
    
#     scale = 1/scaler.scale_[0]
    
#     y_pred = y_pred*scale
#     y_test = y_test*scale
    
#     plt.figure(figsize=(14,5))
#     plt.plot(y_test, color = 'red')
#     plt.xlabel(data.columns[i+1])
#     plt.ylabel("values")
#     plt.plot(y_pred, color = 'blue')

#     sl.pyplot(plt)

    
#     first=np.mean(y_pred[150:180])
#     last=np.mean(y_pred[373:403])
#     percentage=(last-first)/first*100
    
#     print("stock name= "+ data.columns[i+1] )

#     print(percentage)

#     print("")

#     if(i==0):
#         break



sl.write(''' # Analysing the Predictions''')
sl.write('''## Pearson Correlation - ''')
sl.write(data.corr(method='pearson'))





sl.write(''' # Asset Allocation ''')
sl.write(''' ## Efficient Frontier -  ''')

col7,col8 = sl.columns((2,2))
col7.header("")
img3 = Image.open("efchart.png")
col7.image(img3, use_column_width=True)


img5 = Image.open("efoutput.png")
col8.image(img5, use_column_width=True)





sl.write(''' # Results ''')
import ffn
prices = ffn.get('sbin.ns,ICICIBANK.NS,HEROMOTOCO.NS,wipro.ns,jswsteel.ns,SUNPHARMA.NS,tatasteel.ns', start='2005-01-01')


#selected top stocks from that group on the basis of annual return if they were highly correlated
#WE WILL USE RETURNS FROM LSTM AND CORRELATION TO FIND STOCKS TO SELECT IN PORTFOLIO
#plotting stock performace since 2015
ax = prices.rebase().plot()




np.seterr(all='ignore')
stats = prices.calc_stats()





# We put random weight in starting which will be optimized further
weights = np.asarray([0.2,0.2,0.1,0.1,0.1,0.1,0.2])

returns = prices.pct_change()
 
# mean daily return and covariance of daily returns
mean_daily_returns = returns.mean()
cov_matrix = returns.cov()

# portfolio return and volatility
pf_return = round(np.sum(mean_daily_returns * weights) * 252, 3)
pf_std_dev = round(np.sqrt(np.dot(weights.T, np.dot(cov_matrix, weights))) * np.sqrt(252), 3)






print("Volatility: " + "{:.1%}".format(pf_std_dev))

sl.write("Expected annualized return: " + "{:.1%}".format(pf_return))
sl.write("Volatility: " + "{:.1%}".format(pf_std_dev))


from pypfopt import expected_returns
from pypfopt import risk_models
from pypfopt import discrete_allocation
from pypfopt.cla import CLA

from pypfopt.efficient_frontier import EfficientFrontier

import matplotlib
from matplotlib.ticker import FuncFormatter

exp_returns = expected_returns.mean_historical_return(prices)
covar = risk_models.sample_cov(prices)

#have to use sortino here, hopefully that gives us better returns

# Optimise portfolio for maximum Sharpe Ratio
ef = EfficientFrontier(exp_returns, covar)
raw_weights = ef.max_sharpe()
pf = ef.clean_weights()
print(pf)

perf = ef.portfolio_performance(verbose=True)



ef = EfficientFrontier(exp_returns, covar, weight_bounds=(-1, 1))
pf = ef.efficient_return(target_return=perf[0])
print(pf)
sl.write(pf)
sl.write("***")
perf = ef.portfolio_performance(verbose=True)



