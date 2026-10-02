import reservoirpy as rpy
import numpy as np
import matplotlib.pyplot as plt
from reservoirpy.nodes import Reservoir
from reservoirpy.nodes import Ridge
import os
from reservoirpy.datasets import mackey_glass
from reservoirpy.observables import nrmse, rsquare
from reservoirpy.datasets import to_forecasting
from sklearn.linear_model import Ridge as Ridge_sklearn
import json
from reservoirpy.hyper import research
from reservoirpy.hyper import plot_hyperopt_report
import pandas as pd
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense
from sklearn.svm import SVR


print('X timesteps ahead forecast\n\n')
os.chdir('Timesteps_forecasting')
plt. clf()
#Advanced; RC for chaotic timeseries forcasting
timesteps = 2510
tau = 20
X = mackey_glass(timesteps, tau=tau)
# rescale between -1 and 1
X = 2 * (X - X.min()) / (X.max() - X.min()) - 1

def plot_mackey_glass(X, sample, tau):

    fig = plt.figure(figsize=(13, 5))
    N = sample

    ax = plt.subplot((121))
    t = np.linspace(0, N, N)
    for i in range(N-1):
        ax.plot(t[i:i+2], X[i:i+2], color=plt.cm.magma(255*i//N), lw=1.0)

    plt.title(f"Timeseries - {N} timesteps")
    plt.xlabel("$t$")
    plt.ylabel("$P(t)$")

    ax2 = plt.subplot((122))
    ax2.margins(0.05)
    for i in range(N-1):
        ax2.plot(X[i:i+2], X[i+tau:i+tau+2], color=plt.cm.magma(255*i//N), lw=1.0)

    plt.title("Phase diagram: $P(t) = f(P(t-\\tau))$")
    plt.xlabel("$P(t-\\tau)$")
    plt.ylabel("$P(t)$")

    plt.tight_layout()
    #plt.show()
    plt.savefig('Time series and Phase diagram')

#plot_mackey_glass(X, 500, tau)

#Data preprocessing
def plot_train_test(X_train, y_train, X_test, y_test):
    sample = 500
    test_len = X_test.shape[0]
    fig = plt.figure(figsize=(15, 5))
    plt.plot(np.arange(0, 500), X_train[-sample:], color ='red', label="Training data")
    plt.plot(np.arange(0, 500), y_train[-sample:], color ='orange', label="Training ground truth")
    plt.plot(np.arange(500, 500+test_len), X_test, color ='blue', label="Testing data")
    plt.plot(np.arange(500, 500+test_len), y_test, color ='green', label="Testing ground truth")
    plt.title("Data preprocessing for "+str(forecast)+" timesteps", fontsize= '20')
    plt.xlabel("Number of days", fontsize= '16')
    plt.ylabel("Timeseries - Mackey Glass", fontsize= '16')
    plt.legend()
    plt.grid(axis='y',linestyle='--', linewidth=0.35, color='gray')
    #plt.show()
    plt.savefig('Data preprocessing for '+str(forecast)+' timesteps' )


forecast = 10
x, y = to_forecasting(X, forecast=forecast)
split_decimal = 0.8
split_index = int(split_decimal * len(X))
X_train1, y_train1 = x[:split_index], y[:split_index]
X_test1, y_test1 = x[split_index:], y[split_index:]

plot_train_test(X_train1, y_train1, X_test1, y_test1)

#Bulding ESN
units = 200 #number of neurons
leak_rate = 0.3
spectral_radius = 1.25
input_scaling = 1.0
connectivity = 0.1
input_connectivity = 0.2
regularization = 1e-8
seed = 1234
input_dim = None
fb_scaling = None

def ridge_regression(x,split_decimal,window_size):
  #Split data into training and testing sets
  split_index = int(split_decimal * len(x))
  train_data = x[:split_index]
  test_data = x[split_index-window_size:]

  X_train = []
  y_train = []
  for i in range(window_size, len(train_data)):
      X_train.append(train_data[i-window_size:i])
      y_train.append(train_data[i])
  X_train = np.array(X_train).reshape(-1, window_size)
  y_train = np.array(y_train)

  X_test = []
  y_test = x[split_index:]
  for i in range(window_size, len(test_data)):
      X_test.append(test_data[i-window_size:i])
  X_test = np.array(X_test).reshape(-1, window_size)
  y_test = np.array(y_test)

  # Train a ridge regression model
  ridge_model = Ridge_sklearn(alpha=1.0)
  ridge_model.fit(X_train, y_train)
  y_pred_ridge = ridge_model.predict(X_test)

  # Plot the original time series
  plt. clf()
  plt.plot(train_data, lw=3, label="Original-Training")
  plt.plot(y_test, lw=3, label="Original-True Value")

  # Plot the predicted values
  plt.plot(range(split_index, len(x)), y_pred_ridge, label="Predicted Values")

  plt.legend()
  plt.savefig('Results for '+str(window_size)+' timesteps without reservoir')
  plt. clf()
  return (y_pred_ridge,y_test)

def ridge_regression_timesteps(x,split_decimal,forecast):
  #Split data into training and testing sets
  split_index = int(split_decimal * len(x))
  x_data, y_data = to_forecasting(x, forecast)
  X_train, y_train = x_data[:split_index], y_data[:split_index]
  X_test, y_test = x_data[split_index:], y_data[split_index:]

  # Train and test a ridge regression model
  ridge_model = Ridge_sklearn(alpha=1)
  ridge_model.fit(X_train, y_train)
  y_pred_ridge_timesteps = ridge_model.predict(X_test)

  plt. clf()
  plt.figure(figsize=(15, 7.5))
  # Plot the predicted values
  plt.plot(y_pred_ridge_timesteps, lw=3, label="Ridge prediction")
  # Plot the target timeseries
  plt.plot(y_test, linestyle="--", lw=2, label="True value")
  plt.title ("Prediction for "+str(forecast)+" timesteps without reservoir (ridge)")
  plt.legend()
  plt.xlabel ("Number of days")
  plt.ylabel ("Timeseries - Mackey Glass")
  plt.grid(axis='y',linestyle='--', linewidth=0.35, color='gray')
  plt.savefig('Prediction for '+str(forecast)+' timesteps without reservoir (ridge)')
  plt. clf()
  return (y_pred_ridge_timesteps,y_test)

(y_pred_ridge_timesteps,y_test_ridge_timesteps) = ridge_regression_timesteps(X,0.8,forecast)

def lstm_timesteps(x,split_decimal,forecast):
  split_index = int(split_decimal * len(x))
  x_data, y_data = to_forecasting(x, forecast)
  X_train, y_train = x_data[:split_index], y_data[:split_index]
  X_test, y_test = x_data[split_index:], y_data[split_index:]

  #defining and compiling lstm model
  model_lstm = Sequential()
  model_lstm.add(LSTM(128, return_sequences=True, activation='relu', input_shape=(1, 1)))
  model_lstm.add(LSTM(64, return_sequences=False, activation='relu', input_shape=(1, 1)))
  model_lstm.add(Dense(25))
  model_lstm.add(Dense(25))
  model_lstm.add(Dense(1))
  model_lstm.compile(optimizer='adam', loss='mse')

  #train and test an lstm model
  model_lstm.fit(X_train, y_train, epochs=1, batch_size=1, verbose=1)
  y_pred_lstm_timesteps = model_lstm.predict(X_test)

  plt. clf()
  plt.figure(figsize=(15, 7.5))
  # Plot the predicted values
  plt.plot(y_pred_lstm_timesteps, lw=3, label="Lstm prediction")
  # Plot the target timeseries
  plt.plot(y_test, linestyle="--", lw=2, label="True value")
  plt.title ("Prediction for "+str(forecast)+" timesteps without reservoir (lstm)")
  plt.legend()
  plt.xlabel ("Number of days")
  plt.ylabel ("Timeseries - Mackey Glass")
  plt.grid(axis='y',linestyle='--', linewidth=0.35, color='gray')
  plt.savefig('Prediction for '+str(forecast)+' timesteps without reservoir (lstm)')
  plt. clf()
  model_lstm.reset_states()
  return (y_pred_lstm_timesteps,y_test)

(y_pred_lstm_timesteps,y_test_lstm_timesteps) = lstm_timesteps(X,0.8,forecast)

def svr_timesteps(x,split_decimal,forecast):
  #Split data into training and testing sets
  split_index = int(split_decimal * len(x))
  x_data, y_data = to_forecasting(x, forecast)
  X_train, y_train = x_data[:split_index], y_data[:split_index]
  X_test, y_test = x_data[split_index:], y_data[split_index:]

  #Create the SVR model
  model = SVR(kernel='rbf', C=100, gamma=0.1, epsilon=.1)

  #Train and test SVR model
  model.fit(X_train, y_train)
  y_pred_SVR = model.predict(X_test)

  plt. clf()
  plt.figure(figsize=(15, 7.5))
  # Plot the predicted values
  plt.plot(y_pred_SVR, lw=3, label="SVR prediction")
  # Plot the target timeseries
  plt.plot(y_test, linestyle="--", lw=2, label="True value")
  plt.title ("Prediction for "+str(forecast)+" timesteps without reservoir (SVR)")
  plt.legend()
  plt.xlabel ("Number of days")
  plt.ylabel ("Timeseries - Mackey Glass")
  plt.grid(axis='y',linestyle='--', linewidth=0.35, color='gray')
  plt.savefig('Prediction for '+str(forecast)+' timesteps without reservoir (SVR)')
  plt. clf()
  return (y_pred_SVR,y_test)

(y_pred_SVR_timesteps,y_test_SVR_timesteps) = svr_timesteps(X,0.8,forecast)

def reset_esn():
    from reservoirpy.nodes import Reservoir, Ridge

    reservoir = Reservoir(units, input_scaling=input_scaling, sr=spectral_radius,
                          lr=leak_rate, rc_connectivity=connectivity,
                          input_connectivity=input_connectivity, seed=seed, input_dim=input_dim,
                          fb_scaling=fb_scaling)
    readout   = Ridge(1, ridge=regularization)

    return reservoir >> readout

#defining
reservoir = Reservoir(units, input_scaling=input_scaling, sr=spectral_radius,
                      lr=leak_rate, rc_connectivity=connectivity,
                      input_connectivity=input_connectivity, seed=seed)

readout   = Ridge(1, ridge=regularization)

esn = reservoir >> readout

# initialisation
y = esn(X[0])
reservoir.Win is not None, reservoir.W is not None, readout.Wout is not None
np.all(readout.Wout == 0.0)

#ESN training
esn = esn.fit(X_train1, y_train1)

#Plotting the readout(or saving); coefficients values of neurons
def plot_readout(readout):
  Wout = readout.Wout
  bias = readout.bias
  Wout = np.r_[bias, Wout]

  fig = plt.figure(figsize=(15, 5))

  ax = fig.add_subplot(111)
  ax.grid(axis="y")
  ax.set_ylabel("Coefs. of $W_{out}$")
  ax.set_xlabel("reservoir neurons index")
  ax.bar(np.arange(Wout.size), Wout.ravel()[::-1])

  #plt.show()
  plt.savefig('Readout')

#plot_readout(readout)


#testing the ESN
def plot_results(y_pred, y_test, sample):

  plt.figure(figsize=(15, 7.5))
  plt.plot(np.arange(sample), y_pred[:sample], lw=3, label="ESN prediction")
  plt.plot(np.arange(sample), y_test[:sample], linestyle="--", lw=2, label="True value")
  #plt.plot(np.abs(y_test[:sample] - y_pred[:sample]), label="Absolute deviation")
  plt.title('Prediction for '+str(forecast)+' timesteps with reservoir (ESN)')
  plt.xlabel("Number of days")
  plt.ylabel("Timeseries - Mackey Glass")
  plt.legend()
  plt.grid(axis='y',linestyle='--', linewidth=0.35, color='gray')
  plt.savefig('Prediction for '+str(forecast)+' timesteps with reservoir (ESN)')
  plt. clf()

y_pred1 = esn.run(X_test1)
sample = len(X) - forecast - split_index
plot_results(y_pred1, y_test1, sample)

def plot_all(y_test,y_pred_esn,y_pred_ridge,y_pred_lstm,y_pred_SVR,sample):
  plt.figure(figsize=(22.5, 15))
  plt.plot(np.arange(sample), y_pred_esn[:sample], lw=5, linestyle="-", color ='blue', label="ESN prediction")
  plt.plot(y_pred_ridge, lw=5, linestyle="-", color ='orange', label="Ridge prediction")
  plt.plot(y_pred_lstm, lw=5, linestyle="-", color ='green', label="Lstm prediction")
  plt.plot(y_pred_SVR, lw=5, linestyle="-", color ='red', label="SVR prediction")
  plt.plot(y_test, linestyle="--", lw=15, alpha = 0.3, color="red", label="True value")
  plt.title ("Results for "+str(forecast)+" timesteps",fontsize=30)
  plt.legend(fontsize=25)
  plt.xticks(fontsize=20)
  plt.yticks(fontsize=20)
  plt.xlabel ("Number of days",fontsize=30)
  plt.ylabel ("Timeseries - Mackey Glass",fontsize=30)
  plt.grid(axis='y',linestyle='--', linewidth=0.5, color='gray')

  plt.savefig('Predictions for '+str(forecast)+' timesteps (ALL)')
  plt. clf()

plot_all(y_test1,y_pred1,y_pred_ridge_timesteps,y_pred_lstm_timesteps,y_pred_SVR_timesteps,sample)

def MSE_all(x,split_decimal):
  timesteps = []
  timesteps.append(range(1, 100))
  timesteps = np.array(timesteps)
  print(timesteps.shape[1])
  timesteps = timesteps.reshape(timesteps.shape[1],1)
  y_SVR = []
  y_lstm = []
  y_ridge = []
  y_esn = []


  for forecast in range(1, 100):
    #Split data into training and testing sets
    split_index = int(split_decimal * len(x))
    x_data, y_data = to_forecasting(x, forecast)
    X_train, y_train = x_data[:split_index], y_data[:split_index]
    X_test, y_test = x_data[split_index:], y_data[split_index:]

    #Create the SVR model, then train and test
    model = SVR(kernel='rbf', C=100, gamma=0.1, epsilon=.1)
    model.fit(X_train, y_train)
    y_pred_SVR = model.predict(X_test)
    y_pred_SVR = y_pred_SVR.reshape(y_pred_SVR.shape[0],1)
    y_pred_SVR = nrmse(y_test, y_pred_SVR)
    y_SVR.append(y_pred_SVR)

    # Train and test a ridge regression model
    ridge_model = Ridge_sklearn(alpha=1)
    ridge_model.fit(X_train, y_train)
    y_pred_ridge_timesteps = ridge_model.predict(X_test)
    y_pred_ridge_timesteps= nrmse(y_test, y_pred_ridge_timesteps)
    y_ridge.append(y_pred_ridge_timesteps)

    #defining and compiling lstm model, then train and test
    model = Sequential()
    model.add(LSTM(100, activation='relu', input_shape=(1, 1)))
    model.add(Dense(25))
    model.add(Dense(1))
    model.compile(optimizer='adam', loss='mse')
    model.fit(X_train, y_train, epochs=1, batch_size=1, verbose=1)
    y_pred_lstm_timesteps = model.predict(X_test)
    y_pred_lstm_timesteps= nrmse(y_test, y_pred_lstm_timesteps)
    y_lstm.append(y_pred_lstm_timesteps)
    model.reset_states()


    #esn
    #Bulding ESN
    units = 200 #number of neurons
    leak_rate = 0.3
    spectral_radius = 1.25
    input_scaling = 1.0
    connectivity = 0.1
    input_connectivity = 0.2
    regularization = 1e-8
    seed = 1234
    input_dim = None
    fb_scaling = None

    reservoir = Reservoir(units, input_scaling=input_scaling, sr=spectral_radius,
                          lr=leak_rate, rc_connectivity=connectivity,
                          input_connectivity=input_connectivity, seed=seed)

    readout   = Ridge(1, ridge=regularization)

    esn = reservoir >> readout

    esn = esn.fit(X_train, y_train)
    y_pred_esn = esn.run(X_test)
    y_pred_esn= nrmse(y_test, y_pred_esn)
    y_esn.append(y_pred_esn)

  plt.figure(figsize=(22.5, 15))
  plt.plot(timesteps, y_esn, lw=5, linestyle="-", marker="x", markersize=15, label="ESN prediction")
  plt.plot(timesteps, y_ridge, lw=5, linestyle="-", marker="x", markersize=15, label="Ridge prediction")
  plt.plot(timesteps, y_lstm, lw=5, linestyle="-", marker="x", markersize=15, label="Lstm prediction")
  plt.plot(timesteps, y_SVR, lw=5, linestyle="-", marker="x", markersize=15, label="SVR prediction")
  plt.title ("Normalised Root Mean Squared Error (NRMSE) for prediction of the Mackey-glass series for $\t{k}$ timesteps in the future",fontsize=25)
  plt.xticks(fontsize=16)
  plt.yticks(fontsize=16)
  plt.legend(fontsize=20)
  plt.xlabel ("$\t{k}$ Timesteps",fontsize=20)
  plt.ylabel ("NRMSE",fontsize=20)
  plt.grid(linestyle='--', linewidth=0.5, )
  plt.savefig('NRMSE results for 2-100 timesteps with and without reservoir')
  plt. clf()

MSE_all(X,0.8)

def plot_results_topBot(y_test,y_pred_esn,y_pred_ridge_timesteps,sample):
  plt. clf()
  plt.figure(figsize=(22.5, 15))
  ax1 = plt.subplot((312))
  ax1.plot(np.arange(sample), y_pred_esn[:sample], lw=3, label="ESN prediction")
  ax1.plot(np.arange(sample), y_test[:sample], linestyle="--", lw=2, label="True value")
  ax1.set_title('Results for '+str(forecast)+' timesteps with reservoir',fontsize=20)
  ax1.legend(fontsize=18)
  ax1.tick_params(axis='both',labelsize=13)
  ax1.set_ylabel ("Timeseries - Mackey Glass / $P(t)$", fontsize=25)
  ax1.grid(axis='y',linestyle='--', linewidth=0.35, color='gray')

  ax2 = plt.subplot((311))
  ax2.plot(y_pred_ridge_timesteps, lw=3, label="Ridge prediction")
  ax2.plot(y_test, linestyle="--", lw=2, label="True value")
  ax2.set_title ("Results for "+str(forecast)+" timesteps without reservoir",fontsize=20)
  ax2.legend(fontsize=18)
  ax2.tick_params(axis='both',labelsize=13)
  ax2.grid(axis='y',linestyle='--', linewidth=0.35, color='gray')

  ax3 = plt.subplot((313))
  ax3.plot(np.arange(sample), y_pred_esn[:sample], lw=3, label="ESN prediction")
  ax3.plot(y_pred_ridge_timesteps, lw=3, label="Ridge prediction")
  ax3.plot(y_test, linestyle="--", lw=2, label="True value")
  ax3.set_title ("Results for "+str(forecast)+" timesteps with and without reservoir",fontsize=20)
  ax3.legend(fontsize=18)
  ax3.tick_params(axis='both',labelsize=13)
  ax3.set_xlabel ("Number of days",fontsize=20)
  ax3.grid(axis='y',linestyle='--', linewidth=0.35, color='gray')

  plt.savefig('Predictions for '+str(forecast)+' timesteps with and without reservoir')
  plt. clf()

plot_results_topBot(y_test1,y_pred1,y_pred_ridge_timesteps,sample)

#finding the R^2 and NRMSE(normalised RMSE)
def analyse(y_test, y_pred):
  print("The R-squared is:"+str(rsquare(y_test, y_pred)))
  print("The Normalised Root Mean Squared Error is:" +str(nrmse(y_test, y_pred)))
  print("\n")

print("The following shows the R^2 and NRMSE for 10 timesteps ahead")
print("\nWith reservoir:")
analyse(y_test1,y_pred1)
print("\nWithout reservoir:")
analyse(y_test_ridge_timesteps,y_pred_ridge_timesteps)

#predicting 100 time steps ahead with reservoir
forecast = 100
x, y = to_forecasting(X, forecast=forecast)
X_train2, y_train2 = x[:split_index], y[:split_index]
X_test2, y_test2 = x[split_index:], y[split_index:]

plot_train_test(X_train2, y_train2, X_test2, y_test2)
y_pred2 = esn.fit(X_train2, y_train2).run(X_test2)
sample = len(X) - forecast - split_index
plot_results(y_pred2, y_test2, sample)

#ridge
(y_pred_ridge_timesteps2,y_test_ridge_timesteps2) = ridge_regression_timesteps(X,0.8,forecast)

#lstm
(y_pred_lstm_timesteps2,y_test_lstm_timesteps2) = lstm_timesteps(X,0.8,forecast)

#SVR
(y_pred_SVR_timesteps2,y_test_SVR_timesteps2) = svr_timesteps(X,0.8,forecast)


print("The following shows the R^2 and NRMSE for 100 timesteps ahead")
print("\nWith reservoir:")
analyse(y_test2,y_pred2)
print("\nWithout reservoir:")
analyse(y_test_ridge_timesteps2,y_pred_ridge_timesteps2)

plot_results_topBot(y_test2,y_pred2,y_pred_ridge_timesteps2,sample)
plot_all(y_test2,y_pred2,y_pred_ridge_timesteps2,y_pred_lstm_timesteps2,y_pred_SVR_timesteps2,sample)



#We are done with timeshifted time series
#Now we will do time series prediction
os.chdir('..')
os.chdir('Timeseries_forecasting')

#Split data into training and testing sets
def split_data_time_series(X,split_decimal):
  split_index = int(split_decimal * len(X))
  window_size = len(X) - split_index
  train_data = X[:split_index]
  test_data = X[split_index-window_size:]

  X_train = []
  y_train = []
  for i in range(window_size, len(train_data)):
    X_train.append(train_data[i-window_size:i,0])
    y_train.append(train_data[i,0])

  X_train = np.array(X_train).reshape(-1, window_size)
  y_train = np.array(y_train)
  y_train = y_train.reshape(y_train.shape[0],1)

  X_test = []
  y_test = X[split_index:]
  for i in range(window_size, len(test_data)):
    X_test.append(test_data[i-window_size:i,0])
  X_test = np.array(X_test).reshape(-1, window_size)
  y_test = np.array(y_test)


  return(X_train,y_train,X_test,y_test,train_data,split_index,window_size)

X_train,y_train,X_test,y_test,train_data,split_index,window_size = split_data_time_series(X,split_decimal = 0.8)

#Defining
input_dim = window_size
reservoir = Reservoir(units, input_scaling=input_scaling, sr=spectral_radius,
                      lr=leak_rate, rc_connectivity=connectivity,
                      input_connectivity=input_connectivity, seed=seed, input_dim=input_dim)

readout   = Ridge(1, ridge=regularization)

esn = reservoir >> readout

# Plotting
def plot_results_1(X,window_size,split_index,train_data,y_test,y_pred1,y_pred2,title1,title2):
  plt. clf()
  plt.figure(figsize=(15, 15))
  k = int(window_size*0.5)

  ax1 = plt.subplot((211))
  ax1.plot(range(split_index-k, split_index), train_data[split_index-k:], lw=3, label="Training")
  ax1.plot(range(split_index, len(X)), y_test, lw=3, label="True Value")
  ax1.plot(range(split_index, len(X)), y_pred1, label="Predicted Values without reservoir")
  ax1.set_title(title1)
  ax1.set_ylabel("Timeseries - Mackey Glass / arb", fontsize = 20)
  ax1.legend()
  ax1.tick_params(axis='both',labelsize=13)
  ax1.grid(axis='y',linestyle='--', linewidth=0.35, color='gray')

  ax2 = plt.subplot((212))
  ax2.plot(range(split_index-k, split_index), train_data[split_index-k:], lw=3, label="Training")
  ax2.plot(range(split_index, len(X)), y_test, lw=3, label="True Value")
  ax2.plot(range(split_index, len(X)), y_pred2, label="Predicted Values with reservoir")
  ax2.set_title(title2)
  ax2.set_ylabel("Timeseries - Mackey Glass / arb", fontsize = 20)
  ax2.set_xlabel("Number of days")
  ax2.legend()


# Train a ridge regression model
ridge_model = Ridge_sklearn(alpha=1.0)
ridge_model.fit(X_train, y_train)
y_pred_ridge = ridge_model.predict(X_test)
title_1= str("Predictions for "+str(window_size)+" timesteps ahead without reservoir (ridge regression)")

# Train ESN model
y_pred_esn = esn.fit(X_train, y_train).run(X_test)
title_2= str("Predictions for "+str(window_size)+" timesteps ahead with reservoir (before optimisation)")

# Plotting the results
plot_results_1(X,window_size,split_index,train_data,y_test,y_pred_ridge,y_pred_esn,title_1,title_2)
plt.savefig('Results for '+str(window_size)+' timesteps')
plt. clf()

print("The following shows the R^2 and NRMSE for prediction")
print("\nWith reservoir:")
analyse(y_test,y_pred_esn)
print("\nWithout reservoir:")
analyse(y_test,y_pred_ridge)


#Optimse the hyperparameters
os.chdir('..')
os.chdir('Changing_hyperParameters')
#Defining the objective

def objective(dataset,config, *,iss,N,sr,lr,icntvt,rcntvt,idim,ridge,seed):

  train_data, validation_data = dataset
  X_train, y_train = train_data
  X_val, y_val = validation_data

  instances = config["instances_per_trial"]
  variable_seed = seed

  losses = []; r2s = [];
  for n in range(instances):
    # Build your model given the input parameters
    reservoir = Reservoir(units=N, sr=sr, lr=lr, input_scaling=iss, seed=variable_seed,input_connectivity=icntvt,rc_connectivity=rcntvt,input_dim=idim)

    readout = Ridge(ridge=ridge)

    model = reservoir >> readout


    # Train your model and test your model.
    predictions = model.fit(X_train, y_train).run(X_val)

    loss = nrmse(y_test, predictions, norm_value=np.ptp(X_train))
    r2 = rsquare(y_test, predictions)

    # Change the seed between instances
    variable_seed += 1

    losses.append(loss)
    r2s.append(r2)

  return {'loss': np.mean(losses), 'r2': np.mean(r2s)}

#Defining the workspace
#Connectivity has to be in [0,1]
hyperopt_config = {
    "exp": f"hyperopt-mackeyGlass",
    "hp_max_evals": 200,
    "hp_method": "random",
    "seed": 69,
    "instances_per_trial": 3,
    "hp_space": {
        "N": ["choice", 500],
        "sr": ["loguniform", 1e-2, 10],
        "lr": ["loguniform", 1e-3, 1],
        "iss": ["loguniform", 1e-2,10],
        "ridge": ["choice", 1e-7],
        "seed": ["choice", 1234],
        "icntvt": ["loguniform", 1e-2,1],
        "rcntvt": ["loguniform", 1e-2,1],
        "idim": ["choice", window_size]
    }
}

dataset = ((X_train, y_train), (X_test, y_test))

with open(f"{hyperopt_config['exp']}.config.json", "w+") as f:
  json.dump(hyperopt_config, f)

best = research(objective, dataset, f"{hyperopt_config['exp']}.config.json", ".")

fig = plot_hyperopt_report(hyperopt_config["exp"], ("iss","sr","lr","icntvt","rcntvt"), metric="r2")
plt.savefig('Hyperoptimisation for all parameters')

# plot results using optimised hyperparameters
# use the following;
for element in best:
 print(element)

units = 500 #number of neurons
leak_rate = best[0]['lr']
spectral_radius = best[0]['sr']
input_scaling = best[0]['iss']
rc_connectivity = best[0]['rcntvt']
input_connectivity = best[0]['icntvt']
regularization = 1e-8
input_dim = window_size
seed = 1234


reservoir = Reservoir(units, input_scaling=input_scaling, sr=spectral_radius,
                      lr=leak_rate, rc_connectivity=rc_connectivity,
                      input_connectivity=input_connectivity,
                      seed=seed, input_dim=input_dim)

readout   = Ridge(1, ridge=regularization)

esn = reservoir >> readout
y_pred_esn_optimised = esn.fit(X_train, y_train).run(X_test)

def plot_results_2(X,window_size,split_index,train_data,y_test,y_pred1,y_pred2,y_pred3,title1,title2,title3):
  plt. clf()
  plt.figure(figsize=(17, 21))
  k = int(window_size*0.5)

  ax1 = plt.subplot((311))
  ax1.plot(range(split_index-k, split_index), train_data[split_index-k:], lw=3, label="Training data")
  ax1.plot(range(split_index, len(X)), y_test, lw=3, label="True Value")
  ax1.plot(range(split_index, len(X)), y_pred1, label="Prediction without reservoir")
  ax1.text(0.5, 1.1, "Predictions for " + str(window_size) + " timesteps ahead", fontsize=30, fontweight='bold', ha='center', va='bottom', transform=ax1.transAxes)
  ax1.text(0.5, 1.0, "Ridge Regression, NRMSE = " + str(round(nrmse(y_test, y_pred1), 3)), fontsize=25, ha='center', va='bottom', transform=ax1.transAxes)
  ax1.grid(axis='y',linestyle='--', linewidth=0.30, color='gray')
  ax1.set_ylim(top=ax1.get_ylim()[1]*1.1)
  ax1.legend(fontsize=16)
  ax1.tick_params(axis='both', which='major', labelsize=16)

  ax2 = plt.subplot((312))
  ax2.plot(range(split_index-k, split_index), train_data[split_index-k:], lw=3, label="Training data")
  ax2.plot(range(split_index, len(X)), y_test, lw=3, label="True Value")
  ax2.plot(range(split_index, len(X)), y_pred2, label="Prediction with non-optimized hyperparameters")
  ax2.set_title("RC before optimization, NRMSE = "+str(round(nrmse(y_test, y_pred2), 3)),fontsize=25)
  ax2.set_ylabel("Timeseries - Mackey Glass / $P(t)$",fontsize=30)
  ax2.grid(axis='y',linestyle='--', linewidth=0.35, color='gray')
  ax2.legend(fontsize=16)
  ax2.tick_params(axis='both', which='major', labelsize=16)

  ax3 = plt.subplot((313))
  ax3.plot(range(split_index-k, split_index), train_data[split_index-k:], lw=3, label="Training data")
  ax3.plot(range(split_index, len(X)), y_test, lw=3, label="True Value")
  ax3.plot(range(split_index, len(X)), y_pred3, label="Prediction with optimized hyperparameters")
  ax3.set_title("RC after optimazation, NRMSE = "+str(round(nrmse(y_test, y_pred3), 3)),fontsize=25)
  ax3.set_xlabel("Number of days",fontsize=25)
  ax3.legend(fontsize=16)
  ax3.grid(axis='y',linestyle='--', linewidth=0.35, color='gray')
  ax3.tick_params(axis='both', which='major', labelsize=16)


def plot_bar(y_test,y_pred1,y_pred2,y_pred3):
  x = ['(Ridge)']
  x1 =['Non-optimized \nhyperparameters (RC)','Optimized \nhyperparameters (RC)']
  y = [nrmse(y_test, y_pred1)]
  y1 =[nrmse(y_test, y_pred2),nrmse(y_test, y_pred3)]
  plt.bar(x, y, label = "Without Reservoir")
  plt.bar(x1, y1, label = "With Reservoir", color = 'magenta')
  plt.xlabel('Prediction',fontsize=30)
  plt.ylabel('NRMSE',fontsize=30)
  plt.title('NRMSE against predictions',fontsize=30)
  plt.grid(linestyle='--', linewidth=0.35, color='gray')
  plt.legend(fontsize=25)
  plt.xticks(fontsize=20)
  plt.yticks(fontsize=20)



title_3 =  str("Predictions for "+str(window_size)+" timesteps ahead with reservoir (after optimisation)")
plot_results_2(X,window_size,split_index,train_data,y_test,y_pred_ridge,y_pred_esn,y_pred_esn_optimised,title_1,title_2,title_3)
plt.savefig('Results for '+str(window_size)+' timesteps ahead with optimised hyperparameters')
plt. clf()
plot_bar(y_test,y_pred_ridge,y_pred_esn,y_pred_esn_optimised)
plt.savefig('Bar chart for '+str(window_size)+' timesteps ahead with optimised hyperparameters')
plt. clf()



print("The following shows the R^2 and NRMSE for optimised prediction")
print("\nWith optimised reservoir:")
analyse(y_test,y_pred_esn_optimised)

######### AMAZON #########
print('####### Amazon Stock Price 2014-2019 #######\n\n')
os.chdir('..')
plt. clf()

#Reading the csv file
df_train = pd.read_csv('AMZNtrain.csv')
df_test = pd.read_csv('AMZNtest.csv')
#Plotting the stock price for open, high, low, close, adj close.
os.chdir('Amazon_Stock')
os.chdir('Open')
ax = df_train.plot(x = 'Since start of year', y = 'Open', xlabel = 'Number of days',
              ylabel = 'Stock price at the beginning of the trading day / $', title ='Amazon opening stock price from 2014 to 2019' )
ax.grid(linestyle=':', linewidth=0.30, color='gray')
plt.savefig('Opening stock price 2014-2019')
plt. clf()
os.chdir('..')


os.chdir('Close')
df_train.plot(x = 'Since start of year', y = 'Close', xlabel = 'Number of days', ylabel = 'Stock at the end of the trading day')
plt.savefig('Closing stock price 2014-2019')
plt. clf()
os.chdir('..')

os.chdir('High')
df_train.plot(x = 'Since start of year', y = 'High', xlabel = 'Number of days', ylabel = 'Highest stock price during the trading day')
plt.savefig('Highest stock price 2014-2019')
plt. clf()
os.chdir('..')

os.chdir('Low')
df_train.plot(x = 'Since start of year', y = 'Low', xlabel = 'Number of days', ylabel = 'Lowest stock price during the trading day')
plt.savefig('Lowest stock price 2014-2019')
plt. clf()
os.chdir('..')

os.chdir('Adj_close')
df_train.plot(x = 'Since start of year', y = 'Adj Close', xlabel = 'Number of days', ylabel = 'Adjusted closing stock price during the trading day')
plt.savefig('Adjusted closing stock price 2014-2019')
plt. clf()
os.chdir('..')


#Indexing and rescaling data between -1 and 1
X_opening = np.vstack(df_train['Open'])
X_opening = 2 * (X_opening - X_opening.min()) / (X_opening.max() - X_opening.min()) - 1

X_closing = np.vstack(df_train['Close'])
X_closing = 2 * (X_closing - X_closing.min()) / (X_closing.max() - X_closing.min()) - 1

X_high = np.vstack(df_train['High'])
X_high = 2 * (X_high - X_high.min()) / (X_high.max() - X_high.min()) - 1

X_low = np.vstack(df_train['Low'])
X_low = 2 * (X_low - X_low.min()) / (X_low.max() - X_low.min()) - 1

X_adjClose = np.vstack(df_train['Adj Close'])
X_adjClose = 2 * (X_adjClose - X_adjClose.min()) / (X_adjClose.max() - X_adjClose.min()) - 1


#Data preprocessing for opening stock price
X_train,y_train,X_test,y_test,train_data,split_index,window_size = split_data_time_series(X_opening,split_decimal = 0.8)

#Bulding ESN using default values
units = 500
leak_rate = 0.3
spectral_radius = 0.07
input_scaling = 0.05
rc_connectivity = 0.05
input_connectivity = 0.05
regularization = 1e-8
input_dim = window_size
seed = 1234

reservoir = Reservoir(units, input_scaling=input_scaling, sr=spectral_radius,
                      lr=leak_rate, rc_connectivity=rc_connectivity,
                      input_connectivity=input_connectivity,
                      seed=seed, input_dim=input_dim)

readout   = Ridge(1, ridge=regularization)

#Training the reservoir computing before optimsation (using default values)
esn = reservoir >> readout
y_pred_amazon_esn = esn.fit(X_train, y_train).run(X_test)

#Plotting without the reservoir computing
ridge_model = Ridge_sklearn(alpha=1)
ridge_model.fit(X_train, y_train)
y_pred_amazon_ridge = ridge_model.predict(X_test)

#Optimising the hyperparameters
hyperopt_config = {
    "exp": f"hyperopt-amazonOpening",
    "hp_max_evals": 200,
    "hp_method": "random",
    "seed": 69,
    "instances_per_trial": 3,
    "hp_space": {
        "N": ["choice", 500],
        "sr": ["loguniform", 1e-2, 10],
        "lr": ["loguniform", 1e-3, 1],
        "iss": ["loguniform", 1e-2,10],
        "ridge": ["choice", 1e-7],
        "seed": ["choice", 1234],
        "icntvt": ["loguniform", 1e-2,1],
        "rcntvt": ["loguniform", 1e-2,1],
        "idim": ["choice", window_size]
    }
}

dataset = ((X_train, y_train), (X_test, y_test))

with open(f"{hyperopt_config['exp']}.config.json", "w+") as f:
  json.dump(hyperopt_config, f)

best = research(objective, dataset, f"{hyperopt_config['exp']}.config.json", ".")

fig = plot_hyperopt_report(hyperopt_config["exp"], ("iss","sr","lr","icntvt","rcntvt"), metric="r2")
plt.savefig('Hyperoptimisation for Amazon Opening')
plt. clf()

for element in best:
 print(element)

#plot results using optimised hyperparameters
units = 500
leak_rate = best[0]['lr']
spectral_radius = best[0]['sr']
input_scaling = best[0]['iss']
rc_connectivity = best[0]['rcntvt']
input_connectivity = best[0]['icntvt']
regularization = 1e-8
input_dim = window_size
seed = 1234


reservoir = Reservoir(units, input_scaling=input_scaling, sr=spectral_radius,
                      lr=leak_rate, rc_connectivity=rc_connectivity,
                      input_connectivity=input_connectivity,
                      seed=seed, input_dim=input_dim)

readout   = Ridge(1, ridge=regularization)

esn = reservoir >> readout
y_pred_esn_optimised_amazon = esn.fit(X_train, y_train).run(X_test)

def plot_results_2(X,window_size,split_index,train_data,y_test,y_pred1,y_pred2,y_pred3,title1,title2,title3):
  plt. clf()
  plt.figure(figsize=(17, 21))
  k = int(window_size*0.5)

  ax1 = plt.subplot((311))
  ax1.plot(range(split_index-k, split_index), train_data[split_index-k:], lw=3, label="Training data")
  ax1.plot(range(split_index, len(X)), y_test, lw=3, label="True Value")
  ax1.plot(range(split_index, len(X)), y_pred1, label="Prediction without reservoir")
  ax1.text(0.5, 1.1, "Predictions for " + str(window_size) + " timesteps ahead", fontsize=30, fontweight='bold', ha='center', va='bottom', transform=ax1.transAxes)
  ax1.text(0.5, 1.0, "Ridge Regression, NRMSE = " + str(round(nrmse(y_test, y_pred1),3)), fontsize=25, ha='center', va='bottom', transform=ax1.transAxes)
  ax1.grid(axis='y',linestyle='--', linewidth=0.30, color='gray')
  ax1.legend(fontsize=20)
  ax1.set_ylim(top=ax1.get_ylim()[1]*1.1)
  ax1.tick_params(axis='both', which='major', labelsize=16)

  ax2 = plt.subplot((312))
  ax2.plot(range(split_index-k, split_index), train_data[split_index-k:], lw=3, label="Training data")
  ax2.plot(range(split_index, len(X)), y_test, lw=3, label="True Value")
  ax2.plot(range(split_index, len(X)), y_pred2, label="Prediction with non-optimized hyperparameters")
  ax2.set_title("RC before optimization, NRMSE = "+str(round(nrmse(y_test, y_pred2),3)),fontsize=25)
  ax2.set_ylabel("Normalised Amazon Prime Opening Stock",fontsize=30)
  ax2.grid(axis='y',linestyle='--', linewidth=0.35, color='gray')
  ax2.legend(fontsize=20)
  ax2.tick_params(axis='both', which='major', labelsize=16)

  ax3 = plt.subplot((313))
  ax3.plot(range(split_index-k, split_index), train_data[split_index-k:], lw=3, label="Training data")
  ax3.plot(range(split_index, len(X)), y_test, lw=3, label="True Value")
  ax3.plot(range(split_index, len(X)), y_pred3, label="Prediction with optimized hyperparameters")
  ax3.set_title("RC after optimazation, NRMSE = "+str(round(nrmse(y_test, y_pred3),3)),fontsize=25)
  ax3.set_xlabel("Number of days",fontsize=25)
  ax3.grid(axis='y',linestyle='--', linewidth=0.35, color='gray')
  ax3.legend(fontsize=20)
  ax3.tick_params(axis='both', which='major', labelsize=16)


plot_results_2(X_opening,window_size,split_index,train_data,y_test,y_pred_amazon_ridge,
               y_pred_amazon_esn,y_pred_esn_optimised_amazon,title_1,title_2,title_3)
plt.savefig('Results for '+str(window_size)+' timesteps ahead for amazon opening stock price with optimised hyperparameters')
plt. clf()
plot_bar(y_test,y_pred_amazon_ridge,y_pred_amazon_esn,y_pred_esn_optimised_amazon)
plt.savefig('Bar chart for '+str(window_size)+' timesteps ahead for amazon opening stock price with optimised hyperparameters')
plt. clf()



print("The following shows the R^2 and NRMSE for optimised prediction")
print("\nWith reservoir:")
analyse(y_test,y_pred_esn_optimised_amazon)

