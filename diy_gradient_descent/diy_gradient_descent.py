#A DIY/from-scratch implementation of 1D linear regression using gradient descent
#Copyright Jayaraj Poroor 2020
#Released under MIT License

import sys
import time
import math
import numpy as np
import pandas as pd
from matplotlib import pyplot as plt
if len(sys.argv) > 1 and sys.argv[1] == 'raw':
    from linear_fit_gd_raw import linear_fit
else:
    from linear_fit_gd_derivative import linear_fit

#Read training data from CSV file
train_data = pd.read_csv("./train.csv")
train_data = train_data.dropna()

#Randomize order
train_data = train_data.sample(frac=1)

#Split X, Y columns
X = train_data.iloc[:, 0].values #all rows 0th column
Y = train_data.iloc[:, 1].values #all rows 1st column

#Plot
#plt.figure(figsize=(16, 8))
#plt.scatter(
#    X,
#    Y,
#    c='black'
#)
#plt.xlabel("Marketing expense ($)")
#plt.ylabel("Sales ($)")
#plt.show()

#Split into train/validate sets
train_size = int(len(X) * 0.7)
Xtrain = X[0:train_size]
Ytrain = Y[0:train_size]
Xvalidate = X[train_size:]
Yvalidate = Y[train_size:]

#Train - returns 'm' : the learnt parameter, 'iteration_mse': progression of mse over iterations
#'iteration_param' : progression of the learnt parameter over iterations
start_ms = time.time()
m, mse, iteration_mse, iteration_param = linear_fit(
				Xtrain, Ytrain, Xvalidate, Yvalidate, n_iterations=20, learning_rate=0.0001, m=100)
end_ms = time.time()
time_taken = round((end_ms - start_ms), 2)
mse = round(math.sqrt(mse)/max(Yvalidate), 2)
print("Best 'm' is", m, "with error: ", mse, "%", "Training time:", time_taken, "ms")
#Plot the progression of error/learnt parameter over iterations
plt.plot(iteration_param, iteration_mse, color='red')
plt.show()

test_data = pd.read_csv("./test.csv")
test_data = test_data.dropna()

Xtest = test_data.iloc[:, :-1].values
Ytest = test_data.iloc[:, 1].values

#Now predict on the test data and find the error: TODO
