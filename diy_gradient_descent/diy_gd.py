import sys
import numpy as np
import pandas as pd
from matplotlib import pyplot as plt
if len(sys.argv) > 1 and sys.argv[1] == 'raw':
    from linear_fit_gd_raw import linear_fit
else:
    from linear_fit_gd import linear_fit

train_data = pd.read_csv("./train.csv")
train_data = train_data.dropna()

train_data = train_data.sample(frac=1)

X = train_data.iloc[:, 0].values #all rows 0th column
Y = train_data.iloc[:, 1].values #all rows 1st column

plt.figure(figsize=(16, 8))
plt.scatter(
    X,
    Y,
    c='black'
)
plt.xlabel("Marketing expense ($)")
plt.ylabel("Sales ($)")
plt.show()

train_size = int(len(X) * 0.7)
Xtrain = X[0:train_size]
Ytrain = Y[0:train_size]
Xvalidate = X[train_size:]
Yvalidate = Y[train_size:]


m, iteration_mse, iteration_param = linear_fit(
				Xtrain, Ytrain, Xvalidate, Yvalidate, n_iterations=10, learning_rate=0.0001, m=100)

plt.plot(iteration_param, iteration_mse, color='red')
plt.show()

test_data = pd.read_csv("./test.csv")
test_data = test_data.dropna()

Xtest = test_data.iloc[:, :-1].values
Ytest = test_data.iloc[:, 1].values

#Now predict on the test data and find the error: TODO
