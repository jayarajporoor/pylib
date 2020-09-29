from sklearn.metrics import mean_squared_error
import math

#d(yp - y)^2/dm where yp = m * x =  2*(yp - y) * d(yp)/dm = 2*(yp-y)*x

def error(y, x, m):
    y_predict = x * m
    return (y - y_predict)**2

def linear_fit(Xtrain, Ytrain, Xvalidate, Yvalidate, n_iterations, learning_rate, m):
    print("Computing gradients using derivative formula...")
    iteration_mse = []
    iteration_param = []
    least_mse = math.inf
    best_m = None
    for epoch in range(0, n_iterations):

        del_m = 0
        for x, y in zip(Xtrain, Ytrain):
            yp = x * m
            dm = 2* (yp - y) * x
            del_m += dm
        del_m /= len(Ytrain)
        m = m - learning_rate * del_m

        mse = 0
        for x, y in zip(Xvalidate, Yvalidate):
            mse += error(y, x, m)
        mse /= len(Ytrain)
        if mse < least_mse:
            best_m = m
            least_mse = mse

        iteration_mse.append(mse)
        iteration_param.append(m)

    return (best_m, least_mse, iteration_mse, iteration_param)
