from sklearn.metrics import mean_squared_error
import math

#d(Yp - Y)^2/dm where Yp = m * X =  2(Yp - Y) * dYp/dm = 2(Yp-Y)*X

def predict(X, m):
    return [m * x for x in X]

def linear_fit(Xtrain, Ytrain, Xvalidate, Yvalidate, n_iterations, learning_rate, m):
    print("Computing gradients using derivative formula...")
    iteration_mse = []
    iteration_param = []
    least_mse = math.inf
    best_m = None
    for epoch in range(0, n_iterations):
        Yvalidate_pred = predict(Xvalidate, m)
        mse = mean_squared_error(Yvalidate, Yvalidate_pred)
        if mse < least_mse:
            best_m = m
            least_mse = mse
        iteration_mse.append(mse)
        iteration_param.append(m)

        del_m = 0
        for x, y in zip(Xtrain, Ytrain):
            yp = x * m
            dm = 2* (yp - y) * x
            del_m += dm
        del_m /= len(Ytrain)
        m = m - learning_rate * del_m
    return (best_m, least_mse, iteration_mse, iteration_param)
