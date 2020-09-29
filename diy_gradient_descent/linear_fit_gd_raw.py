from sklearn.metrics import mean_squared_error
import math

def error(y, x, m):
    y_predict = x * m
    return (y - y_predict)**2

def linear_fit(Xtrain, Ytrain, Xvalidate, Yvalidate, n_iterations, learning_rate, m):
    print("Computing raw (inefficient) gradients ...")
    iteration_mse = []
    iteration_param = []

    least_mse = math.inf
    best_m = None
    for epoch in range(0, n_iterations):
        batch_derror_dm = 0
        dm = 0.00001
        mse = 0
        mse_delta = 0

        for x, y in zip(Xtrain, Ytrain):
            mse += error(y, x, m)
            mse_delta += error(y, x, m+dm)

        mse /= len(Ytrain)
        mse_delta /= len(Ytrain)
        dmse_dm = (mse_delta - mse)/dm

        if mse < least_mse:
            best_m = m
            least_mse = mse

        m = m - learning_rate * dmse_dm

        iteration_mse.append(mse)
        iteration_param.append(m)
        
    return (best_m, least_mse, iteration_mse, iteration_param)
