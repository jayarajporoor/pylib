from sklearn.metrics import mean_squared_error
import math

def predict(X, m):
	return [m * x for x in X]

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
        Yvalidate_pred = predict(Xvalidate, m)
        mse = mean_squared_error(Yvalidate, Yvalidate_pred)
        if mse < least_mse:
            best_m = m
            least_mse = mse
        iteration_mse.append(mse)
        iteration_param.append(m)

        batch_derror_dm = 0
        dm = 0.00001
        for x, y in zip(Xtrain, Ytrain):
            derror_dm = (error(y, x, m+dm) - error(y, x, m))/dm
            batch_derror_dm += derror_dm

        batch_derror_dm /= len(Ytrain)

        m = m - learning_rate * batch_derror_dm

    return (best_m, least_mse, iteration_mse, iteration_param)
