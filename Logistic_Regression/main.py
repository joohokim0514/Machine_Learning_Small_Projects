import numpy as np
import pandas as pd
import scipy.stats as stats
import time


def find_binary_error(w, X, y):
    # find_binary_error: compute the binary error of a linear classifier w on data set (X, y)
    # Inputs:
    #        w: weight vector
    #        X: data matrix (without an initial column of 1s)
    #        y: data labels (plus or minus 1)
    # Outputs:
    #        binary_error: binary classification error of w on the data set (X, y)
    #           this should be between 0 and 1.

    # Your code here, assign the proper value to binary_error:
    x = np.insert(X, 0, 1, 1)
    N = x.shape[0]
    y_predict = np.sign(np.matmul(x, w))
    error = y_predict != y
    binary_error = np.sum(error)/N
    return binary_error


def logistic_reg(X, y, w_init, max_its, eta, grad_threshold):
    # logistic_reg learn logistic regression model using gradient descent
    # Inputs:
    #        X : data matrix (without an initial column of 1s)
    #        y : data labels (plus or minus 1)
    #        w_init: initial value of the w vector (d+1 dimensional)
    #        max_its: maximum number of iterations to run for
    #        eta: learning rate
    #        grad_threshold: one of the terminate conditions;
    #               terminate if the magnitude of every element of gradient is smaller than grad_threshold
    # Outputs:
    #        t : number of iterations gradient descent ran for
    #        w : weight vector
    #        e_in : in-sample error (the cross-entropy error as defined in LFD)

    # Your code here, assign the proper values to t, w, and e_in:
    start = time.time()
    w = w_init
    x = np.insert(X, 0, 1, 1)
    N = x.shape[0]
    d = x.shape[1]
    t = 0
    while t < max_its:
        gradient = np.multiply(x, y.reshape((-1, 1))) / (1 + np.exp(y * np.matmul(x, w))).reshape((-1, 1))
        g = -np.mean(gradient, 0)
        v = -g
        w = w + np.multiply(eta, v)
        if np.all(np.abs(g) < grad_threshold):
            break
        t += 1
    e_in = np.sum(np.log(1 + np.exp(-y * np.matmul(x, w))))/N
    runtime = time.time()-start
    return t, w, e_in, runtime


def main():
    # Load training data
    raw_train_data = pd.read_csv('clevelandtrain.csv')
    # Load test data
    raw_test_data = pd.read_csv('clevelandtest.csv')

    # Your code here
    train_data = pd.DataFrame.to_numpy(raw_train_data)
    test_data = pd.DataFrame.to_numpy(raw_test_data)
    row = train_data.shape[0]
    row_test = test_data.shape[0]
    col = train_data.shape[1]
    w_init = np.zeros((col, ))

    X_train = train_data[:, :col-1]
    y_train = train_data[:, col-1]
    y_train = y_train - (y_train == 0)
    X_test = test_data[:, :col-1]
    y_test = test_data[:, col-1]
    y_test = y_test - (y_test == 0)
    X_train_mean = np.mean(X_train, axis=0)
    X_train_std = np.std(X_train, axis=0)
    Z_train = (X_train - X_train_mean) / X_train_std
    Z_test = (X_test - X_train_mean) / X_train_std

    eta = 1 / 100000
    grad_threshold = 1 / 1000
    max_its = 10000
    [num_its, w, e_in, runtime] = logistic_reg(X_train, y_train, w_init, max_its, eta, grad_threshold)
    train_error = find_binary_error(w, X_train, y_train)
    test_error = find_binary_error(w, X_test, y_test)
    print("max_its: ", max_its)
    print("num_its: ", num_its)
    print("e_in: ", e_in)
    print("train_error: ", train_error)
    print("test_error: ", test_error)
    print("")

    max_its = 100000
    [num_its, w, e_in, runtime] = logistic_reg(X_train, y_train, w_init, max_its, eta, grad_threshold)
    train_error = find_binary_error(w, X_train, y_train)
    test_error = find_binary_error(w, X_test, y_test)
    print("max_its: ", max_its)
    print("num_its: ", num_its)
    print("train_error: ", train_error)
    print("test_error: ", test_error)
    print("")

    max_its = 1000000
    [num_its, w, e_in, runtime] = logistic_reg(X_train, y_train, w_init, max_its, eta, grad_threshold)
    train_error = find_binary_error(w, X_train, y_train)
    test_error = find_binary_error(w, X_test, y_test)
    print("max_its: ", max_its)
    print("num_its: ", num_its)
    print("e_in: ", e_in)
    print("train_error: ", train_error)
    print("test_error: ", test_error)
    print("")

    etas = [0.01, 0.1, 1, 4, 7, 7.5, 7.6, 7.7]
    for eta in etas:
        grad_threshold = 1/1000000
        max_its = 1000000
        [num_its, w, e_in, runtime] = logistic_reg(Z_train, y_train, w_init, max_its, eta, grad_threshold)
        train_error = find_binary_error(w, Z_train, y_train)
        test_error = find_binary_error(w, Z_test, y_test)
        print("eta: ", eta)
        print("max_its: ", max_its)
        print("num_its: ", num_its)
        print("e_in: ", e_in)
        print("train_error: ", train_error)
        print("test_error: ", test_error)
        print("runtime: ", runtime)
        print("")


if __name__ == "__main__":
    main()