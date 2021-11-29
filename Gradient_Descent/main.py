import numpy as np


def find_binary_error(w, X, y):
    X = np.insert(X, 0, 1, 1)
    N = X.shape[0]
    y_ = np.sign(np.matmul(X, w))
    error = y_ != y
    binary_error = np.sum(error)/N
    return binary_error


def gradient_descent_l1(X, y, w_init, max_its, eta, grad_threshold, lmbda):
    w = w_init
    X = np.insert(X, 0, 1, 1)
    N = X.shape[0]
    t = 0
    while t < max_its:
        gradient = np.multiply(X, y.reshape((-1, 1))) / (1 + np.exp(y * np.matmul(X, w))).reshape((-1, 1))
        g = -np.mean(gradient, 0)
        v = -g
        w_noreg = w + np.multiply(eta, v)
        w_tmp = w_noreg - np.multiply(np.sign(w), eta*lmbda)
        truncated = np.zeros((w.shape[0], ))
        for i in range(truncated.size):
            truncated[i] = 0 if (np.sign(w_noreg[i]) != np.sign(w_tmp[i]) and w_noreg[i] != 0) else w_tmp[i]
        w = truncated
        if np.all(np.abs(g) < grad_threshold):
            break
        t += 1

    e_in = np.sum(np.log(1 + np.exp(-y * np.matmul(X, w))))/N
    return t, w, e_in


def gradient_descent_l2(X, y, w_init, max_its, eta, grad_threshold, lmbda):
    w = w_init
    X = np.insert(X, 0, 1, 1)
    N = X.shape[0]
    t = 0
    decay = 1-2*eta*lmbda
    while t < max_its:
        gradient = np.multiply(X, y.reshape((-1, 1))) / (1 + np.exp(y * np.matmul(X, w))).reshape((-1, 1))
        g = -np.mean(gradient, 0)
        w = np.multiply(w, decay) - np.multiply(g, eta)
        if np.all(np.abs(g) < grad_threshold):
            break
        t += 1

    e_in = np.sum(np.log(1 + np.exp(-y * np.matmul(X, w))))/N
    return t, w, e_in


def main():
    # Load and organize data
    X_train, X_test, y_train, y_test = np.load("digits_preprocess.npy", allow_pickle=True)
    y_train = y_train - (y_train == 0)
    y_test = y_test - (y_test == 0)
    mean = np.mean(X_train, axis=0)
    std = np.std(X_train, axis=0)
    X_train = np.nan_to_num((X_train - mean) / std)
    X_test = np.nan_to_num((X_test - mean) / std)
    d = X_train.shape[1]
    w_init = np.zeros((d+1, ))

    # Define variables
    max_its = 10000
    eta = 0.01
    grad_threshold = 0.000001
    lmbdas = [0, 0.0001, 0.001, 0.005, 0.01, 0.05, 0.1]

    for lmbda in lmbdas:
        [num_its, w, e_in] = gradient_descent_l1(X_train, y_train, w_init, max_its, eta, grad_threshold, lmbda)
        num_zeroes = np.sum(w == 0)
        test_error = find_binary_error(w, X_test, y_test)
        print("L1 logistic regression gradient descent")
        print("lambda: ", lmbda)
        print("num_its: ", num_its)
        print("test_error: ", test_error)
        print("num_zeroes: ", num_zeroes)
        print("")

    for lmbda in lmbdas:
        [num_its, w, e_in] = gradient_descent_l2(X_train, y_train, w_init, max_its, eta, grad_threshold, lmbda)
        num_zeroes = np.sum(w == 0)
        test_error = find_binary_error(w, X_test, y_test)
        print("L2 logistic regression gradient descent")
        print("lambda: ", lmbda)
        print("test_error: ", test_error)
        print("num_zeroes: ", num_zeroes)
        print("")


if __name__ == "__main__":
    main()