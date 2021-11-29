from sklearn.tree import DecisionTreeClassifier
import numpy as np
import math
import matplotlib.pyplot as plt


def adaboost_trees(X_train, y_train, X_test, y_test, n_trees):
    # %AdaBoost: Implement AdaBoost using decision trees
    # %   using information gain as the weak learners.
    # %   X_train: Training set
    # %   y_train: Training set labels
    # %   X_test: Testing set
    # %   y_test: Testing set labels
    # %   n_trees: The number of trees to use
    N = X_train.shape[0]
    N_test = X_test.shape[0]
    train_results = np.zeros((N, ))
    test_results = np.zeros((N_test, ))
    train_error = np.zeros((n_trees, ))
    test_error = np.zeros((n_trees, ))
    weights = np.full((N, ), 1/N)

    for t in range(n_trees):
        classifier = DecisionTreeClassifier(criterion='entropy', max_depth=1)
        classifier.fit(X_train, y_train, sample_weight=weights)

        y_train_predict = classifier.predict(X_train)
        epsilon = np.inner(weights, (y_train_predict != y_train))
        alpha = 0.5 * np.log((1-epsilon)/epsilon)
        gamma = math.sqrt((1-epsilon)/epsilon)
        z = epsilon*gamma + (1-epsilon)/gamma
        weights = (1/z) * weights*np.exp(-alpha * y_train_predict * y_train)

        y_train_predict_weighted = alpha * y_train_predict
        y_train_predict_weighted = np.add(y_train_predict_weighted, train_results)
        error_tr = np.sum(np.sign(y_train_predict_weighted) != y_train) / N
        train_error[t] = error_tr
        train_results = y_train_predict_weighted

        y_test_predict = alpha*classifier.predict(X_test)
        y_test_predict = np.add(y_test_predict, test_results)
        error_te = np.sum(np.sign(y_test_predict) != y_test) / N_test
        test_error[t] = error_te
        test_results = y_test_predict

    return train_error, test_error


def main_hw5():
    # Load data
    train_data = np.genfromtxt('zip.train.txt')
    test_data = np.genfromtxt('zip.test.txt')

    # Extract data
    one_three_train = train_data[np.logical_or(train_data[:, 0] == 1, train_data[:, 0] == 3), :]
    one_three_test = test_data[np.logical_or(test_data[:, 0] == 1, test_data[:, 0] == 3), :]
    three_five_train = train_data[np.logical_or(train_data[:, 0] == 3, train_data[:, 0] == 5), :]
    three_five_test = test_data[np.logical_or(test_data[:, 0] == 3, test_data[:, 0] == 5), :]
    num_trees = 200

    # ONE THREE CLASSIFICATION
    X_train = one_three_train[:, 1:]
    y_train = one_three_train[:, 0]
    y_train = np.where(y_train == 1, y_train, -1)
    X_test = one_three_test[:, 1:]
    y_test = one_three_test[:, 0]
    y_test = np.where(y_test == 1, y_test, -1)

    train_error, test_error = adaboost_trees(X_train, y_train, X_test, y_test, num_trees)
    trees = np.arange(1, 201)
    plt.plot(trees, train_error, color='b', label='train')
    plt.plot(trees, test_error, color='g', label='test')
    plt.xlabel("num_trees")
    plt.ylabel("error")
    plt.title("One Three Classification")
    plt.legend()
    plt.show()

    # THREE FIVE
    X_train = three_five_train[:, 1:]
    y_train = three_five_train[:, 0]
    y_train = np.where(y_train == 3, 1, -1)
    X_test = three_five_test[:, 1:]
    y_test = three_five_test[:, 0]
    y_test = np.where(y_test == 3, 1, -1)

    train_error, test_error = adaboost_trees(X_train, y_train, X_test, y_test, num_trees)
    trees = np.arange(1, 201)
    plt.plot(trees, train_error, color='b', label='train')
    plt.plot(trees, test_error, color='g', label='test')
    plt.xlabel("num_trees")
    plt.ylabel("error")
    plt.title("Three Five Classification")
    plt.legend()
    plt.show()



if __name__ == "__main__":
    main_hw5()