from sklearn.tree import DecisionTreeClassifier
import numpy as np
import matplotlib.pyplot as plt


def bagged_trees(X_train, y_train, X_test, y_test, num_bags):
    N = X_train.shape[0]
    N_test = X_test.shape[0]
    out_of_bag_error = np.zeros((num_bags, ))
    test_error = np.zeros((num_bags, ))
    size = np.arange(N)
    for num_bag in range(1, num_bags+1):
        results = np.zeros((num_bag, N))
        predictions = np.zeros((num_bag, N_test))

        for i in range(num_bag):
            sampled = np.random.choice(X_train.shape[0], N, replace=True)
            X_bag = X_train[sampled, :]
            y_bag = y_train[sampled]
            classifier = DecisionTreeClassifier(criterion='entropy')
            classifier.fit(X_bag, y_bag)

            result = classifier.predict(X_train)
            result[np.unique(sampled)] = 0
            results[i, :] = result
            prediction = classifier.predict(X_test)
            predictions[i, :] = prediction

        S = np.count_nonzero(results, axis=0)
        aggregate = np.sum(results, axis=0)
        np.seterr(divide='ignore', invalid='ignore')
        aggregate = np.nan_to_num(np.sign(aggregate / S))
        index = np.where(aggregate != 0)[0]
        aggregate = aggregate[index]
        target = y_train[index]
        oob_error = np.sum(target != aggregate) / np.count_nonzero(S)
        out_of_bag_error[num_bag-1] = oob_error

        y_predict = np.sign(np.sum(predictions, axis=0))
        error = np.sum(y_predict != y_test) / N_test
        test_error[num_bag-1] = error

    return out_of_bag_error, test_error


def single_decision_tree(X_train, y_train, X_test, y_test):
    classifier = DecisionTreeClassifier(criterion='entropy')
    classifier.fit(X_train, y_train)
    train_error = 1 - classifier.score(X_train, y_train)
    y_predict = classifier.predict(X_test)
    N = y_test.shape[0]
    test_error = np.sum(y_test != y_predict)/N
    return train_error, test_error

def main_hw4():
    # Load data
    train_data = np.genfromtxt('zip.train.txt')
    test_data = np.genfromtxt('zip.test.txt')

    one_three_train = train_data[np.logical_or(train_data[:, 0] == 1, train_data[:, 0] == 3), :]
    one_three_test = test_data[np.logical_or(test_data[:, 0] == 1, test_data[:, 0] == 3), :]
    three_five_train = train_data[np.logical_or(train_data[:, 0] == 3, train_data[:, 0] == 5), :]
    three_five_test = test_data[np.logical_or(test_data[:, 0] == 3, test_data[:, 0] == 5), :]
    num_bags = 200
    np.random.seed(153)

    # ONE THREE
    X_train = one_three_train[:, 1:]
    y_train = one_three_train[:, 0]
    y_train = np.where(y_train == 1, y_train, -1)
    X_test = one_three_test[:, 1:]
    y_test = one_three_test[:, 0]
    y_test = np.where(y_test == 1, y_test, -1)

    out_of_bag_error, bagged_test_error = bagged_trees(X_train, y_train, X_test, y_test, num_bags)
    train_error, single_test_error = single_decision_tree(X_train, y_train, X_test, y_test)
    print("One Three Classification")
    print("out of bag error for 200 bags: ", out_of_bag_error[199])
    print("test error for 200 bags: ", bagged_test_error[199])
    print("test error for single decision tree: ", single_test_error)
    bags = np.arange(1, 201)
    plt.plot(bags, out_of_bag_error)
    plt.title("One Three Classification")
    plt.xlabel("# bags")
    plt.ylabel("out-of-bag error")
    plt.show()

    # THREE FIVE
    X_train = three_five_train[:, 1:]
    y_train = three_five_train[:, 0]
    y_train = np.where(y_train == 3, 1, -1)
    X_test = three_five_test[:, 1:]
    y_test = three_five_test[:, 0]
    y_test = np.where(y_test == 3, 1, -1)

    out_of_bag_error, bagged_test_error = bagged_trees(X_train, y_train, X_test, y_test, num_bags)
    train_error, single_test_error = single_decision_tree(X_train, y_train, X_test, y_test)
    print("Three Five Classification")
    print("out of bag error for 200 bags: ", out_of_bag_error[199])
    print("test error for 200 bags: ", bagged_test_error[199])
    print("test error for single decision tree: ", single_test_error)
    bags = np.arange(1, 201)
    plt.plot(bags, out_of_bag_error)
    plt.title("Three Five Classification")
    plt.xlabel("# bags")
    plt.ylabel("out-of-bag error")
    plt.show()


if __name__ == "__main__":
    main_hw4()
