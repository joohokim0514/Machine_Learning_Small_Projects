import numpy as np
import matplotlib.pyplot as plt


def perceptron_learn(data_in):
    # Run PLA on the input data
    #
    # Inputs: data_in: Assumed to be a matrix with each row representing an
    #                (x,y) pair, with the x vector augmented with an
    #                initial 1 (i.e., x_0), and the label (y) in the last column
    # Outputs: w: A weight vector (should linearly separate the data if it is linearly separable)
    #        iterations: The number of iterations the algorithm ran for

    # Your code here, assign the proper values to w and iterations:
    iterations = 0
    x = data_in[0]
    y = data_in[1]
    N = x.shape[0]
    d = x.shape[1]-1
    y_training = np.zeros((N, ))
    w = np.zeros((d+1, ))
    w[0] = 0

    while not np.array_equal(y, y_training):
        for i in range(N):
            if y[i] != np.sign(np.dot(w, x[i])):
                w = w + np.multiply(y[i], x[i])
                iterations += 1
            y_training[i] = np.sign(np.dot(w, x[i]))

    return w, iterations


def perceptron_experiment(N, d, num_exp):
    # Code for running the perceptron experiment in HW1
    # Implement the dataset construction and call perceptron_learn; repeat num_exp times
    #
    # Inputs: N is the number of training data points
    #         d is the dimensionality of each data point (before adding x_0)
    #         num_exp is the number of times to repeat the experiment
    # Outputs: num_iters is the # of iterations PLA takes for each experiment
    #          bounds_minus_ni is the difference between the theoretical bound and the actual number of iterations
    # (both the outputs should be num_exp long)

    # Initialize the return variables
    num_iters = np.zeros((num_exp,))
    bounds_minus_ni = np.zeros((num_exp,))

    for i in range(num_exp):
        w_star = np.random.uniform(0, 1, d + 1)
        w_star[0] = 0
        x = np.zeros((N * (d + 1),))
        for j in range(N * (d + 1)):
            x[j] = np.random.uniform(-1, 1)
        x = x.reshape(N, d + 1)
        x[:, 0] = 1
        y = np.zeros((N,))
        R = float('-inf')
        p = float('inf')
        for j in range(N):
            y[j] = np.sign(np.dot(w_star, x[j]))
            R = max(R, np.linalg.norm(x[j]))
            p = min(p, y[j] * np.dot(w_star, x[j]))
        data_in = [x, y]
        [w, iteration] = perceptron_learn(data_in)

        num_iters[i] = iteration
        w_norm = np.linalg.norm(w_star)
        bound = pow(R, 2) * pow(w_norm, 2) / pow(p, 2)
        bounds_minus_ni[i] = bound - iteration

    return num_iters, bounds_minus_ni


def main():
    print("Running the experiment...")
    num_iters, bounds_minus_ni = perceptron_experiment(100, 10, 1000)
    print("Printing histogram...")
    plt.hist(num_iters)
    plt.title("Histogram of Number of Iterations")
    plt.xlabel("Number of Iterations")
    plt.ylabel("Count")
    plt.show()

    print("Printing second histogram")
    plt.hist(np.log(bounds_minus_ni))
    plt.title("Bounds Minus Iterations")
    plt.xlabel("Log Difference of Theoretical Bounds and Actual # Iterations")
    plt.ylabel("Count")
    plt.show()

if __name__ == "__main__":
    main()


    for i in range(55000):
        v1s.append(0)
    for i in range(40000):
        v1s.append(0.1)
    for i in range(300):
        v1s.append(0.2)
    plt.hist(v1s, bins=11)
    plt.title("Histogram of distribution of vmin")
    plt.xlabel("Fraction of heads")
    plt.ylabel("Frequency of heads")
    plt.show()