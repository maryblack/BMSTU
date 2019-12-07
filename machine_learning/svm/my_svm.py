import random
import numpy as np
import matplotlib.pyplot as plt
from sklearn import svm
from cvxopt import matrix, solvers


def func(x):
    return (x - 3) ** 2 + 2 * x - 6
#     return math.sin(x) + 10


def create_data(num: int):
    X = []
    y = []
    for i in range(num):
        value_x = float(random.randint(0, 20))
        value_y = float(random.randint(0, 20))
        coord = [value_x, value_y]
        X.append(coord)
        if func(value_x) > value_y:
            y.append(1.0)
        else:
            y.append(-1.0)

    return np.array(X), np.array(y)


def find_lambda(X, y):
    l = X.shape[0]
    K = np.multiply(X.T, y)
    P = matrix(1 / 2 * np.dot(K.T, K))
    q = matrix(-np.ones((l, 1)))  # solver решает задачу минимизации, а мы максимизируем функцию
    G = matrix(-np.eye(l))
    h = matrix(np.zeros(l))
    A = matrix(y.reshape(1, -1))
    b = matrix(0.0)
    sol = solvers.qp(P, q, G, h, A, b)
    lambdas = np.array(sol['x'])
    return lambdas


def sep_by_function(X, sep):
    data_len = X.shape[0]
    X_1 = []
    X_2 = []
    for i in range(data_len):
        if X[i][1] < sep(X[i][0]):
            X_1.append(X[i])
        else:
            X_2.append(X[i])
    return np.array(X_1), np.array(X_2)


def data_separation(X, y):
    data_len = len(y)
    X_1 = []
    X_2 = []
    for i in range(data_len):
        if y[i] == 1:
            X_1.append(X[i])
        else:
            X_2.append(X[i])

    return np.array(X_1), np.array(X_2)


def data_plot(X, y, ax, coeff):
    x = np.linspace(0, 21)
    X_true_1, X_true_2 = data_separation(X, y)
    plt.subplot(311)
    plt.title('Real')
    plt.scatter(X_true_1[:, 0], X_true_1[:, 1], c='red', marker=">")
    plt.scatter(X_true_2[:, 0], X_true_2[:, 1], c='green', marker=(5, 2))
    plt.plot(x, func(x))
    plt.ylim(
        -10, 30
    )

    X_pred_1, X_pred_2 = data_separation(X, ax)
    plt.subplot(312)
    plt.title('Predicted by sklearn')
    plt.scatter(X_pred_1[:, 0], X_pred_1[:, 1], c='red', marker=">")
    plt.scatter(X_pred_2[:, 0], X_pred_2[:, 1], c='green', marker=(5, 2))
    #     plt.plot(x, func(x))
    plt.ylim(
        -10, 30
    )

    w = np.sum(coeff * y.reshape(-1, 1) * X, axis=0)
    cond = (coeff > 1e-4).reshape(-1)
    b = y[cond] - np.dot(X[cond], w)
    b1 = y - np.dot(X, w)
    bias = b[0]
    k = -w[0] / w[1]
    b = -bias / w[1]
    sep_plane = lambda x: x * k + b
    plt.subplot(313)
    plt.title('Predicted by my_svm')
    plt.plot(x, sep_plane(x))
    X_pred_1, X_pred_2 = sep_by_function(X, sep_plane)
    plt.scatter(X_pred_1[:, 0], X_pred_1[:, 1], c='red', marker=">")
    plt.scatter(X_pred_2[:, 0], X_pred_2[:, 1], c='green', marker=(5, 2))
    plt.ylim(
        -10, 30
    )

    plt.tight_layout()


def main():
    data_len = 100
    X, y = create_data(data_len)
    clf = svm.SVC()
    clf.fit(X, y)
    a_x = clf.predict(X)
    lambdas = find_lambda(X, y)
    data_plot(X, y, a_x, lambdas)
    plt.show()


if __name__ == '__main__':
    main()
