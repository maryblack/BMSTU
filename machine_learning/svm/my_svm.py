import random
import numpy as np
import matplotlib.pyplot as plt
import math
from sklearn import svm

def func(x):
    return x**2 + 2*x - 6
    # return math.sin(x) + 10

def create_data(num: int):
    X = []
    y = []
    for i in range(num):
        value_x = random.randint(0,20)
        max_y = round(func(10))
        max_y = round(func(10))
        value_y = random.randint(0,20)
        coord = [value_x, value_y]
        X.append(coord)
        if func(value_x) > value_y:
            y.append(1)
        else:
            y.append(-1)

    return np.array(X), np.array(y)


def scalar_prod(w1,w2):
    s = 0
    for i in range(0,len(w1)):
        s+= w1[i]*w2[i]
    return s


def r_exp(x_i1, x_i2, y_i,w1,w2):
    return 1 - 1/(1+math.exp(-y_i*(w1*x_i1+w2*x_i2)))

def a_sigma(x, w):
    return 1/(1+math.exp(-w[0]*x[0]-w[1]*x[1]))


def sum1_l(x, y, w1, w2):
    s=0
    x1 = x[:,0]
    x2 = x[:,1]
    l = len(y)
    for i in range(0,l):
        s+=y[i]*x1[i]*r_exp(x1[i],x2[i],y[i],w1,w2)
    return s/l

def sum2_l(x, y, w1, w2):
    s=0
    x1 = x[:,0]
    x2 = x[:,1]
    l = len(y)
    for i in range(0,l):
        s+=y[i]*x2[i]*r_exp(x1[i],x2[i],y[i],w1,w2)
    return s/l


def normal_grad(X,y, w_0:list, k, C):
    w1 = w_0[0]
    w2 = w_0[1]
    w1 = w1 + k*sum1_l(X,y,w1,w2)-k*C*w1
    w2 = w2 + k*sum2_l(X,y,w1,w2)-k*C*w2
    w = [w1,w2]
    wx = [w[i]-w_0[i] for i in range(0,len(w))]
    eps = math.sqrt(scalar_prod(wx,wx))
    return w, eps


def LogRegression(X,y,k,C):
    w_0 = [0,0]#initial weights
    e = 1e-5
    N = 1e+3#iterations
    i = 0
    eps = 10
    while (eps > e) and (i < N):
        w, eps = normal_grad(X,y,w_0,k,C)
        w_0 = w
        i+=1
    # a_x = [a_sigma(x,w) for x in X]
    a_x = [np.sign(scalar_prod(x,w)) for x in X]
    return a_x





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



def data_plot(X, y, ax, data_len):
    x = np.linspace(0, 20, data_len)
    f = [func(el) for el in x]

    plt.ylim(
        f[0], 30
    )


    X_true_1, X_true_2 = data_separation(X, y)
    plt.subplot(311)
    plt.title('Real')
    plt.scatter(X_true_1[:, 0], X_true_1[:, 1], c='red',  marker=">")
    plt.scatter(X_true_2[:, 0], X_true_2[:, 1], c='green',  marker=(5, 2))
    plt.plot(x, f)
    plt.ylim(
        -10, 30
    )

    X_pred_1, X_pred_2 = data_separation(X, ax)
    plt.subplot(312)
    plt.title('Predicted by sklearn')
    plt.scatter(X_pred_1[:, 0], X_pred_1[:, 1], c='red',  marker=">")
    plt.scatter(X_pred_2[:, 0], X_pred_2[:, 1], c='green',  marker=(5, 2))
    plt.plot(x, f)
    plt.ylim(
        -10, 30
    )

    # X_pred_1, X_pred_2 = data_separation(X, ax)
    plt.subplot(313)
    plt.title('Predicted by my_svm')
    # plt.scatter(X_pred_1[:, 0], X_pred_1[:, 1], c='red', marker=">")
    # plt.scatter(X_pred_2[:, 0], X_pred_2[:, 1], c='green', marker=(5, 2))
    plt.plot(x, f)
    plt.ylim(
        -10, 30
    )

    plt.tight_layout()


def main():
    data_len = 36
    X, y = create_data(data_len)
    print(X, y)
    clf = svm.SVC(C=100, gamma='scale')
    clf.fit(X,y)
    a_x = clf.predict(X)
    print(a_x)
    data_plot(X, y, a_x, data_len)
    plt.show()




if __name__ == '__main__':
    main()
