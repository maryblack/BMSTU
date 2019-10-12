import operator
import random
import numpy as np
import math, cmath
import matplotlib.pyplot as plt


class Signal:
    def __init__(self, x_min, x_max, K, sigma, func, window_size, alpha, lmbd):
        self.x_min = x_min
        self.x_max = x_max
        self.K = K
        self.sigma = sigma
        self.func = func
        self.signal = []
        self.x = [x_min + k * (x_max - x_min) / K for k in range(K + 1)]
        self.M = window_size // 2
        self.alpha = alpha
        self.lmbd = lmbd

    def clear(self):
        f_clear = [self.func(x) for x in self.x]
        return f_clear

    def add_noise(self):
        self.signal.clear()
        f_clear = self.clear()
        for i in range(self.K + 1):
            sigma = random.uniform(-self.sigma, self.sigma)
            self.signal.append(f_clear[i] + sigma)



    def flattened(self):# среднее геометрическое
        alpha_i = np.float64(self.alpha).item()
        alpha_i_1 = (1 - alpha_i) / 2
        f_flat = []
        f_flat.clear()
        f_flat = [self.signal[i] for i in range(0, self.M)]
        for i in range(self.M, self.K - self.M + 1):
            weight_sum = self.signal[i - 1]**(alpha_i_1) * self.signal[i + 1]**(alpha_i_1) * self.signal[i]**(alpha_i)
            f_flat.append(weight_sum)

        f_flat.extend(self.signal[self.K - self.M + 1:])

        return f_flat

    def omega(self,):# евклидова метрика
        flat = self.flattened()
        fsum = 0
        for i in range(1, self.K):
            fsum += (flat[i] - flat[i-1])**2

        return math.sqrt(np.real(fsum))

    def delta(self):# евклидова метрика
        flat = self.flattened()
        fsum = 0
        for i in range(self.K+1):
            fsum += (flat[i] - self.signal[i])**2

        return math.sqrt(np.real(fsum)/self.K)


    def show_result(self):
        plt.subplot(411)
        plt.title('Clear + noised signal', fontdict={'size': 10})
        plt.plot(self.x, self.clear(), label='clear signal',c='blue')
        self.add_noise()
        plt.plot(self.x, self.signal, label='signal with noise', c='red')
        plt.legend()

        plt.subplot(412)
        plt.title('Noised + flattened signal', fontdict={'size': 10})
        plt.plot(self.x, self.signal, label='signal with noise', c='red')
        plt.plot(self.x, self.flattened(), label='flattened signal', c='green')
        plt.legend()

        plt.subplot(413)
        plt.title(f'Clear + flattened signal before optimization\n for J={self.criteria()}, dist={self.dist()}', fontdict={'size': 10})
        plt.plot(self.x, self.clear(), label='clear signal', c='blue')
        plt.plot(self.x, self.flattened(), label='flattened signal', c='green')
        plt.legend()

        self.find_alpha()
        self.find_lambda()
        plt.subplot(414)
        plt.title(f'Clear + flattened signal after optimization\n for J={self.criteria()}, dist={self.dist()}', fontdict={'size': 10})
        plt.plot(self.x, self.clear(), label='clear signal', c='blue')
        plt.plot(self.x, self.flattened(), label='flattened signal', c='green')
        plt.legend()

        plt.tight_layout()
        plt.show()


    def criteria(self):
        return self.lmbd*self.omega() + (1-self.lmbd)*self.delta()

    def find_alpha(self):
        self.add_noise()
        res = {}
        alph = np.linspace(0,1,50)[1:-1]
        for a in alph:
            self.alpha = a
            res[self.criteria()] = a

        min_J = sorted(res.items(), key=operator.itemgetter(0))[0][0]
        self.alpha = res[min_J]
        return min_J, self.alpha

    def find_lambda(self):
        self.add_noise()
        res = {}
        lmbd = [1/l for l in range(1,20)]
        for l in lmbd:
            self.lmbd = l
            res[self.dist()] = l

        min_J = sorted(res.items(), key=operator.itemgetter(0))[0][0]
        self.lmbd = res[min_J]
        return min_J, self.lmbd

    def dist(self):
        return math.sqrt(self.omega()**2 + self.delta()**2)




def main():
    func = lambda x: math.sin(x) + 0.5

    signal1 = Signal(
        x_min=0,
        x_max=2*math.pi,
        K=100,
        sigma=0.25,
        func=func,
        window_size=3,
        alpha=0.5,
        lmbd=0.5
    )

    signal1.show_result()


if __name__ == '__main__':
    main()
