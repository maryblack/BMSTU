import random
import numpy as np
import math
import matplotlib.pyplot as plt


class Signal:
    def __init__(self, x_min, x_max, K, alpha, func):
        self.x_min = x_min
        self.x_max = x_max
        self.K = K
        self.alpha = alpha
        self.func = func
        self.signal = []
        self.x = [x_min + k * (x_max - x_min) / K for k in range(K + 1)]

    def clear(self):
        f_clear = [self.func(x) for x in self.x]
        return f_clear

    def add_noise(self):
        self.signal.clear()
        f_clear = self.clear()
        for i in range(self.K + 1):
            sigma = random.uniform(-self.alpha, self.alpha)
            self.signal.append(f_clear[i] + sigma)

    def flattened(self, window_size, alpha_i):
        M = window_size // 2
        alpha_i_1 = (1 - alpha_i) / 2
        f_flat = [self.signal[i] for i in range(0, M)]
        for i in range(M, self.K - M + 1):
            weight_sum = alpha_i_1 * (self.signal[i - 1] + self.signal[i + 1]) + alpha_i * self.signal[i]
            f_flat.append(weight_sum)

        f_flat.extend(self.signal[self.K - M + 1:])

        return f_flat

    def plot_result(self):
        plt.subplot(311)
        plt.title('Clear + noised signal')
        plt.plot(self.x, self.clear(), label='clear signal')
        self.add_noise()
        plt.plot(self.x, self.signal, label='signal with noise')
        plt.legend()

        plt.subplot(312)
        plt.title('Noised + flattened signal')
        plt.plot(self.x, self.signal, label='signal with noise')
        plt.plot(self.x, self.flattened(5, 0.5), label='flattened signal w = 5')
        plt.legend()

        plt.subplot(313)
        plt.title('Clear + flattened signal')
        plt.plot(self.x, self.clear(), label='clear signal')
        plt.plot(self.x, self.flattened(5, 0.5), label='flattened signal w = 5')
        plt.legend()

        plt.tight_layout()
        plt.show()




def main():
    func = lambda x: math.sin(x) + 0.5

    signal1 = Signal(
        x_min=0,
        x_max=math.pi,
        K=100,
        alpha=0.25,
        func=func
    )

    signal1.plot_result()


if __name__ == '__main__':
    main()
