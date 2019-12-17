import random
import numpy as np
import pandas as pd


class StohMatrix:
    def __init__(self, N: int):
        self.size = N
        self.A = np.zeros((N, N))
        self.generate()
        self.init_opinions = np.random.randint(19, size=N) + 1
        self.infl_opinions = np.random.randint(19, size=N) + 1

    def generate(self):
        N = self.size
        iter = 0
        for i in range(N - 1):
            for j in range(N - 1):
                self.A[i, j] = round(random.random() / 5, 3)

        for i in range(N - 1):
            self.A[i, N - 1] = round(1 - np.sum(self.A[i, :N - 1]), 3)

        for j in range(N):
            self.A[N - 1, j] = round(1 - np.sum(self.A[:N - 1, j]), 3)
        while np.sign(np.min(self.A)) < 0:
            for i in range(N - 1):
                for j in range(N - 1):
                    self.A[i, j] = round(random.random() / (N / 2), 3)

            for i in range(N - 1):
                self.A[i, N - 1] = round(1 - np.sum(self.A[i, :N - 1]), 3)

            for j in range(N):
                self.A[N - 1, j] = round(1 - np.sum(self.A[:N - 1, j]), 3)
            iter += 1
        print(iter)


    def convergence(self, opinions, eps=1e-6):
        iter = 0
        N = self.size
        curr = np.dot(self.A, opinions)
        delta = np.abs(curr - opinions)
        while len(np.where(delta < eps)[0]) != N:
            pred = curr.copy()
            curr = np.dot(self.A, pred)
            delta = np.abs(curr - pred)
            iter+=1

        print(f'Изначальные мнения агентов:{opinions}')
        print(f'Потребовалось итераций:{iter}')
        print(f'Результирующее мнение агентов:{np.round(curr,3)}')

    def influence(self):
        player1 = []
        player2 = []
        N = self.size
        K = random.randint(3, N)
        P = np.arange(0, N)
        agents = random.sample(list(P), K)
        # print(agents)
        for a in agents:
            p = random.randint(1,2)
            if p==1:
                player1.append(a)
            else:
                player2.append(a)
        print(f'Агенты первого игрока:{player1}')
        print(f'Агенты второго игрока:{player2}')
        o1 = random.randint(0, 100)
        o2 = random.randint(-100, 0)
        print(f'Сформированное начальное мнение агентов первого игрока: {o1}')
        print(f'Сформированное начальное мнение агентов второго игрока: {o2}')
        for a in player1:
            self.infl_opinions[a] = o1

        for a in player2:
            self.infl_opinions[a] = o2



def main():
    M1 = StohMatrix(10)
    df = pd.DataFrame(M1.A)
    print(df)
    print('\n')
    print(f'Без влияния:')
    M1.convergence(M1.init_opinions)
    print(f'\nC учетом влияния:')
    M1.influence()
    M1.convergence(M1.infl_opinions)



if __name__ == '__main__':
    main()
