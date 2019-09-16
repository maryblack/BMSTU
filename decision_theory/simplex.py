import pandas as pd
import numpy as np
from copy import deepcopy


class Simplex:
    def __init__(self, A, b, c, opt):
        n = len(c)
        m = len(A)
        self.A = A
        self.b = b
        self.c = c
        self.matrix = self.simplex_matrix(A, b, c)
        self.free = list(range(n))
        self.basis = list(range(n, n + m))
        self.opt = opt

    def simplex_matrix(self, A, b, c):
        c_0 = [el for el in c]
        A_0 = []
        for i in range(len(A)):
            A_0.append([el for el in A[i]])
        # нужно ли делать отрицательной матрицу А?
        col = len(A[0])
        columns = []
        for i in range(col):
            columns.append(f'{i + 1}')
        A_0.append(c_0)
        df = pd.DataFrame(A_0, index=None, columns=columns)
        b.append(0)
        df.insert(0, 'b', b)
        # print(df)
        m = df.to_numpy(dtype=np.float64)
        return m

    def change_basis(self, r, k):
        old_matrix = deepcopy(self.matrix)
        new_k = self.free[k-1]
        new_r = self.basis[r]
        self.basis[r] = new_k
        self.free[k-1] = new_r
        s_rk = float(1 / old_matrix[r][k])
        self.matrix[r][k] = s_rk
        m = len(self.matrix)
        n = len(self.matrix[0])

        for j in range(m):
            if j != r:
                new_val = float(old_matrix[j][k]*s_rk)
                self.matrix[j][k] = -new_val

        for i in range(n):
            if i != k:
                new_val = float(old_matrix[r][i]*s_rk)
                self.matrix[r][i] = new_val

        for j in range(m):
            for i in range(n):
                if j != r and i != k:
                    new_val = float(old_matrix[j][i] - old_matrix[r][i] * old_matrix[j][k]*s_rk)
                    self.matrix[j][i] = new_val




    def accept_solution(self):
        na = self.na_row()
        while na < (len(self.b)+1):
            # print(r)
            k = self.a_solving_col(na)
            if k < len(self.c)+1:
                r = self.a_solving_row(k)
                self.change_basis(r, k)
                na = self.na_row()
                # r = self.a_solving_col(na)
            else:
                na = len(self.b) + 1
        free_str = [f'x{el+1}' for el in self.free]
        basis_str = [f'x{el+1}' for el in self.basis]
        b_0 = column(self.matrix,0)[:-1]
        F = column(self.matrix,0)[-1]
        new_line = '='*30

        answer = f"Опорное решение: {', '.join(free_str)}=0\n" \
            f"{', '.join([basis_str[i]+'='+str(b_0[i]) for i in range(len(basis_str))])}\nF={F}"

        return answer



    def optimal_solution(self):
        no_col = self.no_col()
        while no_col < (len(self.c) + 1):
            r = self.a_solving_row(no_col)
            if r < len(self.b)+1:
                # k = self.a_solving_row(no_col)
                self.change_basis(r, no_col)
                no_col = self.no_col()
            else:
                no_col = len(self.c) + 1



        free_str = [f'x{el+1}' for el in self.free]
        basis_str = [f'x{el+1}' for el in self.basis]
        b_0 = column(self.matrix, 0)[:-1]
        if self.opt == 'min':
            F = column(self.matrix, 0)[-1]
        else:
            F = - column(self.matrix, 0)[-1]
        new_line = '=' * 30
        answer = f"Оптимальное решение: {', '.join(free_str)}=0\n" \
            f"{', '.join([basis_str[i] + '=' + str(b_0[i]) for i in range(len(basis_str))])}\nF={F}\n{new_line}"

        return answer

    def na_row(self):
        b_0 = column(self.matrix,0)[:-1]
        for i in range(len(b_0)):
            if b_0[i] < 0:
                return i
        return len(b_0)+2 # значит нет отрицательных значений в столбце


    def a_solving_col(self, k):
        row = self.matrix[k][1:]
        for i in range(len(row)):
            if row[i] < 0:
                return i + 1
        return len(row) + 2  # значит нет отрицательных значений в строке

    def a_solving_row(self, r):
        b_0 = column(self.matrix, 0)[:-1]
        ind = 0
        min = max(b_0) # ищем положительный минимум
        sol_col = column(self.matrix, r)[:-1]
        # что делать с делением на 0?
        for i in range(len(b_0)):
            if sol_col[i] != 0:
                s = b_0[i]/sol_col[i]
            else:
                s = sol_col[i]
            if s > 0 and s < min:
                min = s
                ind = i
        return ind


    def no_col(self):
        F = self.matrix[-1]
        for i in range(len(F)):
            # if F[i] < 0:
            if F[i] > 0:
                return i
        return len(F)+1 # значит нет отрицательных значений в строке



def column(matrix, j):
    N = len(matrix)
    # print(N)
    col = []
    col.clear()
    for i in range(N):
        col.append(matrix[i][j])

    return col

def simplex_method(A, b, c, opt):
    solution = Simplex(A, b, c, opt)
    print(solution.matrix)
    print(solution.accept_solution())
    print(solution.optimal_solution())



def main():

    c2 = [7, 5, 3]
    A2 = [[4, 1, 1],
        [1, 2, 0],
        [0, 0.5, 1]
        ]
    b2 = [4, 3, 2]

    c1 = [1, -1] # пример из лекции
    A1 = [[1, -2],
         [-2, 1],
         [1, 1]
         ]
    b1 = [2, -2, 5]

    simplex_method(A1, b1, c1, 'min')
    simplex_method(A2, b2, c2, 'max')

    # matrix, F = simplex_init(c, A, b, opt[1])
    # print_matrix(matrix, F, b)


if __name__ == '__main__':
    main()
