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

    def change_basis(self, r, k, ind):
        old_matrix = deepcopy(self.matrix)
        new_k = self.free[k-1]
        new_r = self.basis[r]
        self.basis[r] = new_k
        self.free[k-1] = new_r
        if old_matrix[r][k] == 0:
            return -1
        s_rk = float(1 / old_matrix[r][k])
        self.matrix[r][k] = s_rk
        m = len(self.matrix)
        n = len(self.matrix[0])

        for i in range(m):
            if i != r:
                new_val = float(old_matrix[i][k]*s_rk)
                self.matrix[i][k] = -new_val

        for j in range(n):
            if j != k:
                new_val = float(old_matrix[r][j]*s_rk)
                self.matrix[r][j] = new_val

        for i in range(m):
            for j in range(n):
                if i != r and j != k:
                    new_val = float(old_matrix[i][j] - old_matrix[r][j] * old_matrix[i][k]*s_rk)
                    self.matrix[i][j] = new_val

        return ind




    def accept_solution(self):
        na = self.na_row()
        ind_sol = 0
        ind_break = 0
        while na < (len(self.b)+1):
            # print(r)
            k = self.a_solving_col(na)
            if k < len(self.c)+1:
                r = self.a_solving_row(k)
                ind_break = self.change_basis(r, k, ind_sol)
                na = self.na_row()
                # r = self.a_solving_col(na)
                ind_sol += 1
            else:
                ind_sol = -1
                na = len(self.b) + 1

        if ind_sol >= 0:
            free_str = [f'x{el+1}' for el in self.free]
            basis_str = [f'x{el+1}' for el in self.basis]
            b_0 = column(self.matrix,0)[:-1]
            F = column(self.matrix,0)[-1]

            answer = f"Опорное решение: {', '.join(free_str)}=0\n" \
                f"{', '.join([basis_str[i]+'='+str(b_0[i]) for i in range(len(basis_str))])}\nF={F}"

            return answer
        else:
            return -1



    def optimal_solution(self):
        str = self.accept_solution()
        new_line = '=' * 30
        if str == -1:
            return f'Допустимых решений нет!\n{new_line}'
        print(str)
        no_col = self.no_col()
        iter = 0
        ind_sol = 0
        ind_break = 0
        r = self.a_solving_row(no_col)
        while no_col < (len(self.c) + 1) and ind_break != -1 and r != -1:
            r = self.a_solving_row(no_col)
            if r < len(self.b)+1:
                # k = self.a_solving_row(no_col)
                ind_break = self.change_basis(r, no_col, ind_sol)
                no_col = self.no_col()
                # r = self.a_solving_row(no_col)
                ind_sol += 1
            else:
                no_col = len(self.c) + 1
                ind_sol = -1

        if ind_break == -1 or r == -1:
            return f'Неограниченное решение\n{new_line}'
        else:
            free_str = [f'x{el+1}' for el in self.free]
            basis_str = [f'x{el+1}' for el in self.basis]
            b_0 = column(self.matrix, 0)[:-1]
            if self.opt == 'min':
                F = column(self.matrix, 0)[-1]
            else:
                F = - column(self.matrix, 0)[-1]
            solution = []
            for i in range(len(basis_str)):
                sol = f'{basis_str[i]}={b_0[i]}'
                solution.append(sol)
            answer = f"Оптимальное решение: {', '.join(free_str)}=0\n" \
                f"{', '.join(solution)}\nF={F}\n{new_line}"

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
        check = 0
        # что делать с делением на 0?
        for i in range(len(b_0)):
            if sol_col[i] != 0:
                s = b_0[i]/sol_col[i]
            else:
                s = sol_col[i]
            if s > 0:
                check += 1
            if s > 0 and s <= min:
                min = s
                ind = i
        if check == 0:
            return -1
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
    # print(solution.accept_solution())
    print(solution.optimal_solution())



def main():

    c2 = [7, 5, 3]# 10 вариант
    A2 = [[4, 1, 1],
        [1, 2, 0],
        [0, 0.5, 1]
        ]
    b2 = [4, 3, 2]

    c3 = [5, 3, 8]  # 16 вариант
    A3 = [[2, 1, 1],
          [1, 1, 0],
          [0, 0.5, 2]
          ]
    b3 = [3, 6, 3]

    c1 = [1, -1] # пример из лекции
    A1 = [[1, -2],
         [-2, 1],
         [1, 1]
         ]
    b1 = [2, -2, 5]

    c4 = [2, 4]
    A4 = [[1, 2],
          [1, 1]
          ]
    b4 = [5, 4]

    c5 = [1, 2] # неограниченное решение
    A5 = [[1, -1],
          [1, 0]
          ]
    b5 = [10, 20]

    c6 = [3, 2]  # нет допустимых решений
    A6 = [[2, 1],
          [-3, -4]
          ]
    b6 = [2, -12]

    c7 = [1, 1]  # неограниченное решение
    A7 = [[-2, -2],
          [-1, 1],
          [1, -1],
          ]
    b7 = [-1, 1, 1]

    simplex_method(A1, b1, c1, 'min')
    simplex_method(A2, b2, c2, 'max')
    simplex_method(A6, b6, c6, 'max')
    simplex_method(A3, b3, c3, 'max')
    simplex_method(A4, b4, c4, 'max')
    simplex_method(A5, b5, c5, 'max')
    simplex_method(A7, b7, c7, 'max')


    # matrix, F = simplex_init(c, A, b, opt[1])
    # print_matrix(matrix, F, b)


if __name__ == '__main__':
    main()
