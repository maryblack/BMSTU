import pandas as pd
import numpy as np
from copy import deepcopy
from itertools import permutations

class Solutions:
    def __init__(self, child: list, parents: list, iter: int):
        self.child = child
        self.parents = parents
        self.iter = iter

    @property
    def variant(self):
        var = []
        var.clear()
        curr_variants = self.child
        pred_variants = self.parents
        for i in range(len(curr_variants)):
            for j in range(len(pred_variants)):
                new = [curr_variants[i]]
                toadd = pred_variants[j]
                # print(type(toadd))
                if self.iter == 1:
                    new.extend([toadd])
                else:
                    new.extend(toadd)
                var.append(new)

        return var


class Simplex:
    def __init__(self, A, b, c, opt, dual=None):
        n = len(c)
        m = len(A)
        self.A = A
        self.b = b
        self.c = c
        self.matrix = self.simplex_matrix(A, b, c, opt)
        self.free = list(range(n))
        self.basis = list(range(n, n + m))
        self.opt = opt
        self.d = dual

    def simplex_matrix(self, A, b, c, opt):
        if opt == 'min':
            c_0 = [-el for el in c]
        else:
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
        b_0 = [el for el in b]
        b_0.append(0)
        df.insert(0, 'b', b_0)
        # print(df)
        m = df.to_numpy(dtype=np.float64)
        return m

    def change_basis(self, r, k, ind):
        old_matrix = deepcopy(self.matrix)
        new_k = self.free[k - 1]
        new_r = self.basis[r]
        self.basis[r] = new_k
        self.free[k - 1] = new_r
        if old_matrix[r][k] == 0:
            return -1
        s_rk = float(1 / old_matrix[r][k])
        self.matrix[r][k] = s_rk
        m = len(self.matrix)
        n = len(self.matrix[0])

        for i in range(m):
            if i != r:
                new_val = float(old_matrix[i][k] * s_rk)
                self.matrix[i][k] = -new_val

        for j in range(n):
            if j != k:
                new_val = float(old_matrix[r][j] * s_rk)
                self.matrix[r][j] = new_val

        for i in range(m):
            for j in range(n):
                if i != r and j != k:
                    new_val = float(old_matrix[i][j] - old_matrix[r][j] * old_matrix[i][k] * s_rk)
                    self.matrix[i][j] = new_val

        return ind

    def accept_solution(self):
        na = self.na_row()
        ind_sol = 0
        while na < (len(self.b) + 1):
            # print(r)
            k = self.a_solving_col(na)
            if k < len(self.c) + 1:
                r = self.a_solving_row(k)
                ind_break = self.change_basis(r, k, ind_sol)
                na = self.na_row()
                # r = self.a_solving_col(na)
                ind_sol += 1
            else:
                ind_sol = -1
                na = len(self.b) + 1

        if ind_sol >= 0:
            # print(f'iterations step 1: {ind_sol}')
            if self.d:
                var = 'y'
            else:
                var = 'x'
            free_str = [f'{var}{el + 1}' for el in self.free]
            basis_str = [f'{var}{el + 1}' for el in self.basis]
            b_0 = column(self.matrix, 0)[:-1]
            if self.opt == 'min':
                F = column(self.matrix, 0)[-1]
            else:
                F = - column(self.matrix, 0)[-1]

            answer = f"Опорное решение: {', '.join(free_str)}=0\n" \
                f"{', '.join([basis_str[i] + '=' + str(b_0[i]) for i in range(len(basis_str))])}\nF={F}"

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
        no_init = no_col
        if no_col == len(self.c) + 1:
            r = -1
        else:
            r = self.a_solving_row(no_col)
        while no_col < (len(self.c) + 1) and ind_break != -1 and r != -1:
            r = self.a_solving_row(no_col)
            if r < len(self.b) + 1:
                # k = self.a_solving_row(no_col)
                ind_break = self.change_basis(r, no_col, ind_sol)
                no_col = self.no_col()
                # r = self.a_solving_row(no_col)
                ind_sol += 1
                # print(self.matrix)
            else:
                no_col = len(self.c) + 1
                ind_sol = -1

        if (ind_break == -1 or r == -1) and (no_init != len(self.c) + 1):
            return f'Неограниченное решение\n{new_line}'
        else:
            # print(f'iterations step 2: {ind_sol}')
            if self.d:
                var = 'y'
            else:
                var = 'x'
            free_str = [f'{var}{el + 1}' for el in self.free]
            basis_str = [f'{var}{el + 1}' for el in self.basis]
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
        b_0 = column(self.matrix, 0)[:-1]
        for i in range(len(b_0)):
            if b_0[i] < 0:
                return i
        return len(b_0) + 2  # значит нет отрицательных значений в столбце

    def a_solving_col(self, k):
        row = self.matrix[k][1:]
        for i in range(len(row)):
            if row[i] < 0:
                return i + 1
        return len(row) + 2  # значит нет отрицательных значений в строке

    def a_solving_row(self, r):
        b_0 = column(self.matrix, 0)[:-1]
        ind = 0
        min = 9999999  # ищем положительный минимум
        sol_col = column(self.matrix, r)[:-1]
        check = 0
        # что делать с делением на 0?
        for i in range(len(b_0)):
            if sol_col[i] != 0:
                s = b_0[i] / sol_col[i]
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
        F = self.matrix[-1][1:]
        for i in range(len(F)):
            # if F[i] < 0:
            if F[i] > 0:
                return i + 1
        return len(F) + 1  # значит нет отрицательных значений в строке

    def brute_force(self):
        max_vect = max(self.b)
        conditions = []
        r = len(self.c)
        for el in range(r):
            x_range = list(np.linspace(0, max_vect, int(max_vect)+1))
            conditions.append(x_range)

        for i in range(r):
            if i == 0:
                combinations = conditions[0]
            else:
                curr_digit = Solutions(conditions[i], combinations, i)
                combinations = curr_digit.variant

        filtered = [el for el in combinations if self.check_integer_solution(el)]
        A = np.array(self.A)
        # for el in filtered:
        #     print('============')
        #     print(self.check_integer_solution(el), el)
        #     print(f'b vect: {self.b}\nres vect:{np.dot(A, np.array(el).T)}')

        c = np.array(self.c)
        result = [np.dot(c, np.array(el))for el in filtered]
        ans = np.argmax(result)
        # print(filtered[ans])

        return filtered[ans], max(result)

    def check_integer_solution(self, solution):
        A = np.array(self.A)
        b = np.array(self.b)
        sol = np.array(solution)
        res = np.dot(A, sol.T)
        # print(f'b vect: {b}\nres vect:{res}')
        for i in range(len(b)):
            if res[i] > b[i]:
                return False

        # print(f'b вектор: {res}')
        return True



    def gomori(self):
        pass


def column(matrix, j):
    N = len(matrix)
    col = []
    col.clear()
    for i in range(N):
        col.append(matrix[i][j])

    return col


def simplex_method(A, b, c, opt, dual = None):
    if dual:
        print("Решение двойственной задачи:")
        solution = Simplex(A, b, c, opt, dual)
    else:
        print("Решение прямой задачи:")
        solution = Simplex(A, b, c, opt)
    print(solution.matrix)
    print(solution.optimal_solution())




def duality_simplex_method(A, b, c, opt):
    A_matrix = np.array(A)
    if opt == 'min':
        opt_dual = 'max'
        c_dual = np.negative(b)
        A_dual = np.negative(A_matrix.T)
        b_dual = c

    else:
        opt_dual = 'min'
        c_dual = b
        A_dual = np.negative(A_matrix.T)
        b_dual = np.negative(c)

    return A_dual, b_dual, c_dual, opt_dual


def primal_n_dual(A, b, c, opt):
    simplex_method(A, b, c, opt)
    A_dual, b_dual, c_dual, opt_dual = duality_simplex_method(A, b, c, opt)
    simplex_method(A_dual, b_dual, c_dual, opt_dual, True)

    # print('Двойственная к двойственной')
    # duality_simplex_method(A_dual, b_dual, c_dual, opt_dual)

# def brute_force(A, b, c, opt):
#     pass
#
# def gomori(A, b, c, opt):
#     pass
#
def integer(A, b, c, opt):
    solution = Simplex(A, b, c, opt)
    print('Целочисленное решение прямой задачи:')
    print(solution.matrix)
    sol, F = solution.brute_force()
    new_line = '=' * 30
    if solution.d:
        var = 'y'
    else:
        var = 'x'
    free_str = [f'{var}{el + 1}' for el in solution.free]
    basis_str = [f'{var}{el + 1}' for el in solution.basis]
    answer = []
    for i in range(len(free_str)):
        s = f'{free_str[i]}={sol[i]}'
        answer.append(s)
    print(f"Ответ: {', '.join(basis_str)}=0\n" \
        f"{', '.join(answer)}\nF={F}\n{new_line}")

    # A_dual, b_dual, c_dual, opt_dual = duality_simplex_method(A, b, c, opt)
    # solution = Simplex(A_dual, b_dual, c_dual, opt_dual)
    # print('Целочисленное решение обратной задачи:')
    # print(solution.matrix)
    # sol, F = solution.brute_force()
    # new_line = '=' * 30
    # if solution.d:
    #     var = 'y'
    # else:
    #     var = 'x'
    # free_str = [f'{var}{el + 1}' for el in solution.free]
    # basis_str = [f'{var}{el + 1}' for el in solution.basis]
    # answer = []
    # for i in range(len(basis_str)):
    #     s = f'{basis_str[i]}={sol[i]}'
    #     answer.append(s)
    # print(f"Оптимальное решение: {', '.join(free_str)}=0\n" \
    #           f"{', '.join(answer)}\nF={F}\n{new_line}")


def main():
    c1 = [-1, 1]
    A1 = [[1, -2],
          [-2, 1],
          [1, 1]]

    b1 = [2, -2, 5]

    c2 = [7, 5, 3]  # 10 вариант
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

    c4 = [2, 4]
    A4 = [[1, 2],
          [1, 1]
          ]
    b4 = [5, 4]

    c5 = [1, 2]  # неограниченное решение
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

    c8 = [1, -2]  # неограниченное решение
    A8 = [[-1, -1],
          [1, -2]
          ]
    b8 = [-1, 4]

    c9 = [-4, -18, -30, -5]
    A9 = [
        [3, 1, -4, -1],
        [-2, -4, -1, 1]
    ]
    b9 = [-3, -3]

    c = [2, 3, 0, 0, 0]
    A = [
        [4, 8, 1, 0, 0],
        [2, 1, 0, 1, 0],
        [3, 2, 0, 0, 1]
    ]
    b = [12, 3, 4]

    c11 = [12, -1]
    A11 = [
        [6, -1],
        [2, 5]
    ]
    b11 = [12, 20]

    # primal_n_dual(A1, b1, c1, 'min')
    primal_n_dual(A, b, c, 'max')
    integer(A, b, c, 'max')

    primal_n_dual(A2, b2, c2, 'max')
    integer(A2, b2, c2, 'max')
    #
    primal_n_dual(A11, b11, c11, 'max')
    integer(A11, b11, c11, 'max')
    # primal_n_dual(A6, b6, c6, 'max')
    # primal_n_dual(A3, b3, c3, 'max')
    # primal_n_dual(A4, b4, c4, 'max')
    # primal_n_dual(A5, b5, c5, 'max')
    # primal_n_dual(A7, b7, c7, 'max')
    # primal_n_dual(A8, b8, c8, 'min')
    # primal_n_dual(A9, b9, c9, 'max')

    # matrix, F = simplex_init(c, A, b, opt[1])
    # print_matrix(matrix, F, b)


if __name__ == '__main__':
    main()
