from typing import Optional

import pandas as pd
import numpy as np
from copy import deepcopy
from itertools import permutations


class NoAcceptedSolution(Exception):
    def __init__(self):
        super().__init__("Нет опорного решения!")


class NoOptimalSolution(Exception):
    def __init__(self):
        super().__init__("Решение не ограничено!")


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
    def __init__(self, A, b, c, opt, dual: bool = None):
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

    def accept_solution(self) -> list:
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
            # if self.d:
            #     var = 'y'
            # else:
            #     var = 'x'
            # free_str = [f'{var}{el + 1}' for el in self.free]
            # basis_str = [f'{var}{el + 1}' for el in self.basis]
            b_0 = column(self.matrix, 0)[:-1]
            # if self.opt == 'min':
            #     F = column(self.matrix, 0)[-1]
            # else:
            #     F = - column(self.matrix, 0)[-1]
            #
            # answer = f"Опорное решение: {', '.join(free_str)}=0\n" \
            #     f"{', '.join([basis_str[i] + '=' + str(b_0[i]) for i in range(len(basis_str))])}\nF={F}"

            return b_0
        else:
            raise NoAcceptedSolution()

    def optimal_solution(self) -> (list, float):
        accepted_solution = self.accept_solution()
        new_line = '=' * 30
        print("Опорное решение:")
        print(self.print_answer())
        print("Оптимальное решение:")
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
        b_0 = column(self.matrix, 0)[:-1]
        check = 0
        for i in range(len(b_0)):
            if b_0[i] < 0:
                check += 1

        if (check != 0) or ((ind_break == -1 or r == -1) and (no_init != len(self.c) + 1)):
            raise NoOptimalSolution()
        else:
            print(self.print_answer())
            return b_0, self.get_f()

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
            x_range = list(np.linspace(0, max_vect, int(max_vect) + 1))
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
        result = [np.dot(c, np.array(el)) for el in filtered]
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

    def print_answer(self):
        new_line = '=' * 30
        if self.d:
            var = 'y'
        else:
            var = 'x'
        free_str = [f'{var}{el + 1}' for el in self.free]
        basis_str = [f'{var}{el + 1}' for el in self.basis]
        b_0 = column(self.matrix, 0)[:-1]
        F = self.get_f()
        solution = []
        for i in range(len(basis_str)):
            sol = f'{basis_str[i]}={b_0[i]}'
            solution.append(sol)
        answer = f"{', '.join(free_str)}=0\n" \
            f"{', '.join(solution)}\nF={F}"
        return answer

    def get_f(self) -> float:
        if self.opt == 'min':
            return column(self.matrix, 0)[-1]
        else:
            return - column(self.matrix, 0)[-1]


class DualSimplex(Simplex):
    def __init__(self, A, b, c, opt):
        A_dual, b_dual, c_dual, opt_dual = DualSimplex.duality_simplex_method(A, b, c, opt)
        super().__init__(A_dual, b_dual, c_dual, opt_dual, dual=True)

    @staticmethod
    def duality_simplex_method(A, b, c, opt) -> (np.ndarray, np.ndarray, np.ndarray, str):
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


def column(matrix, j):
    N = len(matrix)
    col = []
    col.clear()
    for i in range(N):
        col.append(matrix[i][j])

    return col


def bb_method(A, b, c, opt):
    new_line = '=' * 30
    print("Решение прямой задачи методом ветвей и границ:")
    answer = bb(A, b, c, opt)
    if answer == 0:
        pass
    print(new_line)


def integer_answer(b):
    for i in range(len(b)):
        if b[i] % 1 != 0:
            return False, i, b[i]
    return True, 0, b[-1]


def bb(A, b, c, opt):
    solution = Simplex(A, b, c, opt)
    answer, _ = solution.optimal_solution()
    if answer == -1:
        print(f'Неограниченное решение!')
        return -1
    elif answer == -2:
        print(f'Допустимых решений нет!')
        return -1
    cond, ind, value = integer_answer(answer)
    if cond:
        print(f'Целочисленное решение найдено:\n{solution.print_answer()}')
        return 0
    else:
        val_r = round(value)
        a_new = np.zeros(len(A[0]))
        if val_r > value:
            val_gr = val_r
            val_ls = val_r - 1
        else:
            val_gr = val_r + 1
            val_ls = val_r
        b_1 = deepcopy(b)
        b_1.append(-val_ls)
        a_new[ind] = -1
        A_1 = deepcopy(A)
        A_1.append(list(a_new))
        bb(A_1, b_1, c, opt)
        # b_2 = deepcopy(b)
        # b_2.append(val_gr)
        # a_new[ind] = 1
        # A_2 = deepcopy(A)
        # A_2.append(list(a_new))
        # bb(A_2, b_2, c, opt)


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


def main():
    c = [-1, 1]
    A = [[1, -2],
         [-2, 1],
         [1, 1]]
    b = [2, -2, 5]
    _, f_primal = Simplex(A, b, c, 'min').optimal_solution()
    _, f_dual = DualSimplex(A, b, c, 'min').optimal_solution()
    assert -3.0 == f_primal == f_dual
    print('=' * 30)

    c = [7, 5, 3]  # 10 вариант
    A = [[4, 1, 1],
         [1, 2, 0],
         [0, 0.5, 1]
         ]
    b = [4, 3, 2]
    _, f_primal = Simplex(A, b, c, 'max').optimal_solution()
    _, f_dual = DualSimplex(A, b, c, 'max').optimal_solution()
    assert 13.0 == f_primal == f_dual

    c = [5, 3, 8]  # 16 вариант
    A = [[2, 1, 1],
         [1, 1, 0],
         [0, 0.5, 2]
         ]
    b = [3, 6, 3]
    _, f_primal = Simplex(A, b, c, 'max').optimal_solution()
    _, f_dual = DualSimplex(A, b, c, 'max').optimal_solution()
    assert 15.75 == f_primal == f_dual

    c = [2, 4]
    A = [[1, 2],
         [1, 1]
         ]
    b = [5, 4]
    _, f_primal = Simplex(A, b, c, 'max').optimal_solution()
    _, f_dual = Simplex(A, b, c, 'max').optimal_solution()
    assert 10.0 == f_primal

    c = [1, 2]  # неограниченное решение
    A = [[1, -1],
         [1, 0]
         ]
    b = [10, 20]
    try:
        Simplex(A, b, c, 'max').optimal_solution()
    except NoOptimalSolution as e:
        print(e)
        assert True
    else:
        assert False
    try:
        DualSimplex(A, b, c, 'max').optimal_solution()
    except NoAcceptedSolution as e:
        print(e)
        assert True
    else:
        assert False

    c = [3, 2]  # нет допустимых решений
    A = [[2, 1],
         [-3, -4]
         ]
    b = [2, -12]
    try:
        Simplex(A, b, c, 'max').optimal_solution()
    except NoAcceptedSolution as e:
        print(e)
        assert True
    else:
        assert False
    try:
        DualSimplex(A, b, c, 'max').optimal_solution()
    except NoOptimalSolution as e:
        print(e)
        assert True
    else:
        assert False

    c = [1, 1]  # неограниченное решение
    A = [[-2, -2],
         [-1, 1],
         [1, -1],
         ]
    b = [-1, 1, 1]
    try:
        Simplex(A, b, c, 'max').optimal_solution()
    except NoOptimalSolution as e:
        print(e)
        assert True
    else:
        assert False


    try:
        DualSimplex(A, b, c, 'max').optimal_solution()
    except NoAcceptedSolution as e:
        print(e)
        assert True
    else:
        assert False


    c = [1, -2]  # неограниченное решение
    A = [[-1, -1],
         [1, -2]
         ]
    b = [-1, 4]
    try:
        Simplex(A, b, c, 'min').optimal_solution()
    except NoOptimalSolution as e:
        print(e)
        assert True
    else:
        assert False


    try:
        DualSimplex(A, b, c, 'min').optimal_solution()
    except NoAcceptedSolution as e:
        print(e)
        assert True
    else:
        assert False



    c = [-4, -18, -30, -5]
    A = [
        [3, 1, -4, -1],
        [-2, -4, -1, 1]
    ]
    b = [-3, -3]
    _, actual_f_primal = Simplex(A, b, c, 'max').optimal_solution()
    _, actual_f_dual = DualSimplex(A, b, c, 'max').optimal_solution()
    comparison_delta = 0.01
    expected_f = -36
    assert abs(expected_f - actual_f_primal) < comparison_delta
    assert abs(expected_f - actual_f_dual) < comparison_delta

    c = [2, 3, 0, 0, 0]
    A = [
        [4, 8, 1, 0, 0],
        [2, 1, 0, 1, 0],
        [3, 2, 0, 0, 1]
    ]
    b = [12, 3, 4]
    _, f_primal = Simplex(A, b, c, 'max').optimal_solution()
    _, f_dual = DualSimplex(A, b, c, 'max').optimal_solution()
    assert 4.75 == f_primal == f_dual

    c = [12, -1]
    A = [
        [6, -1],
        [2, 5]
    ]
    b = [12, 20]
    _, f_primal = Simplex(A, b, c, 'max').optimal_solution()
    _, f_dual = DualSimplex(A, b, c, 'max').optimal_solution()
    assert 27 == f_primal == f_dual

    c = [12, -1]
    A = [
        [6, -1],
        [2, 5],
        [1, 0]
    ]
    b = [12, 20, 2]
    _, f_primal = Simplex(A, b, c, 'max').optimal_solution()
    _, f_dual = DualSimplex(A, b, c, 'max').optimal_solution()
    assert 24 == f_primal == f_dual

    # primal_n_dual(A1, b1, c1, 'min')
    # bb_method(A1, b1, c1, 'min')
    # primal_n_dual(A, b, c, 'max')
    # integer(A, b, c, 'max')
    #
    # primal_n_dual(A2, b2, c2, 'max')
    # integer(A2, b2, c2, 'max')
    # bb_method(A2, b2, c2, 'max')
    #
    # primal_n_dual(A11, b11, c11, 'max')
    # bb_method(A11, b11, c11, 'max')
    # primal_n_dual(A12, b12, c12, 'max')
    # bb_method(A12, b12, c12, 'max')
    # bb_method(A11, b11, c11, 'max')
    # integer(A11, b11, c11, 'max')


if __name__ == '__main__':
    main()
