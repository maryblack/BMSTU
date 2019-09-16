import pandas as pd



class Simplex:
    def __init__(self, A, b, c):
        n = len(c)
        m = len(A)
        self.A = A
        self.b = b
        self.c = c
        self.matrix = self.simplex_matrix(A, b, c)
        self.free = list(range(n))
        self.basis = list(range(n, n + m))

    def simplex_matrix(self, A, b, c):
        c_0 = [-el for el in c]
        A_0 = []
        for i in range(len(A)):
            A_0.append([-el for el in A[i]])
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
        m = df.to_numpy()
        return m

    def change_basis(self, r, k):
        new_k = self.free[r]
        new_r = self.basis[k]
        self.basis[k] = new_k
        self.free[r] = new_r
        s_rk = 1 / self.matrix[k][r]
        m = len(self.matrix)
        n = len(self.matrix[0])

        for j in range(m):
            if j != k:
                new_val = self.matrix[j][r]/s_rk
                self.matrix[j][r] = new_val

        for i in range(n):
            if i != r:
                new_val = self.matrix[k][i]/s_rk
                self.matrix[k][i] = new_val

        for j in range(m):
            for i in range(n):
                if j != k and i != r:
                    new_val = self.matrix[j][i] - self.matrix[k][i] * self.matrix[j][r]/ s_rk
                    self.matrix[j][i] = new_val




    def accept_solution(self):
        na = self.na_row()
        while na < (len(self.b)+1):
            r = self.a_solving_col(na)
            print(r)
            if r < len(self.c)+1:
                k = self.a_solving_row(na, r)
                self.change_basis(r, k)
                na = self.na_row()
            else:
                na = len(self.b) + 1
        free_str = [f'x{el}' for el in self.free]
        basis_str = [f'x{el}' for el in self.basis]
        b_0 = column(self.matrix,0)[:-1]
        F = column(self.matrix,0)[-1]

        answer = f"Опорное решение: {', '.join(free_str)}=0\n" \
            f"{', '.join([basis_str[i]+'='+str(b_0[i]) for i in range(len(basis_str))])}\nF={F}"

        return answer



    def optimal_solution(self):
        pass

    def na_row(self):
        for i in range(len(self.b)):
            if self.b[i] < 0:
                return i
        return len(self.b)+1 # значит нет отрицательных значений в столбце

    def a_solving_col(self, k):
        row = self.matrix[k][1:]
        for i in range(len(row)):
            if row[i] < 0:
                return i + 1
        return len(row) + 1  # значит нет отрицательных значений в строке

    def a_solving_row(self, na, r):
        b_0 = self.b[:-1]
        ind = 0
        min = max(b_0) # ищем положительный минимум
        sol_col = column(self.matrix, r)
        for i in range(len(b_0)):
            s = b_0[i]/sol_col[i]
            if s > 0 and s < min:
                min = s
                ind = i
        return ind


    def o_solving_col(self):
        for i in range(len(self.c)):
            if self.c[i] < 0:
                return i
        return len(self.c)+1 # значит нет отрицательных значений в строке







def column(matrix, j):
    N = len(matrix)
    # print(N)
    col = []
    col.clear()
    for i in range(N):
        col.append(matrix[i][j])

    return col


def main():
    c = [7, 5, 3]
    A = [[4, 1, 1],
        [1, 2, 0],
        [0, 0.5, 1]
        ]
    b = [4, 3, 2]
    solution = Simplex(A, b, c)
    print(solution.matrix)
    # print(f'free: {solution.free}')
    # print(f'basis: {solution.basis}')
    print(solution.accept_solution())

    # matrix, F = simplex_init(c, A, b, opt[1])
    # print_matrix(matrix, F, b)


if __name__ == '__main__':
    main()
