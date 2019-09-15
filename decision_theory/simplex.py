import pandas as pd

def simplex_init(F, cond, b, opt: str):
    F_0 = [-el for el in F]
    m = len(cond)
    n = len(cond[0])
    cond1 = []
    for i in range(m):
        cond1.append([-el for el in cond[i]])
    for i in range(m):
        F_0.append(0)
        for j in range(m):
            if i == j:
                cond1[i].append(1)
            else:
                cond1[i].append(0)

    # return F_0, m, n
    return cond1, F_0


def column(matrix, j):
    N = len(matrix)
    # print(N)
    col = []
    col.clear()
    for i in range(N):
        col.append(matrix[i][j])

    return col



def basis(matrix)->list:
    base = []
    M = len(matrix[0])
    N = len(matrix)
    for j in range(M):
        col = column(matrix, j)
        if col.count(0) == (N - 1):
            for i in range(len(col)):
                if i == 1:
                    base.append(f'x{j + 1}')


    return base


def print_matrix(matrix, F, b):
    ind = basis(matrix)
    ind.append('F')
    # print(column(matrix,3))
    col = len(matrix[0])
    base = len(matrix)
    columns = []
    for i in range(col):
        columns.append(f'x{i+1}')

    matrix.append(F)
    b.append(0)
    df = pd.DataFrame(matrix, index=ind,columns=columns)
    df.insert(0, 'b', b)
    print(df)



def main():
    opt = ['min', 'max']
    F = [20, 0, 10]
    cond = [[-4, -3, 6],
            [-3, -2, 17],
            [-1, -1, 3]
            ]
    b = [-23, 33, -12]
    matrix, F = simplex_init(F, cond, b, opt[1])
    print_matrix(matrix, F, b)


if __name__ == '__main__':
    main()
