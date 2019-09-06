def simplex(F, cond, b, opt: str):
    if opt == 'max':
        F_0 = [-el for el in F]
    else:
        F_0 = F
    m = len(cond)
    n = len(cond[0])
    for i in range(m):
        for j in range(m):
            if i == j:
                cond[i].append(1)
            else:
                cond[i].append(0)

    # return F_0, m, n
    return cond

def basis(cond):
    pass


def main():
    opt = ['min', 'max']
    F = [20, 0, 10]
    cond = [[-4, -3],
            [-3, -2],
            [-1, -1]
            ]
    b = [-23, -33, -12]
    print(simplex(F, cond, b, opt[1]))

if __name__ == '__main__':
    main()
