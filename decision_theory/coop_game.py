from math import factorial

V = {
    frozenset([]): 0,
    frozenset([1]): 3,
    frozenset([2]): 1,
    frozenset([3]): 2,
    frozenset([4]): 4,
    frozenset([1, 2]): 4,
    frozenset([1, 3]): 5,
    frozenset([1, 4]): 8,
    frozenset([2, 3]): 3,
    frozenset([2, 4]): 5,
    frozenset([3, 4]): 6,
    frozenset([1, 2, 3]): 7,
    frozenset([1, 2, 4]): 10,
    frozenset([1, 3, 4]): 11,
    frozenset([2, 3, 4]): 8,
    frozenset([1, 2, 3, 4]): 13
}

def x(i:int):
    N = 4
    res = 0
    for coalition, profit in V.items():
        pow_S = len(coalition)
        if pow_S==0:
            continue
        res += factorial(pow_S - 1) * factorial(N - pow_S) * (profit - V[coalition.difference([i])])

    return res/factorial(N)

def is_convex(V):
    for coalition1, profit1 in V.items():
        for coalition2, profit2 in V.items():
            union_ij = frozenset(coalition1.union(coalition2))
            intersection_ij = frozenset(coalition1.intersection(coalition2))
            if (V[union_ij] + V[intersection_ij]) < (profit1 + profit2):
                return False
    return True

def is_superadditive(V):
    for coalition1, profit1 in V.items():
        for coalition2, profit2 in V.items():
            intersection_ij = frozenset(coalition1.intersection(coalition2))
            if intersection_ij == frozenset([]):
                union_ij = frozenset(coalition1.union(coalition2))
                if (profit1 + profit2) > V[union_ij]:
                    # print(coalition1, coalition2)
                    # print(profit1 + profit2)
                    # print(V[union_ij])
                    return False
    return True



def main():
    print(f'x(1)={x(1)}')
    print(f'x(2)={x(2)}')
    print(f'x(3)={x(3)}')
    print(f'x(4)={x(4)}')
    print(f'Check convex: {is_convex(V)}')
    print(f'Check super additive: {is_superadditive(V)}')

if __name__ == '__main__':
    main()