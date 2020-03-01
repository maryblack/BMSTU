def finder(arr: list, len_arr: int, x: int) -> int:
    left = 0
    right = len_arr - 1
    search = -1
    while left <= right:                        #condLoop
        print('A')
        mid_ind = (left + right) // 2           #statementA
        if arr[mid_ind] == x:                   #condEq
            print('B')
            search, left = mid_ind, right + 1   #statementB
        elif arr[mid_ind] < x:                  #condLess
            print('C')
            left = mid_ind + 1                  #statementC
        elif arr[mid_ind] > x:                  #condGreater
            print('D')
            right = mid_ind - 1                 #statementD

    return search


if __name__ == '__main__':
    print("\nModel 0\n")
    print(finder([1], 1, 1))

    print("\nModel 1\n")
    print(finder([1], 1, 2))

    print("\nModel 2\n")
    print(finder([1, 2, 3], 3, 3))

    print("\nModel 3\n")
    print(finder([1, 2, 13, 24, 75, 123], 6, 24))
