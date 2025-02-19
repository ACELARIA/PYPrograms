def max_pieces(K):
    return K + 1

T = int(input("Input number: "))

for _ in range(T):
    K = int(input("Input number of cuts: "))
    print(max_pieces(K))
