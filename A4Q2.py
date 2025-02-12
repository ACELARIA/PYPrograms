import math

def count_square_integers(A, B):
    l = math.ceil(math.sqrt(A))
    r = math.floor(math.sqrt(B))
    
    return max(0, r - l + 1)

print("Enter numbers:")
T = int(input())

for _ in range(T):
    A, B = map(int, input().split()) 
    print(count_square_integers(A, B))
