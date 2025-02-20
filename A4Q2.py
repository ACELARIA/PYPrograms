import math

def count_square_integers(a,b):
    l = math.ceil(math.sqrt(a))
    r = math.floor(math.sqrt(b))
    
    return max(0,r-l+1)

T = int(input("Enter number of test cases and numbers:"))

for _ in range(T):
    a,b = map(int, input().split()) 
    print(count_square_integers(a,b))
