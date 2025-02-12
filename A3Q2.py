import math

def perfect_square(x):
    a = int(math.sqrt(x))
    return a*a == x

def is_fibonacci(n):
    return perfect_square(5*n*n+4) or perfect_square(5*n*n-4)

n = int(input("Enter a number: "))
if is_fibonacci(n):
    print(f"{n} is a Fibonacci number.")
else:
    print(f"{n} is not a Fibonacci number.")


