import math

def is_perfect_square(x):
    s = int(math.sqrt(x))
    return s * s == x

def is_fibonacci(n):
    return is_perfect_square(5 * n * n + 4) or is_perfect_square(5 * n * n - 4)
# A number is Fibonacci if and only if one or both of (5*n*n + 4) or (5*n*n - 4) is a perfect square

n = int(input("Enter a number: "))
if is_fibonacci(n):
    print(f"{n} is a Fibonacci number.")
else:
    print(f"{n} is not a Fibonacci number.")


