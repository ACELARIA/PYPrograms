import numpy as np

def f(x):
    return x**3 - 7*x - 5

def bisection_method(f, a, b, tol=1e-6, max_iter=100):
    if f(a) * f(b) > 0:
        print("The function must have opposite signs at the endpoints a and b.")
        return None
    
    updates = []  
    iter_count = 0

    while (b - a) / 2.0 > tol and iter_count < max_iter:
        c = (a + b) / 2.0
        updates.append(c)

        print(f"Iteration {iter_count + 1}:")
        print(f"  Interval: [{a}, {b}]")
        print(f"  Midpoint: {c}, f(c) = {f(c)}")

        if f(c) == 0:  
            print(f"Exact root found: {c}")
            break
        elif f(a) * f(c) < 0:
            b = c
        else:
            a = c
        
        iter_count += 1

    return np.array(updates)

a = 2
b = 3

updates = bisection_method(f, a, b)

print("\nAll Midpoints (Updates) during the Bisection Method:")
print(updates)
