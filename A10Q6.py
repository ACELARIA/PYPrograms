import numpy as np
import matplotlib.pyplot as plt

def f(x):
    return x**3 - 6*x**2 + 11*x - 6

def bisection_method(f, a, b, tol=1e-6, max_iter=100):
    iterations = []
    if f(a) * f(b) > 0:
        print("Function has the same sign at both ends of the interval.")
        return None
    
    for i in range(max_iter):
        m = (a + b) / 2
        iterations.append(m)
        
        if abs(f(m)) < tol:
            print(f"Root found at {m} with f(m) = {f(m)}")
            break
        
        if f(a) * f(m) < 0:
            b = m
        else:
            a = m
    
    return np.array(iterations)

a = 0.0
b = 3.0

updates = bisection_method(f, a, b)

x = np.linspace(a - 1, b + 1, 400)
y = f(x)

plt.plot(x, y, label='f(x)', color='blue')
plt.axhline(0, color='black',linewidth=1)
plt.axvline(0, color='black',linewidth=1)

if updates is not None:
    plt.plot(updates, f(updates), 'ro-', label='Bisection Method Updates')

plt.xlabel('x')
plt.ylabel('f(x)')
plt.title('Bisection Method for Root Finding')
plt.legend()
plt.grid(True)
plt.show()
