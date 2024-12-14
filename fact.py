def factorial(a):
    if a==1:
        return 1
    factorial_op = a*factorial(a-1)
    return factorial_op
    

a = int(input("enter number:\n"))
fact = factorial(a)
print("factorial is",fact)