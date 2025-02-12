def digital_root(n):
    while n >= 10:  
        n = sum(int(digit) for digit in str(n))
    return n

n=int(input("Enter value of n to find digital root: "))
print(digital_root(n)) 
   
