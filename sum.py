def add(num1,num2):
    print ("inside function")
    print (num1,num2)
    sum = num1+num2
    print(sum)
    return sum


num1 = int(input("Enter the first number: "))
num2 = int(input("Enter the second number: "))

sum = add(num1,num2)

print("The sum is", sum);
