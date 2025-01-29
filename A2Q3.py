#def find_digits(n):
    
    #count = 0
    #temp = n
    #while temp > 0:
        #digit = temp % 10
        #if digit!= 0 and n % digit == 0:
         #   count += 1
        #temp//= 10
    #return count

def count_divisible_digits(N):
    count = 0
    # Convert the number to string to extract each digit
    for digit in str(N):
        int_digit = int(digit)
        # Check if the digit divides N exactly and is not zero
        if int_digit != 0 and N % int_digit == 0:
            count += 1
    return count

# Reading number of test cases
T = int(input())

# Processing each test case
for _ in range(T):
    N = int(input())
    result = count_divisible_digits(N)
    print(result)


