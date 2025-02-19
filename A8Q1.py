class BankAccount:
    def __init__(self, account_number, balance=0):
        self.account_number = account_number  
        self.balance = balance  
    def deposit(self, amount):
        if amount > 0:
            self.balance += amount
            print(f"Deposited {amount}. New balance is {self.balance}.")
        else:
            print("Deposit amount must be positive.")

    def withdraw(self, amount):
        if amount <= 0:
            print("Withdrawal amount must be positive.")
        elif amount > self.balance:
            print("Insufficient funds. Cannot withdraw.")
        else:
            self.balance -= amount
            print(f"Withdrew {amount}. New balance is {self.balance}.")

    def display(self):
        print(f"Account Number: {self.account_number}")
        print(f"Current Balance: {self.balance}")

if __name__ == "__main__":
    account = BankAccount(account_number=123456, balance=1000)
    
    account.display()
    account.deposit(500)
    account.withdraw(300)
    account.withdraw(1500)
    account.display()
