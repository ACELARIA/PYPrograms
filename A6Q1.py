class Password_manager:
    def __init__(self):
        self.old_passwords = []

    def get_password(self):
        return self.old_passwords[-1]

    def set_password(self, new_password):
        if new_password not in self.old_passwords:
            self.old_passwords.append(new_password)
            return True
        return False

    def is_correct(self, password):
        return password == self.get_password()

manager = Password_manager()
password = input("Set the initial password : ")
manager.set_password(password) 
print("Share the current Password: ")
print(manager.get_password()) 
password = input("Check if the current password is correct: ")
print(manager.is_correct(password))  
password = input ("Try setting a used password: ")
print(manager.set_password(password))  
password = input ("Try setting a new password:")
print(manager.set_password(password)) 
print("Share the latest password:")
print(manager.get_password())  