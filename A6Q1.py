class Password_manager:
    old_passwords = []
    
    def get_password(self):
        if self.old_passwords:
            return self.old_passwords[-1]
        return None 
    
    def set_password(self, new_password):
        if new_password not in self.old_passwords:
            self.old_passwords.append(new_password)
        else:
            print("Error: The new password cannot be the same as any past passwords.")
    
    def is_correct(self, password_to_check):
        password_to_check == self.get_password()
           
manager = Password_manager()

manager.set_password("password123")
manager.set_password("password456")

print(manager.get_password()) 

print(manager.is_correct("password123"))  
print(manager.is_correct("password456"))   

manager.set_password("password")  
