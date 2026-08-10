def brute_force_caesar(ciphertext):
    for shift in range(1, 26):
        decrypted = ""

        for char in ciphertext:
            if char.isalpha():
                if char.isupper():
                    decrypted += chr((ord(char) - ord('A') - shift) % 26 + ord('A'))
                else:
                    decrypted += chr((ord(char) - ord('a') - shift) % 26 + ord('a'))
            else:
                decrypted += char

        print("Shift", shift, ":", decrypted)


ciphertext = input("Enter ciphertext: ")
brute_force_caesar(ciphertext)