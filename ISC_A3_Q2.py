import string

key = "QWERTYUIOPASDFGHJKLZXCVBNM"
alphabet = string.ascii_uppercase


def encrypt(text):
    result = ""

    for char in text:
        if char.isalpha():
            if char.isupper():
                result += key[alphabet.index(char)]
            else:
                result += key[alphabet.index(char.upper())].lower()
        else:
            result += char

    return result


def decrypt(text):
    result = ""

    for char in text:
        if char.isalpha():
            if char.isupper():
                result += alphabet[key.index(char)]
            else:
                result += alphabet[key.index(char.upper())].lower()
        else:
            result += char

    return result


text = input("Enter patient data: ")

encrypted = encrypt(text)
decrypted = decrypt(encrypted)

print("Original  :", text)
print("Encrypted :", encrypted)
print("Decrypted :", decrypted)