# Q1. ONE TIME PAD

def otp_encrypt(plaintext, key):
    ciphertext = ""

    for p, k in zip(plaintext.upper(), key.upper()):
        # Convert letters to numbers and add modulo 26
        c = (ord(p) - ord('A') + ord(k) - ord('A')) % 26
        ciphertext += chr(c + ord('A'))

    return ciphertext


def otp_decrypt(ciphertext, key):
    plaintext = ""

    for c, k in zip(ciphertext.upper(), key.upper()):
        # Subtract key value modulo 26
        p = (ord(c) - ord('A') - (ord(k) - ord('A'))) % 26
        plaintext += chr(p + ord('A'))

    return plaintext


# Input
plaintext = input("Enter plaintext: ").replace(" ", "")
key = input("Enter key: ").replace(" ", "")

# Check key length
if len(plaintext) != len(key):
    print("Error: Key must be the same length as plaintext.")
else:
    ciphertext = otp_encrypt(plaintext, key)
    decrypted = otp_decrypt(ciphertext, key)

    print("Ciphertext :", ciphertext)
    print("Decrypted  :", decrypted)