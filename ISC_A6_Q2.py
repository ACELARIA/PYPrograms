# Q2. AFFINE CIPHER

from math import gcd


def mod_inverse(a, m):
    # Find multiplicative inverse of a modulo m
    for i in range(1, m):
        if (a * i) % m == 1:
            return i
    return None


def affine_encrypt(plaintext, a, b):
    ciphertext = ""

    for ch in plaintext.upper():
        if ch.isalpha():
            # P = 0 to 25
            p = ord(ch) - ord('A')

            # Encryption formula
            c = (a * p + b) % 26

            ciphertext += chr(c + ord('A'))

    return ciphertext


def affine_decrypt(ciphertext, a, b):
    plaintext = ""

    # Find a inverse
    a_inv = mod_inverse(a, 26)

    for ch in ciphertext.upper():
        if ch.isalpha():
            c = ord(ch) - ord('A')

            # Decryption formula
            p = (a_inv * (c - b)) % 26

            plaintext += chr(p + ord('A'))

    return plaintext


# Input
plaintext = input("Enter plaintext: ").replace(" ", "")
a = int(input("Enter value of a: "))
b = int(input("Enter value of b: "))

# Check whether a is valid
if gcd(a, 26) != 1:
    print("Error: 'a' must be coprime with 26.")
else:
    ciphertext = affine_encrypt(plaintext, a, b)
    decrypted = affine_decrypt(ciphertext, a, b)

    print("Ciphertext :", ciphertext)
    print("Decrypted  :", decrypted)