# ============================================================
# 2 × 2 HILL CIPHER
# ============================================================

import numpy as np
from math import gcd

# ------------------------------------------------------------
# 1. Construct and display the 2 × 2 Hill Cipher key matrix
# ------------------------------------------------------------

K = np.array([
    [3, 3],
    [2, 5]
])

print("Hill Cipher Key Matrix:")
print(K)


# ------------------------------------------------------------
# 2. Convert plaintext to uppercase, remove spaces,
#    and map letters to numbers using A=0, B=1, ..., Z=25
# ------------------------------------------------------------

plaintext = "MEET AT BASE"

plaintext = plaintext.upper().replace(" ", "")

print("\nOriginal Plaintext:", plaintext)

# Convert each letter to its numerical value
numbers = [ord(ch) - ord('A') for ch in plaintext]

print("Numerical Plaintext:", numbers)


# ------------------------------------------------------------
# 3. Divide plaintext into valid 2-letter blocks
# ------------------------------------------------------------

blocks = []

for i in range(0, len(numbers), 2):
    block = numbers[i:i+2]
    blocks.append(block)

print("\n2-Letter Blocks:")

for block in blocks:
    print(block)


# ------------------------------------------------------------
# 4. Encrypt each block using Hill Cipher formula
#
# Ciphertext = K × Plaintext (mod 26)
# ------------------------------------------------------------

encrypted_blocks = []
ciphertext = ""

print("\nEncryption:")

for block in blocks:

    # Convert block into column vector
    P = np.array(block).reshape(2, 1)

    # Matrix multiplication
    C = np.dot(K, P) % 26

    # Convert numpy values to normal integers
    c1 = int(C[0][0])
    c2 = int(C[1][0])

    encrypted_blocks.append([c1, c2])

    # Convert numbers back to letters
    ciphertext += chr(c1 + ord('A'))
    ciphertext += chr(c2 + ord('A'))

    print(f"{block} -> {[c1, c2]} -> "
          f"{chr(c1 + 65)}{chr(c2 + 65)}")


# ------------------------------------------------------------
# 5. Display encrypted numerical blocks and ciphertext
# ------------------------------------------------------------

print("\nEncrypted Numerical Blocks:")

for block in encrypted_blocks:
    print(block)

print("\nCiphertext:", ciphertext)


# ------------------------------------------------------------
# 6. Find modular inverse of key matrix modulo 26
#
# For:
# K = [ a b ]
#     [ c d ]
#
# det(K) = ad - bc
#
# K^-1 = det(K)^-1 × [ d  -b ]
#                       [ -c  a ]
#                       mod 26
# ------------------------------------------------------------

det = int(round(np.linalg.det(K)))

det = det % 26

print("\nDeterminant of K:", det)


# Find multiplicative inverse of determinant modulo 26
def modular_inverse(a, m):

    for x in range(1, m):
        if (a * x) % m == 1:
            return x

    return None


det_inverse = modular_inverse(det, 26)

print("Multiplicative Inverse of Determinant:",
      det_inverse)


# Adjugate matrix
adjugate = np.array([
    [K[1][1], -K[0][1]],
    [-K[1][0], K[0][0]]
])

# Modular inverse of K
K_inverse = (det_inverse * adjugate) % 26

print("\nModular Inverse of Key Matrix:")
print(K_inverse)


# ------------------------------------------------------------
# 7. Decrypt the ciphertext
#
# Plaintext = K^-1 × Ciphertext (mod 26)
# ------------------------------------------------------------

decrypted_numbers = []
decrypted_text = ""

print("\nDecryption:")

for block in encrypted_blocks:

    # Ciphertext block as column vector
    C = np.array(block).reshape(2, 1)

    # P = K^-1 × C mod 26
    P = np.dot(K_inverse, C) % 26

    p1 = int(P[0][0])
    p2 = int(P[1][0])

    decrypted_numbers.extend([p1, p2])

    decrypted_text += chr(p1 + ord('A'))
    decrypted_text += chr(p2 + ord('A'))

    print(f"{block} -> {[p1, p2]} -> "
          f"{chr(p1 + 65)}{chr(p2 + 65)}")


# ------------------------------------------------------------
# 8. Convert decrypted numerical values back to letters
#    and verify original plaintext
# ------------------------------------------------------------

print("\nDecrypted Numerical Values:")
print(decrypted_numbers)

print("\nDecrypted Plaintext:", decrypted_text)

print("\nOriginal Plaintext :", plaintext)

if decrypted_text == plaintext:
    print("Verification: SUCCESS")
    print("Original plaintext is successfully recovered.")
else:
    print("Verification: FAILED")