# ============================================================
# Information Security and Cryptography - AI 331
# Lab Assignment 4
# Playfair Cipher
# Key: DRONE
# Plaintext: ATTACK AT THE STATION
# ============================================================


# ------------------------------------------------------------
# 1. Construct the 5 x 5 Playfair Key Matrix
# ------------------------------------------------------------

def create_matrix(key):
    # Combine key with remaining alphabet
    key = key.upper().replace("J", "I")

    alphabet = "ABCDEFGHIKLMNOPQRSTUVWXYZ"

    # Remove duplicate letters while preserving order
    combined = key + alphabet

    letters = []
    for char in combined:
        if char.isalpha() and char not in letters:
            letters.append(char)

    # Create 5 x 5 matrix
    matrix = [letters[i:i + 5] for i in range(0, 25, 5)]

    return matrix


# ------------------------------------------------------------
# 2. Find position of a letter in the matrix
# ------------------------------------------------------------

def find_position(matrix, letter):
    for row in range(5):
        for col in range(5):
            if matrix[row][col] == letter:
                return row, col


# ------------------------------------------------------------
# 3. Format plaintext into digraphs
# ------------------------------------------------------------

def prepare_plaintext(text):
    text = ''.join(char for char in text.upper() if char.isalpha())

    # Playfair uses I/J in the same cell
    text = text.replace("J", "I")

    digraphs = []
    i = 0

    while i < len(text):
        first = text[i]

        if i + 1 < len(text):
            second = text[i + 1]

            # Repeated letters: insert X
            if first == second:
                digraphs.append(first + "X")
                i += 1
            else:
                digraphs.append(first + second)
                i += 2
        else:
            # Odd length: append X
            digraphs.append(first + "X")
            i += 1

    return digraphs


# ------------------------------------------------------------
# 4. Encrypt a digraph
# ------------------------------------------------------------

def encrypt_pair(pair, matrix):
    r1, c1 = find_position(matrix, pair[0])
    r2, c2 = find_position(matrix, pair[1])

    # Same row -> move one column right
    if r1 == r2:
        return (
            matrix[r1][(c1 + 1) % 5] +
            matrix[r2][(c2 + 1) % 5]
        )

    # Same column -> move one row down
    elif c1 == c2:
        return (
            matrix[(r1 + 1) % 5][c1] +
            matrix[(r2 + 1) % 5][c2]
        )

    # Rectangle rule
    else:
        return (
            matrix[r1][c2] +
            matrix[r2][c1]
        )


# ------------------------------------------------------------
# 5. Decrypt a digraph
# ------------------------------------------------------------

def decrypt_pair(pair, matrix):
    r1, c1 = find_position(matrix, pair[0])
    r2, c2 = find_position(matrix, pair[1])

    # Same row -> move one column left
    if r1 == r2:
        return (
            matrix[r1][(c1 - 1) % 5] +
            matrix[r2][(c2 - 1) % 5]
        )

    # Same column -> move one row up
    elif c1 == c2:
        return (
            matrix[(r1 - 1) % 5][c1] +
            matrix[(r2 - 1) % 5][c2]
        )

    # Rectangle rule
    else:
        return (
            matrix[r1][c2] +
            matrix[r2][c1]
        )


# ------------------------------------------------------------
# MAIN PROGRAM
# ------------------------------------------------------------

key = "DRONE"
plaintext = "ATTACK AT THE STATION"


# Create matrix
matrix = create_matrix(key)

print("=" * 60)
print("PLAYFAIR CIPHER")
print("=" * 60)

print("\nKey:", key)

print("\n5 x 5 Key Matrix:")
for row in matrix:
    print(" ".join(row))


# Prepare plaintext
digraphs = prepare_plaintext(plaintext)

print("\nOriginal Plaintext:", plaintext)
print("Formatted Digraphs:", " ".join(digraphs))


# Encrypt
ciphertext = ""

for pair in digraphs:
    ciphertext += encrypt_pair(pair, matrix)

print("\nCiphertext:", ciphertext)


# Decrypt
decrypted = ""

for i in range(0, len(ciphertext), 2):
    pair = ciphertext[i:i + 2]
    decrypted += decrypt_pair(pair, matrix)

print("Decrypted Text:", decrypted)

print("\n" + "=" * 60)