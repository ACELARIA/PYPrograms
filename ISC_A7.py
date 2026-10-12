
from Crypto.Cipher import DES
from Crypto.Util.Padding import pad, unpad

# Accept plaintext from the user
plaintext = input("Enter plaintext message: ")

# Accept secret key (DES requires exactly 8 bytes)
key = input("Enter secret key (8 characters): ")

# Validate the key length
if len(key.encode("utf-8")) != 8:
    print("Error: DES key must be exactly 8 bytes.")
else:
    # Convert plaintext and key into bytes
    plaintext_bytes = plaintext.encode("utf-8")
    key_bytes = key.encode("utf-8")

    # Create DES encryption object using CBC mode
    cipher = DES.new(key_bytes, DES.MODE_CBC)

    # Store the initialization vector (IV)
    iv = cipher.iv

    # Encrypt the plaintext with padding
    padded_plaintext = pad(plaintext_bytes, DES.block_size)
    ciphertext = cipher.encrypt(padded_plaintext)

    # Display ciphertext in hexadecimal format
    print("\nOriginal Plaintext:", plaintext)
    print("Ciphertext (Hex):", ciphertext.hex())

    # Create DES decryption object using the same key and IV
    decipher = DES.new(key_bytes, DES.MODE_CBC, iv=iv)

    # Decrypt and remove padding
    decrypted_padded = decipher.decrypt(ciphertext)
    decrypted_plaintext = unpad(
        decrypted_padded, DES.block_size
    ).decode("utf-8")

    # Display recovered plaintext
    print("Decrypted Plaintext:", decrypted_plaintext)

    # Verify the result
    if decrypted_plaintext == plaintext:
        print("Verification: Successful")
        print("Decrypted text matches original plaintext.")
    else:
        print("Verification: Failed")
