
import hashlib

# Function to generate message digests
def generate_digests(message):
    # Convert message into bytes
    message_bytes = message.encode()

    # Generate MD5, SHA-1, and SHA-256 digests
    md5_digest = hashlib.md5(message_bytes).hexdigest()
    sha1_digest = hashlib.sha1(message_bytes).hexdigest()
    sha256_digest = hashlib.sha256(message_bytes).hexdigest()

    return md5_digest, sha1_digest, sha256_digest


# Main program
def main():
    print("===== MESSAGE DIGEST AND INTEGRITY VERIFICATION =====")

    # Accept original message
    original_message = input("\nEnter the original message: ")

    # Generate original digests
    md5_original, sha1_original, sha256_original = generate_digests(
        original_message
    )

    # Display original message and digests
    print("\n--- Original Message Details ---")
    print("Original Message:", original_message)
    print("MD5 Digest:     ", md5_original)
    print("SHA-1 Digest:   ", sha1_original)
    print("SHA-256 Digest: ", sha256_original)

    # Accept received or modified message
    received_message = input("\nEnter the received/modified message: ")

    # Generate received message digests
    md5_received, sha1_received, sha256_received = generate_digests(
        received_message
    )

    print("\n--- Received Message Details ---")
    print("Received Message:", received_message)
    print("MD5 Digest:     ", md5_received)
    print("SHA-1 Digest:   ", sha1_received)
    print("SHA-256 Digest: ", sha256_received)

    # Verify integrity using each algorithm
    print("\n--- Message Integrity Verification ---")

    algorithms = [
        ("MD5", md5_original, md5_received),
        ("SHA-1", sha1_original, sha1_received),
        ("SHA-256", sha256_original, sha256_received)
    ]

    for name, original_digest, received_digest in algorithms:
        if original_digest == received_digest:
            print(name, ": Message integrity verified.")
        else:
            print(name, ": Message has been modified!")

    # Demonstrate effect of changing one character
    print("\n--- One-Character Modification Demonstration ---")

    modified_message = original_message + "X"

    print("Original Message :", original_message)
    print("Modified Message :", modified_message)

    md5_modified, sha1_modified, sha256_modified = generate_digests(
        modified_message
    )

    print("Original SHA-256 :", sha256_original)
    print("Modified SHA-256 :", sha256_modified)

    if sha256_original != sha256_modified:
        print("Result: Even a one-character change produces a different digest.")


if __name__ == "__main__":
    main()
