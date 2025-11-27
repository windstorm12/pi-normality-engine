from mpmath import mp

# Set precision: 1 million digits + 1 for safety
mp.dps = 1_000_001  

# Get pi as string
pi_str = str(mp.pi)

# Remove the leading "3" and decimal point
pi_digits = pi_str[2:]  # skips "3."

# Check first 50 digits as a sample
print(pi_digits[:50])

# Save to a file if needed
with open("pi_1million_digits.txt", "w") as f:
    f.write(pi_digits)
