products = {}

while True:
    product_name = input("Enter product name (or 'done' to finish): ")
    if product_name.lower() == 'done':
        break
    try:
        price = float(input("Enter price: "))
    except ValueError:
        print("Invalid price. Please enter a number.")
        continue
    products[product_name] = price

while True:
    product_name = input("Enter product name to find price (or 'done' to finish): ")
    if product_name.lower() == 'done':
        break
    if product_name in products:
        print(f"Price of {product_name}: ${products[product_name]}")
    else:
        print(f"{product_name} not found in the list.")