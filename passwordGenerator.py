#Password Generator Project
import random
letters = ['a', 'b', 'c', 'd', 'e', 'f', 'g', 'h', 'i', 'j', 'k', 'l', 'm', 'n', 'o', 'p', 'q', 'r', 's', 't', 'u', 'v', 'w', 'x', 'y', 'z', 'A', 'B', 'C', 'D', 'E', 'F', 'G', 'H', 'I', 'J', 'K', 'L', 'M', 'N', 'O', 'P', 'Q', 'R', 'S', 'T', 'U', 'V', 'W', 'X', 'Y', 'Z']
numbers = ['0', '1', '2', '3', '4', '5', '6', '7', '8', '9']
symbols = ['!', '#', '$', '%', '&', '(', ')', '*', '+']

print("Welcome to the PyPassword Generator!")
nr_letters= int(input("How many letters would you like in your password?\n")) 
nr_symbols = int(input(f"How many symbols would you like?\n"))
nr_numbers = int(input(f"How many numbers would you like?\n"))

# Basic passwprd generator
basicPassword = " "
for char in range(0, nr_letters):
    basicPassword += random.choice(letters)
for char in range(0, nr_symbols):
    basicPassword += random.choice(symbols)
for char in range(0, nr_numbers):
    basicPassword += random.choice(numbers)
print(f"Your basic password is: {basicPassword}")

# Random password generator
passList = []
for char in range(0 , nr_letters):
    passList.append(random.choice(letters))
for char in range(0 , nr_letters):
    passList.append(random.choice(numbers))
for char in range(0 , nr_letters):
    passList.append(random.choice(symbols))

hardPassword = " "
random.shuffle(passList)
for char in passList:
    hardPassword += char
print(f"Your hard password is: {hardPassword}")