# write a program that returns the first repeated character in a string


def first_repeated_char(str):
    repeat = []
    for letter in str:
        if letter in repeat:
            return letter
        else:
            repeat.append(letter)
    return None


ex1 = 'abca'
ex2 = 'bcaba'
ex3 = 'abc'
ex4 = 'dbcaba'

print(first_repeated_char(ex1))
print(first_repeated_char(ex2))
print(first_repeated_char(ex3))
print(first_repeated_char(ex4))

# given an array, treat all entries as a number, then increment


def array_incrementer(given_array):
    number = 0
    for i, val in enumerate(reversed(given_array)):
        number = number + 10**i * val
    number = number + 1
    new_array = []
    for val in str(number):
        new_array.append(int(val))
    return new_array


print(array_incrementer([1, 3, 2, 9]))
