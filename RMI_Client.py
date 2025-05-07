import Pyro5.api

# Use the exact URI printed by the server
uri = input("Enter the server URI (e.g. PYRO:obj_abc123@localhost:port): ")
concatenator = Pyro5.api.Proxy(uri)

str1 = input("Enter first string: ")
str2 = input("Enter second string: ")

result = concatenator.concatenate(str1, str2)
print("Concatenated Result:", result)
