import Pyro5.api
from concatenator import Concatenator  # This uses the file you created

# def Concatenator(str1, str2):
#         return str1 + str2

def main():
    daemon = Pyro5.api.Daemon()
    uri = daemon.register(Concatenator)
    print("Server ready. Object uri =", uri)
    daemon.requestLoop()

if __name__ == "__main__":
    main()
