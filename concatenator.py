import Pyro5.api

@Pyro5.api.expose
class Concatenator:
    def concatenate(self, str1, str2):
        return str1 + str2
