
class GPIO:
    BOARD = 0
    OUT = 1
    
    @staticmethod
    def setboard(self, *args):
        pass

    @staticmethod
    def setup(self, *args):
        pass

    @staticmethod
    def setmode(self, *args):
        pass

    @staticmethod
    def output(pin, value, verbose = False):
        if verbose:
            print(f"GPIO pin {pin:02} set to {value}")
        else:
            return

    @staticmethod
    def cleanup(self, *args):
        pass
