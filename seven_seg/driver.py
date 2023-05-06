try:
    import RPi.GPIO as GPIO
    rp = True
except:
    from .GPIOdummy import GPIO
    rp = False
    

def digits(num):
        digits = format(num,'#06b')[-4:]
        return [int(i) for i in digits]


class SevenSegment:
        def __init__(self,pin0,pin1,pin2,pin3):
                GPIO.setmode(GPIO.BOARD)
                self.pins = [pin0,pin1,pin2,pin3]
                for pin in self.pins:
                        print(f"Setting pin {pin} to OUT")
                        GPIO.setup(pin,GPIO.OUT)        
        def display(self,num):
                for pin, digit in zip(self.pins,digits(num)):
                        GPIO.output(pin,digit)
                        
        def exit(self):
            GPIO.cleaup()


