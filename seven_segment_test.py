from seven_seg.driver import SevenSegment
from time import sleep
import numpy as np

sevseg = SevenSegment(7, 11, 13, 15)

for i in range(20):
    sevseg.display(i % 10)
    sleep(1)

for _ in range(100):
    sevseg.display(np.random.randint(0,10))
    sleep(0.1)    

