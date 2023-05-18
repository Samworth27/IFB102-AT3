try:
    import RPi.RPIO as RPIO
    rp = True
except:
    rp = False
finally:
    from seven_seg.driver import SevenSegment
    from NN.network import Network
    from camera.camera import Camera
    import cv2 as cv
    import statistics

print(f"Running on a pi: {rp}")


sevseg = SevenSegment(7, 11, 13, 15)

network = Network()
network.load_network('net1')
cam = Camera()
# cam.capture_continuous()
results = [0]
while True:
    captured, detected, digit, annotated = cam.capture()
    try:
        cam.display_capture(digit, annotated)
    except:
        pass
    predicted = network.predict([digit.reshape((1, 28*28))])[0]
    if len(results) > 100:
        results.pop(0)
    results.append(predicted)
    outcome = statistics.mean(results)
    sevseg.display(outcome)
    print(outcome)
    if cv.waitKey(1) == ord('q'):
        break
cam.exit()
