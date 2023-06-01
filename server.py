# https://pyimagesearch.com/2019/09/02/opencv-stream-video-to-web-browser-html-page/

try:
    import RPi.RPIO as RPIO

    rp = True
except:
    rp = False
finally:
    from SevenSeg.driver import SevenSegment
    from NN.network import Network
    from camera.camera import Camera

    from flask import Flask
    from flask import Response
    from flask import render_template

    import time
    import threading
    import argparse
    import cv2 as cv

outputFrame = None
lock = threading.Lock()
cam = Camera()
app = Flask(__name__)


@app.route("/")
def index():
    return render_template("index.html")


def detect_digit():
    global cam, outputFrame, lock
    sevseg = SevenSegment(7, 11, 13, 15)

    network = Network()
    network.load_network("net2")

    while True:
        captured, detected, digit, annotated = cam.capture()
        if detected:
            predicted = network.predict([digit.reshape((1, 28 * 28))])[0]
            # predicted = network.predict([digit.reshape((28, 28))])[0]
            sevseg.display(predicted)
            print(predicted)
            cv.putText(
                annotated,
                f"Predicted Value: {predicted}",
                (10, annotated.shape[0] - 10),
                cv.FONT_HERSHEY_SIMPLEX,
                1,
                (0, 0, 255),
                1,
            )
        else:
            sevseg.display(0)

        with lock:
            outputFrame = annotated.copy()


def generate():
    global outputFrame, lock
    while True:
        time.sleep(0.1)
        with lock:
            if outputFrame is None:
                continue
            flag, encodedImage = cv.imencode(".jpg", outputFrame)

            if not flag:
                print("error encoding")
                continue

            yield (
                b"--frame\r\n" b"Content-Type: image/jpeg\r\n\r\n" + bytearray(encodedImage) + b"\r\n"
            )


@app.route("/video_feed")
def video_feed():
    return Response(generate(), mimetype="multipart/x-mixed-replace; boundary=frame")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("-i", "--ip", type=str, required=True, help="ip address of device")
    ap.add_argument("-0", "--port", type=int, required=True, help="port of server")
    args = vars(ap.parse_args())

    t = threading.Thread(target=detect_digit)
    t.daemon = True
    t.start()

    app.run(
        host=args["ip"],
        port=args["port"],
        debug=True,
        threaded=True,
        use_reloader=False,
    )
cam.exit()

