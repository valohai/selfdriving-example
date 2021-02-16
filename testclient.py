import base64
import logging
import random
import sys

import threading
import socketio

endpoint = 'https://valohai.cloud/juha/selfdriving/drive/vroom/drive'
socket_io_path = '/juha/selfdriving/drive/vroom/drive/socket.io'

logging.basicConfig(level=logging.DEBUG)

sio = socketio.Client()

response_event = threading.Event()


def send_test_data(sio: socketio.Client):
    with open("test_image.jpg", "rb") as f:
        imagestr = base64.b64encode(f.read()).decode()

    data = {
        "steering_angle": random.uniform(-1, 1),
        "throttle": random.uniform(0, 1),
        "speed": random.uniform(1, 5),
        "image": imagestr,
    }

    response_event.clear()
    sio.emit('telemetry', data=data)
    print("Emitted")
    response_event.wait()
    response_event.clear()
    print("Okay, got response. Disconnecting...")
    sio.disconnect()
    sio.wait()



@sio.event
def connect():
    print('connection established')
    send_test_data(sio)


@sio.on('steer')
def receive_steer(data):
    print('Steer', data)
    response_event.set()


@sio.event
def disconnect():
    print('disconnected from server')


def main():
    sio.connect(
        endpoint,
        socketio_path=socket_io_path,
    )


if __name__ == '__main__':
    main()
