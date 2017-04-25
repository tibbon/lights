import redis
import time
import random
import struct
import cPickle
import os

from pixelpusher import pixel, build_strip, send_strip

import sys

IP = '192.168.1.100' # Pixelpusher IP address
PORT = 9897

MAX = 255
MID = 128
OFF = 0

FRAME_MASTER = 'master'
FRAME_KEY = 'frame'

def redis_conn():
    host = os.environ.get('REDISHOST', '127.0.0.1')
    print(host)
    return redis.Redis(host=host)

def prep_client(client):
    client.delete('q1')

def main():
    client = redis_conn()
    prep_client(client)

    pixel_width = 360 # Number per strip
    delay = 0.005
    print("Starting")

    while True:
        _, q1_frame = client.blpop('q1')

        q1_frame = cPickle.loads(q1_frame)


        frame = q1_frame

        lines = []

        # For all the strips?
        for i in range(0, 8):
            lines.append(frame[pixel_width*i:pixel_width*(i+1)])

        # Sends two strips at a time?
        for index in range(0, 6):
            s = struct.pack('!xxxxB', index) + ''.join(lines[index])
            send_strip(''.join(s), (IP, PORT))
            time.sleep(delay)

if __name__ == "__main__":
    main()
