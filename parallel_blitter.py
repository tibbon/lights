import redis
import time
import random
import struct
import cPickle

from pixelpusher import pixel, build_strip, send_strip

import sys

IP = '192.168.1.100' # Skipping DHCP
PORT = 9897

MAX = 255
MID = 128
OFF = 0

FRAME_MASTER = 'master'
FRAME_KEY = 'frame'

def prep_client(client):
    client.delete('q1')
    client.delete('q2')
    client.delete('q3')
    client.delete('q4')

def main():
    client = redis.Redis()
    prep_client(client)

    pixel_width = 360 # Number per strip
    delay = 0.001
    print("Starting")

    while True:
        _, q1_frame = client.blpop('q1')
        _, q2_frame = client.blpop('q2')
        _, q3_frame = client.blpop('q3')
        _, q4_frame = client.blpop('q4')

        full_frame = q1_frame + q2_frame + q3_frame + q4_frame

        # print("Loop")
        frame = cPickle.loads(full_frame)
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
