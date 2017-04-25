#!/usr/bin/python

from __future__ import print_function
import numpy as np
from numpy import unravel_index
import sys
import itertools
from collections import deque
from PIL import Image
from io import BytesIO
import PIL
import time
import cPickle
import os
import random
import redis
from pixelpusher import pixel, build_strip, send_strip, bound
from postprocess import BlurLeft, BlurRight
from service import Service
from util import redis_conn

HEIGHT = 60        # Number of LEDS (60 per meter)
WIDTH = 9          # Number of LED strips left/right. 36 strips total

def initialize_empty_bitmap():
    bitmap = deque(maxlen=WIDTH*HEIGHT)
    empty_element = 127, 0, 0
    empty_row = [empty_element for n in range(WIDTH)]

    for n in range(HEIGHT):
        bitmap.appendleft(empty_row)
    return bitmap


# @profile
def display_to_pixelpusher(image, strip, redis_client):
    for led_strip_index in xrange(WIDTH): # Strip number
    # led_strip_index = 8 # Starts with 0th. Numbers appear to be off
        for led_length_index in xrange(HEIGHT): # The position up and down a image, from the bottom, up to 60
            value = image[led_length_index][led_strip_index]
            strip.set_pixel(led_length_index, led_strip_index, value[0], value[1], value[2])

    new_frame = strip.step()
    redis_client.rpush('q1', cPickle.dumps(new_frame))

# @profile
def render_bitmap(bitmap, strip, redis_client):
    display_to_pixelpusher(bitmap, strip, redis_client)

# @profile
def main(argv):
    global FRAME_KEY

    # This is which PI. Need to be configured per unit in lightpusher.service
    FRAME_KEY = sys.argv[1]

    redis_client = redis_conn()
    strip = Service(width=HEIGHT, height=WIDTH) # Yes, this seems backwards

    bitmap = initialize_empty_bitmap()
    while True:
        np_bitmap = np.array(bitmap, dtype=np.uint8)
        render_bitmap(np_bitmap, strip, redis_client)
        time.sleep(5)


if __name__ == '__main__':
   main(sys.argv[1:])
