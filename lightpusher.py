#!/usr/bin/python

from __future__ import print_function
import numpy as np
from numpy import unravel_index
import cv2
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
BLUR_AMOUNT = 29    # Blur amount. More is better for accuracy.

# Unused, but for helping punch up the colors a bit. Faster as a lookup than calculation
SAT_TABLE = np.array([
    0,1,2,3,4,5,6,7,8,9,11,12,13,14,15,16,17,18,19,20,22,23,24,25,26,27,28,29,30,31,33,34,35,36,37,38,39,40,41,42,44,45,
    46,47,48,49,50,51,52,53,55,56,57,58,59,60,61,62,63,64,66,67,68,69,70,71,72,73,74,75,77,78,79,80,81,82,83,84,85,86,
    88,89,90,91,92,93,94,95,96,97,99,100,101,102,103,104,105,106,107,108,110,111,112,113,114,115,116,117,118,119,121,
    122,123,124,125,126,127,128,129,130,132,133,134,135,136,137,138,139,140,141,143,144,145,146,147,148,149,150,151,152,
    154,155,156,157,158,159,160,161,162,163,165,166,167,168,169,170,171,172,173,174,176,177,178,179,180,181,182,183,184,
    185,187,188,189,190,191,192,193,194,195,196,198,199,200,201,202,203,204,205,206,207,209,210,211,212,213,214,215,216,
    217,218,220,221,222,223,224,225,226,227,228,229,231,232,233,234,235,236,237,238,239,240,242,243,244,245,246,247,248,
    249,250,251,253,254,255,255,255,255,255,255,255,255,255,255,255,255,255,255,255,255,255,255,255,255,255,255,255,255])

# @profile
def blur(image):
    return cv2.blur(image,(BLUR_AMOUNT,BLUR_AMOUNT)) #865923

# @profile
def convert_color(image):
    return cv2.cvtColor(image, cv2.COLOR_RGB2HSV)

# Finds the index of the maximum saturated pixel
# hsv_image:Image -> (hue:Int, sat:Int, lum:Int)
# @profile
def find_dominant_color(hsv_image):
    hue_channel = hsv_image[:,:,0]
    _, sat_channel = cv2.threshold(hsv_image[:,:,1], 210, 256, cv2.THRESH_TOZERO) # Was 80, 256, wanna push this amount up!

    # sat_channel = cv2.LUT(sat_channel, SAT_TABLE)
    _, lum_channel = cv2.threshold(hsv_image[:,:,2], 50, 256, cv2.THRESH_TOZERO) # Not too much, not too little
    _, lum_channel = cv2.threshold(lum_channel, 140, 256, cv2.THRESH_TOZERO_INV)

    max_index = unravel_index(sat_channel.argmax(), sat_channel.shape)
    hue = hue_channel[max_index]
    # sat = sat_channel[max_index]

    sat = 255 # Max saturation. Ideally, I want this to be the actual saturation, but kinda washed out?
    lum = lum_channel[max_index]
    return hue, sat, lum

# Creates an empty bitmap queue
# () -> image:Deque
# @profile
def initialize_empty_bitmap():
    bitmap = deque(maxlen=WIDTH*HEIGHT)
    empty_element = 0, 0, 0
    empty_row = [empty_element for n in range(WIDTH)]

    for n in range(HEIGHT):
        bitmap.appendleft(empty_row)
    return bitmap

# @profile
def process_frame(frame):
    frame = frame[1] # Why is it a tuple now?
    ready_frame = blur(convert_color(frame))

    (_, width, _) = ready_frame.shape

    # 160 / 9 slices = 17 pixels. Alignment with a few kinda off, but thats ok
    slice_size = width / WIDTH

    image_slices = deque()
    for n in xrange(WIDTH):
        left = int(n * slice_size)
        right = int((n + 1) * slice_size)
        image_slices.append(ready_frame[0:width, left:right])

    return [find_dominant_color(x) for x in image_slices]

# @profile
def display_to_pixelpusher(image, strip, redis_client):
    for led_strip_index in xrange(WIDTH): # Strip number
    # led_strip_index = 8 # Starts with 0th. Numbers appear to be off
        for led_length_index in xrange(HEIGHT): # The position up and down a image, from the bottom, up to 60
            value = image[led_length_index][led_strip_index]
            strip.set_pixel(led_length_index, led_strip_index, value[0], value[1], value[2])

    new_frame = strip.step()
    redis_client.rpush(FRAME_KEY, cPickle.dumps(new_frame))

# @profile
def render_bitmap(bitmap, strip, redis_client):
    img = cv2.cvtColor(bitmap, cv2.COLOR_HSV2RGB)
    display_to_pixelpusher(img, strip, redis_client)

# @profile
def main(argv):
    global FRAME_KEY

    # This is which PI. Need to be configured per unit in lightpusher.service
    FRAME_KEY = sys.argv[1]

    redis_client = redis_conn()
    strip = Service(width=HEIGHT, height=WIDTH) # Yes, this seems backwards
    strip.add_post_process(BlurRight)
    strip.add_post_process(BlurLeft)

    bitmap = initialize_empty_bitmap()

    print("Initializing Camera")
    camera= cv2.VideoCapture(0)

    camera.set(cv2.CAP_PROP_FRAME_WIDTH, 160.0)
    camera.set(cv2.CAP_PROP_FRAME_HEIGHT, 90.0)

    while True:
        frame = camera.read()

        colors = process_frame(frame)
        bitmap.appendleft(colors)
        bitmap.pop()
        np_bitmap = np.array(bitmap, dtype=np.uint8)
        render_bitmap(np_bitmap, strip, redis_client)

        if (cv2.waitKey(5) == 27):
            break

if __name__ == '__main__':
   main(sys.argv[1:])
