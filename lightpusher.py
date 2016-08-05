#!/usr/bin/python

from __future__ import print_function
import numpy as np
from numpy import unravel_index
# from imutils.video import VideoStream
import cv2
import sys
# import imutils
import itertools
from collections import deque
from PIL import Image
from io import BytesIO
import PIL
import time
import cPickle
import random


# New ones
import redis
from pixelpusher import pixel, build_strip, send_strip, bound
from postprocess import BlurLeft, BlurRight
from service import Service
from util import redis_conn

import pdb
# pdb.set_trace()

NUM_CAMERAS = 4
HEIGHT = 60        # Number of LEDS (60 per meter)
WIDTH = 9          # Number of LED strips left/right
BLUR_AMOUNT = 29    # Blur amount. More is better for accuracy.
OFFSET = 0
SAT_TABLE = np.array([
    0,1,2,3,4,5,6,7,8,9,11,12,13,14,15,16,17,18,19,20,22,23,24,25,26,27,28,29,30,31,33,34,35,36,37,38,39,40,41,42,44,45,46,47,48,49,50,51,52,53,55,56,57,58,59,60,61,62,63,64,66,67,68,69,70,71,72,73,74,75,77,78,79,80,81,82,83,84,85,86,88,89,90,91,92,93,94,95,96,97,99,100,101,102,103,104,105,106,107,108,110,111,112,113,114,115,116,117,118,119,121,122,123,124,125,126,127,128,129,130,132,133,134,135,136,137,138,139,140,141,143,144,145,146,147,148,149,150,151,152,154,155,156,157,158,159,160,161,162,163,165,166,167,168,169,170,171,172,173,174,176,177,178,179,180,181,182,183,184,185,187,188,189,190,191,192,193,194,195,196,198,199,200,201,202,203,204,205,206,207,209,210,211,212,213,214,215,216,217,218,220,221,222,223,224,225,226,227,228,229,231,232,233,234,235,236,237,238,239,240,242,243,244,245,246,247,248,249,250,251,253,254,255,255,255,255,255,255,255,255,255,255,255,255,255,255,255,255,255,255,255,255,255,255,255,255])

# Performs a strong GaussianBlur
# image:Image -> Image
# takes 20fps
# @profile
def blur(image):
    # cv2.GaussianBlur(image, (BLUR_AMOUNT, BLUR_AMOUNT),25) #4873963
    return cv2.blur(image,(BLUR_AMOUNT,BLUR_AMOUNT)) #865923


# Converts from BGR to HSV colorspace
# image:Image -> Image
# @profile
def convert_color(image):
    return cv2.cvtColor(image, cv2.COLOR_RGB2HSV)

# Processes one image frame
# frame:Image, bitmap:Deque, strip:Adafruit_DotStar -> False
# 80% of time is in here
# 10.0365 s
# @profile
def flow(frame, bitmap, strip, redis_client):
    bitmap.appendleft(process_frame(frame))
    bitmap.pop()
    render_bitmap(bitmap, strip, redis_client)


# Finds the index of the maximum saturated pixel
# hsv_image:Image -> (hue:Int, sat:Int, lum:Int)
# 0.309177 s
# @profile
def find_dominant_color(hsv_image):
    hue_channel = hsv_image[:,:,0]
    _, sat_channel = cv2.threshold(hsv_image[:,:,1], 210, 256, cv2.THRESH_TOZERO) # Was 80, 256, wanna push this amount up!

    # sat_channel = cv2.LUT(sat_channel, SAT_TABLE)

    _, lum_channel = cv2.threshold(hsv_image[:,:,2], 50, 256, cv2.THRESH_TOZERO) # Not too much, not too little
    _, lum_channel = cv2.threshold(lum_channel, 140, 256, cv2.THRESH_TOZERO_INV)

    max_index = unravel_index(sat_channel.argmax(), sat_channel.shape)
    hue = hue_channel[max_index]

    sat = 255 # Max up. Could shape better
    lum = lum_channel[max_index]
    return hue, sat, lum

# Creates an empty bitmap
# () -> image:Deque
# @profile
def initialize_empty_bitmap():
    bitmap = deque(maxlen=WIDTH*HEIGHT)
    empty_element = 0, 0, 0
    empty_row = [empty_element for n in range(WIDTH)]

    for n in range(HEIGHT):
        bitmap.appendleft(empty_row)
    return bitmap


# Finds the max hue/saturation across the image
# frame:Image -> Deque
# 5.79235 s
# @profile
def process_frame(frame):
    frame = frame[1] # Why is it a tuple now?
    ready_frame = blur(convert_color(shrink(frame)))

    (_, width, _) = ready_frame.shape
    print(width)
    slice_size = width / WIDTH # 9 pixel width slices
    image_slices = deque()
    for n in xrange(WIDTH):
        left = int(n * slice_size)
        right = int((n + 1) * slice_size)
        image_slices.append(ready_frame[0:width, left:right])

    # return p.map(find_dominant_color, image_slices)
    return [find_dominant_color(x) for x in image_slices]

# Need to split into sub 480-section segments, with each being a strip to address.
# Store all strips as a global array to iterate through?
def display_to_pixelpusher(image, strip, redis_client):
    for led_strip_index in xrange(WIDTH): # Strip number
    # led_strip_index = 8 # Starts with 0th. Numbers appear to be off
        for led_length_index in xrange(HEIGHT): # The position up and down a image, from the bottom, up to 60
            value = image[led_length_index][led_strip_index]
            strip.set_pixel(led_length_index, led_strip_index, value[0], value[1], value[2])
    new_frame = strip.step()
    redis_client.rpush(FRAME_KEY, cPickle.dumps(new_frame))
    # redis_client.rpush('q2', cPickle.dumps(new_frame))
    # redis_client.rpush('q3', cPickle.dumps(new_frame))
    # redis_client.rpush('q4', cPickle.dumps(new_frame))

# Renders to LEDs or screen
# The bitmap coming in should be smaller right?
# @profile
def render_bitmap(bitmap, strip, redis_client):
    # print(bitmap)
    # bitmap_list = list(bitmap)
    # np_bitmap = np.array(bitmap_list, np.uint8)
    img = cv2.cvtColor(bitmap, cv2.COLOR_HSV2RGB)

    # For swapping colors, might not need since PP can do this for me!
    # b,g,r = cv2.split(img)
    # img = cv2.merge((r,g,b))

    display_to_pixelpusher(img, strip, redis_client)

# Shrinks image to 30% of original size. Makes faster and more accurate
# image:Image -> Image
# @profile
def shrink(image):
    return cv2.resize(image,None,fx=0.25, fy=0.25, interpolation = cv2.INTER_NEAREST)

# This is the main function
# It is kinda messy because it sets up alternate paths for various modes
# (args) -> Boolean
# @profile
def main(argv):
    global MODE
    global INPUT
    global p
    global FRAME_KEY
    INPUT = sys.argv[1] # camera / image / video
    FRAME_KEY = sys.argv[2] # This is which PI. Need to be configured per unit!

    redis_client = redis_conn()
    strip = Service(width=HEIGHT, height=WIDTH) # Yes, this seems backwards

    bitmap = initialize_empty_bitmap()

    print("Initializing Camera 1")
    camera= cv2.VideoCapture(0)

    print("Setting camera properties")
    # pdb.set_trace()

    camera.set(cv2.cv.CV_CAP_PROP_FRAME_WIDTH, 81.0)
    camera.set(cv2.cv.CV_CAP_PROP_FRAME_HEIGHT, 50.0)
    # time.sleep(2)
    # v4l2-ctl --set-fmt-video=width=81,height=50

    while True:
        print("frame")
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
