#!/usr/bin/python

from __future__ import print_function
import numpy as np
from numpy import unravel_index
import cv2
import sys
from collections import deque
from PIL import Image
from io import BytesIO
import PIL
import time


HEIGHT = 240        # Number of LEDS (60 per meter)
WIDTH = 6           # Number of LED strips
DATAPIN   = 10      # Data Pin for SPI. Probably needs a 3.3v to 5v level stepper
CLOCKPIN  = 11      # Clock Pin for SPI. Probably needs a 3.3v to 5v level stepper
BLUR_AMOUNT = 29    # Blur amount. More is better for accuracy.
LED_GLOBAL_BRIGHTNESS = 12 # Global LED Brightness. Causes flickering. Might be easier to dim all pixels in RGB. Or just have better power supplies
OFFSET = 0

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


# Displays an HSV image on screen for debugging
# Not as needed with cv2.imshow
# Image -> False
# @profile
def display_hsv_image_array(hsv_image):
    rgb_image = cv2.cvtColor(hsv_image, cv2.COLOR_HSV2RGB)
    im = PIL.Image.fromarray(rgb_image)
    bio = BytesIO()
    im.save(bio, format='png')
    display(Image(bio.getvalue(), format='png'))


# Processes one image frame
# frame:Image, bitmap:Deque, strip:Adafruit_DotStar -> False
# 80% of time is in here
# 10.0365 s
# @profile
def flow(frame, bitmap, strip):
    solution = process_frame(frame)
    bitmap.appendleft(solution)
    bitmap.pop()
    render_bitmap(bitmap, strip)


# Finds the index of the maximum saturated pixel
# hsv_image:Image -> (hue:Int, sat:Int, lum:Int)
# 0.309177 s
# @profile
def find_dominant_color(hsv_image):
    hue_channel = hsv_image[:,:,0]
    _, sat_channel = cv2.threshold(hsv_image[:,:,1], 80, 256, cv2.THRESH_TOZERO)
    lum_channel = hsv_image[:,:,2]

    # What's happening is that then the max saturation becomes meh, and becomes a blah one for all

    max_index = unravel_index(sat_channel.argmax(), sat_channel.shape)
    hue = hue_channel[max_index]
    sat = sat_channel[max_index]
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

    ready_frame = blur(convert_color(shrink(frame))) # 90% of the time here is spent in this
    # cv2.imshow('cameraPreview', cv2.cvtColor(ready_frame, cv2.COLOR_HSV2RGB))
    bright_points = deque()
    (height, width, _) = ready_frame.shape
    slice_size = width / WIDTH
    image_slices = deque()
    for n in xrange(WIDTH):
        left = int(n * slice_size)
        right = int((n + 1) * slice_size)
        image_slices.append(ready_frame[0:width, left:right])
        dominant_color_pixel = find_dominant_color(image_slices[n])
        bright_points.append(dominant_color_pixel)

    return bright_points


# Zig-zags around for the led strips in rows.
# @profile
def wrap_and_display_to_leds(image, strip):
    for led_strip_index in xrange(WIDTH):
        for led_length_index in xrange(HEIGHT):
            value = image[led_length_index][led_strip_index]
            if led_strip_index % 2 == 0:
                pixel_index = HEIGHT * led_strip_index + led_length_index + OFFSET
            else:
                pixel_index = HEIGHT * led_strip_index + (HEIGHT - led_length_index - 1) + OFFSET
            strip.setPixelColor(pixel_index, value[0], value[1], value[2])
    strip.show()


# Renders to LEDs or screen
# 2.95871 s
# @profile
def render_bitmap(bitmap, strip):
    # 70% of this is spent in the following. Can I get around that?
    # Surely some way to turn those queues around. Or maybe just array them to begin with?
    np_bitmap = np.array(bitmap, np.uint8) # A faster than below. Accurate?
    # np_bitmap = np.zeros((HEIGHT, WIDTH, 3), np.uint8)
    # for row_idx, row in enumerate(bitmap):
    #     for column_idx, column in enumerate(row):
    #         np_bitmap[row_idx][column_idx] = bitmap[row_idx][column_idx]

    rgb_image = cv2.cvtColor(np_bitmap, cv2.COLOR_HSV2RGB) # This is 25% of the time
    b,g,r = cv2.split(rgb_image)
    img = cv2.merge((g,r,b))

    if MODE == 'pi':
        wrap_and_display_to_leds(img, strip)
    else:
        cv2.imshow('preview', enlarge(rgb_image))

# Enlarges the image. Used for drawing pixels on computer for debugging
# @profile
def enlarge(image):
    return cv2.resize(image,None,fx=40, fy=3, interpolation = cv2.INTER_AREA)


# Shrinks image to 30% of original size. Makes faster and more accurate
# image:Image -> Image
# 4.61985 s Lots of time here. Faster way? Different interpolation?
# @profile
def shrink(image):
    # cv2.resize(image,None,fx=0.3, fy=0.3, interpolation = cv2.INTER_LINEAR) # 1000869
    # cv2.resize(image,None,fx=0.3, fy=0.3, interpolation = cv2.INTER_AREA) # 4782976
    # cv2.resize(image,None,fx=0.3, fy=0.3, interpolation = cv2.INTER_CUBIC) # 2115175
    # cv2.resize(image,None,fx=0.3, fy=0.3, interpolation = cv2.INTER_LANCZOS4) # 5460249
    return cv2.resize(image,None,fx=0.3, fy=0.3, interpolation = cv2.INTER_NEAREST) #  359332 (way fastest)



# This is the main function
# It is kinda messy because it sets up alternate paths for various modes
# (args) -> Boolean
# @profile
def main(argv):
    global MODE
    global INPUT
    MODE = sys.argv[1]  # debug / pi
    INPUT = sys.argv[2] # camera / image / video

    # MODE SETUP FOR LEDs or Display
    if MODE == 'debug':
        print('Running in Debug mode')
        cv2.namedWindow('preview')
        strip = True

    # If you're running on the PI, you want to setup the LED strip
    if MODE == 'pi':
        from dotstar import Adafruit_DotStar
        strip = Adafruit_DotStar(HEIGHT*WIDTH + OFFSET, DATAPIN, CLOCKPIN)
        strip.begin()

        # Lower power consumption, but makes it flicker.
        strip.setBrightness(LED_GLOBAL_BRIGHTNESS)

    bitmap = initialize_empty_bitmap()
    render_bitmap(bitmap, strip)

    # INPUT SELECTION SETUP
    # If you're using a USB camera for input
    # TODO: Allow for arg use for different cameras
    if INPUT == 'camera':
        if MODE == 'debug':
            cv2.namedWindow('cameraPreview', cv2.WINDOW_NORMAL)
        vc = cv2.VideoCapture(0)
        if vc.isOpened():
            # vc.set(15, -10)
            vc.set(3,200) # These aren't accurate, but help
            vc.set(4,100)
            rval, frame = vc.read()
        else:
            rval = False

        while rval:
            rval, frame = vc.read()
            # if MODE == 'debug':
                # cv2.imshow('cameraPreview', frame)
            start = time.time()
            flow(frame, bitmap, strip)
            key = cv2.waitKey(15)
            if key == 27: # exit on ESC
                break
            end = time.time()
            print(end - start)

    # If you're using a static image for debugging
    if INPUT == 'image':
        if len(sys.argv) == 4:
            frame = cv2.imread(sys.argv[3])
        else:
            frame = cv2.imread('bars.jpg')
        rval = True

        # while True:
        # For 1000 frames
        start = time.time()
        while True:
            flow(frame, bitmap, strip)
            key = cv2.waitKey(15)
            if key == 27: # exit on ESC
                break
        end = time.time()
        fps = 1000 / (end - start)
        print('fps:', fps)



    # If you're using a pre-recorded video for debugging set it here
    if INPUT == 'video':
        cv2.namedWindow('videoPreview', cv2.WINDOW_NORMAL)
        vc = cv2.VideoCapture('WaveCanon2.mp4')
        if vc.isOpened():
            rval, frame = vc.read()
        else:
            rval = False

        while rval:
            rval, frame = vc.read()
            frame = shrink(frame) # 1080p video too big coming in
            cv2.imshow('videoPreview', frame)
            flow(frame, bitmap, strip)
            key = cv2.waitKey(15)
            if key == 27: # exit on ESC
                break

        return False

# This is python magic I don't understand. Wat.
if __name__ == '__main__':
   main(sys.argv[1:])
