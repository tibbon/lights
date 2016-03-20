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


HEIGHT = 240        # Number of LEDS (60 per meter)
WIDTH = 4           # Number of LED strips
DATAPIN   = 10      # Data Pin
CLOCKPIN  = 11      # Clock Pin
BLUR_AMOUNT = 29    # Blur amount
GLOBAL_LUM = 30     # Global Lum. Currently unused
LED_GLOBAL_BRIGHTNESS = 12 # Global LED Brightness. Causes flickering. Might be easier to dim all pixels in RGB. Or just have better power supplies

# Performs a strong GaussianBlur
# image:Image -> Image
def blur(image):
    return cv2.GaussianBlur(image, (BLUR_AMOUNT, BLUR_AMOUNT),25)


# Converts from BGR to HSV colorspace
# image:Image -> Image
def convert_color(image):
    return cv2.cvtColor(image, cv2.COLOR_RGB2HSV)


# Displays an HSV image on screen for debugging
# Not as needed with cv2.imshow
# Image -> False
def display_hsv_image_array(hsv_image):
    rgb_image = cv2.cvtColor(hsv_image, cv2.COLOR_HSV2RGB)
    im = PIL.Image.fromarray(rgb_image)
    bio = BytesIO()
    im.save(bio, format='png')
    display(Image(bio.getvalue(), format='png'))


# Processes one image frame
# frame:Image, bitmap:Deque, strip:Adafruit_DotStar -> False
def flow(frame, bitmap, strip):
    solution = process_frame(frame)
    bitmap.appendleft(solution)
    bitmap.pop()
    render_bitmap(bitmap, strip)


# Finds the index of the maximum saturated pixel
# hsv_image:Image -> (hue:Int, sat:Int, lum:Int)
def find_dominant_color(hsv_image):
    hue_channel = hsv_image[:,:,0]
    sat_channel = hsv_image[:,:,1]
    lum_channel = hsv_image[:,:,2]

    max_index = unravel_index(sat_channel.argmax(), sat_channel.shape)
    hue = hue_channel[max_index]
    sat = sat_channel[max_index]
    lum = lum_channel[max_index]

    return hue, sat, lum


# Creates an empty bitmap
# () -> image:Deque
def initialize_empty_bitmap():
    bitmap = deque(maxlen=WIDTH*HEIGHT)
    empty_element = 0, 0, 0
    empty_row = [empty_element for n in range(WIDTH)]

    for n in range(HEIGHT):
        bitmap.appendleft(empty_row)

    return bitmap


# Finds the max hue/saturation across the image
# frame:Image -> Deque
def process_frame(frame):

    # Camera already comes in pre-shrunk
    if INPUT == 'camera':
        ready_frame = blur(convert_color(frame))
    else:
        ready_frame = blur(shrink(convert_color(frame)))

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
def wrap_and_display_to_leds(image, strip):
    for led_strip_index in xrange(WIDTH):
        for led_length_index in xrange(HEIGHT):
            value = image[led_length_index][led_strip_index]
            if led_strip_index % 2 == 0:
                pixel_index = HEIGHT * led_strip_index + led_length_index
            else:
                pixel_index = HEIGHT * led_strip_index + (HEIGHT - led_length_index)
            strip.setPixelColor(pixel_index, value[0], value[1], value[2])
    strip.show()


# Renders to LEDs or screen
def render_bitmap(bitmap, strip):
    np_bitmap = np.zeros((HEIGHT, WIDTH, 3), np.uint8)
    for row_idx, row in enumerate(bitmap):
        for column_idx, column in enumerate(row):
            np_bitmap[row_idx][column_idx] = bitmap[row_idx][column_idx]

    rgb_image = cv2.cvtColor(np_bitmap, cv2.COLOR_HSV2RGB)
    b,g,r = cv2.split(rgb_image)
    img = cv2.merge((g,r,b))

    if MODE == 'pi':
        wrap_and_display_to_leds(img, strip)
    else:
        cv2.imshow('preview', enlarge(rgb_image))

# Enlarges the image. Used for drawing pixels on computer for debugging
def enlarge(image):
    return cv2.resize(image,None,fx=40, fy=3, interpolation = cv2.INTER_AREA)


# Shrinks image to 30% of original size. Makes faster and more accurate
# image:Image -> Image
def shrink(image):
    return cv2.resize(image,None,fx=0.3, fy=0.3, interpolation = cv2.INTER_AREA)


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

    if MODE == 'pi':
        from dotstar import Adafruit_DotStar
        strip = Adafruit_DotStar(HEIGHT*WIDTH, DATAPIN, CLOCKPIN)
        strip.begin()

        # Lower power consumption, but makes it flicker.
        strip.setBrightness(LED_GLOBAL_BRIGHTNESS)

    bitmap = initialize_empty_bitmap()
    render_bitmap(bitmap, strip)

    # INPUT SELECTION SETUP
    if INPUT == 'camera':
        if MODE == 'debug':
            cv2.namedWindow('cameraPreview', cv2.WINDOW_NORMAL)
        vc = cv2.VideoCapture(0)
        if vc.isOpened():
            vc.set(3,200) # These aren't accurate, but help
            vc.set(4,100)
            rval, frame = vc.read()
        else:
            rval = False

        while rval:
            rval, frame = vc.read()
            if MODE == 'debug':
                cv2.imshow('cameraPreview', frame)
            flow(frame, bitmap, strip)
            key = cv2.waitKey(15)
            if key == 27: # exit on ESC
                break

    if INPUT == 'image':
        if len(sys.argv) == 4:
            frame = cv2.imread(sys.argv[3])
        else:
            frame = cv2.imread('bars.jpg')
        rval = True

        while True:
            flow(frame, bitmap, strip)
            key = cv2.waitKey(15)
            if key == 27: # exit on ESC
                break

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


if __name__ == '__main__':
   main(sys.argv[1:])
