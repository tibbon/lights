#!/usr/bin/python
import cv2
import sys

def main(argv):
  camera= cv2.VideoCapture(0)

  for i in range(120, 1920):
    result = camera.set(cv2.CAP_PROP_FRAME_WIDTH, i)
    print(result, ": ", i)

if __name__ == '__main__':
   main(sys.argv[1:])
