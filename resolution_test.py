#!/usr/bin/python
import cv2
import sys

def main(argv):
  camera= cv2.VideoCapture(0)
  camera.set(cv2.CAP_PROP_AUTOFOCUS, False)
  print("FPS: ", int(camera.get(cv2.CAP_PROP_FPS)))
  print(camera.get(cv2.CAP_PROP_CONVERT_RGB))


  for i in range(120, 1920):
    result = camera.set(3, (i))
    print(result, float(i))

if __name__ == '__main__':
   main(sys.argv[1:])
