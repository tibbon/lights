#!/usr/bin/python
import cv2
import sys

def main(argv):
  print('OpenCV version: ', cv2.__version__)
  camera= cv2.VideoCapture(0)
  print("format ", camera.set(cv2.CAP_PROP_CONVERT_RGB, 0))
  camera.set(cv2.CAP_PROP_AUTOFOCUS, False)
  print("FPS: ", int(camera.get(cv2.CAP_PROP_FPS)))
  print(camera.get(cv2.CAP_PROP_CONVERT_RGB))

  for x in range(0, 40):
    print("prop ",x , camera.get(x))

  for x in range(0, 40):
    print("prop set: ",x , camera.set(x, 0))

  for i in range(120, 1920):
    result = camera.set(3, (i))
    print(result, float(i))

if __name__ == '__main__':
   main(sys.argv[1:])
