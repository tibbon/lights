To run Pixelpusher, open tabs with the following:

- `redis-server`
- `redis-cli monitor`
- `python lights.py camera q1` for quadrant 1
- `python blitter.py`

## Initial Python setup

OpenCV will probably give installation problems on OS X.

Install Requirements:
`pip install -r requirements.txt`
`pip freeze >> requirements.txt `

## Debug Pixelpusher via serial.

1) Hook up via USB cable
2) Run `screen /dev/tty.usbmodem12341 115200`

## Updating Pixelpusher firmware:

`./configtool /dev/tty.usbmodem12341 ./pixel.rc`

## Network Layout

- 192.168.1.1, Router
- 192.168.1.100, Pixelpusher
- 192.168.1.101, rPi Redis and Blitter
- 192.168.1.102, rPi quadrant 1
- 192.168.1.103, rPi quadrant 2
- 192.168.1.104, rPi quadrant 3
- 192.168.1.105, rPi quadrant 4

