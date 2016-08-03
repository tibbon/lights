To run Pixelpusher, open tabs with the following:

- `redis-server`
- `redis-cli monitor`
- `python lights.py camera`
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

