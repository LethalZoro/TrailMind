import gpiozero
from gpiozero import LED
from time import sleep

# Use the default pin factory (RPi.GPIO) for real hardware
led = LED(17)

try:
    while True:
        led.on()
        sleep(1)
        led.off()
        sleep(1)
except KeyboardInterrupt:
    led.off()  # Turn off the LED when the program is interrupted
    print("Program stopped")