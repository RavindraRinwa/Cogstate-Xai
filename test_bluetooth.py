import serial
import time

ports = [
    "/dev/cu.MindWaveMobile",
    "/dev/tty.MindWaveMobile"
]

for port in ports:
    print(f"\nTesting {port}")

    try:
        ser = serial.Serial(port, 57600, timeout=2)

        start = time.time()
        total = 0

        while time.time() - start < 10:
            data = ser.read(128)

            if data:
                total += len(data)
                print(data.hex())

        print("Total bytes:", total)
        ser.close()

    except Exception as e:
        print("Error:", e)