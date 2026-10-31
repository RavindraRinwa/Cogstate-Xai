
import argparse
import glob
import time
import numpy as np

SYNC = 0xAA
RAW = 0x80
POOR = 0x02
ATTENTION = 0x04
MEDITATION = 0x05


class ThinkGearParser:

    def __init__(self):
        self.buf = bytearray()
        self.poor = 200
        self.attention = None
        self.meditation = None

        self.bytes_received = 0
        self.valid_packets = 0
        self.bad_checksums = 0
        self.raw_samples = 0
        self.poor_updates = 0

    def feed(self, data):
        self.bytes_received += len(data)
        self.buf.extend(data)

        raw = []

        while True:
            i = self.buf.find(b"\xaa\xaa")

            if i < 0:
                self.buf = self.buf[-1:]
                break

            del self.buf[:i]

            if len(self.buf) < 4:
                break

            n = self.buf[2]

            if n >= 170:
                del self.buf[0]
                continue

            if len(self.buf) < 4 + n:
                break

            payload = bytes(self.buf[3:3 + n])
            chk = self.buf[3 + n]

            del self.buf[:4 + n]

            if ((~sum(payload)) & 0xFF) == chk:
                self.valid_packets += 1
                samples = self._payload(payload)
                self.raw_samples += len(samples)
                raw.extend(samples)
            else:
                self.bad_checksums += 1

        return raw

    def _payload(self, p):
        raw = []
        i = 0

        while i < len(p):

            while i < len(p) and p[i] == 0x55:
                i += 1

            if i >= len(p):
                break

            code = p[i]
            i += 1

            if code >= 0x80:

                if i >= len(p):
                    break

                n = p[i]
                i += 1

                if i + n > len(p):
                    break

                val = p[i:i + n]
                i += n

                if code == RAW and n == 2:
                    v = (val[0] << 8) | val[1]

                    if v >= 32768:
                        v -= 65536

                    raw.append(v)

            else:

                if i >= len(p):
                    break

                v = p[i]
                i += 1

                if code == POOR:
                    self.poor = v
                    self.poor_updates += 1

                elif code == ATTENTION:
                    self.attention = v

                elif code == MEDITATION:
                    self.meditation = v

        return raw


class MindWaveEEG:

    fs = 512.0

    def __init__(self, port="", baud=57600):

        import serial

        if not port:

            candidates = (
                glob.glob("/dev/cu.*")
                + glob.glob("/dev/ttyUSB*")
                + glob.glob("/dev/rfcomm*")
            )

            found = [
                p for p in candidates
                if any(k in p.lower() for k in ("mindwave", "neurosky"))
            ]

            if not found:
                raise SystemExit(
                    "MindWave serial port not found.\n"
                    "Available ports:\n" + "\n".join(candidates)
                )

            port = found[0]

        print("MindWave port:", port)
        print("Baud rate:", baud)

        self.ser = serial.Serial(
            port=port,
            baudrate=baud,
            timeout=0.05
        )

        self.parser = ThinkGearParser()

    @property
    def poor(self):
        return self.parser.poor

    def read(self):

        waiting = self.ser.in_waiting

        data = self.ser.read(waiting if waiting else 1)

        if not data:
            return np.empty((0, 1), dtype=np.float32)

        raw = self.parser.feed(data)

        return np.asarray(raw, dtype=np.float32).reshape(-1, 1)

    def close(self):
        self.ser.close()


def main():

    ap = argparse.ArgumentParser()

    ap.add_argument("--port", default="")
    ap.add_argument("--baud", type=int, default=57600)
    ap.add_argument("--secs", type=float, default=15)

    args = ap.parse_args()

    dev = MindWaveEEG(args.port, args.baud)

    samples = []
    start = time.time()

    print("\nCollecting EEG data...\n")

    try:

        while time.time() - start < args.secs:

            data = dev.read()

            if len(data):
                samples.extend(data[:, 0])

            time.sleep(0.005)

    except KeyboardInterrupt:
        print("\nCollection interrupted.")

    finally:
        elapsed = time.time() - start
        parser = dev.parser

        print("\n========== EEG DIAGNOSTICS ==========")

        print("Elapsed time:", round(elapsed, 2), "seconds")
        print("Bytes received:", parser.bytes_received)
        print("Valid packets:", parser.valid_packets)
        print("Bad checksums:", parser.bad_checksums)
        print("Raw samples received:", parser.raw_samples)
        print("Poor signal updates:", parser.poor_updates)
        print("Current poor signal:", parser.poor)

        if elapsed > 0:
            print(
                "Raw sampling rate:",
                round(len(samples) / elapsed, 2),
                "Hz"
            )

        if len(samples):

            x = np.asarray(samples)

            print("\n========== RAW EEG SUMMARY ==========")

            print("Sample count:", len(x))
            print("Minimum:", x.min())
            print("Maximum:", x.max())
            print("Mean:", round(float(x.mean()), 2))
            print("Standard deviation:", round(float(x.std()), 2))

            np.savetxt(
                "mindwave_eeg.csv",
                x,
                delimiter=",",
                header="raw_eeg",
                comments=""
            )

            print("\nSaved EEG data to mindwave_eeg.csv")

        else:
            print("\nNo raw EEG samples received.")

        if parser.bytes_received == 0:
            print("Diagnosis: No serial bytes received.")

        elif parser.valid_packets == 0:
            print("Diagnosis: No valid ThinkGear packets.")

        elif parser.raw_samples == 0:
            print("Diagnosis: Valid packets received, but no raw EEG.")

        elif parser.poor == 200:
            print("Diagnosis: Raw EEG received; sensor contact is poor.")

        else:
            print("Diagnosis: Raw EEG acquisition is working.")

        dev.close()


if __name__ == "__main__":
    main()
