"""Level-invariance probe: write copies of one WAV peak-normalized to a set
of dBFS targets, for wake_test --wav. 16-bit mono PCM in/out."""
import struct
import sys
import wave

src = sys.argv[1]
out_prefix = sys.argv[2]
targets_db = [float(a) for a in sys.argv[3:]] or [-40.0, -20.0, -6.0]

with wave.open(src, "rb") as w:
    assert w.getnchannels() == 1 and w.getsampwidth() == 2
    rate = w.getframerate()
    n = w.getnframes()
    pcm = struct.unpack(f"<{n}h", w.readframes(n))

peak = max(abs(s) for s in pcm) / 32768.0
for db in targets_db:
    g = (10.0 ** (db / 20.0)) / peak
    scaled = [max(-32768, min(32767, round(s * g))) for s in pcm]
    path = f"{out_prefix}_{int(-db)}db.wav"
    with wave.open(path, "wb") as w:
        w.setnchannels(1)
        w.setsampwidth(2)
        w.setframerate(rate)
        w.writeframes(struct.pack(f"<{n}h", *scaled))
    print(path)
