"""Write a small synthetic dataset in DocTamper's LMDB format, for smoke-testing every phase.

Each image is a page of random text lines. Tampered pages have a few words re-rendered with a
slightly different font, size or ink colour after a JPEG round trip, as a crude stand-in for
DocTamper's copy-move/splicing/generation forgeries. The mask covers the re-rendered glyphs.
Not a benchmark: it only checks that the pipeline runs end to end.
"""
import argparse
import io
import random
from pathlib import Path

import lmdb
import numpy as np
from PIL import Image, ImageDraw, ImageFont

FONTS = [
    "/usr/share/fonts/truetype/liberation/LiberationSans-Regular.ttf",
    "/usr/share/fonts/truetype/liberation/LiberationSerif-Regular.ttf",
    "/usr/share/fonts/truetype/liberation/LiberationMono-Regular.ttf",
    "/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf",
]
WORDS = ("invoice total amount date account number payment due balance customer order "
         "receipt tax subtotal reference address phone bank transfer 2024 2025 1,250.00 "
         "38.90 #4471 USD EUR approved signature contract clause section page").split()


def fonts():
    fs = [f for f in FONTS if Path(f).exists()]
    if not fs:
        raise SystemExit("no TTF fonts found; edit FONTS in make_synthetic.py")
    return fs


def page(rng, fs, tampered):
    w, h = rng.choice([(768, 1024), (1024, 768), (900, 900), (640, 880)])
    img = Image.new("RGB", (w, h), tuple(rng.randint(235, 255) for _ in range(3)))
    d = ImageDraw.Draw(img)
    font_path, size = rng.choice(fs), rng.randint(16, 26)
    font = ImageFont.truetype(font_path, size)
    ink = tuple(rng.randint(0, 50) for _ in range(3))
    words = []  # (x, y, text)
    y = rng.randint(20, 50)
    while y < h - 2 * size:
        x = rng.randint(20, 60)
        while x < w - 120:
            t = rng.choice(WORDS)
            d.text((x, y), t, font=font, fill=ink)
            words.append((x, y, t))
            x += int(d.textlength(t + " ", font=font))
        y += int(size * rng.uniform(1.4, 2.2))

    buf = io.BytesIO()
    img.save(buf, "JPEG", quality=rng.randint(75, 95))
    img = Image.open(buf).convert("RGB")
    mask = np.zeros((h, w), np.uint8)
    if tampered and words:
        d = ImageDraw.Draw(img)
        for x, y, t in rng.sample(words, k=min(len(words), rng.randint(1, 3))):
            l, top, r, b = d.textbbox((x, y), t, font=font)
            d.rectangle([l - 1, top - 1, r + 1, b + 1], fill=img.getpixel((max(l - 3, 0), top)))
            new = rng.choice(WORDS)
            f2 = ImageFont.truetype(rng.choice(fs), size + rng.choice([-1, 0, 1]))
            ink2 = tuple(min(255, c + rng.randint(5, 40)) for c in ink)
            d.text((x, y), new, font=f2, fill=ink2)
            l2, t2, r2, b2 = d.textbbox((x, y), new, font=f2)
            mask[min(top, t2):max(b, b2), min(l, l2):max(r, r2)] = 255
    return img, mask


def write(path, n, rng, fs, tampered_frac):
    path.mkdir(parents=True, exist_ok=True)
    env = lmdb.open(str(path), map_size=1 << 32)
    with env.begin(write=True) as txn:
        for i in range(n):
            img, mask = page(rng, fs, rng.random() < tampered_frac)
            b = io.BytesIO(); img.save(b, "JPEG", quality=95)
            txn.put(b"image-%09d" % i, b.getvalue())
            b = io.BytesIO(); Image.fromarray(mask).save(b, "PNG")
            txn.put(b"label-%09d" % i, b.getvalue())
        txn.put(b"num-samples", str(n).encode())
    env.close()
    print(f"wrote {n} samples to {path}")


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--out", default="data/synthetic")
    p.add_argument("--n_train", type=int, default=240)
    p.add_argument("--n_test", type=int, default=40)
    p.add_argument("--seed", type=int, default=0)
    a = p.parse_args()
    rng, fs, out = random.Random(a.seed), fonts(), Path(a.out)
    write(out / "TrainingSet", a.n_train, rng, fs, 0.9)
    for split in ["TestingSet", "FCD", "SCD"]:
        write(out / split, a.n_test, rng, fs, 0.9)


if __name__ == "__main__":
    main()
