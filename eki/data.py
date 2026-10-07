"""DocTamper LMDB reader and the jsonl artefacts shared between phases.

DocTamper stores ``num-samples``, ``image-%09d`` (JPEG bytes) and ``label-%09d`` (PNG mask,
nonzero = tampered), indexed from 0. Sample ids are ``"<split>:<index>"``.
"""
import io
import json
from pathlib import Path

import lmdb
import numpy as np
from PIL import Image


class DocTamper:
    def __init__(self, path):
        self.path = str(path)
        self._env = None
        env = self._open()
        with env.begin() as txn:
            self.n = int(txn.get(b"num-samples"))
        self.split = Path(path).name

    def _open(self):
        # opened lazily so the object can be pickled into DataLoader workers
        if self._env is None:
            self._env = lmdb.open(self.path, readonly=True, lock=False, readahead=False, meminit=False)
        return self._env

    def __getstate__(self):
        return {**self.__dict__, "_env": None}

    def __len__(self):
        return self.n

    def ids(self):
        return [f"{self.split}:{i}" for i in range(self.n)]

    def raw(self, idx):
        with self._open().begin() as txn:
            img = Image.open(io.BytesIO(txn.get(b"image-%09d" % idx))).convert("RGB")
            m = txn.get(b"label-%09d" % idx)
        mask = np.zeros(img.size[::-1], bool) if m is None else \
            np.array(Image.open(io.BytesIO(m)).convert("L")) > 0
        return img, mask

    def get(self, idx, size):
        """Image and mask resized together to size x size (nearest-neighbour for the mask)."""
        img, mask = self.raw(idx)
        img = img.resize((size, size), Image.BICUBIC)
        mask = np.array(Image.fromarray(mask.astype(np.uint8) * 255).resize((size, size), Image.NEAREST)) > 0
        return img, mask


class Splits:
    """Lazy registry of all LMDBs under data.root, addressed by sample id."""

    def __init__(self, cfg):
        self.root = cfg.data_root
        self.size = cfg.data.img_size
        self._dbs = {}

    def db(self, split):
        if split not in self._dbs:
            self._dbs[split] = DocTamper(self.root / split)
        return self._dbs[split]

    def get(self, sid):
        split, idx = sid.rsplit(":", 1)
        return self.db(split).get(int(idx), self.size)

    def image(self, sid):
        return self.get(sid)[0]


def read_jsonl(path):
    with open(path) as f:
        return [json.loads(line) for line in f if line.strip()]


def write_jsonl(path, rows):
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w") as f:
        for r in rows:
            f.write(json.dumps(r) + "\n")


def read_ids(path):
    return [l.strip() for l in open(path) if l.strip()]


def write_ids(path, ids):
    Path(path).write_text("\n".join(ids) + "\n")
