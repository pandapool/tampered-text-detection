"""Phase 3a: PaddleOCR over the training subset + validation ids. Run in the `eki-ocr` env.

  conda run -n eki-ocr python scripts/p3_ocr.py [--device gpu]

OCR runs at original resolution; boxes are rescaled into the img_size frame. Writes ocr.jsonl:
{"id", "lines": [{"text", "conf", "box": [x1, y1, x2, y2]}]} (all lines; the conf filter is applied
in p3_tf.py). Resumes by skipping ids already in the file.
"""
import json

import numpy as np
from tqdm import tqdm

from eki.config import cli
from eki.data import Splits, read_ids, read_jsonl


def make_ocr(device, lang, det=None, rec=None):
    from paddleocr import PaddleOCR
    try:  # PaddleOCR >= 3
        # oneDNN is off: paddlepaddle 3.3's oneDNN path fails on PP-OCRv5 (PirAttribute error)
        models = {k: v for k, v in (("text_detection_model_name", det), ("text_recognition_model_name", rec)) if v}
        o = PaddleOCR(lang=None if models else lang, device=device, use_doc_orientation_classify=False,
                      use_doc_unwarping=False, use_textline_orientation=False, enable_mkldnn=False, **models)
        return lambda img: _v3(o.predict(img))
    except TypeError:  # PaddleOCR 2.x
        o = PaddleOCR(lang=lang, use_angle_cls=False, use_gpu=device != "cpu", show_log=False)
        return lambda img: _v2(o.ocr(img, cls=False))


def _v3(res):
    out = []
    for r in res:
        for t, s, b in zip(r["rec_texts"], r["rec_scores"], r["rec_boxes"]):
            out.append((t, float(s), [float(v) for v in b]))
    return out


def _v2(res):
    out = []
    for page in res or []:
        for poly, (t, s) in page or []:
            xs, ys = [p[0] for p in poly], [p[1] for p in poly]
            out.append((t, float(s), [min(xs), min(ys), max(xs), max(ys)]))
    return out


def main():
    cfg, a = cli(__doc__, lambda p: (p.add_argument("--device", default="cpu"),
                                     p.add_argument("--lang", default="ch")))
    S, size = Splits(cfg), cfg.data.img_size
    out = cfg.work / "ocr.jsonl"
    ids = read_ids(cfg.work / "subset_ids.txt") + read_ids(cfg.work / "val_ids.txt")
    done = {r["id"] for r in read_jsonl(out)} if out.exists() else set()
    ocr = make_ocr(a.device, a.lang, cfg.ocr.get("det_model"), cfg.ocr.get("rec_model"))
    with open(out, "a") as f:
        for sid in tqdm([i for i in ids if i not in done], desc="ocr"):
            split, idx = sid.rsplit(":", 1)
            img, _ = S.db(split).raw(int(idx))
            W, H = img.size
            bgr = np.asarray(img)[:, :, ::-1].copy()
            lines = []
            for t, s, (x1, y1, x2, y2) in ocr(bgr):
                b = [round(x1 * size / W), round(y1 * size / H), round(x2 * size / W), round(y2 * size / H)]
                b = [min(max(v, 0), size) for v in b]
                if b[2] > b[0] and b[3] > b[1]:
                    lines.append({"text": t, "conf": round(s, 4), "box": b})
            f.write(json.dumps({"id": sid, "lines": lines}, ensure_ascii=False) + "\n")


if __name__ == "__main__":
    main()
