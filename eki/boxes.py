"""Mask <-> box conversion, answer formatting/parsing, and pixel IoU/F1 (paper Sec. III-A, V-A5).

Boxes are [x1, y1, x2, y2] in absolute integer pixels of the square img_size frame, with x2/y2
exclusive, so rasterising a box covers exactly the pixels of the rectangle it came from.
"""
import re

import numpy as np
from skimage.measure import label, regionprops

REAL = "Real."
_QUAD = re.compile(r"\[\s*(-?\d+)\s*,\s*(-?\d+)\s*,\s*(-?\d+)\s*,\s*(-?\d+)\s*\]")


def mask_to_boxes(mask, min_area=4, merge_gap=0, max_boxes=None):
    """Binary mask -> min enclosing rectangles of its connected components (8-connectivity)."""
    lab = label(np.asarray(mask) > 0, connectivity=2)
    boxes = []
    for r in regionprops(lab):
        if r.area < min_area:
            continue
        y1, x1, y2, x2 = r.bbox
        boxes.append([int(x1), int(y1), int(x2), int(y2)])
    if merge_gap > 0:
        boxes = merge_close(boxes, merge_gap)
    boxes = sort_boxes(boxes)
    if max_boxes is not None and len(boxes) > max_boxes:
        # keep the largest, then restore reading order
        boxes = sorted(boxes, key=area, reverse=True)[:max_boxes]
        boxes = sort_boxes(boxes)
    return boxes


def area(b):
    return max(0, b[2] - b[0]) * max(0, b[3] - b[1])


def sort_boxes(boxes):
    return sorted(boxes, key=lambda b: (b[1], b[0]))


def merge_close(boxes, gap):
    boxes = [list(b) for b in boxes]
    merged = True
    while merged:
        merged = False
        for i in range(len(boxes)):
            for j in range(i + 1, len(boxes)):
                a, b = boxes[i], boxes[j]
                if (a[0] - gap <= b[2] and b[0] - gap <= a[2]
                        and a[1] - gap <= b[3] and b[1] - gap <= a[3]):
                    boxes[i] = [min(a[0], b[0]), min(a[1], b[1]), max(a[2], b[2]), max(a[3], b[3])]
                    del boxes[j]
                    merged = True
                    break
            if merged:
                break
    return boxes


def fmt_box(b):
    return "[{}, {}, {}, {}]".format(*b)


def format_answer(boxes):
    """Target text: "Tampered. [x1, y1, x2, y2], [...]" or "Real."."""
    if not boxes:
        return REAL
    return "Tampered. " + ", ".join(fmt_box(b) for b in boxes)


def parse_answer(text, size):
    """Model output -> (boxes, ok). Unparseable output is treated as "Real." with ok=False."""
    t = text.strip()
    quads = _QUAD.findall(t)
    if t.lower().startswith("real") and not quads:
        return [], True
    if not quads:
        return [], False
    boxes = []
    for q in quads:
        x1, y1, x2, y2 = (min(max(int(v), 0), size) for v in q)
        x1, x2 = sorted((x1, x2))
        y1, y2 = sorted((y1, y2))
        if x2 > x1 and y2 > y1:
            boxes.append([x1, y1, x2, y2])
    return boxes, True


def rasterise(boxes, size):
    m = np.zeros((size, size), dtype=bool)
    for x1, y1, x2, y2 in boxes:
        m[y1:y2, x1:x2] = True
    return m


def pixel_iou_f1(pred, gt):
    """Per-image pixel IoU and F1 of two binary masks.

    Empty GT and empty prediction count as IoU = F1 = 1; any false positive on an empty GT
    (or a miss on a tampered image) gives 0.
    """
    p, g = np.asarray(pred, bool), np.asarray(gt, bool)
    ps, gs = p.sum(), g.sum()
    if ps == 0 and gs == 0:
        return 1.0, 1.0
    inter = np.logical_and(p, g).sum()
    union = ps + gs - inter
    return float(inter / union), float(2 * inter / (ps + gs))


def box_iou_union(pred_boxes, gt_boxes, size):
    """Pixel IoU between the unions of two box sets (hard-sample criterion, no boxes = 0)."""
    if not pred_boxes:
        return 0.0 if gt_boxes else 1.0
    return pixel_iou_f1(rasterise(pred_boxes, size), rasterise(gt_boxes, size))[0]


def intersect(a, b):
    x1, y1 = max(a[0], b[0]), max(a[1], b[1])
    x2, y2 = min(a[2], b[2]), min(a[3], b[3])
    return [x1, y1, x2, y2] if x2 > x1 and y2 > y1 else None
