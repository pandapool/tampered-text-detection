import numpy as np

from eki import boxes as B


def test_mask_roundtrip():
    m = np.zeros((504, 504), bool)
    m[10:20, 30:60] = True
    m[100:140, 5:9] = True
    m[300, 300] = True  # 1 px noise, dropped
    bx = B.mask_to_boxes(m, min_area=4)
    assert bx == [[30, 10, 60, 20], [5, 100, 9, 140]]
    assert B.pixel_iou_f1(B.rasterise(bx, 504), m & ~np.pad(np.ones((1, 1), bool), ((300, 203), (300, 203))))[0] == 1.0


def test_format_parse():
    bx = [[1, 2, 3, 4], [10, 20, 30, 40]]
    s = B.format_answer(bx)
    assert s == "Tampered. [1, 2, 3, 4], [10, 20, 30, 40]"
    assert B.parse_answer(s, 504) == (bx, True)
    assert B.parse_answer("Real.", 504) == ([], True)
    assert B.parse_answer("garbage", 504) == ([], False)
    assert B.parse_answer("Tampered. [600, 5, 2, 9]", 504) == ([[2, 5, 504, 9]], True)


def test_metrics():
    e = np.zeros((8, 8), bool)
    assert B.pixel_iou_f1(e, e) == (1.0, 1.0)
    p = e.copy(); p[0, 0] = True
    assert B.pixel_iou_f1(p, e) == (0.0, 0.0)
    g = e.copy(); g[0, :2] = True
    assert B.pixel_iou_f1(p, g) == (0.5, 2 / 3)
    assert B.box_iou_union([], [[0, 0, 2, 2]], 8) == 0.0


def test_merge():
    assert B.merge_close([[0, 0, 5, 5], [7, 0, 9, 5]], 2) == [[0, 0, 9, 5]]
    assert B.merge_close([[0, 0, 5, 5], [9, 0, 12, 5]], 2) == [[0, 0, 5, 5], [9, 0, 12, 5]]
