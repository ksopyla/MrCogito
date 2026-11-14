"""E33n copy split: each position is none / near / far by where its (k+1)-gram last occurred."""
from evaluation.copy_split import copy_categories


def test_near_far_none():
    W, k = 10, 2
    # 5 6 7 occurs at 0..2 and again at 3..5 (near), and 1 2 3 at 6..8 then again at 22..24 (far: 16 tokens back)
    ids = [5, 6, 7, 5, 6, 7, 1, 2, 3] + list(range(100, 113)) + [1, 2, 3]
    cat = copy_categories(ids, k, W).tolist()
    assert cat[:5] == [0, 0, 0, 0, 0]          # first occurrences / (5,6,7)'s 3-gram (7,5,6) is new
    assert cat[5] == 1                          # (5,6,7) ended at 2, now at 5: 3 back <= W - k
    assert cat[24] == 2                         # (1,2,3) ended at 8, now at 24: 16 back > W - k
    assert cat[22] == 0 and cat[23] == 0        # (112,1,2)/(…) never seen before: none
    assert sum(c == 1 for c in cat) == 1 and sum(c == 2 for c in cat) == 1


def test_boundary_is_window_minus_k():
    W, k = 8, 2
    ids = [1, 2, 3] + [50 + i for i in range(3)] + [1, 2, 3]  # (1,2,3) ends at 2, again at 8: 6 back = W - k → near
    assert copy_categories(ids, k, W)[8] == 1
    ids = [1, 2, 3] + [50 + i for i in range(4)] + [1, 2, 3]  # 7 back > W - k → far
    assert copy_categories(ids, k, W)[9] == 2
