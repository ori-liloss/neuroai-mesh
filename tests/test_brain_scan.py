"""brain_scan's low-rank norms against the full matrices built explicitly (needs numpy + safetensors)."""
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import brain_scan as bs  # noqa: E402

rng = np.random.default_rng(7)
passed = 0


def ok(cond, label):
    global passed
    if not cond:
        raise AssertionError(label)
    passed += 1
    print("ok  ", label)


r, out_dim, in_dim = 16, 256, 384
a, b = rng.normal(size=(r, in_dim)), rng.normal(size=(out_dim, r))
full = b @ a
ok(np.isclose(bs.total_sq(a, b), np.sum(full ** 2)), "total: ||BA||^2 from r x r products equals the full matrix")
e_r, e_c = bs.edges(out_dim), bs.edges(in_dim)
rows = np.array([np.sum(full[e_r[i]:e_r[i + 1]] ** 2) for i in range(bs.GROUPS)])
cols = np.array([np.sum(full[:, e_c[j]:e_c[j + 1]] ** 2) for j in range(bs.GROUPS)])
ok(np.allclose(bs.row_sq(a, b), rows), "row groups equal the full matrix's row-group norms")
ok(np.allclose(bs.col_sq(a, b), cols), "column groups equal the full matrix's column-group norms")
blocks = np.array([[np.sum(full[e_r[i]:e_r[i + 1], e_c[j]:e_c[j + 1]] ** 2) for j in range(bs.GROUPS)] for i in range(bs.GROUPS)])
ok(np.allclose(bs.block_sq(a, b), blocks), "32 x 32 block norms equal the full matrix's blocks")
ok(np.isclose(bs.block_sq(a, b).sum(), np.sum(full ** 2)), "the blocks add up to the total")
# A stage's own change: cur - prev, as one low-rank pair of factors.
a0, b0 = rng.normal(size=(r, in_dim)), rng.normal(size=(out_dim, r))
key = (0, "mlp", "up_proj")
fa = bs.factors({key: (a, b)}, 2.0, {key: (a0, b0)}, 2.0, key)
diff = 2.0 * (b @ a) - 2.0 * (b0 @ a0)
ok(np.allclose(fa[1] @ fa[0], diff) and np.isclose(bs.total_sq(*fa), np.sum(diff ** 2)),
   "the stage's own change equals (cur - prev) built in full, scaling included")
ok(np.isclose(bs.total_sq(*bs.factors({key: (a, b)}, 2.0, None, 1.0, key)), np.sum((2.0 * b @ a) ** 2)),
   "the total change equals the scaled adapter built in full")
print(f"\nALL {passed} CHECKS PASSED")
