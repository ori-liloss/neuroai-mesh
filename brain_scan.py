"""Measure what training changed inside the model, from the LoRA adapter files, for the dashboard's
Brain page. Exact and cheap: a LoRA change is dW = s * B @ A with rank r (16), so every norm below is
computed from r x r products - the full 4096 x 12288 matrices are never built.

    python -I brain_scan.py --runs D:\\neuro-models\\runs --plan D:\\neuro-models\\training-plan.json --out brain

For each stage of the plan it reads runs/neuro-<stage>/adapter (or the newest checkpoint-N while the
stage still trains) and writes <out>/<stage>.json with two modes:
  "stage": what this stage changed by itself = its adapter minus the previous stage's adapter (each
           stage starts from the one before, so an adapter holds the total change since the base model;
           the difference of two low-rank changes is again low rank, 2r, so it stays exact);
  "total": the change since the original base model.
Per layer: 32 groups of the 4096 residual-stream dimensions ("nodes": how much the layer's writes
into those dimensions changed; "reads": how much its reads from them changed), totals for attention
and MLP, and 32 x 32 block maps of the MLP and attention changes for the detail view. Values are
Frobenius norms (square root of the summed squares). Needs numpy and safetensors; no GPU.
"""
from __future__ import annotations

import argparse
import json
import os
import re
import sys
import time
from pathlib import Path

import numpy as np
from safetensors import safe_open

GROUPS = 32
KEY = re.compile(r"layers\.(\d+)\.(self_attn|linear_attn|mlp)\.(\w+)\.lora_([AB])\.weight$")
WRITERS = {"o_proj", "out_proj", "down_proj"}           # write into the residual stream
ATTN_READERS = ["q_proj", "k_proj", "v_proj", "in_proj_qkv", "in_proj_z", "in_proj_a", "in_proj_b"]
MLP_READERS = ["gate_proj", "up_proj"]


def edges(n: int, groups: int = GROUPS) -> np.ndarray:
    return np.linspace(0, n, groups + 1).round().astype(int)


def load(path: Path) -> dict:
    """{(layer, block, module): (A [r, in], B [out, r])} in float64."""
    parts: dict = {}
    with safe_open(str(path), "np") as handle:
        for key in handle.keys():
            match = KEY.search(key)
            if match:
                parts.setdefault((int(match[1]), match[2], match[3]), {})[match[4]] = handle.get_tensor(key).astype(np.float64)
    return {k: (v["A"], v["B"]) for k, v in parts.items() if "A" in v and "B" in v}


def scaling(adapter_dir: Path) -> float:
    config = json.loads((adapter_dir / "adapter_config.json").read_text(encoding="utf-8"))
    return float(config.get("lora_alpha", 16)) / float(config.get("r", 8))


def factors(cur: dict, s_cur: float, prev: dict | None, s_prev: float, key) -> tuple[np.ndarray, np.ndarray] | None:
    """A, B with dW = B @ A: the stage's own change (cur - prev) or the total (prev None)."""
    if key not in cur:
        return None
    a, b = cur[key]
    b = b * s_cur
    if prev is not None and key in prev:
        a0, b0 = prev[key]
        a = np.concatenate([a, a0], 0)
        b = np.concatenate([b, -b0 * s_prev], 1)
    return a, b


def total_sq(a: np.ndarray, b: np.ndarray) -> float:
    # ||B A||_F^2 = sum((B^T B) * (A A^T)) for the symmetric r x r Gram matrices.
    return float(np.sum((b.T @ b) * (a @ a.T)))


def row_sq(a: np.ndarray, b: np.ndarray, groups: int = GROUPS) -> np.ndarray:
    """Squared norms of the output-row groups of B A."""
    per_row = np.einsum("ir,rs,is->i", b, a @ a.T, b)
    e = edges(b.shape[0], groups)
    return np.array([per_row[e[i]:e[i + 1]].sum() for i in range(groups)])


def col_sq(a: np.ndarray, b: np.ndarray, groups: int = GROUPS) -> np.ndarray:
    """Squared norms of the input-column groups of B A."""
    btb = b.T @ b
    e = edges(a.shape[1], groups)
    return np.array([float(np.sum(btb * (a[:, e[j]:e[j + 1]] @ a[:, e[j]:e[j + 1]].T))) for j in range(groups)])


def block_sq(a: np.ndarray, b: np.ndarray, row_groups: int = GROUPS, col_groups: int = GROUPS) -> np.ndarray:
    """[row group, col group] squared norms of B A: sum((B_i^T B_i) * (A_j A_j^T))."""
    er, ec = edges(b.shape[0], row_groups), edges(a.shape[1], col_groups)
    g = np.stack([(b[er[i]:er[i + 1]].T @ b[er[i]:er[i + 1]]).ravel() for i in range(row_groups)])
    h = np.stack([(a[:, ec[j]:ec[j + 1]] @ a[:, ec[j]:ec[j + 1]].T).ravel() for j in range(col_groups)])
    return g @ h.T


def r4(values) -> list:
    return [float(f"{float(v):.4g}") for v in np.asarray(values).ravel()]


def scan(cur: dict, s_cur: float, prev: dict | None, s_prev: float) -> dict:
    layers = sorted({k[0] for k in cur})
    out = []
    for layer in layers:
        keys = {(blk, mod): (layer, blk, mod) for (lay, blk, mod) in cur if lay == layer}
        block = "linear_attn" if any(blk == "linear_attn" for blk, _ in keys) else "self_attn"
        get = lambda mod, blk=None: factors(cur, s_cur, prev, s_prev, (layer, blk or (block if mod != "gate_proj" and mod not in {"up_proj", "down_proj"} else "mlp"), mod))  # noqa: E731
        attn_sq = mlp_sq = 0.0
        nodes = np.zeros(GROUPS)
        reads = np.zeros(GROUPS)
        modules = {}
        for (blk, mod), key in keys.items():
            fa = factors(cur, s_cur, prev, s_prev, key)
            if fa is None:
                continue
            a, b = fa
            sq = total_sq(a, b)
            modules[mod] = float(np.sqrt(max(sq, 0.0)))
            if blk == "mlp":
                mlp_sq += sq
            else:
                attn_sq += sq
            if mod in WRITERS and b.shape[0] % GROUPS == 0:
                nodes += row_sq(a, b)
            elif a.shape[1] % GROUPS == 0:
                reads += col_sq(a, b)
        # Detail maps. MLP: residual groups -> MLP neuron groups (gate+up) -> residual groups (down).
        mlp_in = np.zeros((GROUPS, GROUPS))
        for mod in MLP_READERS:
            fa = get(mod, "mlp")
            if fa is not None:
                mlp_in += block_sq(*fa)
        down = get("down_proj", "mlp")
        mlp_out = block_sq(*down) if down is not None else np.zeros((GROUPS, GROUPS))
        # Attention reads: residual groups -> output groups of each projection, groups split by size.
        readers = [(mod, get(mod)) for mod in ATTN_READERS]
        readers = [(mod, fa) for mod, fa in readers if fa is not None]
        sizes = [fa[1].shape[0] for _, fa in readers]
        shares = [max(1, round(GROUPS * s / max(1, sum(sizes)))) for s in sizes]
        while sum(shares) > GROUPS:
            shares[shares.index(max(shares))] -= 1
        while sum(shares) < GROUPS and shares:
            shares[shares.index(max(shares))] += 1
        attn_rows, attn_labels = [], []
        for (mod, fa), groups in zip(readers, shares):
            if groups < 1:
                continue
            attn_rows.append(block_sq(fa[0], fa[1], groups, GROUPS))
            attn_labels += [mod.replace("_proj", "").replace("in_", "")] * groups
        attn_in = np.concatenate(attn_rows, 0) if attn_rows else np.zeros((GROUPS, GROUPS))
        writer = get("o_proj") or get("out_proj")
        attn_out = block_sq(*writer) if writer is not None else np.zeros((GROUPS, GROUPS))
        out.append({
            "layer": layer + 1,
            "type": "deltanet" if block == "linear_attn" else "attention",
            "attn": float(f"{np.sqrt(attn_sq):.4g}"),
            "mlp": float(f"{np.sqrt(mlp_sq):.4g}"),
            "modules": {k: float(f"{v:.4g}") for k, v in sorted(modules.items())},
            "nodes": r4(np.sqrt(np.maximum(nodes, 0))),
            "reads": r4(np.sqrt(np.maximum(reads, 0))),
            "mlp_in": r4(np.sqrt(np.maximum(mlp_in, 0))),
            "mlp_out": r4(np.sqrt(np.maximum(mlp_out, 0))),
            "attn_in": r4(np.sqrt(np.maximum(attn_in, 0))),
            "attn_in_labels": attn_labels,
            "attn_out": r4(np.sqrt(np.maximum(attn_out, 0))),
        })
    return {"layers": out, "groups": GROUPS}


def source_of(run: Path) -> tuple[Path | None, int | None]:
    """The finished adapter, else the newest checkpoint (a stage that is still training)."""
    if (run / "adapter" / "adapter_model.safetensors").exists():
        return run / "adapter", None
    best = None
    for child in run.glob("checkpoint-*"):
        if (child / "adapter_model.safetensors").exists() and child.name[11:].isdigit():
            step = int(child.name[11:])
            if best is None or step > best[1]:
                best = (child, step)
    return best if best else (None, None)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--runs", type=Path, required=True)
    parser.add_argument("--plan", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--only", help="scan only this stage")
    args = parser.parse_args()
    plan = json.loads(args.plan.read_text(encoding="utf-8"))
    names = [str(s.get("name")) for s in plan.get("stages", []) if re.fullmatch(r"[a-z0-9-]{1,30}", str(s.get("name")))]
    args.out.mkdir(parents=True, exist_ok=True)
    prev_dir = None
    for name in names:
        src, step = source_of(args.runs / f"neuro-{name}")
        if src is None:
            break                       # later stages build on this one: nothing to compare yet
        target = args.out / f"{name}.json"
        stamp = os.path.getmtime(src / "adapter_model.safetensors")
        fresh = False
        if target.exists():
            try:
                old = json.loads(target.read_text(encoding="utf-8"))
                fresh = old.get("source") == str(src) and abs(float(old.get("source_mtime", 0)) - stamp) < 1 and \
                    old.get("previous") == (str(prev_dir) if prev_dir else None)
            except (OSError, ValueError):
                fresh = False
        if (args.only in (None, name)) and not fresh:
            started = time.time()
            cur, s_cur = load(src / "adapter_model.safetensors"), scaling(src)
            prev, s_prev = (load(prev_dir / "adapter_model.safetensors"), scaling(prev_dir)) if prev_dir else (None, 1.0)
            body = {
                "stage": name, "source": str(src), "source_mtime": stamp, "step": step,
                "previous": str(prev_dir) if prev_dir else None, "scanned_at": int(time.time()),
                "modes": {"stage": scan(cur, s_cur, prev, s_prev), "total": scan(cur, s_cur, None, 1.0)},
            }
            body["seconds"] = round(time.time() - started, 1)
            tmp = target.with_suffix(".tmp")
            tmp.write_text(json.dumps(body, separators=(",", ":")), encoding="utf-8")
            os.replace(tmp, target)
            print(f"{name}: scanned {src.name} in {body['seconds']} s", flush=True)
        if step is not None:
            break                       # a stage still training: the next ones have no adapter yet
        prev_dir = src
    return 0


if __name__ == "__main__":
    sys.exit(main())
