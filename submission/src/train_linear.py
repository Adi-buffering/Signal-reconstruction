#!/usr/bin/env python3
"""Linear ene_lo reconstruction pipeline (stdlib-only).

Loads .pt shards written by torch.save, reconstructs sample_lo windows,
fits linear models, selects regularization on validation split,
and evaluates on held-out test split.
"""

from __future__ import annotations

import argparse
import array
import io
import json
import math
import os
import pickle
import random
import struct
import zipfile
from collections import OrderedDict
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Iterable, List, Sequence, Tuple


class FloatStorage:
    """Placeholder class for torch FloatStorage in pickle stream."""


def rebuild_tensor_v2(storage, storage_offset, size, stride, requires_grad, backward_hooks):
    """Rebuild torch tensor as nested Python lists."""
    if isinstance(size, int):
        size = (size,)
    if isinstance(stride, int):
        stride = (stride,)

    def rec(dim: int, base: int):
        if dim == len(size) - 1:
            st = stride[dim]
            return [storage[base + i * st] for i in range(size[dim])]
        st = stride[dim]
        return [rec(dim + 1, base + i * st) for i in range(size[dim])]

    return rec(0, storage_offset) if size else storage[storage_offset]


class TorchZipUnpickler(pickle.Unpickler):
    def __init__(self, fp: io.BytesIO, zf: zipfile.ZipFile, root: str):
        super().__init__(fp)
        self.zf = zf
        self.root = root

    def find_class(self, module: str, name: str):
        if (module, name) == ("torch._utils", "_rebuild_tensor_v2"):
            return rebuild_tensor_v2
        if (module, name) == ("torch", "FloatStorage"):
            return FloatStorage
        if (module, name) == ("collections", "OrderedDict"):
            return OrderedDict
        raise pickle.UnpicklingError(f"Unsupported class: {module}.{name}")

    def persistent_load(self, pid):
        kind, cls, key, loc, size = pid
        if kind != "storage":
            raise pickle.UnpicklingError(f"Unsupported persistent kind: {kind}")
        raw = self.zf.read(f"{self.root}/data/{key}")
        vals = array.array("f")
        vals.frombytes(raw)
        if len(vals) != size:
            raise ValueError(f"Storage size mismatch for key {key}: {len(vals)} != {size}")
        return vals


def load_pt(path: Path) -> Dict[str, list]:
    with zipfile.ZipFile(path) as zf:
        root = zf.namelist()[0].split("/")[0]
        data = zf.read(f"{root}/data.pkl")
        return TorchZipUnpickler(io.BytesIO(data), zf, root).load()


def load_npy_from_bytes(blob: bytes) -> Tuple[dict, Tuple[float, ...]]:
    if blob[:6] != b"\x93NUMPY":
        raise ValueError("Not a .npy file")
    major, minor = blob[6], blob[7]
    if (major, minor) == (1, 0):
        hlen = struct.unpack("<H", blob[8:10])[0]
        start = 10
    else:
        hlen = struct.unpack("<I", blob[8:12])[0]
        start = 12
    header_raw = blob[start : start + hlen].decode("latin1").strip()
    header = eval(header_raw, {"__builtins__": None}, {})
    data = blob[start + hlen :]
    if header["descr"] != "<f4":
        raise ValueError(f"Expected <f4 in npy, got {header['descr']}")
    vals = struct.unpack("<" + "f" * (len(data) // 4), data)
    return header, vals


def load_y_stats(path: Path) -> Tuple[float, float, float, float]:
    with zipfile.ZipFile(path) as zf:
        _, mean_vals = load_npy_from_bytes(zf.read("mean.npy"))
        _, std_vals = load_npy_from_bytes(zf.read("std.npy"))
    return mean_vals[0], mean_vals[1], std_vals[0], std_vals[1]


def build_dataset(split_dir: Path, target_col: int, y_stats: Tuple[float, float, float, float]):
    files = sorted(split_dir.glob("*.pt"))
    if not files:
        raise FileNotFoundError(f"No .pt files found in {split_dir}")

    m0, m1, s0, s1 = y_stats
    mean = [m0, m1]
    std = [s0, s1]

    X_all: List[List[float]] = []
    y_all: List[float] = []

    for fp in files:
        shard = load_pt(fp)
        for req in ("X", "y"):
            if req not in shard:
                raise KeyError(f"Missing key '{req}' in {fp}")

        X = shard["X"]
        y = shard["y"]
        if len(X) != len(y):
            raise ValueError(f"X/y length mismatch in {fp}: {len(X)} vs {len(y)}")

        for i in range(len(X)):
            x_row = X[i]
            if len(x_row) < 2:
                raise ValueError(f"Expected gain dimension >=2 in {fp}, row {i}")
            lo = x_row[1]
            if len(lo) != 7:
                raise ValueError(
                    f"Expected 7 low-gain samples in {fp}, row {i}; got {len(lo)}. "
                    "This repository dataset does not expose 8-sample windows."
                )
            # Repository supplies 7-sample low-gain windows.
            X_all.append([float(v) for v in lo])
            y_all.append(float(y[i][target_col]) * std[target_col] + mean[target_col])

    return X_all, y_all, files


def mat_solve(a: List[List[float]], b: List[float]) -> List[float]:
    n = len(a)
    aug = [row[:] + [b[i]] for i, row in enumerate(a)]
    for col in range(n):
        piv = max(range(col, n), key=lambda r: abs(aug[r][col]))
        aug[col], aug[piv] = aug[piv], aug[col]
        if abs(aug[col][col]) < 1e-12:
            raise ValueError("Singular system")
        fac = aug[col][col]
        for j in range(col, n + 1):
            aug[col][j] /= fac
        for r in range(n):
            if r == col:
                continue
            f = aug[r][col]
            if f == 0:
                continue
            for j in range(col, n + 1):
                aug[r][j] -= f * aug[col][j]
    return [aug[i][n] for i in range(n)]


def fit_ridge(X: Sequence[Sequence[float]], y: Sequence[float], alpha: float, fit_bias: bool = True):
    p = len(X[0]) + (1 if fit_bias else 0)
    ata = [[0.0] * p for _ in range(p)]
    aty = [0.0] * p

    for xr, yt in zip(X, y):
        feat = list(xr) + ([1.0] if fit_bias else [])
        for i in range(p):
            aty[i] += feat[i] * yt
            fi = feat[i]
            row = ata[i]
            for j in range(i, p):
                row[j] += fi * feat[j]
    for i in range(p):
        for j in range(i):
            ata[i][j] = ata[j][i]
    for i in range(p - (1 if fit_bias else 0)):
        ata[i][i] += alpha

    w = mat_solve(ata, aty)
    return w[:-1] if fit_bias else w, (w[-1] if fit_bias else 0.0)


def predict(X: Sequence[Sequence[float]], w: Sequence[float], b: float) -> List[float]:
    return [sum(wj * xj for wj, xj in zip(w, xr)) + b for xr in X]


def residual_ratio(yhat: Sequence[float], y: Sequence[float], eps: float) -> List[float]:
    out = []
    for yp, yt in zip(yhat, y):
        denom = yt if abs(yt) > eps else (eps if yt >= 0 else -eps)
        out.append((yp - yt) / denom)
    return out


def metric_mean(v: Sequence[float]) -> float:
    return sum(v) / len(v)


def metric_rms(v: Sequence[float]) -> float:
    return math.sqrt(sum(x * x for x in v) / len(v))


def write_hist_svg(values: Sequence[float], path: Path, bins: int = 80):
    vmin, vmax = min(values), max(values)
    if vmax <= vmin:
        vmax = vmin + 1.0
    bw = (vmax - vmin) / bins
    counts = [0] * bins
    for v in values:
        idx = int((v - vmin) / bw)
        idx = max(0, min(bins - 1, idx))
        counts[idx] += 1
    cmax = max(counts) or 1

    W, H, M = 900, 500, 50
    plot_w, plot_h = W - 2 * M, H - 2 * M
    bars = []
    for i, c in enumerate(counts):
        x = M + i * plot_w / bins
        h = (c / cmax) * plot_h
        y = H - M - h
        w = plot_w / bins - 1
        bars.append(f'<rect x="{x:.2f}" y="{y:.2f}" width="{w:.2f}" height="{h:.2f}" fill="#1f77b4"/>')
    svg = f'''<svg xmlns="http://www.w3.org/2000/svg" width="{W}" height="{H}">
<rect width="100%" height="100%" fill="white"/>
<line x1="{M}" y1="{H-M}" x2="{W-M}" y2="{H-M}" stroke="black"/>
<line x1="{M}" y1="{M}" x2="{M}" y2="{H-M}" stroke="black"/>
{''.join(bars)}
<text x="{W/2}" y="30" text-anchor="middle" font-size="18">Residual ratio distribution</text>
<text x="{W/2}" y="{H-10}" text-anchor="middle" font-size="14">(reco-target)/target</text>
<text x="20" y="{H/2}" transform="rotate(-90,20,{H/2})" text-anchor="middle" font-size="14">Count</text>
</svg>'''
    path.write_text(svg)


def write_scatter_svg(x: Sequence[float], y: Sequence[float], path: Path, max_points: int = 40000):
    idx = list(range(len(x)))
    random.Random(42).shuffle(idx)
    idx = idx[:max_points]
    xs = [x[i] for i in idx]
    ys = [y[i] for i in idx]

    xmin, xmax = min(xs), max(xs)
    ymin, ymax = min(ys), max(ys)
    if xmax <= xmin:
        xmax = xmin + 1.0
    if ymax <= ymin:
        ymax = ymin + 1.0

    W, H, M = 900, 500, 50
    pw, ph = W - 2 * M, H - 2 * M

    circles = []
    for xv, yv in zip(xs, ys):
        px = M + (xv - xmin) * pw / (xmax - xmin)
        py = H - M - (yv - ymin) * ph / (ymax - ymin)
        circles.append(f'<circle cx="{px:.2f}" cy="{py:.2f}" r="1" fill="#d62728" fill-opacity="0.2"/>')

    svg = f'''<svg xmlns="http://www.w3.org/2000/svg" width="{W}" height="{H}">
<rect width="100%" height="100%" fill="white"/>
<line x1="{M}" y1="{H-M}" x2="{W-M}" y2="{H-M}" stroke="black"/>
<line x1="{M}" y1="{M}" x2="{M}" y2="{H-M}" stroke="black"/>
{''.join(circles)}
<text x="{W/2}" y="30" text-anchor="middle" font-size="18">Residual ratio vs target</text>
<text x="{W/2}" y="{H-10}" text-anchor="middle" font-size="14">target ene_lo</text>
<text x="20" y="{H/2}" transform="rotate(-90,20,{H/2})" text-anchor="middle" font-size="14">(reco-target)/target</text>
</svg>'''
    path.write_text(svg)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data-root", type=Path, default=Path("."))
    ap.add_argument("--out-dir", type=Path, default=Path("submission"))
    ap.add_argument("--target-col", type=int, default=1, help="Column in y corresponding to ene_lo")
    ap.add_argument("--eps", type=float, default=1e-6)
    args = ap.parse_args()

    train_dir = args.data_root / "train" / "train"
    val_dir = args.data_root / "val" / "val"
    test_dir = args.data_root / "test" / "test"
    y_stats_file = args.data_root / "y_stats.npz"

    y_stats = load_y_stats(y_stats_file)
    Xtr, ytr, train_files = build_dataset(train_dir, args.target_col, y_stats)
    Xva, yva, val_files = build_dataset(val_dir, args.target_col, y_stats)
    Xte, yte, test_files = build_dataset(test_dir, args.target_col, y_stats)

    alphas = [0.0, 1e-6, 1e-4, 1e-2, 1e-1, 1.0, 10.0, 100.0]
    val_table = []
    best = None
    for a in alphas:
        w, b = fit_ridge(Xtr, ytr, alpha=a, fit_bias=True)
        pred = predict(Xva, w, b)
        rr = residual_ratio(pred, yva, args.eps)
        m, r = metric_mean(rr), metric_rms(rr)
        val_table.append({"alpha": a, "mean": m, "rms": r})
        if best is None or r < best["val_rms"]:
            best = {"alpha": a, "w": w, "b": b, "val_mean": m, "val_rms": r}

    yhat_val = predict(Xva, best["w"], best["b"])
    rr_val = residual_ratio(yhat_val, yva, args.eps)
    yhat_test = predict(Xte, best["w"], best["b"])
    rr_test = residual_ratio(yhat_test, yte, args.eps)

    (args.out_dir / "figures").mkdir(parents=True, exist_ok=True)
    (args.out_dir / "results").mkdir(parents=True, exist_ok=True)

    write_hist_svg(rr_test, args.out_dir / "figures" / "residual_distribution.svg")
    write_scatter_svg(yte, rr_test, args.out_dir / "figures" / "residual_vs_target.svg")

    metrics = {
        "dataset": {
            "train_files": len(train_files),
            "val_files": len(val_files),
            "test_files": len(test_files),
            "train_rows": len(Xtr),
            "val_rows": len(Xva),
            "test_rows": len(Xte),
            "window_length_available": len(Xtr[0]),
            "note": "Repository tensors provide 7 low-gain samples per row.",
        },
        "selection": {"criterion": "lowest validation RMS residual ratio", "candidates": val_table},
        "model": {"type": "ridge_linear", "alpha": best["alpha"], "weights": best["w"], "bias": best["b"]},
        "residual_policy": {
            "definition": "(reconstructed-target)/target with signed epsilon guard when |target|<=eps",
            "epsilon": args.eps,
        },
        "validation": {"mean": metric_mean(rr_val), "rms": metric_rms(rr_val)},
        "test": {"mean": metric_mean(rr_test), "rms": metric_rms(rr_test)},
    }
    (args.out_dir / "metrics.json").write_text(json.dumps(metrics, indent=2))

    print(json.dumps({"best_alpha": best["alpha"], "val_rms": best["val_rms"], "test_rms": metric_rms(rr_test)}, indent=2))


if __name__ == "__main__":
    main()
