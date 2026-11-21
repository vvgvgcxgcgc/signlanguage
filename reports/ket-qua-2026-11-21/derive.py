"""Curves for the report charts. Usage: python -I derive.py OUT"""
import json, sys
from pathlib import Path
import numpy as np
OUT = Path(sys.argv[1])
meta = json.loads((OUT / "meta.json").read_text()); rep = json.loads((OUT / "report.json").read_text())
idx = {n: i for i, n in enumerate(meta["labels"])}
def sm(z): z = z - z.max(1, keepdims=True); e = np.exp(z); return e / e.sum(1, keepdims=True)
P = {m: {s: sm(np.load(OUT / f"{m}.npz")[f"{s}_logits"]) for s in ("vsl", "oov", "qip")} for m in meta["models"]}
z0 = np.load(OUT / f"{next(iter(meta['models']))}.npz")
Y = {"vsl": np.array([idx[g] for g in z0["vsl_gloss"]]), "qip": np.array([idx[g] for g in z0["qip_gloss"]])}
best = rep["best_ensemble"].split(" + ")
P["ENS4"] = {s: np.mean([P[m][s] for m in best], 0) for s in ("vsl", "oov", "qip")}
taus = np.round(np.arange(0, 1.0, 0.02), 2)
out = {}
for m in ["j76_MS-TCN", "j76_ST-GCN", "j76_CTR-GCN", "j68_AAGCN", "j76_CTR-GCN-BT", "j76_MultiRate-Attn-STGCN", "j76_AnchorGCN-A", "ENS4"]:
    d = {}
    for s in ("vsl", "qip"):
        p, y = P[m][s], Y[s]
        rank = (np.argsort(-p, 1) == y[:, None]).argmax(1)
        d[f"{s}_topk"] = [float((rank < k).mean()) for k in range(1, 11)]
        conf, hit = p.max(1), p.argmax(1) == y
        d[f"{s}_cov"] = [float((conf >= t).mean()) for t in taus]
        d[f"{s}_acc"] = [float(hit[conf >= t].mean()) if (conf >= t).any() else None for t in taus]
        bins = np.linspace(0, 1, 11); rel = []
        for lo, hi in zip(bins[:-1], bins[1:]):
            k = (conf > lo) & (conf <= hi)
            rel.append([float(conf[k].mean()) if k.any() else None, float(hit[k].mean()) if k.any() else None, int(k.sum())])
        d[f"{s}_rel"] = rel
    d["oov_accept"] = [float((P[m]["oov"].max(1) >= t).mean()) for t in taus]
    out[m] = d
json.dump({"taus": taus.tolist(), "models": out, "ens4": best}, open(OUT / "derived.json", "w"), indent=0)
print("ok")
