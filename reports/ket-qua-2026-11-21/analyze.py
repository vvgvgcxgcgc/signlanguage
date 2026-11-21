"""Metrics from the logits that run_eval.py saved. Usage: python -I analyze.py OUT"""
import itertools, json, sys
from pathlib import Path
import numpy as np

OUT = Path(sys.argv[1])
meta = json.loads((OUT / "meta.json").read_text())
labels = meta["labels"]; idx = {n: i for i, n in enumerate(labels)}
qip_set = meta["qipedc_glosses"]; qip_idx = np.array([idx[g] for g in qip_set])
N = len(labels); rng = np.random.default_rng(0); TAUS = (0.5, 0.7, 0.9)


def softmax(z):
    z = z - z.max(1, keepdims=True); e = np.exp(z); return e / e.sum(1, keepdims=True)


def macro(y, p, classes):
    f1s, ps, rs = [], [], []
    for c in classes:
        tp = np.sum((p == c) & (y == c)); fp = np.sum((p == c) & (y != c)); fn = np.sum((p != c) & (y == c))
        pc = tp / (tp + fp) if tp + fp else 0.0; rc = tp / (tp + fn) if tp + fn else 0.0
        ps.append(pc); rs.append(rc); f1s.append(2 * pc * rc / (pc + rc) if pc + rc else 0.0)
    return float(np.mean(f1s)), float(np.mean(ps)), float(np.mean(rs)), np.array(f1s)


def ece(prob, y, bins=15):
    conf = prob.max(1); hit = prob.argmax(1) == y; e = 0.0
    edges = np.linspace(0, 1, bins + 1)
    for lo, hi in zip(edges[:-1], edges[1:]):
        m = (conf > lo) & (conf <= hi)
        if m.any():
            e += m.mean() * abs(hit[m].mean() - conf[m].mean())
    return float(e)


def boot_f1(y, p, classes, n=1000):
    vals = []
    for _ in range(n):
        s = rng.integers(0, len(y), len(y))
        vals.append(macro(y[s], p[s], sorted(set(y[s].tolist())))[0])
    return float(np.percentile(vals, 2.5)), float(np.percentile(vals, 97.5))


def scores(prob, y, classes, restrict=None, boot=False):
    pr = prob if restrict is None else np.where(np.isin(np.arange(N), restrict)[None], prob, 0.0)
    p = prob.argmax(1)  # outside-set prediction is a miss: p stays the global argmax
    top5 = np.argsort(-prob, 1)[:, :5]
    f1, P, R, f1s = macro(y, p, classes)
    out = {"n": int(len(y)), "top1": float((p == y).mean()), "top5": float((top5 == y[:, None]).any(1).mean()),
           "macro_f1": f1, "macro_p": P, "macro_r": R, "ece": ece(prob, y)}
    if boot:
        out["f1_ci"] = boot_f1(y, p, classes)
    return out, p, f1s


data, probs = {}, {}
for mid in meta["models"]:
    z = np.load(OUT / f"{mid}.npz")
    data[mid] = z
    probs[mid] = {s: softmax(z[f"{s}_logits"]) for s in ("vsl", "oov", "qip")}
any_mid = next(iter(data))
y_vsl = np.array([idx[g] for g in data[any_mid]["vsl_gloss"]])
y_qip = np.array([idx[g] for g in data[any_mid]["qip_gloss"]])
for mid in data:  # same clips for every model?
    assert len(data[mid]["vsl_gloss"]) == len(y_vsl), mid
    assert len(data[mid]["qip_gloss"]) == len(y_qip), mid
in147 = np.isin(y_vsl, qip_idx)
classes_all = sorted(set(y_vsl.tolist())); classes_q = sorted(set(y_qip.tolist()))

report = {"n_vsl": len(y_vsl), "n_oov": int(len(data[any_mid]["oov_gloss"])),
          "n_oov_gloss": len(set(data[any_mid]["oov_gloss"].tolist())), "n_qip": len(y_qip),
          "n_vsl147": int(in147.sum()), "models": {}, "ensembles": {}}


def full(prob_v, prob_o, prob_q, boot=True):
    r = {}
    r["vsl"], p_v, f1s = scores(prob_v, y_vsl, classes_all, boot=boot)
    r["vsl147"], _, _ = scores(prob_v[in147], y_vsl[in147], classes_q)
    r["qip"], p_q, _ = scores(prob_q, y_qip, classes_q, boot=boot)
    r["drop_top1"] = 1 - r["qip"]["top1"] / r["vsl147"]["top1"]
    r["drop_f1"] = 1 - r["qip"]["macro_f1"] / r["vsl147"]["macro_f1"]
    sel = {}
    for t in TAUS:
        cv = prob_v.max(1) >= t; cq = prob_q.max(1) >= t
        sel[str(t)] = {"cov_vsl": float(cv.mean()), "acc_vsl": float((p_v[cv] == y_vsl[cv]).mean()) if cv.any() else None,
                       "cov_qip": float(cq.mean()), "acc_qip": float((p_q[cq] == y_qip[cq]).mean()) if cq.any() else None,
                       "oov_accept": float((prob_o.max(1) >= t).mean())}
    r["selective"] = sel
    r["qip_outside147"] = float((~np.isin(p_q, qip_idx)).mean())
    bins = [0, .5, .8, .9, .95, 1.01]
    r["f1_bins"] = [int(((f1s >= lo) & (f1s < hi)).sum()) for lo, hi in zip(bins[:-1], bins[1:])]
    r["f1_one"] = int((f1s == 1).sum())
    conf = {}
    for t, q in zip(y_vsl, p_v):
        if t != q:
            conf[(t, q)] = conf.get((t, q), 0) + 1
    sup = {c: int((y_vsl == c).sum()) for c in classes_all}
    pairs = sorted(conf.items(), key=lambda kv: (-kv[1] / sup[kv[0][0]], -kv[1]))[:12]
    r["pairs"] = [(labels[a], labels[b], n, sup[a]) for (a, b), n in pairs]
    worst = np.argsort(f1s)[:10]
    r["worst"] = [(labels[classes_all[i]], float(f1s[i]), sup[classes_all[i]]) for i in worst]
    return r


for mid, m in meta["models"].items():
    print("model", mid, flush=True)
    report["models"][mid] = {**m, **full(probs[mid]["vsl"], probs[mid]["oov"], probs[mid]["qip"])}

# ensembles among j76 single-T models plus multirate, by mean probability
j76 = [m for m in meta["models"] if m.startswith("j76")]
top = sorted(j76, key=lambda m: -report["models"][m]["vsl"]["macro_f1"])[:7]
combos = [tuple(c) for k in (2, 3, 4) for c in itertools.combinations(top, k)]
named = [("j76_MS-TCN", "j76_ST-GCN"), ("j76_CTR-GCN", "j76_Transformer"),
         ("j76_CTR-GCN", "j76_ST-GCN", "j76_Transformer"), tuple(j76)]
best = None
for c in combos + named:
    pv = np.mean([probs[m]["vsl"] for m in c], 0)
    f1 = macro(y_vsl, pv.argmax(1), classes_all)[0]
    if best is None or f1 > best[1]:
        best = (c, f1)
keep = set(named) | {best[0]}
for c in keep:
    pv, po, pq = (np.mean([probs[m][s] for m in c], 0) for s in ("vsl", "oov", "qip"))
    r = full(pv, po, pq)
    r["params"] = int(sum(meta["models"][m]["params"] for m in c))
    r["cpu_ms_median"] = float(sum(meta["models"][m]["cpu_ms_median"] for m in c))
    report["ensembles"][" + ".join(c)] = r
report["best_ensemble"] = " + ".join(best[0])

# agreement j68 vs j76 for CH2
pairs = {}
for base in ("MS-TCN", "ST-GCN", "CTR-GCN", "AAGCN", "MultiRate-Attn-STGCN"):
    a, b = f"j68_{base}", f"j76_{base}"
    if a in report["models"] and b in report["models"]:
        pairs[base] = {"j68": report["models"][a]["vsl"], "j76": report["models"][b]["vsl"],
                       "j68_qip": report["models"][a]["qip"], "j76_qip": report["models"][b]["qip"]}
report["legs"] = pairs
(OUT / "report.json").write_text(json.dumps(report, ensure_ascii=False, indent=1, default=float))
print("written", OUT / "report.json")

# CH3: confusion rate of the hard pairs, per model
HARD = [("Dơ", "Con heo"), ("Con heo", "Dơ"), ("Chị", "Cô"), ("Cô", "Chị"), ("Nhẹ", "Bây giờ"),
        ("Không cần", "Đắng"), ("Xôi", "Ướt"), ("Thú vị", "Rửa tay"), ("Thứ ba", "Thứ tư"), ("Thứ tư", "Thứ ba"),
        ("Bà nội", "Ông nội"), ("Bánh chưng", "Bánh tét"), ("Bánh tét", "Bánh chưng"), ("Hát", "Nhầm lẫn")]
hard = {}
sets = {**{m: probs[m]["vsl"] for m in probs},
        **{"ENS:" + k: np.mean([probs[m]["vsl"] for m in k.split(" + ")], 0) for k in report["ensembles"]}}
for name, pv in sets.items():
    p = pv.argmax(1); row = {}
    for a, b in HARD:
        ya = y_vsl == idx[a]
        row[f"{a}→{b}"] = [int((p[ya] == idx[b]).sum()), int(ya.sum())]
    row["_total"] = sum(v[0] for v in row.values())
    hard[name] = row
report["hard_pairs"] = hard
(OUT / "report.json").write_text(json.dumps(report, ensure_ascii=False, indent=1, default=float))
print("hard pairs added")
