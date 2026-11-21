"""Các mô hình đề xuất có tốt hơn mốc ở đúng trường hợp chúng nhắm tới không?

Đọc logits mà run_eval.py đã lưu, cùng keypoint thô của VSL400 test, rồi ghi
de-xuat.json cạnh file này. Usage:

    python -I de_xuat.py OUT PROJECT

OUT là thư mục .npz của run_eval.py, PROJECT là thư mục gốc của dự án.

Bốn phép đo:
1. Cặp khó: 202 clip thuộc 14 chiều nhầm của Bảng 10. McNemar chính xác so với mốc.
2. Bổ sung khi tổ hợp: ghép mỗi mô hình với ST-GCN / CTR-GCN, bao nhiêu lỗi của mốc được sửa.
3. Tốc độ ký: độ dài clip chia cho trung vị độ dài của cùng từ trên tập train.
4. Tay gần mặt: tỷ lệ khung có điểm tay cách tâm miệng dưới 0,75 lần độ rộng vai.
Bootstrap ghép cặp 2.000 lần cho khoảng tin cậy 95% của chênh lệch.
"""
import json
import math
import sys
from pathlib import Path

import numpy as np

OUT, PROJECT = Path(sys.argv[1]), Path(sys.argv[2])
HERE = Path(__file__).resolve().parent
meta = json.loads((OUT / "meta.json").read_text())
labels = meta["labels"]
idx = {n: i for i, n in enumerate(labels)}
MODELS = [
    "j76_MS-TCN", "j76_ST-GCN", "j76_CTR-GCN", "j76_AAGCN", "j76_Transformer",
    "j76_MultiRate-Attn-STGCN", "j76_CTR-GCN-BT", "j76_AnchorGCN-A", "j76rest_AnchorGCN-AS", "j76rest_AnchorGCN-T32",
]
PAIRS = [
    ("Dơ", "Con heo"), ("Con heo", "Dơ"), ("Chị", "Cô"), ("Cô", "Chị"), ("Nhẹ", "Bây giờ"), ("Không cần", "Đắng"),
    ("Xôi", "Ướt"), ("Thú vị", "Rửa tay"), ("Thứ ba", "Thứ tư"), ("Thứ tư", "Thứ ba"), ("Bà nội", "Ông nội"),
    ("Bánh chưng", "Bánh tét"), ("Bánh tét", "Bánh chưng"), ("Hát", "Nhầm lẫn"),
]
GROUPS = {
    "nhịp, động tác ngón": ["Nhẹ", "Bánh tét", "Bánh chưng"],
    "điểm chạm trên mặt": ["Chị", "Cô", "Xôi"],
    "hình dạng bàn tay": ["Dơ", "Con heo"],
}
rng = np.random.default_rng(0)


def softmax(z):
    z = z - z.max(1, keepdims=True)
    e = np.exp(z)
    return e / e.sum(1, keepdims=True)


def mcnemar(a, b):
    """Số clip chỉ a đúng, chỉ b đúng, và p hai phía của McNemar chính xác."""
    only_a, only_b = int(np.sum(a & ~b)), int(np.sum(~a & b))
    n, k = only_a + only_b, min(only_a, only_b)
    p = min(1.0, 2 * sum(math.comb(n, i) for i in range(k + 1)) / 2 ** n) if n else 1.0
    return only_a, only_b, p


def boot_diff(a, b, n=2000):
    ii = np.arange(len(a))
    d = [a[s].mean() - b[s].mean() for s in (rng.choice(ii, len(ii)) for _ in range(n))]
    return [float(x) for x in np.percentile(d, [2.5, 97.5]) * 100]


ref = np.load(OUT / "j76_CTR-GCN.npz")
files = [str(f) for f in ref["vsl_files"]]
gloss = np.array([str(g) for g in ref["vsl_gloss"]])
y = np.array([idx[g] for g in gloss])
prob = {}
for m in MODELS:
    z = np.load(OUT / f"{m}.npz")
    assert [str(f) for f in z["vsl_files"]] == files, m
    prob[m] = softmax(z["vsl_logits"])
pred = {m: prob[m].argmax(1) for m in MODELS}
hit = {m: pred[m] == y for m in MODELS}

# ---- đặc trưng từng clip
median_len = {}
for g in sorted(set(gloss)):
    lens = [np.load(f, mmap_mode="r").shape[0] for f in (PROJECT / "vsl400-keypoint" / "train" / g).glob("*.npy")]
    median_len[g] = float(np.median(lens))
ratio, face = np.zeros(len(files)), np.zeros(len(files))
for i, (f, g) in enumerate(zip(files, gloss)):
    a = np.load(f).astype(np.float32)
    ratio[i] = a.shape[0] / median_len[g]
    ls, rs = a[:, 11, :2], a[:, 12, :2]
    ok = (np.abs(ls).sum(1) > 0) & (np.abs(rs).sum(1) > 0)
    width = np.linalg.norm(ls - rs, axis=1)[:, None] + 1e-6
    mouth = ((a[:, 9, :2] + a[:, 10, :2]) / 2)[:, None, :]
    pts = np.concatenate([a[:, 15:17, :2], a[:, 34:76, :2]], axis=1)
    present = np.abs(pts).sum(2) > 0
    near = ((np.linalg.norm(pts - mouth, axis=2) / width < 0.75) & present).any(1)
    face[i] = near[ok].mean() if ok.any() else 0.0

out = {"n_clips": len(files), "models": MODELS}

# ---- 1. cặp khó
src = sorted({a for a, _ in PAIRS})
mask = np.isin(gloss, src)
hard = {"n": int(mask.sum()), "acc": {}, "confusions": {}, "groups": {}, "mcnemar": {}}
for m in MODELS:
    hard["acc"][m] = float(hit[m][mask].mean() * 100)
    hard["confusions"][m] = sum(int(np.sum((gloss == a) & (pred[m] == idx[b]))) for a, b in PAIRS)
for name, glosses in GROUPS.items():
    mk = np.isin(gloss, glosses)
    hard["groups"][name] = {"n": int(mk.sum()), "acc": {m: float(hit[m][mk].mean() * 100) for m in MODELS}}
for a, b in [("j76_MultiRate-Attn-STGCN", "j76_MS-TCN"), ("j76_MultiRate-Attn-STGCN", "j76_ST-GCN"),
             ("j76_MultiRate-Attn-STGCN", "j76_CTR-GCN"), ("j76_CTR-GCN-BT", "j76_CTR-GCN"),
             ("j76_AnchorGCN-A", "j76_CTR-GCN")]:
    hard["mcnemar"][f"{a} | {b}"] = mcnemar(hit[a][mask], hit[b][mask])
out["hard_pairs"] = hard

# ---- 2. bổ sung khi tổ hợp
comp = {}
for base in ["j76_ST-GCN", "j76_CTR-GCN"]:
    err = ~hit[base]
    comp[base] = {"errors": int(err.sum()), "partners": {}}
    for m in MODELS:
        if m == base:
            continue
        ens = ((prob[base] + prob[m]) / 2).argmax(1) == y
        comp[base]["partners"][m] = {
            "fixed": int(np.sum(err & hit[m])), "both_wrong": int(np.sum(err & ~hit[m])),
            "ensemble_top1": float(ens.mean() * 100),
        }
out["complement"] = comp

# ---- 3, 4. tập con theo tốc độ và vị trí tay (để ghi nhận cả trường hợp không hơn)
fq = np.percentile(face, [50, 75])
subsets = {
    "rất nhanh (r < 0,85)": ratio < 0.85,
    "bình thường (0,92 ≤ r ≤ 1,08)": (ratio >= 0.92) & (ratio <= 1.08),
    "rất chậm (r > 1,17)": ratio > 1.17,
    "tay gần mặt (25% clip cao nhất)": face >= fq[1],
    "tay xa mặt (50% clip thấp nhất)": face < fq[0],
}
sub = {}
for name, mk in subsets.items():
    sub[name] = {"n": int(mk.sum()), "acc": {m: float(hit[m][mk].mean() * 100) for m in MODELS}, "diff": {}}
    for a, b in [("j76_MultiRate-Attn-STGCN", "j76_ST-GCN"), ("j76_CTR-GCN-BT", "j76_CTR-GCN"), ("j76_AnchorGCN-A", "j76_CTR-GCN")]:
        d = float((hit[a][mk].mean() - hit[b][mk].mean()) * 100)
        sub[name]["diff"][f"{a} | {b}"] = [d] + boot_diff(hit[a][mk], hit[b][mk])
out["subsets"] = sub

# ---- độ tin cậy, lấy từ derived.json (τ = 0,7)
derived = json.loads((HERE / "derived.json").read_text())
k = derived["taus"].index(0.7)
out["selective_tau07"] = {
    m: {"qip_cov": d["qip_cov"][k] * 100, "qip_acc": d["qip_acc"][k] * 100, "oov_accept": d["oov_accept"][k] * 100,
        "vsl_cov": d["vsl_cov"][k] * 100, "vsl_acc": d["vsl_acc"][k] * 100}
    for m, d in derived["models"].items()
}
# ---- thắng / thua so với cả hai mốc, và tỷ lệ thắng rơi vào các từ của cặp khó
pair_glosses = sorted({g for pr in PAIRS for g in pr})
in_pairs = np.isin(gloss, pair_glosses)
both_right = hit["j76_ST-GCN"] & hit["j76_CTR-GCN"]
both_wrong = ~hit["j76_ST-GCN"] & ~hit["j76_CTR-GCN"]
out["wins"] = {"pair_glosses": len(pair_glosses), "pair_clips": int(in_pairs.sum()), "both_wrong": int(both_wrong.sum()), "models": {}}
for m in ["j76_MultiRate-Attn-STGCN", "j76_AnchorGCN-A", "j76_CTR-GCN-BT"]:
    win = hit[m] & both_wrong
    out["wins"]["models"][m] = {"wins": int(win.sum()), "wins_in_pairs": int((win & in_pairs).sum()),
                                "losses": int((~hit[m] & both_right).sum())}

# ---- ví dụ: clip mô hình đề xuất đúng, mô hình gốc sai (dùng cho hình có khung video)
EXAMPLES = ["Nhẹ/024459", "Con heo/016671", "Cô/022498", "Thứ ba/021565", "Xôi/011770", "Nhẹ/012199",
            "Ướt/023723", "Thú vị/013358", "Bánh chưng/020844", "Cô/015032", "Không cần/023058", "Thứ ba/018744"]
key = {f"{g}/{Path(f).stem}": i for i, (f, g) in enumerate(zip(files, gloss))}
out["examples"] = {}
for ex in EXAMPLES:
    i = key[ex]
    out["examples"][ex] = {m: [[labels[j], float(prob[m][i, j])] for j in np.argsort(-prob[m][i])[:3]] for m in MODELS}
(HERE / "de-xuat.json").write_text(json.dumps(out, ensure_ascii=False, indent=1))
print(json.dumps({"hard_acc": hard["acc"], "conf": hard["confusions"], "mcnemar": hard["mcnemar"]}, ensure_ascii=False, indent=1))
