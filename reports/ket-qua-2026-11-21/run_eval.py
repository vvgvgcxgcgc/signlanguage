"""Score the latest Kaggle checkpoints with the repo code that trained them.

Usage: python run_eval.py REPO PROJECT OUT   (REPO = PROJECT = thư mục gốc repo)
Writes OUT/<id>.npz (logits per split) and OUT/meta.json.
"""
import json, re, sys, time
from multiprocessing import Pool
from pathlib import Path

REPO, PROJECT, OUT = (Path(p).resolve() for p in sys.argv[1:4])
sys.path.insert(0, str(REPO))

import numpy as np
import torch

from preprocess.dataset import (normalize_clip, drop_legs, interpolate_clip,
                                pad_prepare, PREPARE_PAD, resample_clip)
from models.factory import build_model, count_parameters

sys.path.insert(0, str(PROJECT))
import assets  # noqa: E402  - tải checkpoint public từ Kaggle (huynguang/vsl-sign-checkpoints)

# Mọi thử nghiệm trong assets/manifest.json; nguồn (nick, notebook, version) ghi trong đó.
CKPTS = {name: assets.checkpoint(name) for name in assets.manifest()}
MULTIRATE = (32, 64, 96)
labels = assets.labels("j76_CTR-GCN")
index = {name: i for i, name in enumerate(labels)}
qip_names = list(json.loads(assets.label_file("signconnect_j76_common.json").read_text()))


def prep_vsl(job):
    path, is_leg, lengths = job
    arr = np.load(path)
    norm = normalize_clip(arr, 1e-3)
    if norm is None:
        return None
    if not is_leg:
        norm = drop_legs(norm)
    return tuple(interpolate_clip(norm, n) for n in lengths)


def prep_qip(job):
    path, is_leg, lengths = job
    padded = pad_prepare(np.load(path).astype(np.float32), PREPARE_PAD)
    out = tuple(resample_clip(padded, n, is_leg, 1e-3) for n in lengths)
    return None if any(o is None for o in out) else out


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    test = assets.ensure_keypoints() / "test"
    vsl_in, vsl_oov = [], []
    for d in sorted(test.iterdir()):
        if not d.is_dir():
            continue
        for f in sorted(d.glob("*.npy")):
            (vsl_in if d.name in index else vsl_oov).append((f, d.name))
    qip = []
    for f in sorted(assets.signconnect_keypoints().glob("*.npy")):
        g = re.sub(r"\(\d+\)$", "", f.stem)
        if g in qip_names:
            qip.append((f, g))
    print(f"vsl in-vocab {len(vsl_in)}  oov {len(vsl_oov)}  qipedc {len(qip)}", flush=True)

    cache = {}

    def features(split, rows, is_leg, lengths, fn):
        key = (split, is_leg, lengths)
        if key not in cache:
            t = time.time()
            with Pool(8) as pool:
                got = pool.map(fn, [(str(p), is_leg, lengths) for p, _ in rows], chunksize=64)
            keep = [i for i, g in enumerate(got) if g is not None]
            branches = tuple(torch.from_numpy(np.stack([got[i][b] for i in keep]).astype(np.float32))
                             for b in range(len(lengths)))
            cache[key] = (keep, branches)
            print(f"  prepared {split} leg={is_leg} T={lengths}: {len(keep)}/{len(rows)} in {time.time()-t:.0f}s", flush=True)
        return cache[key]

    device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")
    meta = {"labels": labels, "qipedc_glosses": qip_names, "device": str(device), "models": {}}
    for mid, path in CKPTS.items():
        payload = torch.load(path, map_location="cpu", weights_only=False)
        name, is_leg, T = payload["model_name"], bool(payload["is_leg"]), int(payload["max_frames"])
        model = build_model(name, int(payload["num_classes"]), is_leg=is_leg, max_frames=T)
        model.load_state_dict({k.replace("module.", ""): v for k, v in payload["state_dict"].items()})
        model.eval()
        lengths = MULTIRATE if name == "MultiRate-Attn-STGCN" else (T,)

        def run(branches, dev):
            m = model.to(dev)
            outs = []
            with torch.no_grad():
                for s in range(0, branches[0].shape[0], 64):
                    xs = [b[s:s + 64].to(dev) for b in branches]
                    outs.append(m(xs[0] if len(xs) == 1 else tuple(xs)).float().cpu())
            return torch.cat(outs).numpy()

        res = {}
        t0 = time.time()
        for split, rows, fn in (("vsl", vsl_in, prep_vsl), ("oov", vsl_oov, prep_vsl), ("qip", qip, prep_qip)):
            keep, branches = features(split, rows, is_leg, lengths, fn)
            try:
                logits = run(branches, device)
            except Exception as e:  # unsupported op on MPS
                print(f"  {mid}: {device} failed ({str(e)[:60]}), using cpu", flush=True)
                logits = run(branches, torch.device("cpu"))
            res[f"{split}_logits"] = logits
            res[f"{split}_files"] = np.array([str(rows[i][0]) for i in keep])
            res[f"{split}_gloss"] = np.array([rows[i][1] for i in keep])
        np.savez_compressed(OUT / f"{mid}.npz", **res)

        # CPU latency, one clip per call, 4 threads
        torch.set_num_threads(4)
        m = model.to("cpu")
        one = tuple(b[:1] for b in features("vsl", vsl_in, is_leg, lengths, prep_vsl)[1])
        x = one[0] if len(one) == 1 else one
        with torch.no_grad():
            for _ in range(3):
                m(x)
            times = []
            for _ in range(30):
                t = time.perf_counter(); m(x); times.append((time.perf_counter() - t) * 1000)
        meta["models"][mid] = {
            "checkpoint": str(path.relative_to(PROJECT)), "model": name, "is_leg": is_leg, "frames": T,
            "epoch": int(payload.get("epoch", -1)), "params": count_parameters(model),
            "size_mb": round(path.stat().st_size / 2**20, 2),
            "cpu_ms_median": float(np.median(times)), "cpu_ms_p95": float(np.percentile(times, 95)),
            "stored": {k: v for k, v in (payload.get("scores") or payload.get("metrics") or {}).items()
                       if isinstance(v, (int, float))},
        }
        print(f"{mid}: done in {time.time()-t0:.0f}s, cpu {np.median(times):.1f} ms", flush=True)
        (OUT / "meta.json").write_text(json.dumps(meta, ensure_ascii=False, indent=1))


if __name__ == "__main__":
    main()
