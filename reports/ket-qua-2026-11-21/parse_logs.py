"""Join per-epoch metrics from every Kaggle version log into one curve per run.

Usage: python -I parse_logs.py LOGS_DIR SCAN_DIR REPO_RUNS OUT_JSON
"""
import json, re, sys
from datetime import datetime
from pathlib import Path

LOGS, SCAN, RUNS, OUT = (Path(p) for p in sys.argv[1:5])
EPOCH = re.compile(
    r"(\d\d:\d\d:\d\d) INFO\s+training\.engine \| epoch\s+(\d+)/(\d+) \| train loss=([\d.]+) acc=([\d.]+) "
    r"top5=([\d.]+) macroF1=([\d.]+) \| val loss=([\d.]+) acc=([\d.]+) top5=([\d.]+) macroF1=([\d.]+)")
WROTE = re.compile(r"Wrote \S+ checkpoint to checkpoints/(\S+?)_(?:last|best)\.pth")
SCRIPT = re.compile(r"=== (\S+\.sh) ===")

script_run = {}
for sh in RUNS.glob("*.sh"):
    t = sh.read_text()
    m, g = re.search(r"--model (\S+)", t), re.search(r"--tag (\S+)", t)
    if m and g:
        script_run[sh.name] = f"{g.group(1)}_{m.group(1)}"

dates = {}  # (account, version) -> run date
for f in SCAN.glob("*.txt"):
    acc = f.stem.split("_")[0]
    for line in f.read_text().splitlines():
        m = re.match(r"v(\d+) \|\s*(.+?)\|", line)
        if m and m.group(2).strip():
            dates[(acc, int(m.group(1)))] = datetime.strptime(m.group(2).strip(), "%I:%M %p, %A %d %B %Y UTC")

runs = {}
for d in sorted(LOGS.iterdir()):
    m = re.match(r"(\w+?)_v(\d+)$", d.name)
    logs = list(d.glob("*.log"))
    if not m or not logs:
        continue
    acc, ver = m.group(1), int(m.group(2))
    try:
        txt = "".join(x["data"] for x in json.loads(logs[0].read_text()))
    except Exception:
        txt = logs[0].read_text(errors="ignore")
    parts = SCRIPT.split(txt)  # [head, name1, body1, name2, body2, ...]
    segments = [(None, parts[0])] + [(parts[i], parts[i + 1]) for i in range(1, len(parts) - 1, 2)]
    when = dates.get((acc, ver), datetime(2000, 1, 1)).isoformat()
    for script, seg in segments:
        w = WROTE.search(seg)
        run = w.group(1) if w else script_run.get(script)
        rows = {}
        prev = None
        for e in EPOCH.finditer(seg):
            ep = int(e.group(2))
            if ep in rows:
                continue
            t = datetime.strptime(e.group(1), "%H:%M:%S")
            dur = (t - prev).total_seconds() % 86400 if prev is not None else None
            prev = t
            rows[ep] = {"total": int(e.group(3)), "train_loss": float(e.group(4)), "train_acc": float(e.group(5)),
                        "train_f1": float(e.group(7)), "val_loss": float(e.group(8)), "val_acc": float(e.group(9)),
                        "val_top5": float(e.group(10)), "val_f1": float(e.group(11)), "sec": dur,
                        "src": f"{acc} v{ver}"}
        if not run or not rows:
            continue
        for ep, r in rows.items():
            cur = runs.setdefault(run, {}).get(ep)
            if cur is None or when >= cur["_when"]:
                runs[run][ep] = {**r, "_when": when}

out = {}
for run, eps in runs.items():
    keys = sorted(eps)
    out[run] = [{"epoch": k, **{a: b for a, b in eps[k].items() if a != "_when"}} for k in keys]
    secs = [eps[k]["sec"] for k in keys if eps[k]["sec"]]
    print(f"{run:34s} epochs {keys[0]}-{keys[-1]} ({len(keys)}) best valF1 {max(eps[k]['val_f1'] for k in keys):.4f}"
          f" median epoch {sorted(secs)[len(secs)//2]/60 if secs else 0:.1f} min  src {sorted({eps[k]['src'] for k in keys})}")
OUT.write_text(json.dumps(out, indent=1))
