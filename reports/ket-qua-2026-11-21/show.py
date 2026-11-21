import json,sys
r=json.load(open(sys.argv[1]))
print({k:r[k] for k in ('n_vsl','n_oov','n_oov_gloss','n_qip','n_vsl147','best_ensemble')})
def pc(x): return f"{x*100:.2f}"
def row(n,m):
    v,q,s=m['vsl'],m['qip'],m['selective']
    st=m.get('stored',{}).get('macro_f1')
    print(f"{n[:40]:40s} ep{m.get('epoch','-')} P{m['params']/1e6:.2f}M {m.get('size_mb','-')}MB cpu{m['cpu_ms_median']:.1f}"
          f" | t1 {pc(v['top1'])} t5 {pc(v['top5'])} F1 {pc(v['macro_f1'])} [{pc(v['f1_ci'][0])},{pc(v['f1_ci'][1])}] P{pc(v['macro_p'])} R{pc(v['macro_r'])} ECE{pc(v['ece'])} stored{pc(st) if st else '-'}"
          f" | v147 {pc(m['vsl147']['top1'])}/{pc(m['vsl147']['macro_f1'])} qip {pc(q['top1'])}/{pc(q['top5'])}/{pc(q['macro_f1'])} [{pc(q['f1_ci'][0])},{pc(q['f1_ci'][1])}] ECEq{pc(q['ece'])} drop {pc(m['drop_top1'])}/{pc(m['drop_f1'])} out{pc(m['qip_outside147'])}")
    print("     sel " + " ".join(f"t{t}: v{x['cov_vsl']*100:.1f}%/{(x['acc_vsl'] or 0)*100:.2f} q{x['cov_qip']*100:.1f}%/{(x['acc_qip'] or 0)*100:.1f} oov{x['oov_accept']*100:.1f}" for t,x in s.items()) + f" bins{m['f1_bins']} one{m['f1_one']}")
for n,m in r['models'].items(): row(n,m)
for n,m in r['ensembles'].items(): print('ENS',n); row('ens',m)
