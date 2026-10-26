"""report_real.py: the common evaluation and printing of the real-data round (INSTRUCTIONS_REAL_2026-10-24.md), used by
vss/run_reader_real.py and models/bert_real.py so that both are scored by the same function (eval/metrics.score_real) on the same rows.

evaluate() takes one trained model's predictions on the condition's calibration rows and eval sets, sets the thresholds on the
CALIBRATION rows only (eval/metrics.set_taus, the written rule: the smallest tau with confident-wrong <= 0.05, per decision kind:
intent, slot, frame), scores every eval set, and writes logs/results_real_<dataset>_<condition>_<tag>_<size>_seed<k>.json for
eval/compare.py. summarize() prints mean ± std over seeds with the per-seed list (rule 27)."""
import json, os
import numpy as np
from . import mainrepo, real_data as rd
from eval import metrics

def gold_flat(item): return rd.flatten(rd.head_frame(item["tree"], item["tokens"]))
def gold_spans(item): return [(t, s, e) for t, s, e, _ in item["slots"]]

def schema_roles(cond, ids=None):
    return sorted({t for i in (ids or cond["train"]) for t, *_ in cond["items"][i]["slots"]})

def is_test(name): return name.startswith("test") or name.startswith("ood:test")

def eval_sets(cond, test=False, max_items=None):
    """the eval sets of the condition that are scored in this run (test sets only with --test), each cut to max_items for smoke tests"""
    return {k: (v[:max_items] if max_items else v) for k, v in cond["eval"].items() if test or not is_test(k)}

def evaluate(name, tag, cond, seed, size, preds, cal_ids, cal_preds, roles, variant=None, extra=None):
    """preds: {set name: [prediction]} in the order of the set's ids (eval_sets); a prediction may carry 'spans' (exact-span F1) and
    'status' / 'residual' (the reader's tool fields)"""
    items = cond["items"]; ev = {k: v for k, v in eval_sets(cond, True).items() if k in preds}
    cg = [gold_flat(items[i]) for i in cal_ids]; taus = metrics.set_taus(cg, cal_preds)
    row = {"model": name, "tag": tag, "dataset": cond["dataset"], "condition": cond["condition"], "size": size, "seed": seed, "variant": variant,
           "taus": taus, "n_train": len(cond["train"]) if size is None else size, "sets": {}, "spans": {}, "tool": {}}
    for k, p in preds.items():
        ids = ev[k][:len(p)]; g = [gold_flat(items[i]) for i in ids]
        newp = [cond["new_roles"].get(i, []) for i in ids] if "new_slot_values" in k else None
        row["sets"][k] = metrics.score_real(g, p, roles, taus, newp, [items[i]["depth"] for i in ids])
        if p and "spans" in p[0]: row["spans"][k] = metrics.span_f1([gold_spans(items[i]) for i in ids], [x["spans"] for x in p])
        if p and "spans_rule_on_gold" in p[0]: row["spans"][k + " [span rule on GOLD heads: its ceiling]"] = metrics.span_f1([gold_spans(items[i]) for i in ids], [x["spans_rule_on_gold"] for x in p])
        for extra_k, label in (("spans_by_rule", " [spans by the rule, second line, R71]"), ("spans_typer_only", " [typer only, R71 item 1]")):   # additive (R71)
            if p and extra_k in p[0]: row["spans"][k + label] = metrics.span_f1([gold_spans(items[i]) for i in ids], [x[extra_k] for x in p])
        if p and "status" in p[0]:
            st = {}
            for x in p: st[x["status"]] = st.get(x["status"], 0) + 1
            row["tool"][k] = {"status": st, "residual_mean": float(np.mean([x["residual"] for x in p]))}
    for k in [x for x in preds if x.startswith("ood:")]:        # threshold-free: does the intent margin separate in-domain from out-of-domain?
        ind = k.split(":", 1)[1] + ":in_domain"
        if ind in preds: row.setdefault("ood_auroc_intent_margin", {})[k] = auroc([x["margins"].get("intent", 0.0) for x in preds[ind]], [x["margins"].get("intent", 0.0) for x in preds[k]])
        if ind in preds and preds[ind] and "residual" in preds[ind][0]:   # additive (R71): in-domain should have the SMALLER residual
            row.setdefault("ood_auroc_residual", {})[k] = auroc([-x["residual"] for x in preds[ind]], [-x["residual"] for x in preds[k]])
    row.update(extra or {})
    for k, s in row["sets"].items():
        if s is None: continue
        c = s["calibration"]
        print(f"  seed {seed} size {size or 'full'} {name}{' [' + variant + ']' if variant else ''} | {k} (n {s['n']}): FULL {s['full']:.3f} intent {s['intent']:.3f} slot heads {s['slot_heads_on_gold'] if s['slot_heads_on_gold'] is not None else float('nan'):.3f}"
              + (f" new-value heads {s['new_value_heads']:.3f} (n {s['n_new_value_heads']})" if s.get("new_value_heads") is not None else "")
              + (f" nested intents {s['nested_intents']:.3f}" if s.get("nested_intents") is not None else "")
              + f" | frame tau {taus['frame']:.3f}: coverage {c['frame']['coverage']:.2f} confident-wrong {c['frame']['confident_wrong']:.3f} | intent abstain {s['abstain']['intent_margin_below_tau']:.3f}"
              + (f" | span F1 {row['spans'][k]['f1']:.3f}" if k in row["spans"] else ""), flush=True)
    vt = f"_{variant}" if variant else ""
    path = os.path.join(mainrepo.ROOT, "logs", f"results_real_{cond['dataset']}_{cond['condition']}_{tag}{vt}_{size if size else 'full'}_seed{seed}.json")
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w") as f: json.dump(row, f)
    return row

def auroc(pos, neg):
    """P(margin of an in-domain item > margin of an out-of-domain item), ties counted one half (rank form)"""
    if not pos or not neg: return None
    a = np.array(pos)[:, None]; b = np.array(neg)[None, :]; return float((a > b).mean() + 0.5 * (a == b).mean())

def ms(v):
    v = [x for x in v if x is not None]
    return f"{np.mean(v):.3f}±{np.std(v):.3f} [{' '.join(f'{x:.3f}' for x in v)}]" if v else "n/a"

def summarize(rows, name):
    """mean ± std over seeds [per seed] for every eval set: FULL, intent, slot heads, new-value heads, frame coverage / confident-wrong,
    intent abstention, span F1"""
    if not rows: return
    print(f"\n== {name}: {rows[0]['dataset']} / {rows[0]['condition']} / size {rows[0]['size'] or 'full'}{' [' + rows[0]['variant'] + ']' if rows[0].get('variant') else ''}: mean ± std over {len(rows)} seeds [per seed] ==")
    for k in rows[0]["sets"]:
        S = [r["sets"][k] for r in rows if r["sets"].get(k)]
        if not S: continue
        print(f"  {k} (n {S[0]['n']}): FULL {ms([s['full'] for s in S])} | intent {ms([s['intent'] for s in S])} | slot heads {ms([s['slot_heads_on_gold'] for s in S])}"
              + (f" | new-value heads {ms([s.get('new_value_heads') for s in S])}" if S[0].get("new_value_heads") is not None else "")
              + (f" | nested intents {ms([s.get('nested_intents') for s in S])}" if S[0].get("nested_intents") is not None else ""))
        print(f"      frame coverage {ms([s['calibration']['frame']['coverage'] for s in S])} confident-wrong {ms([s['calibration']['frame']['confident_wrong'] for s in S])}"
              f" | intent abstain {ms([s['abstain']['intent_margin_below_tau'] for s in S])} | absent read absent {ms([s['occupancy']['absent_read_absent'] for s in S])}"
              + (f" | span F1 {ms([r['spans'][k]['f1'] for r in rows if k in r['spans']])}" if k in rows[0]["spans"] else ""))
    print("  written interpretation: INSTRUCTIONS_REAL_2026-10-24.md table R7 (the axes and what each outcome will be said to mean)")
