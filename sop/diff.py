"""
diff.py — `sop diff a.sop b.sop`: the verifier (spec 3.7), the four verdicts of experiment/frame.py's diff with the margins.

Each top-level wire is flattened into slots (event, negation, speaker.<k>, <role>, <role>.attr, <role>.det); frame.diff's three sets are
computed (missing = only in a, extra = only in b, different = both, unequal). Verdict per compared pair:
  contradiction          a slot differs and the margins on that slot are >= tau on both sides (a side without a margin — e.g. a human or
                         LLM annotation — counts as certain); _role names the first such slot;
  unverifiable           every differing slot has a margin under tau on some side (the margin the verdict rests on is reported), or one
                         side is an `unknown` wire;
  compatible-incomplete  nothing differs, but a slot is filled on one side only;
  agreement              the flattened frames are equal.
Pairs: by `_sentence` when both files carry it, else by order. Wires are compared only within the same world (`_world`, else the file's
!world): a pair from two worlds gets `_verdict=unverifiable` with `_role=_world` (the mismatch is reported, never scored).
"""
from . import soplang, world

INF = float("inf")


def flatten(wire, by_id, legend=None):
    f = world.frame_of(wire, by_id, legend); slots, margins = {}, {}
    if f["shape"] != "ok": return f, None, None
    slots["event"] = f["event"]; slots["negation"] = "yes" if f["neg"] else "no"
    for k, s in enumerate(f["speakers"]): slots[f"speaker.{k}"] = s
    def rep(fl):
        if fl is None: return None
        if fl[0] == "word": return fl[1]
        if fl[0] == "negword": return f"not({fl[1]})"
        if fl[0] in ("set", "alt"): return (" & " if fl[0] == "set" else " | ").join(sorted(str(rep(x)) for x in fl[1:]))
        return "/".join(str(x) for x in fl)
    for r, fl in f["roles"].items():
        if fl[0] == "entity":
            slots[r] = fl[1]
            if fl[2] is not None: slots[r + ".attr"] = rep(fl[2])
            if fl[3] is not None: slots[r + ".det"] = fl[3]
        else: slots[r] = rep(fl)
    # margins: from the wires along the chain
    w, k = wire, 0
    def mval(wi, key):
        v = wi.value(key)
        try: return float(v) if v is not None else None
        except ValueError: return None
    top_struct = mval(wire, "_margin.structure")
    if top_struct is not None: margins["negation"] = top_struct
    while w is not None and w.type in ("says", "not"):
        if w.type == "says":
            m = mval(w, "_margin.speaker")
            if m is not None: margins[f"speaker.{k}"] = m
            k += 1
        s = w.get("scope"); w = by_id.get(s.atoms[0].text) if s is not None and s.atoms and s.atoms[0].kind == "ref" else None
    if w is not None:
        m = mval(w, "_margin.event")
        if m is not None: margins["event"] = m
        for l in w.lines:
            if l.key.startswith("_margin.") and l.key[8:] in f["roles"]: margins[l.key[8:]] = float(l.text)
            if not l.key.startswith("_") and len(l.atoms) == 1 and l.atoms[0].kind == "ref" and l.atoms[0].text in by_id:
                en = by_id[l.atoms[0].text]
                for part in ("attr", "det"):
                    m = mval(en, f"_margin.{part}")
                    if m is not None: margins[f"{l.key}.{part}"] = m
    return f, slots, margins


def verdict(sa, ma, sb, mb, tau):
    missing = {k: sa[k] for k in sa if k not in sb}; extra = {k: sb[k] for k in sb if k not in sa}
    different = {k: (sa[k], sb[k]) for k in sa if k in sb and sa[k] != sb[k]}
    order = sorted(different, key=lambda k: (k != "event", k != "negation", k))
    for k in order:
        la, lb = ma.get(k, INF), mb.get(k, INF)
        if min(la, lb) >= tau: return "contradiction", k, different[k], (la, lb)
    if different:
        k = order[0]; return "unverifiable", k, different[k], (ma.get(k, INF), mb.get(k, INF))
    if missing or extra:
        k = sorted(list(missing) + list(extra))[0]; return "compatible-incomplete", k, (sa.get(k), sb.get(k)), (ma.get(k, INF), mb.get(k, INF))
    return "agreement", None, None, None


def pairs(da, db):
    ta, tb = da.top_level(), db.top_level()
    sa = {w.value("_sentence"): w for w in ta}; sb = {w.value("_sentence"): w for w in tb}
    if None not in sa and None not in sb and len(sa) == len(ta) and len(sb) == len(tb):
        return [(sa[k], sb.get(k)) for k in sa] + [(None, sb[k]) for k in sb if k not in sa]
    n = max(len(ta), len(tb)); return [(ta[i] if i < len(ta) else None, tb[i] if i < len(tb) else None) for i in range(n)]


def fmtm(x): return "none" if x == INF else f"{x:.3f}"


def diff_docs(da, db, name_a, name_b, tau=0.05, legend=None):
    out = soplang.Doc(directives=[("left", name_a, None), ("right", name_b, None)]); ba, bb = da.by_id(), db.by_id(); counts = {}
    for i, (wa, wb) in enumerate(pairs(da, db), 1):
        v = soplang.Wire(f"d{i}", "verdict"); out.wires.append(v)
        v.set("left", wa.id if wa is not None else "none"); v.set("right", wb.id if wb is not None else "none")
        if wa is None or wb is None:
            v.set("_verdict", "compatible-incomplete"); v.set("_role", "wire"); counts["compatible-incomplete"] = counts.get("compatible-incomplete", 0) + 1; continue
        wa_w, wb_w = wa.value("_world") or da.directive("world"), wb.value("_world") or db.directive("world")
        if wa_w != wb_w:
            v.set("_verdict", "unverifiable"); v.set("_role", "_world"); v.set("_left", str(wa_w)); v.set("_right", str(wb_w))
            counts["world mismatch (unverifiable)"] = counts.get("world mismatch (unverifiable)", 0) + 1; continue
        fa, sa, ma = flatten(wa, ba, legend); fb, sb, mb = flatten(wb, bb, legend)
        if sa is None or sb is None:
            vd, role, vals, mg = "unverifiable", "event", (fa["shape"] if sa is None else sa.get("event"), fb["shape"] if sb is None else sb.get("event")), None
        else: vd, role, vals, mg = verdict(sa, ma, sb, mb, tau)
        v.set("_verdict", vd); counts[vd] = counts.get(vd, 0) + 1
        if role is not None:
            v.set("_role", role); v.set("_left", str(vals[0]) if vals[0] is not None else "none"); v.set("_right", str(vals[1]) if vals[1] is not None else "none")
        if mg is not None: v.set("_margin.left", fmtm(mg[0])); v.set("_margin.right", fmtm(mg[1]))
    return out, counts
