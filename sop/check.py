"""
check.py — `sop check`: validate SOP files against one or several legends (spec 3.6, plus the `_world` amendment).

Checks: syntax (soplang.parse); ids unique; every $id resolves; no reference cycle; every key is a legend role (on event wires), an
operator/entity key (on those wires) or reserved (_...); every type is a legend event, `entity`, an operator the legend admits, or `unknown`
(`verdict` only in files with !left/!right, i.e. `sop diff` output); & and | never mix (parser); exactly one _text per top-level wire;
_asked only with _mood=question; the legend hash matches; no duplicate key in a wire.

Several worlds in one file (amendment of 2026-10-16): a top-level wire may carry `_world=<name>`; the wires it references belong to the same
world. Each wire is checked against the legend of ITS world. World of a top-level wire = its `_world` line, else the file's `!world`
directive, else the default world. Legend hashes, one directive per world:
    !legend sha256:<hex>             the hash of the default world's legend (the single-legend form)
    !legend <world> sha256:<hex>     the hash of the legend of <world>
Legends are given by name (`--legend name=path`, `--legends <dir>` = <dir>/<name>/legend.json) or by path (the name is the folder that holds
legend.json, skipping `model/` and `versions/v<N>/`). Single-legend form: when ONE legend is given and the file names at most one world,
that legend is the file's legend whatever the names. A file that names two worlds needs a legend for each.
Every problem is reported as (line number, reason).
"""
import os, re
from . import soplang


class Report:
    def __init__(self, path): self.path, self.errors, self.warnings, self.n_wires = path, [], [], 0
    def err(self, lineno, msg): self.errors.append((lineno, msg))
    def warn(self, lineno, msg): self.warnings.append((lineno, msg))
    @property
    def ok(self): return not self.errors
    def lines(self):
        out = [f"{'OK   ' if self.ok else 'ERROR'} {self.path} ({self.n_wires} wires, {len(self.errors)} errors, {len(self.warnings)} warnings)"]
        out += [f"  {self.path}:{l or '-'}: error: {m}" for l, m in self.errors] + [f"  {self.path}:{l or '-'}: warning: {m}" for l, m in self.warnings]
        return out
    def as_json(self): return {"file": self.path, "ok": self.ok, "wires": self.n_wires, "errors": [{"line": l, "reason": m} for l, m in self.errors],
                               "warnings": [{"line": l, "reason": m} for l, m in self.warnings]}


def legend_name(path):
    d = os.path.dirname(os.path.abspath(path))
    while os.path.basename(d) == "model" or re.match(r"^v\d+$", os.path.basename(d)) or os.path.basename(d) == "versions":
        d = os.path.dirname(d)
    return os.path.basename(d)


def legend_hashes(doc):
    """{world name or None (default): (hash, lineno)} from the !legend directives"""
    out = {}
    for n, v, l in doc.directives:
        if n != "legend": continue
        parts = v.split()
        if len(parts) == 1: out[None] = (parts[0], l)
        elif len(parts) == 2: out[parts[0]] = (parts[1], l)
        else: out.setdefault("__bad__", []).append(l)
    return out


def wire_worlds(doc):
    """{wire id: world name or None}; a referenced wire takes the world of the top-level wire that reaches it; conflicts listed"""
    by = doc.by_id(); default = doc.directive("world"); world_of, conflicts = {}, []
    for top in doc.top_level():
        wv = top.value("_world"); wname = wv if wv is not None else default
        stack, seen = [top.id], set()
        while stack:
            x = stack.pop()
            if x in seen or x not in by: continue
            seen.add(x)
            if x in world_of and world_of[x] != wname: conflicts.append((by[x], world_of[x], wname))
            world_of.setdefault(x, wname); stack += by[x].refs()
    for w in doc.wires: world_of.setdefault(w.id, default)
    return world_of, conflicts


def resolve_legends(doc, legends, rep):
    """legends: {name: Legend}. Returns {world name or None: Legend or None}"""
    world_of, conflicts = wire_worlds(doc)
    for w, a, b in conflicts: rep.err(w.lineno, f"wire {w.id} is referenced from two worlds ({a!r} and {b!r})")
    names = {n for n in world_of.values()}
    out = {}
    if len(legends) == 1 and len({n for n in names if n is not None}) <= 1:
        (lname, lg), = legends.items()
        return {n: lg for n in names | {None}}, world_of
    for n in names:
        if n is not None and n in legends: out[n] = legends[n]
        elif n is None:
            out[None] = legends.get(doc.directive("world")) if doc.directive("world") in legends else (next(iter(legends.values())) if len(legends) == 1 else None)
        else: out[n] = None
    return out, world_of


def check_doc(doc, legends, rep, require_text=True, force=False):
    """legends: a Legend (single-legend form) or {name: Legend}"""
    if not isinstance(legends, dict): legends = {"_given": legends}
    rep.n_wires = len(doc.wires)
    is_diff = doc.directive("left") is not None and doc.directive("right") is not None
    lmap, world_of = resolve_legends(doc, legends, rep)
    # hash checks, per world
    lh = legend_hashes(doc)
    for l in lh.get("__bad__", []): rep.err(l, "bad !legend directive (expected '!legend sha256:<hex>' or '!legend <world> sha256:<hex>')")
    for key, (h, l) in lh.items():
        if key == "__bad__": continue
        lg = lmap.get(key) if key in lmap else (legends.get(key) if key is not None else None)
        if key is None and lg is None: lg = lmap.get(doc.directive("world"))
        if lg is None:
            if key is not None and key not in lmap: rep.warn(l, f"!legend for world {key!r}, which no wire of this file uses and no legend given names")
            continue
        if h.strip() != lg.sha:
            (rep.warn if force else rep.err)(l, f"legend hash mismatch{' for world ' + repr(key) if key else ''}: file has {h.strip()}, legend given is {lg.sha}")
    used = {world_of[w.id] for w in doc.wires}
    for n in used:
        lg = lmap.get(n)
        hkey = n if n in lh else (None if None in lh else None)
        if lg is not None and n not in lh and None not in lh and not is_diff:
            rep.warn(None, f"no !legend directive for world {n!r}: its legend hash cannot be checked")
    ids = {}
    for w in doc.wires:
        if w.id in ids: rep.err(w.lineno, f"duplicate wire id {w.id!r} (first declared at line {ids[w.id].lineno})")
        else: ids[w.id] = w
    for w in doc.wires:
        legend = lmap.get(world_of.get(w.id))
        if legend is None:
            rep.err(w.lineno, f"no legend given for world {world_of.get(w.id)!r} (wire {w.id}); pass --legend {world_of.get(w.id)}=<legend.json>")
            continue
        types = legend.wire_types() | ({"verdict"} if is_diff else set())
        if w.type not in types:
            rep.err(w.lineno, f"unknown wire type {w.type!r} (legend events {sorted(legend.events)}, operators {legend.operators}, entity, unknown)")
            continue
        allowed = legend.keys_for(w.type); seen = {}
        for l in w.lines:
            if l.key in seen: rep.err(l.lineno, f"duplicate key {l.key!r} in wire {w.id} (first at line {seen[l.key]})")
            seen.setdefault(l.key, l.lineno)
            if not l.key.startswith("_") and l.key not in allowed:
                rep.err(l.lineno, f"unknown key {l.key!r} for wire type {w.type!r} (allowed: {sorted(allowed)} or reserved _...)")
            if l.key == "_text": continue
            for r in l.refs():
                if r not in ids: rep.err(l.lineno, f"dangling reference ${r} (no wire {r!r} in this file)")
            if w.type == "verdict" and l.key in ("left", "right") and any(a.kind == "ref" for a in l.atoms):
                rep.err(l.lineno, "a verdict names compared wires by plain id, not $ref")
        sc = w.get("_scope")
        if sc is not None:
            if w.type not in legend.events: rep.err(sc.lineno, f"_scope is only allowed on event wires (wire {w.id} is {w.type!r})")
            elif sc.text not in legend.roles and sc.text != "both": rep.err(sc.lineno, f"_scope={sc.text!r}: must be a legend role ({legend.roles}) or 'both'")
        asked = w.get("_asked")
        if asked is not None and w.value("_mood") != "question":
            rep.err(asked.lineno, "_asked is only allowed with _mood=question")
    state = {}
    def dfs(wid, stack):
        state[wid] = 1
        for r in ids[wid].refs():
            if r not in ids: continue
            if state.get(r) == 1:
                cyc = stack[stack.index(r):] + [r] if r in stack else [wid, r]
                rep.err(ids[r].lineno, "reference cycle: " + " -> ".join("$" + x for x in cyc)); continue
            if r not in state: dfs(r, stack + [r])
        state[wid] = 2
    for wid in ids:
        if wid not in state: dfs(wid, [wid])
    tops = {w.id for w in doc.top_level()}
    for w in doc.wires:
        if w.get("_world") is not None and w.id not in tops: rep.err(w.get("_world").lineno, f"_world on wire {w.id}, which is not top-level (it is referenced)")
    if require_text and not is_diff:
        for w in doc.top_level():
            n = sum(1 for l in w.lines if l.key == "_text")
            if n != 1: rep.err(w.lineno, f"top-level wire {w.id} has {n} _text lines (exactly one required)")
    return rep


def check_file(path, legends, require_text=True, force=False):
    rep = Report(path)
    try:
        with open(path, encoding="utf-8") as f: text = f.read()
    except (OSError, UnicodeDecodeError) as e:
        rep.err(None, f"cannot read: {e}"); return rep, None
    try: doc = soplang.parse(text)
    except soplang.SopError as e:
        rep.err(e.lineno, e.msg); return rep, None
    return check_doc(doc, legends, rep, require_text, force), doc
