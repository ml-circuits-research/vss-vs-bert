"""
export_web.py — `sop export-web`: a trained model/ folder -> the files app/src/vss.js loads in the browser (VSS Diagram Editor spec section 6).

  model.json     config (D, seeds, star_version, thresholds, tau, depth estimator, presence readout), the legend (raw json) and its sha256,
                 the kinds lexicon (kinds.tsv) with lemmas and the legend's value patterns, the catalog words with kinds / lemmas, the schema,
                 the legend's reader block, the residual background statistics (scalars only)
  routebook.bin  every array of infer.Routebook (path, bg0, bg, A, L, rel, pos, pc as PHASES quantized to --bits; logits, rel_mask,
                 taker_mask as float32); cats, codes, pos_max in the header
  stars.bin      the stars the reader needs (catalog identities, legend verbs, role vectors, the decoder's operator vectors, the occupancy
                 marker, closed-value symbols), quantized to --bits
  manifest.json  sha256 of the three files, legend sha, training commit, bits, sizes

Binary layout of routebook.bin / stars.bin: 8 bytes magic b"VSSBIN1\\n", uint32 LE header length, the header (UTF-8 JSON), zero padding to
a multiple of 4, then the arrays at the offsets the header gives (relative to the start of the data block). A phase array with bits 8 stores
q = round((wrap(phase) + pi) / (2 pi) * 256) mod 256 per byte, dequantized as q * 2 pi / 256 - pi; bits 4 packs two 16-level values per
byte (low nibble first); bits 32 stores float32 phases unchanged. Phases are wrapped to [-pi, pi): every phase array is used through
exp(i phase) only (path through exp(i cumsum), invariant to multiples of 2 pi), so wrapping changes nothing but the stored range.
Usage: sop export-web --model <model dir> --out <dir> [--bits 8|4|32]
"""
import argparse, hashlib, json, os, sys
import numpy as np

MAGIC = b"VSSBIN1\n"
PHASE_ARRAYS = ("path", "bg0", "bg", "A", "L", "rel", "pos", "pc", "vo")
FLOAT_ARRAYS = ("logits", "rel_mask", "taker_mask", "verb_mask", "cue_mask")
DECODER_OPS = ("pred", "says.scope", "says.speaker", "not", "attr", "det", "wide", "q", "quantity", "must", "may", "if.condition",
               "if.consequence", "and.next")


def wrap(ph): return (np.asarray(ph, dtype=np.float64) + np.pi) % (2 * np.pi) - np.pi


def quantize(ph, bits):
    ph = wrap(ph).ravel()
    if bits == 32: return ph.astype("<f4").tobytes()
    L = 1 << bits; q = (np.round((ph + np.pi) / (2 * np.pi) * L).astype(np.int64) % L).astype(np.uint8)
    if bits == 8: return q.tobytes()
    if bits == 4:
        if len(q) % 2: q = np.concatenate([q, [0]]).astype(np.uint8)
        return (q[0::2] | (q[1::2] << 4)).astype(np.uint8).tobytes()
    raise ValueError(f"bits {bits}: expected 8, 4 or 32")


def dequantize(buf, bits, count):
    if bits == 32: return np.frombuffer(buf, dtype="<f4", count=count).astype(np.float64)
    if bits == 8: q = np.frombuffer(buf, dtype=np.uint8, count=count).astype(np.float64)
    elif bits == 4:
        b = np.frombuffer(buf, dtype=np.uint8, count=(count + 1) // 2); q = np.empty(2 * len(b)); q[0::2] = b & 15; q[1::2] = b >> 4; q = q[:count]
    else: raise ValueError(bits)
    return q * (2 * np.pi / (1 << bits)) - np.pi


def write_bin(path, header, blobs):
    data = b""; arrays = []
    for meta, blob in blobs:
        pad = (-len(data)) % 4; data += b"\0" * pad
        arrays.append(dict(meta, offset=len(data), nbytes=len(blob))); data += blob
    header = dict(header, arrays=arrays); h = json.dumps(header, separators=(",", ":")).encode("utf-8")
    pad = (-(len(MAGIC) + 4 + len(h))) % 4
    with open(path, "wb") as f: f.write(MAGIC + len(h).to_bytes(4, "little") + h + b"\0" * pad + data)


def read_bin(path):
    """-> (header, {name: numpy array}) with phases DEQUANTIZED to float64 (what the browser reads)"""
    raw = open(path, "rb").read()
    assert raw[:8] == MAGIC, f"{path}: not a vss binary"
    n = int.from_bytes(raw[8:12], "little"); header = json.loads(raw[12:12 + n]); start = 12 + n + ((-(12 + n)) % 4)
    out = {}
    for a in header["arrays"]:
        buf = raw[start + a["offset"]: start + a["offset"] + a["nbytes"]]; count = int(np.prod(a["shape"])) if a["shape"] else 1
        v = dequantize(buf, a["bits"], count) if a["kind"] == "phase" else np.frombuffer(buf, dtype="<f4", count=count).astype(np.float64)
        out[a["name"]] = v.reshape(a["shape"])
    return header, out


def star_keys(reader):
    """every (key, vector) the decoder can ask for. Keys: id:<token>, role:<r>, op:<name>, marker, sym:<value>"""
    st, L = reader.st, reader.legend; out = {}
    words = set(reader.words) | set(L.lexicon) | set(L.all_verbs()) | set(L.lemma.values())
    from .world import value_token
    for w in sorted(words): out["id:" + value_token(w)] = st.id(w)
    for r in L.roles: out["role:" + r] = st.role(r)
    for o in DECODER_OPS: out["op:" + o] = st.op(o)
    out["marker"] = st.marker()
    for r, vals in (L.reader.get("closed_values") or {}).items():
        for v in vals: out["sym:" + str(v)] = st.get("sym:" + str(v), st.star_seed)
    return out


def export(model_dir, out_dir, bits=8):
    from .infer import Reader
    rd = Reader(model_dir); os.makedirs(out_dir, exist_ok=True); rb = rd.rb
    # ---- routebook.bin
    blobs = []
    for name in PHASE_ARRAYS:
        v = getattr(rb, name, None)
        if v is None: continue
        blobs.append(({"name": name, "kind": "phase", "bits": bits, "shape": list(v.shape)}, quantize(v, bits)))
    for name in FLOAT_ARRAYS:
        v = getattr(rb, name, None)
        if v is None: continue
        blobs.append(({"name": name, "kind": "float", "bits": 32, "shape": list(v.shape)}, np.asarray(v, dtype="<f4").tobytes()))
    write_bin(os.path.join(out_dir, "routebook.bin"), {"cats": rb.cats, "codes": rb.codes, "pos_max": rb.pos_max, "vo_nmax": rb.vo_nmax, "D": rb.D, "bits": bits}, blobs)
    # ---- stars.bin
    stars = star_keys(rd); keys = sorted(stars); M = np.stack([np.angle(stars[k]) for k in keys])
    write_bin(os.path.join(out_dir, "stars.bin"), {"keys": keys, "D": rd.D, "bits": bits},
              [({"name": "stars", "kind": "phase", "bits": bits, "shape": list(M.shape)}, quantize(M, bits))])
    # ---- model.json
    L = rd.legend; bg = {}
    if rd.background is not None:
        for k, v in rd.background.items():
            if np.ndim(v) == 0: bg[k] = float(v)
    model = {"format": "vss-web-1", "world": rd.cfg.get("world"), "config": rd.cfg, "legend": L.raw, "legend_sha256": L.sha,
             "lexicon": {w: ({"kind": k, "lemma": L.lemma[w]} if w in L.lemma else {"kind": k}) for w, k in sorted(L.lexicon.items())},
             "values": L.raw.get("values", []), "reader": L.reader, "catalog": {"words": rd.words, "star_seed": rd.cat.get("star_seed"),
             "star_version": rd.cfg.get("star_version", 1), "identity_table": bool(rd.cat.get("identity_table"))},
             "schema": rd.schema, "background": bg, "bits": bits}
    with open(os.path.join(out_dir, "model.json"), "w") as f: json.dump(model, f, separators=(",", ":"))
    files = ["model.json", "routebook.bin", "stars.bin"]
    man = {"format": "vss-web-1", "legend_sha256": L.sha, "training_code_commit": rd.cfg.get("training_code_commit"), "bits": bits,
           "star_version": rd.cfg.get("star_version", 1), "D": rd.D, "n_stars": len(keys),
           "files": {fn: {"sha256": hashlib.sha256(open(os.path.join(out_dir, fn), "rb").read()).hexdigest(), "bytes": os.path.getsize(os.path.join(out_dir, fn))} for fn in files}}
    with open(os.path.join(out_dir, "manifest.json"), "w") as f: json.dump(man, f, indent=1)
    return man


def main(argv=None):
    ap = argparse.ArgumentParser(prog="sop export-web"); ap.add_argument("--model", required=True); ap.add_argument("--out", required=True)
    ap.add_argument("--bits", type=int, default=8, choices=[4, 8, 32]); a = ap.parse_args(argv)
    man = export(a.model, a.out, a.bits)
    tot = sum(f["bytes"] for f in man["files"].values())
    print(f"export-web: {a.model} -> {a.out} (bits {a.bits}, D {man['D']}, star_version {man['star_version']}, {man['n_stars']} stars): "
          + ", ".join(f"{k} {v['bytes'] / 1e6:.3f} MB" for k, v in man["files"].items()) + f"; total {tot / 1e6:.3f} MB")
    if man["star_version"] == 1:
        print("export-web: NOTE star_version 1 — the browser reads only the exported stars; words outside the catalog get no identity "
              "(train with --star_version 2 for a model whose new words, labels and numbers are read in the browser)")
    return 0


if __name__ == "__main__":
    sys.exit(main())
