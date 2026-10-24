"""download.py: fetch a real-data source into data/real/raw/<dataset>/ (untracked, .gitignore) and check it against the sha256 pinned in
data/real/checksums.json. No dataset file is ever committed to this repository (the author's rule): only this script, the checksums, the
loaders and the split definitions (ids, seed and rule; no text).

  python3 data/real/download.py snips      # raw.githubusercontent.com (sonos/nlu-benchmark, 2017-06-custom-intent-engines), CC0 1.0
  python3 data/real/download.py massive    # the MASSIVE 1.1 tarball (Amazon S3), CC BY 4.0; extracts every locale
  python3 data/real/download.py topv2      # TOPv2_Dataset.zip (dl.fbaipublicfiles.com), CC BY-SA (check the LICENSE the archive carries)

A file whose pin is null in checksums.json (MASSIVE and TOPv2 could not be fetched from the machine where this repository was prepared)
is pinned on its first download: the script writes its sha256 into checksums.json and says so; commit that file, so that every later
download is checked against it (trust on first use, stated). A mismatch with an existing pin stops the script (exit 1) and nothing is
extracted. The LICENSE file found in an archive is printed.
Options: --force (re-download), --pin_only (verify and pin files already present)."""
import argparse, hashlib, json, os, ssl, sys, tarfile, urllib.request, zipfile

HERE = os.path.dirname(os.path.abspath(__file__)); RAW = os.environ.get("REAL_DATA", os.path.join(HERE, "raw"))
SUMS = os.path.join(HERE, "checksums.json")

def sha256(path):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for b in iter(lambda: f.read(1 << 20), b""): h.update(b)
    return h.hexdigest()

def fetch(url, out):
    os.makedirs(os.path.dirname(out), exist_ok=True)
    ca = os.environ.get("SSL_CERT_FILE") or os.environ.get("REQUESTS_CA_BUNDLE")
    ctx = ssl.create_default_context(cafile=ca) if ca else ssl.create_default_context()
    tmp = out + ".part"
    with urllib.request.urlopen(url, context=ctx, timeout=120) as r, open(tmp, "wb") as f:
        while True:
            b = r.read(1 << 20)
            if not b: break
            f.write(b)
    os.replace(tmp, out)

def check(sums, ds, rel, path):
    got = sha256(path); want = sums[ds]["files"].get(rel)
    if want is None:
        sums[ds]["files"][rel] = got; json.dump(sums, open(SUMS, "w"), indent=1); open(SUMS, "a").write("\n")
        print(f"  PINNED {rel}: sha256 {got} written to data/real/checksums.json (first download; commit that file)"); return True
    if got != want: print(f"  MISMATCH {rel}: sha256 {got}, pinned {want}: stop"); return False
    print(f"  ok {rel} sha256 {got[:16]}..."); return True

def safe_members_tar(t, dest):
    for m in t.getmembers():
        p = os.path.realpath(os.path.join(dest, m.name))
        if not p.startswith(os.path.realpath(dest) + os.sep) or m.issym() or m.islnk(): raise RuntimeError(f"unsafe archive member {m.name}")
        yield m

def main():
    ap = argparse.ArgumentParser(); ap.add_argument("dataset", choices=["snips", "massive", "topv2"]); ap.add_argument("--force", action="store_true")
    ap.add_argument("--pin_only", action="store_true"); a = ap.parse_args()
    sums = json.load(open(SUMS)); spec = sums[a.dataset]; dest = os.path.join(RAW, a.dataset); os.makedirs(dest, exist_ok=True)
    print(f"download {a.dataset} -> {dest} (untracked); licence: {spec['licence']}; source: {spec['source']}")
    ok = True
    for rel, url in spec["urls"].items():
        p = os.path.join(dest, rel)
        if a.force or not os.path.exists(p):
            if a.pin_only: print(f"  missing {rel}"); ok = False; continue
            print(f"  fetching {url}"); fetch(url, p)
        ok &= check(sums, a.dataset, rel, p)
    if not ok: sys.exit(1)
    for rel in spec["urls"]:
        p = os.path.join(dest, rel)
        if rel.endswith(".tar.gz"):
            with tarfile.open(p) as t: t.extractall(dest, members=safe_members_tar(t, dest))
        elif rel.endswith(".zip"):
            with zipfile.ZipFile(p) as z:
                for n in z.namelist():
                    q = os.path.realpath(os.path.join(dest, n))
                    if not q.startswith(os.path.realpath(dest) + os.sep): raise RuntimeError(f"unsafe archive member {n}")
                z.extractall(dest)
        else: continue
        print(f"  extracted {rel}")
    lic = [os.path.join(dp, f) for dp, _, fs in os.walk(dest) for f in fs if f.upper().startswith("LICENSE")]
    for l in lic[:3]: print(f"--- {os.path.relpath(l, dest)} (first lines) ---\n" + "".join(open(l, errors="replace").readlines()[:12]))
    if not lic: print("  no LICENSE file found in the download: record the licence of the primary source in data/real/README.md")
    print(f"done: {a.dataset} in {dest}")

if __name__ == "__main__":
    main()
