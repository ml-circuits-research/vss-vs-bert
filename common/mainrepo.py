"""mainrepo.py: import path to the main repository (third_party/vsa-semantic-space at the pinned commit) and the world it defines.
Nothing in the main repository is modified; this module only makes its experiment/ modules importable and fixes the regime used in the
paper's controls (English world, prefixed outer forms, as v9_transformer.py / v22_transformer_categories.py, RESULTS 7b and 7b-bis)."""
import os, sys, subprocess

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
MAIN = os.path.join(ROOT, "third_party", "vsa-semantic-space")
EXP = os.path.join(MAIN, "experiment")
PINNED = "e531283"

def check_pinned():
    """the submodule must be at the pinned commit (INSTRUCTIONS section 2); a mismatch is an error, not a warning"""
    h = subprocess.run(["git", "rev-parse", "--short", "HEAD"], cwd=MAIN, capture_output=True, text=True).stdout.strip()
    if not h.startswith(PINNED): raise RuntimeError(f"third_party/vsa-semantic-space is at {h}, expected {PINNED}; run: git submodule update --init")
    return h

TOOL = os.path.join(MAIN, "tool")

def import_sop():
    """the main repository's reader-as-a-tool package (tool/sop), imported unmodified; returns the package"""
    if TOOL not in sys.path: sys.path.insert(0, TOOL)
    import sop
    return sop

def import_experiment():
    """returns (v5_scale, v7_nested) with the paper's regime set: English, prefixed reporting forms"""
    if EXP not in sys.path: sys.path.insert(0, EXP)
    import v5_scale as v5, v7_nested as v7
    v5.set_lang("en"); v5.OUTER = [o for o in v5.OUTER if o.index("<P>") > 0]
    return v5, v7
