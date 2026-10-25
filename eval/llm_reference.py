"""llm_reference.py (deviation R27 item 2, responses R27, 2026-10-25): the LLM reference row ("reference, not a baseline") on the basic
world. `--write_prompt` writes the FROZEN prompt eval/llm_reference_prompt.txt (the legend's questions as in models/bert_mrc.QUESTIONS, the
answer format, ten solved examples from the TRAINING sentences of seed 0, two per depth 0-2 and four more of mixed depth, chosen by a
fixed rule) and the fixed item list (200 report items per axis of seed 0, the first 200 of each axis in file order) with a token estimate.
`--call` (responses R32): the Anthropic API through the official SDK, model claude-sonnet-5-5, the frozen prompt as a cached system block
(cache_control ephemeral; below the model's 512-token cacheable minimum it silently does not cache, and the usage fields say so), one
user message per sentence, three repeats, a budget cap in USD from the usage fields and the prices passed on the command line (read on
the provider's pricing page at call time and logged), the spend logged every 100 calls, stop at the cap. No sampling parameter is sent
(questions Q22: the API rejects temperature since 2026-10-21); thinking {"type": "between_tools"} at effort low, the lowest setting
claude-sonnet-5-5 accepts (R33); no server-side fallback (a refusal is recorded as such, so that every answer
comes from the named model). The key is read from ANTHROPIC_API_KEY only; the script never prints, logs or stores it, the request
headers or the environment. Writes logs/results_llm_reference_repeat<k>.json and prints the scores.
Run: python3 eval/llm_reference.py --write_prompt | ./run.sh eval/llm_reference.py --call --price_in P --price_out P --price_cache_write P --price_cache_read P [--cap 20] [--effort low]"""
import argparse, json, os, re, sys, time
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from common import mainrepo
v5, v7 = mainrepo.import_experiment()
from common import basic_world as bwm, report
from models.bert_basic import make_train
from models.bert_mrc import QUESTIONS
import numpy as np

HEAD = """You read sentences of a small constructed world into frames. Answer each question of the legend from the sentence only.
Legend questions (all six are asked of every sentence; level 1 is the outermost speaker):
""" + "\n".join(f"  {r}: {q}" for r, q in QUESTIONS.items()) + """
Answer format: six lines, A, B, C, X1, X2, X3 in this order, each "ROLE: token", where token is one word copied from the sentence, or
"ROLE: not given" if the sentence does not answer that question. Give nothing else.
"""

def fmt(tokens, frame, depth):
    return "Sentence: " + " ".join(tokens) + "\n" + "\n".join(f"{r}: {frame[r] if frame[r] is not None else 'not given'}" for r in bwm.ROLES)

def parse(text):
    f = {r: None for r in bwm.ROLES}
    for line in text.splitlines():
        m = re.match(r"\s*(A|B|C|X1|X2|X3)\s*:\s*(.+?)\s*$", line)
        if m: f[m.group(1)] = None if m.group(2).lower().startswith("not given") else m.group(2).split()[0]
    return f

def call(a, prompt, split, items):
    import anthropic
    if not os.environ.get("ANTHROPIC_API_KEY"): sys.exit("ANTHROPIC_API_KEY is not set: source the key file in the launching shell (see R32); nothing was called")
    client = anthropic.Anthropic(); spent = 0.0; n = 0; out = {}
    price = lambda u: (u.input_tokens * a.price_in + u.output_tokens * a.price_out + (u.cache_creation_input_tokens or 0) * a.price_cache_write + (u.cache_read_input_tokens or 0) * a.price_cache_read) / 1e6
    print(f"llm_reference --call: model {a.model}, thinking between_tools, effort {a.effort}, no sampling parameter, prices per M tokens in {a.price_in} out {a.price_out} cache write {a.price_cache_write} cache read {a.price_cache_read} (read on the provider's page at call time), cap {a.cap} USD, {a.repeats} repeats", flush=True)
    for k in range(1, a.repeats + 1):
        res = {}
        for ax, idx in items.items():
            for i in idx:
                it = split["axes"][ax][i]
                try:
                    r = client.messages.create(model=a.model, max_tokens=a.max_tokens, output_config={"effort": a.effort}, thinking={"type": "between_tools"},
                                               system=[{"type": "text", "text": prompt, "cache_control": {"type": "ephemeral"}}],
                                               messages=[{"role": "user", "content": "Sentence: " + " ".join(it["tokens"])}])
                except anthropic.APIStatusError as e:
                    print(f"  repeat {k} {ax}[{i}]: API error status {e.status_code}; stopping", flush=True); return out, spent, n
                text = "".join(b.text for b in r.content if b.type == "text"); u = r.usage; c = price(u); spent += c; n += 1
                res[f"{ax}:{i}"] = {"frame": parse(text), "stop_reason": r.stop_reason, "usage": {"in": u.input_tokens, "out": u.output_tokens, "cache_write": u.cache_creation_input_tokens or 0, "cache_read": u.cache_read_input_tokens or 0}, "cost": c, "model": r.model}
                if n % 100 == 0: print(f"  {n} calls, spent {spent:.4f} USD (cache read tokens this call {u.cache_read_input_tokens or 0})", flush=True)
                if spent >= a.cap:
                    print(f"  budget cap {a.cap} USD reached after {n} calls (repeat {k}); stopping and reporting", flush=True); out[k] = res; return out, spent, n
        out[k] = res
        json.dump(res, open(os.path.join(mainrepo.ROOT, "logs", f"results_llm_reference_repeat{k}.json"), "w"))
    return out, spent, n

def score(out, split, items):
    for k, res in out.items():
        for ax, idx in items.items():
            got = [(split["axes"][ax][i], res.get(f"{ax}:{i}")) for i in idx if f"{ax}:{i}" in res]
            if not got: continue
            slot = np.mean([p["frame"][r] == it["frame"][r] for it, p in got for r in bwm.roles_at(it["depth"])])
            em = np.mean([all(p["frame"][r] == it["frame"][r] for r in bwm.ROLES) for it, p in got])
            ff = sum(p["frame"][r] is not None and it["frame"][r] is None for it, p in got for r in bwm.ROLES)
            wrong = sum(p["frame"][r] is not None and it["frame"][r] is not None and p["frame"][r] != it["frame"][r] for it, p in got for r in bwm.ROLES)
            ref = sum(p["stop_reason"] == "refusal" for _, p in got)
            print(f"  repeat {k} {ax:24s} n {len(got)}: slot accuracy {slot:.3f}, exact match (six roles) {em:.3f}, false fills {ff}, wrong values {wrong}, refusals {ref}")
    ks = sorted(out)
    if len(ks) > 1:
        keys = set.intersection(*[set(out[k]) for k in ks]); same = np.mean([all(out[k][x]["frame"] == out[ks[0]][x]["frame"] for k in ks) for x in keys]) if keys else float("nan")
        print(f"  determinism: frames identical across the {len(ks)} repeats on {same:.3f} of {len(keys)} items")

if __name__ == "__main__":
    ap = argparse.ArgumentParser(); ap.add_argument("--write_prompt", action="store_true"); ap.add_argument("--n", type=int, default=200)
    ap.add_argument("--call", action="store_true"); ap.add_argument("--model", default="claude-sonnet-5-5"); ap.add_argument("--effort", default="low"); ap.add_argument("--max_tokens", type=int, default=1024)
    ap.add_argument("--repeats", type=int, default=3); ap.add_argument("--cap", type=float, default=20.0)
    for p_ in ("price_in", "price_out", "price_cache_write", "price_cache_read"): ap.add_argument(f"--{p_}", type=float, default=None)
    a = ap.parse_args()
    split = bwm.load(0); bw = bwm.build(0, split["D"]); data = make_train(bw, 0, 2000); by = {0: [], 1: [], 2: []}
    for t, g in data: by[sum(x.startswith("X") for x in g)].append((t, g))
    ex = by[0][:2] + by[1][:2] + by[2][:2] + [d for d in data[:40] if d not in by[0][:2] + by[1][:2] + by[2][:2]][:4]
    shots = "\n\n".join(fmt(t, bwm.frame_from_tags(t, g), sum(x.startswith("X") for x in g)) for t, g in ex)
    prompt = HEAD + "\nSolved examples:\n\n" + shots + "\n\nNow the sentence to read:\n"
    items = {ax: [i for i in range(min(a.n, len(split["axes"][ax])))] for ax in report.AXES if split["axes"][ax]}
    n_items = sum(len(v) for v in items.values()); p_tok = len(prompt) // 4; i_tok = 30; o_tok = 40
    print(f"prompt {len(prompt)} characters (about {p_tok} tokens); items {n_items} over {len(items)} axes; with 3 repeats {3 * n_items} calls; "
          f"about {3 * n_items * (p_tok + i_tok) / 1e6:.2f} M input and {3 * n_items * o_tok / 1e6:.2f} M output tokens")
    if a.write_prompt:
        open(os.path.join(mainrepo.ROOT, "eval", "llm_reference_prompt.txt"), "w").write(prompt)
        json.dump({"seed": 0, "items": items, "note": "first n items of each axis in file order (report rows; seed 0)"}, open(os.path.join(mainrepo.ROOT, "eval", "llm_reference_items.json"), "w"))
        print("written eval/llm_reference_prompt.txt and eval/llm_reference_items.json (frozen; no call made)")
    if a.call:
        assert all(getattr(a, p_) is not None for p_ in ("price_in", "price_out", "price_cache_write", "price_cache_read")), "pass the four prices read at call time"
        frozen = open(os.path.join(mainrepo.ROOT, "eval", "llm_reference_prompt.txt")).read(); assert frozen == prompt, "the frozen prompt file differs from the generated prompt"
        items = json.load(open(os.path.join(mainrepo.ROOT, "eval", "llm_reference_items.json")))["items"]
        out, spent, n = call(a, frozen, split, items); print(f"llm_reference: {n} calls, spent {spent:.4f} USD (label: reference (in-context LLM), not a baseline; 10-shot, not data-matched)")
        score(out, split, items)
