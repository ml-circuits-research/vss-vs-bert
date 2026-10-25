"""llm_reference.py (deviation R27 item 2, responses R27, 2026-10-25): the LLM reference row ("reference, not a baseline") on the basic
world. `--write_prompt` writes the FROZEN prompt eval/llm_reference_prompt.txt (the legend's questions as in models/bert_mrc.QUESTIONS, the
answer format, ten solved examples from the TRAINING sentences of seed 0, two per depth 0-2 and four more of mixed depth, chosen by a
fixed rule) and the fixed item list (200 report items per axis of seed 0, the first 200 of each axis in file order) with a token estimate.
NO external call is made by this script until questions.md Q20 is answered (model, provider, budget cap); `--call` is not implemented
until then. The row stays PLANNED with the prompt committed.
Run: python3 eval/llm_reference.py --write_prompt"""
import argparse, json, os, sys
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from common import mainrepo
v5, v7 = mainrepo.import_experiment()
from common import basic_world as bwm, report
from models.bert_basic import make_train
from models.bert_mrc import QUESTIONS

HEAD = """You read sentences of a small constructed world into frames. Answer each question of the legend from the sentence only.
Legend questions (all six are asked of every sentence; level 1 is the outermost speaker):
""" + "\n".join(f"  {r}: {q}" for r, q in QUESTIONS.items()) + """
Answer format: six lines, A, B, C, X1, X2, X3 in this order, each "ROLE: token", where token is one word copied from the sentence, or
"ROLE: not given" if the sentence does not answer that question. Give nothing else.
"""

def fmt(tokens, frame, depth):
    return "Sentence: " + " ".join(tokens) + "\n" + "\n".join(f"{r}: {frame[r] if frame[r] is not None else 'not given'}" for r in bwm.ROLES)

if __name__ == "__main__":
    ap = argparse.ArgumentParser(); ap.add_argument("--write_prompt", action="store_true"); ap.add_argument("--n", type=int, default=200); a = ap.parse_args()
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
