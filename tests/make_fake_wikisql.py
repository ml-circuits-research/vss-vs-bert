"""make_fake_wikisql.py: a tiny synthetic corpus in WikiSQL's file format for the CPU smoke tests of vss/run_reader_wikisql.py,
models/bert_wikisql.py and eval/error_analysis.py (written to tests/wikisql_fake/data, untracked). Never a result."""
import json, os, random
random.seed(0); d = os.path.join(os.path.dirname(os.path.abspath(__file__)), "wikisql_fake", "data"); os.makedirs(d, exist_ok=True)
tabs = [{"id": "t1", "header": ["Player", "Team", "Score"]}, {"id": "t2", "header": ["Year", "Country", "Winner", "Points"]}]
qs = [("what is the score of the player {x}", "t1", 2, 0), ("how many players are in team {x}", "t1", 0, 3), ("who won in {x}", "t2", 2, 0),
      ("what is the maximum points for {x}", "t2", 3, 1), ("which team does {x} play for", "t1", 1, 0)]
def rows(n):
    out = []
    for k in range(n):
        q, t, sel, agg = qs[k % len(qs)]; x = random.choice(["john", "blue", "1999", "france", "mary", "red", "2004", "italy"])
        out.append({"question": q.format(x=x), "table_id": t, "sql": {"sel": sel, "agg": agg, "conds": [[0, 0, x]]}})
    return out
for sp, n in (("train", 120), ("dev", 40), ("test", 40)):
    open(f"{d}/{sp}.jsonl", "w").write("\n".join(json.dumps(r) for r in rows(n)) + "\n")
    open(f"{d}/{sp}.tables.jsonl", "w").write("\n".join(json.dumps(t) for t in tabs) + "\n")
print("written", d)
