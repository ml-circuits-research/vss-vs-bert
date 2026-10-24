"""make_fake_real.py: tiny FAKE MASSIVE and TOPv2 files, written by this script (no third-party text), in the layout the loaders of
common/real_data.py read, for CPU tests of the loaders, the conversion and the scripts. Output: tests/real_fake/ (untracked). Use with
REAL_DATA=tests/real_fake. The TOPv2 fake has depths 0, 1 and 2 (nested intents inside slots) so that the depth split can be built.
Run: python3 tests/make_fake_real.py [--n 300]"""
import argparse, json, os
import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__)); OUT = os.path.join(HERE, "real_fake")
TIMES = ["five am", "six pm", "noon", "seven thirty", "midnight", "eight"]; DATES = ["tomorrow", "this week", "monday", "friday", "today"]
PLACES = ["paris", "new york", "the office", "home", "london"]; SONGS = ["blue moon", "yellow", "hey jude", "imagine"]
MASSIVE_T = [("alarm_set", "wake me up at [time : {T}] [date : {D}]"), ("alarm_set", "set an alarm for [time : {T}]"),
             ("weather_query", "what is the weather in [place_name : {P}] [date : {D}]"), ("weather_query", "will it rain [date : {D}]"),
             ("play_music", "play [song_name : {S}]"), ("play_music", "put on [song_name : {S}] please"),
             ("calendar_set", "remind me [date : {D}] at [time : {T}] to call [place_name : {P}]"), ("general_quirky", "tell me something funny")]
EVENTS = ["the concert", "my meeting", "the game", "dinner"]; NAV = ["directions to", "how do i get to", "navigate to"]

def massive(n, rng):
    os.makedirs(os.path.join(OUT, "massive", "1.1", "data"), exist_ok=True); rows = []
    for k in range(n):
        intent, t = MASSIVE_T[rng.integers(len(MASSIVE_T))]
        annot = t.format(T=rng.choice(TIMES), D=rng.choice(DATES), P=rng.choice(PLACES), S=rng.choice(SONGS))
        utt = annot
        import re
        utt = re.sub(r"\[[^:\]]+ : ([^\]]+)\]", r"\1", annot)
        part = "train" if k < 0.7 * n else ("dev" if k < 0.85 * n else "test")
        rows.append({"id": str(k), "locale": "en-US", "partition": part, "scenario": intent.split("_")[0], "intent": intent, "utt": utt, "annot_utt": annot})
    with open(os.path.join(OUT, "massive", "1.1", "data", "en-US.jsonl"), "w") as f:
        for r in rows: f.write(json.dumps(r) + "\n")

def top_parse(rng, depth):
    if depth == 0:
        if rng.random() < 0.5: return f"[IN:GET_WEATHER weather in [SL:LOCATION {rng.choice(PLACES)} ] ]"
        return f"[IN:CREATE_ALARM alarm for [SL:DATE_TIME {rng.choice(TIMES)} ] ]"
    if depth == 1 and rng.random() < 0.4:
        return f"[IN:GET_DIRECTIONS {rng.choice(NAV)} [SL:DESTINATION [IN:GET_LOCATION the place of [SL:EVENT {rng.choice(EVENTS)} ] ] ] ]"
    if depth == 1:
        return f"[IN:GET_DIRECTIONS {rng.choice(NAV)} [SL:DESTINATION [IN:GET_EVENT [SL:NAME_EVENT {rng.choice(EVENTS)} ] [SL:DATE_TIME {rng.choice(DATES)} ] ] ] ]"
    return (f"[IN:GET_DIRECTIONS {rng.choice(NAV)} [SL:DESTINATION [IN:GET_LOCATION the place of [SL:EVENT [IN:GET_EVENT "
            f"[SL:NAME_EVENT {rng.choice(EVENTS)} ] ] ] ] ] ]")

def topv2(n, rng):
    d = os.path.join(OUT, "topv2", "TOPv2_Dataset"); os.makedirs(d, exist_ok=True)
    for part, m in (("train", n), ("eval", n // 5), ("test", n // 5)):
        with open(os.path.join(d, f"navigation_{part}.tsv"), "w") as f:
            f.write("domain\tutterance\tsemantic_parse\n")
            for k in range(m):
                dep = int(rng.choice([0, 1, 2], p=[0.5, 0.35, 0.15])); p = top_parse(rng, dep)
                words = [w for w in p.split() if not w.startswith("[") and w != "]"]
                f.write(f"navigation\t{' '.join(words)}\t{p}\n")

if __name__ == "__main__":
    ap = argparse.ArgumentParser(); ap.add_argument("--n", type=int, default=300); a = ap.parse_args(); rng = np.random.default_rng(0)
    massive(a.n, rng); topv2(a.n, rng); print(f"fake MASSIVE and TOPv2 written under {OUT} (use REAL_DATA={OUT})")
