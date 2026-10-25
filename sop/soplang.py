"""
soplang.py — reader and writer for SOP Lang (spec section 3).

A file is a sequence of directives ("!name value") followed by wires. A wire is a header "@id type" followed by "key=value" lines.
Values (3.2): a bare token, a quoted string ("..." with \\" and \\\\ escapes), a reference $id, the occupancy marker ~, the asked role ?,
and sets (a & b) or alternatives (a | b) of those atoms; & and | never mix in one value. `_text` is free text to the end of the line.

Round trip: every parsed line keeps its raw value text; `write()` uses it unchanged, so a file in the canonical layout (directives, one
blank line, wires separated by one blank line) is written back byte for byte. Comments (# ...) are not preserved.
"""
import re

ID_RE = re.compile(r"^[A-Za-z0-9_-]+$")
BARE_OK = re.compile(r'^[^\s$~?"&|][^\s"&|]*$')          # one token, no spaces, does not start with $ ~ ?, no & | "
RESERVED_KEY = re.compile(r"^_[A-Za-z][A-Za-z0-9_]*(\.[A-Za-z0-9_-]+)?$")
ROLE_KEY = re.compile(r"^[A-Za-z][A-Za-z0-9_]*$")
FREE_TEXT_KEYS = {"_text"}                                   # the answer runs to the end of the line, spaces included


class SopError(Exception):
    def __init__(self, msg, lineno=None):
        super().__init__(msg); self.msg, self.lineno = msg, lineno


class Atom:
    """kind: 'word' (text, quoted flag) | 'ref' (text = wire id) | 'unknown' (~) | 'asked' (?)"""
    __slots__ = ("kind", "text", "quoted")
    def __init__(self, kind, text=None, quoted=False): self.kind, self.text, self.quoted = kind, text, quoted
    def __eq__(self, o): return isinstance(o, Atom) and (self.kind, self.text) == (o.kind, o.text)
    def __hash__(self): return hash((self.kind, self.text))
    def __repr__(self): return format_atom(self)


class Line:
    __slots__ = ("key", "raw", "atoms", "sep", "lineno")
    def __init__(self, key, raw=None, atoms=None, sep=None, lineno=None):
        self.key, self.raw, self.atoms, self.sep, self.lineno = key, raw, atoms or [], sep, lineno
    @property
    def text(self):
        """the value as plain text (free-text keys, or a single word atom)"""
        if self.key in FREE_TEXT_KEYS: return self.raw if not self.atoms else self.atoms[0].text
        if len(self.atoms) == 1 and self.atoms[0].kind == "word": return self.atoms[0].text
        return self.raw if self.raw is not None else format_value(self.atoms, self.sep)
    def refs(self): return [a.text for a in self.atoms if a.kind == "ref"]


def line(key, value):
    """a Line built from a python value (quoted on output when needed), never from raw text"""
    if key in FREE_TEXT_KEYS: return Line(key, str(value).replace("\n", " "))
    atoms, sep = to_atoms(value); return Line(key, None, atoms, sep)


class Wire:
    __slots__ = ("id", "type", "lines", "lineno")
    def __init__(self, id, type, lines=None, lineno=None): self.id, self.type, self.lines, self.lineno = id, type, lines or [], lineno
    def get(self, key):
        for l in self.lines:
            if l.key == key: return l
        return None
    def value(self, key, default=None):
        l = self.get(key); return default if l is None else l.text
    def set(self, key, value):
        """set a line from a python value: str (word), ('ref', id), list of those with sep; raw is rebuilt"""
        atoms, sep = to_atoms(value)
        l = self.get(key)
        if l is None: l = Line(key); self.lines.append(l)
        l.atoms, l.sep, l.raw = atoms, sep, None
        if key in FREE_TEXT_KEYS: l.raw = str(value).replace("\n", " "); l.atoms = []
        return l
    def refs(self): return [r for l in self.lines for r in l.refs()]


class Doc:
    def __init__(self, directives=None, wires=None): self.directives, self.wires = directives or [], wires or []
    def directive(self, name):
        for n, v, _ in self.directives:
            if n == name: return v
        return None
    def by_id(self): return {w.id: w for w in self.wires}
    def top_level(self):
        """wires not referenced by any other wire, in file order"""
        referenced = {r for w in self.wires for r in w.refs()}
        return [w for w in self.wires if w.id not in referenced]


def to_atoms(value):
    if isinstance(value, Atom): return [value], None
    if isinstance(value, tuple) and len(value) == 2 and value[0] == "ref": return [Atom("ref", value[1])], None
    if isinstance(value, (list, tuple)) and len(value) == 2 and value[0] in ("&", "|"):
        atoms = [to_atoms(v)[0][0] for v in value[1]]; return atoms, (value[0] if len(atoms) > 1 else None)
    if value == "~": return [Atom("unknown")], None
    if value == "?": return [Atom("asked")], None
    return [Atom("word", str(value))], None


def parse_value(s, lineno=None):
    """value := atom (sep atom)* ; returns (atoms, sep)"""
    i, n, atoms, seps = 0, len(s), [], set()
    def skip(i):
        while i < n and s[i] in " \t": i += 1
        return i
    i = skip(i)
    if i >= n: raise SopError("empty value", lineno)
    while True:
        i = skip(i)
        if i >= n: raise SopError("value ends with a separator", lineno)
        c = s[i]
        if c == '"':
            j, buf = i + 1, []
            while j < n and s[j] != '"':
                if s[j] == "\\":
                    if j + 1 >= n or s[j + 1] not in '"\\': raise SopError("bad escape in quoted value (only \\\" and \\\\)", lineno)
                    buf.append(s[j + 1]); j += 2
                else: buf.append(s[j]); j += 1
            if j >= n: raise SopError("unterminated quoted value", lineno)
            atoms.append(Atom("word", "".join(buf), quoted=True)); i = j + 1
        elif c == "$":
            m = re.match(r"\$([A-Za-z0-9_-]+)", s[i:])
            if not m: raise SopError("bad reference after $", lineno)
            atoms.append(Atom("ref", m.group(1))); i += m.end()
        elif c in "~?":
            atoms.append(Atom("unknown" if c == "~" else "asked")); i += 1
        elif c in "&|":
            raise SopError(f"separator {c!r} without an atom before it", lineno)
        else:
            j = i
            while j < n and s[j] not in " \t&|": j += 1
            tok = s[i:j]
            if '"' in tok: raise SopError(f"bare value {tok!r} contains a quote (quote the value)", lineno)
            atoms.append(Atom("word", tok)); i = j
        i = skip(i)
        if i >= n: break
        if s[i] not in "&|": raise SopError(f"unexpected text {s[i:]!r} (a value with spaces must be quoted)", lineno)
        seps.add(s[i]); i += 1
    if len(seps) > 1: raise SopError("'&' and '|' mixed in one value", lineno)
    return atoms, (seps.pop() if seps else None)


def quote(t): return '"' + t.replace("\\", "\\\\").replace('"', '\\"') + '"'

def format_atom(a):
    if a.kind == "ref": return "$" + a.text
    if a.kind == "unknown": return "~"
    if a.kind == "asked": return "?"
    return a.text if (BARE_OK.match(a.text) and not a.quoted) else quote(a.text)

def format_value(atoms, sep=None):
    return f" {sep} ".join(format_atom(a) for a in atoms) if len(atoms) > 1 else format_atom(atoms[0])


def parse(text, strict=True):
    """text -> Doc. Raises SopError on the first syntax error (with its line number). strict=False collects nothing more: syntax is
    always strict; semantic checks are in check.py."""
    doc = Doc(); cur = None
    for lineno, line in enumerate(text.split("\n"), 1):
        if line.endswith("\r"): line = line[:-1]
        s = line.strip()
        if not s or s.startswith("#"): continue
        if s.startswith("!"):
            if cur is not None: raise SopError("directive after the first wire (directives belong to the file header)", lineno)
            m = re.match(r"^!([A-Za-z][A-Za-z0-9_-]*) (.*)$", s)
            if not m: raise SopError("bad directive (expected '!name value')", lineno)
            doc.directives.append((m.group(1), m.group(2), lineno)); continue
        if s.startswith("@"):
            m = re.match(r"^@([^\s]+) ([^\s]+)$", s)
            if not m: raise SopError("bad header (expected '@id type')", lineno)
            if not ID_RE.match(m.group(1)): raise SopError(f"bad wire id {m.group(1)!r} (allowed: [A-Za-z0-9_-]+)", lineno)
            cur = Wire(m.group(1), m.group(2), lineno=lineno); doc.wires.append(cur); continue
        if "=" not in s: raise SopError("bad line (expected 'key=value')", lineno)
        if cur is None: raise SopError("question line before any wire header", lineno)
        key, raw = s.split("=", 1)
        if not (RESERVED_KEY.match(key) or ROLE_KEY.match(key)): raise SopError(f"bad key {key!r}", lineno)
        if key in FREE_TEXT_KEYS:
            atoms = [Atom("word", parse_value(raw, lineno)[0][0].text, quoted=True)] if raw.startswith('"') and raw.endswith('"') and len(raw) > 1 else []
            cur.lines.append(Line(key, raw, atoms, None, lineno)); continue
        atoms, sep = parse_value(raw, lineno)
        cur.lines.append(Line(key, raw, atoms, sep, lineno))
    return doc


def write(doc):
    """Doc -> text in the canonical layout (directives, blank line, wires separated by one blank line)"""
    out = [f"!{n} {v}" for n, v, _ in doc.directives]
    blocks = []
    for w in doc.wires:
        b = [f"@{w.id} {w.type}"]
        for l in w.lines:
            if l.raw is not None: b.append(f"{l.key}={l.raw}")
            elif l.key in FREE_TEXT_KEYS: b.append(f"{l.key}=" + (l.atoms[0].text if l.atoms else "").replace("\n", " "))
            else: b.append(f"{l.key}={format_value(l.atoms, l.sep)}")
        blocks.append("\n".join(b))
    parts = (["\n".join(out)] if out else []) + blocks
    return "\n\n".join(parts) + ("\n" if parts else "")


def load(path):
    with open(path, encoding="utf-8") as f: return parse(f.read())
