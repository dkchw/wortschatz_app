from flask import Flask, request, jsonify, render_template_string
import re
import json
from collections import defaultdict
import os
from pathlib import Path

app = Flask(__name__)
app.config['MAX_CONTENT_LENGTH'] = 50 * 1024 * 1024

# ── Session persistence ────────────────────────────────────────────────────────
SESSIONS_DIR = Path.home() / ".wortschatz" / "sessions"
SESSIONS_DIR.mkdir(parents=True, exist_ok=True)

# ── Global Session Storage ─────────────────────────────────────────────────────
# Everything stored here vanishes when you kill the CLI server.
IN_MEMORY_DB = {}
nlp = None

def load_model():
    global nlp
    if nlp is not None:
        return True, ""
    try:
        import spacy
        try:
            nlp = spacy.load("de_core_news_sm")
            nlp.max_length = 5_000_000
            return True, ""
        except OSError:
            return False, "Modell nicht gefunden. Führe aus: python -m spacy download de_core_news_sm"
    except ImportError:
        return False, "spaCy fehlt. Führe aus: pip install spacy"

# ── POS & Parsers ──────────────────────────────────────────────────────────────
POS_MAP = {
    "NOUN":  ("Nomen",       "N"),
    "PROPN": ("Eigenname",   "E"),
    "VERB":  ("Verb",        "V"),
    "AUX":   ("Hilfsverb",   "H"),
    "ADJ":   ("Adjektiv",    "Adj"),
    "ADV":   ("Adverb",      "Adv"),
    "DET":   ("Artikel",     "Art"),
    "PRON":  ("Pronomen",    "Pro"),
    "ADP":   ("Präposition", "Prp"),
    "CCONJ": ("Konj.",       "K"),
    "SCONJ": ("Konj.",       "K"),
    "PART":  ("Partikel",    "Par"),
    "NUM":   ("Zahl",        "Zhl"),
    "INTJ":  ("Interjektion","Inj"),
    "X":     ("Sonstige",    "?"),
    "PUNCT": ("Satzzeichen", "."),
    "SPACE": ("Leerzeichen", "_"),
    "SYM":   ("Symbol",      "Sym"),
}

def pos_label(pos):
    return POS_MAP.get(pos, ("Sonstige", "?"))

def parse_srt(content: str) -> str:
    lines = []
    for line in content.splitlines():
        line = line.strip()
        if not line or re.fullmatch(r"\d+", line) or re.search(r"\d{2}:\d{2}:\d{2}[,\.]\d{3}\s*-->", line):
            continue
        line = re.sub(r"<[^>]+>", "", line)
        line = re.sub(r"\{[^}]+\}", "", line)
        if line: lines.append(line)
    return " ".join(lines)

def parse_text(content: str, filename: str) -> str:
    if filename.lower().endswith(".srt"):
        return parse_srt(content)
    return content

# ── Analysis Engine ────────────────────────────────────────────────────────────
SKIP_POS = {"PUNCT", "SPACE", "SYM", "X", "NUM"}

def analyze_text(text: str):
    doc = nlp(text)
    raw = defaultdict(lambda: {"count": 0, "pos": "", "pos_label": "", "pos_short": "", "lemma": "", "examples": []})
    lemma = defaultdict(lambda: {"count": 0, "pos": "", "pos_label": "", "pos_short": "", "forms": defaultdict(int), "examples": []})
    total_tokens = 0

    for token in doc:
        if token.pos_ in SKIP_POS or not token.is_alpha or len(token.text) < 2:
            continue

        total_tokens += 1
        word, lm, pos = token.text.lower(), token.lemma_.lower(), token.pos_
        if pos in ('NOUN', 'PROPN'):
            word = word[0].upper() + word[1:] if word else word
            lm   = lm[0].upper()   + lm[1:]   if lm   else lm
        lbl, sh = pos_label(pos)
        sent_text = re.sub(r"\s+", " ", token.sent.text.strip())

        r = raw[word]
        r["count"] += 1
        r["pos"], r["pos_label"], r["pos_short"], r["lemma"] = pos, lbl, sh, lm
        if sent_text not in r["examples"]: r["examples"].append(sent_text)

        g = lemma[lm]
        g["count"] += 1
        g["pos"], g["pos_label"], g["pos_short"] = pos, lbl, sh
        g["forms"][word] += 1
        if sent_text not in g["examples"]: g["examples"].append(sent_text)

    raw_list = sorted([{"word": w, **d} for w, d in raw.items()], key=lambda x: -x["count"])
    lemma_list = sorted([{
        "lemma": l, "count": d["count"], "pos": d["pos"], "pos_label": d["pos_label"],
        "pos_short": d["pos_short"], "examples": d["examples"],
        "forms": sorted([{"form": f, "count": c} for f, c in d["forms"].items()], key=lambda x: -x["count"])
    } for l, d in lemma.items()], key=lambda x: -x["count"])

    return {"raw": raw_list, "lemma": lemma_list, "total_tokens": total_tokens, "unique_forms": len(raw_list), "unique_lemmas": len(lemma_list)}


# ── CLI Pre-Loader ─────────────────────────────────────────────────────────────
def pre_load_path(path):
    if os.path.isfile(path) and path.lower().endswith(('.txt', '.srt')):
        _load_file(path)
    elif os.path.isdir(path):
        for root, dirs, files in os.walk(path):
            for f in files:
                if f.lower().endswith(('.txt', '.srt')):
                    _load_file(os.path.join(root, f))
    else:
        print(f"  ! Ignoriert (kein Text/Ordner): {path}")

def _load_file(filepath):
    try:
        with open(filepath, 'r', encoding='utf-8', errors='replace') as f:
            content = f.read()
        filename = os.path.basename(filepath)

        text = parse_text(content, filename)
        if len(text.strip()) < 10: return

        res = analyze_text(text)
        res["filename"] = filename

        # Deduplicate filename for the dropdown if needed
        base_name = filename
        idx = 1
        while base_name in IN_MEMORY_DB:
            base_name = f"{filename} ({idx})"
            idx += 1

        IN_MEMORY_DB[base_name] = res
        print(f"  ✓ {base_name} analysiert ({res['total_tokens']} Tokens)")
    except Exception as e:
        print(f"  ✗ Fehler bei {filepath}: {e}")


# ── Routes ─────────────────────────────────────────────────────────────────────
@app.route("/")
def index():
    ok, err = load_model()
    files = list(IN_MEMORY_DB.keys())
    return render_template_string(HTML, model_ok=ok, model_error=err, initial_files=json.dumps(files))

@app.route("/api/upload", methods=["POST"])
def upload():
    ok, err = load_model()
    if not ok: return jsonify({"error": err}), 500

    files = request.files.getlist("files")
    if not files: return jsonify({"error": "Keine Dateien hochgeladen."}), 400

    loaded = []
    for f in files:
        fname = f.filename or "upload.txt"
        if not fname.lower().endswith((".txt", ".srt")): continue

        raw_bytes = f.read()
        try: content = raw_bytes.decode("utf-8")
        except UnicodeDecodeError: content = raw_bytes.decode("latin-1", errors="replace")

        text = parse_text(content, fname)
        if len(text.strip()) < 10: continue

        res = analyze_text(text)
        res["filename"] = fname

        base_name = fname
        idx = 1
        while base_name in IN_MEMORY_DB:
            base_name = f"{fname} ({idx})"
            idx += 1

        IN_MEMORY_DB[base_name] = res
        loaded.append(base_name)

    if not loaded: return jsonify({"error": "Dateien leer oder ungültig."}), 400
    return jsonify({"files": list(IN_MEMORY_DB.keys()), "newest": loaded[0]})

@app.route("/api/files/<path:fname>", methods=["GET"])
def get_file(fname):
    if fname in IN_MEMORY_DB:
        return jsonify(IN_MEMORY_DB[fname])
    return jsonify({"error": "Datei nicht im Speicher."}), 404


# ── Session persistence routes ─────────────────────────────────────────────────
@app.route("/api/sessions", methods=["GET"])
def list_sessions():
    sessions = []
    for p in sorted(SESSIONS_DIR.glob("*.json"), key=lambda x: x.stat().st_mtime, reverse=True):
        stat = p.stat()
        sessions.append({
            "name": p.stem,
            "size_kb": round(stat.st_size / 1024, 1),
            "mtime": stat.st_mtime,
        })
    return jsonify(sessions)

@app.route("/api/sessions/save", methods=["POST"])
def save_session():
    data = request.get_json(silent=True) or {}
    name = (data.get("name") or "").strip()
    if not name:
        return jsonify({"error": "Kein Name angegeben."}), 400
    # Sanitise filename
    safe = re.sub(r'[^\w\-. ]', '_', name)[:80]
    path = SESSIONS_DIR / f"{safe}.json"
    if not IN_MEMORY_DB:
        return jsonify({"error": "Keine Daten im Speicher."}), 400
    with open(path, "w", encoding="utf-8") as f:
        json.dump(IN_MEMORY_DB, f, ensure_ascii=False)
    return jsonify({"ok": True, "name": safe})

@app.route("/api/sessions/load", methods=["POST"])
def load_session():
    global IN_MEMORY_DB
    data = request.get_json(silent=True) or {}
    name = (data.get("name") or "").strip()
    path = SESSIONS_DIR / f"{name}.json"
    if not path.exists():
        return jsonify({"error": "Session nicht gefunden."}), 404
    with open(path, "r", encoding="utf-8") as f:
        IN_MEMORY_DB = json.load(f)
    return jsonify({"ok": True, "files": list(IN_MEMORY_DB.keys())})

@app.route("/api/sessions/<name>", methods=["DELETE"])
def delete_session(name):
    path = SESSIONS_DIR / f"{name}.json"
    if not path.exists():
        return jsonify({"error": "Session nicht gefunden."}), 404
    path.unlink()
    return jsonify({"ok": True})


# ── HTML Frontend ──────────────────────────────────────────────────────────────
HTML = r"""<!DOCTYPE html>
<html lang="de">
<head>
<meta charset="UTF-8">
<meta name="viewport" content="width=device-width, initial-scale=1.0">
<title>Wortschatz · Frequenzanalyse</title>
<link rel="preconnect" href="https://fonts.googleapis.com">
<link href="https://fonts.googleapis.com/css2?family=Playfair+Display:ital,wght@0,700;1,400&family=IBM+Plex+Mono:wght@300;400;500&family=DM+Sans:wght@300;400;500&display=swap" rel="stylesheet">
<style>
*, *::before, *::after { box-sizing: border-box; margin: 0; padding: 0; }
:root {
  --bg: #0c0c0c; --bg2: #141414; --bg3: #1c1c1c; --border: #2a2a2a; --border2: #333;
  --red: #d42b3a; --red-dim: #8a1a23; --gold: #c9a84c; --cream: #f0ebe0;
  --muted: #666; --muted2: #888;
  --mono: 'IBM Plex Mono', monospace; --serif: 'Playfair Display', serif; --sans: 'DM Sans', sans-serif;
}
html, body { height: 100%; background: var(--bg); color: var(--cream); font-family: var(--sans); }
::-webkit-scrollbar { width: 6px; height: 6px; }
::-webkit-scrollbar-track { background: var(--bg); }
::-webkit-scrollbar-thumb { background: var(--border2); border-radius: 3px; }
::-webkit-scrollbar-thumb:hover { background: var(--muted); }
#app { min-height: 100vh; display: flex; flex-direction: column; }
header {
  display: flex; align-items: center; justify-content: space-between;
  padding: 20px 32px; border-bottom: 1px solid var(--border); background: var(--bg);
  position: sticky; top: 0; z-index: 100;
}
.logo { font-family: var(--serif); font-size: 1.4rem; letter-spacing: -0.02em; color: var(--cream); }
.logo span { color: var(--red); font-style: italic; }
.logo-sub { font-family: var(--mono); font-size: 0.62rem; color: var(--muted); letter-spacing: 0.15em; text-transform: uppercase; margin-top: 2px; }
.header-right { display: flex; align-items: center; gap: 12px; }
.model-badge { font-family: var(--mono); font-size: 0.65rem; padding: 4px 10px; border: 1px solid; border-radius: 2px; letter-spacing: 0.05em; }
.model-ok { border-color: #2a5a2a; color: #6dbd6d; background: #0d1f0d; }
.model-err { border-color: var(--red-dim); color: var(--red); background: #1a0d0e; }

.file-select {
  background: var(--bg3); color: var(--cream);
  border: 1px solid var(--border2); padding: 5px 28px 5px 12px;
  font-family: var(--mono); font-size: 0.72rem;
  border-radius: 2px; outline: none; cursor: pointer; appearance: none;
  background-image: url('data:image/svg+xml;utf8,<svg fill="%23888" height="20" viewBox="0 0 24 24" width="20" xmlns="http://www.w3.org/2000/svg"><path d="M7 10l5 5 5-5z"/></svg>');
  background-repeat: no-repeat; background-position-x: calc(100% - 4px); background-position-y: center;
  max-width: 250px; text-overflow: ellipsis; white-space: nowrap; overflow: hidden;
}
.file-select:hover { border-color: var(--muted); }
.file-select:focus { border-color: var(--gold); }

.new-btn {
  font-family: var(--mono); font-size: 0.7rem; padding: 6px 14px; background: none;
  border: 1px solid var(--border2); color: var(--muted2); cursor: pointer;
  letter-spacing: 0.05em; border-radius: 2px; transition: all .15s;
}
.new-btn:hover { border-color: var(--cream); color: var(--cream); }

#upload-screen { flex: 1; display: flex; flex-direction: column; align-items: center; justify-content: center; padding: 40px 20px; }
.upload-hero { text-align: center; margin-bottom: 48px; }
.upload-hero h1 { font-family: var(--serif); font-size: clamp(2.4rem, 6vw, 4.5rem); font-weight: 700; line-height: 1.05; letter-spacing: -0.03em; color: var(--cream); }
.upload-hero h1 em { color: var(--red); font-style: italic; }
.upload-hero p { font-family: var(--mono); font-size: 0.75rem; color: var(--muted); letter-spacing: 0.1em; text-transform: uppercase; margin-top: 16px; }

.drop-zone { width: min(560px, 90vw); border: 1px dashed var(--border2); background: var(--bg2); border-radius: 4px; padding: 56px 40px; text-align: center; cursor: pointer; transition: all .2s; position: relative; }
.drop-zone:hover, .drop-zone.dragging { border-color: var(--red); background: #180c0d; }
.drop-icon { font-size: 2.5rem; margin-bottom: 20px; opacity: 0.5; }
.drop-zone h3 { font-family: var(--mono); font-size: 0.85rem; color: var(--muted2); font-weight: 400; letter-spacing: 0.05em; }
.drop-zone p { font-family: var(--mono); font-size: 0.7rem; color: var(--muted); margin-top: 10px; }
.drop-zone input { position: absolute; inset: 0; opacity: 0; cursor: pointer; }

.upload-btn { margin-top: 24px; padding: 14px 36px; background: var(--red); color: #fff; border: none; cursor: pointer; font-family: var(--mono); font-size: 0.8rem; letter-spacing: 0.1em; text-transform: uppercase; border-radius: 2px; transition: background .15s; }
.upload-btn:hover { background: #b82232; }
.upload-btn:disabled { background: var(--border2); color: var(--muted); cursor: not-allowed; }
.selected-file { font-family: var(--mono); font-size: 0.72rem; color: var(--gold); margin-top: 16px; padding: 8px 16px; background: #1a1508; border: 1px solid #3a2f12; border-radius: 2px; display: none; }

#loading-screen { flex: 1; display: none; flex-direction: column; align-items: center; justify-content: center; gap: 32px; }
.loading-label { font-family: var(--mono); font-size: 0.75rem; color: var(--muted); letter-spacing: 0.15em; text-transform: uppercase; }
.pulse-bar { width: 280px; height: 2px; background: var(--border); position: relative; overflow: hidden; border-radius: 1px; }
.pulse-bar::after { content: ''; position: absolute; left: -40%; width: 40%; height: 100%; background: linear-gradient(90deg, transparent, var(--red), transparent); animation: pulse 1.2s ease-in-out infinite; }
@keyframes pulse { to { left: 140%; } }

#results-screen { flex: 1; display: none; flex-direction: column; overflow: hidden; }
.stats-bar { display: flex; align-items: center; gap: 0; border-bottom: 1px solid var(--border); background: var(--bg); padding: 0 32px; flex-wrap: wrap; }
.stat { padding: 16px 24px; border-right: 1px solid var(--border); display: flex; flex-direction: column; gap: 3px; }
.stat:last-child { border-right: none; }
.stat-val { font-family: var(--mono); font-size: 1.3rem; font-weight: 500; color: var(--cream); }
.stat-key { font-family: var(--mono); font-size: 0.6rem; color: var(--muted); letter-spacing: 0.12em; text-transform: uppercase; }
.stat-file { margin-left: auto; padding: 16px 0; }
.stat-file .stat-val { font-size: 0.8rem; color: var(--gold); }

.controls { display: flex; align-items: center; gap: 12px; flex-wrap: wrap; padding: 14px 32px; border-bottom: 1px solid var(--border); background: var(--bg2); }
.mode-toggle { display: flex; border: 1px solid var(--border2); border-radius: 2px; overflow: hidden; }
.mode-btn { padding: 7px 16px; font-family: var(--mono); font-size: 0.7rem; letter-spacing: 0.06em; cursor: pointer; background: none; border: none; color: var(--muted2); transition: all .15s; white-space: nowrap; }
.mode-btn.active { background: var(--red); color: #fff; }
.pos-filter { display: flex; gap: 6px; flex-wrap: wrap; }
.pos-btn { padding: 5px 12px; font-family: var(--mono); font-size: 0.65rem; letter-spacing: 0.06em; cursor: pointer; border: 1px solid var(--border2); background: none; color: var(--muted2); border-radius: 2px; transition: all .15s; white-space: nowrap; }
.pos-btn.active { border-color: var(--gold); color: var(--gold); background: #1a1508; }
.pos-btn:hover:not(.active) { border-color: var(--border2); color: var(--cream); }
.search-wrap { margin-left: auto; position: relative; }
.search-input { background: var(--bg3); border: 1px solid var(--border2); border-radius: 2px; padding: 7px 12px 7px 32px; font-family: var(--mono); font-size: 0.72rem; color: var(--cream); width: 200px; outline: none; transition: border-color .15s; }
.search-input:focus { border-color: var(--muted2); }
.search-input::placeholder { color: var(--muted); }
.search-icon { position: absolute; left: 10px; top: 50%; transform: translateY(-50%); color: var(--muted); font-size: 0.75rem; pointer-events: none; }
.export-btn { padding: 7px 14px; font-family: var(--mono); font-size: 0.67rem; letter-spacing: 0.06em; cursor: pointer; background: none; border: 1px solid var(--border2); color: var(--muted2); border-radius: 2px; transition: all .15s; white-space: nowrap; }
.export-btn:hover { border-color: var(--cream); color: var(--cream); }

.table-wrap { flex: 1; overflow: auto; }
table { width: 100%; border-collapse: collapse; font-family: var(--mono); }
thead { position: sticky; top: 0; z-index: 10; background: var(--bg3); }
th { padding: 12px 20px; text-align: left; font-size: 0.65rem; letter-spacing: 0.12em; text-transform: uppercase; color: var(--muted); border-bottom: 1px solid var(--border); font-weight: 400; cursor: pointer; white-space: nowrap; user-select: none; }
th:hover { color: var(--cream); }
th.sorted-asc::after { content: ' ↑'; color: var(--red); }
th.sorted-desc::after { content: ' ↓'; color: var(--red); }
th.no-sort { cursor: default; } th.no-sort:hover { color: var(--muted); }
td { padding: 11px 20px; font-size: 0.78rem; border-bottom: 1px solid var(--border); color: var(--cream); vertical-align: top; }
tr.clickable-row:hover td { background: var(--bg2); }

.index-cell { color: var(--muted); font-size: 0.7rem; width: 45px; }
.word-cell { font-weight: 500; font-size: 0.82rem; }
.count-cell { color: var(--gold); font-size: 0.82rem; font-weight: 500; }
.pos-pill { display: inline-block; padding: 2px 8px; border-radius: 2px; font-size: 0.62rem; letter-spacing: 0.08em; border: 1px solid; }
.pos-NOUN, .pos-PROPN { background: #0d1a2e; border-color: #1e3d6e; color: #5b9bd5; }
.pos-VERB, .pos-AUX { background: #1a0d0e; border-color: #4a1820; color: var(--red); }
.pos-ADJ { background: #0d1a0d; border-color: #1e4a1e; color: #6dbd6d; }
.pos-ADV { background: #1a150d; border-color: #4a3a1e; color: var(--gold); }
.pos-OTHER { background: #1a1a1a; border-color: #333; color: var(--muted2); }
.lemma-cell { color: var(--muted2); font-size: 0.72rem; }
.forms-cell { display: flex; flex-wrap: wrap; gap: 6px; }
.form-tag { font-size: 0.68rem; padding: 2px 8px; background: var(--bg3); border: 1px solid var(--border2); border-radius: 2px; color: var(--muted2); display: inline-flex; align-items: center; gap: 6px; }
.form-tag .fc { color: var(--gold); font-size: 0.65rem; }
.rank-bar { display: inline-block; height: 2px; background: var(--red-dim); vertical-align: middle; margin-left: 8px; border-radius: 1px; max-width: 80px; min-width: 2px; }

.tag-btn { background: none; border: 1px solid var(--border2); border-radius: 2px; color: var(--muted); cursor: pointer; font-size: 0.75rem; padding: 2px 7px; line-height: 1; transition: all .15s; }
.tag-btn:hover { border-color: var(--gold); color: var(--gold); }
.tag-btn.tagged { border-color: var(--gold); color: var(--gold); background: #1a1508; }
.tag-cell { width: 42px; text-align: center; }

.tagged-badge { font-family: var(--mono); font-size: 0.65rem; padding: 5px 12px; border: 1px solid #3a2f12; background: #1a1508; color: var(--gold); border-radius: 2px; letter-spacing: 0.05em; display: none; white-space: nowrap; }
.anki-btn { padding: 7px 14px; font-family: var(--mono); font-size: 0.67rem; letter-spacing: 0.06em; cursor: pointer; background: none; border: 1px solid #2a3a5a; color: #6090c0; border-radius: 2px; transition: all .15s; white-space: nowrap; }
.anki-btn:hover:not(:disabled) { border-color: #6090c0; color: #90c0f0; }
.anki-btn:disabled { opacity: 0.35; cursor: not-allowed; }

#toast { position: fixed; bottom: 24px; right: 24px; z-index: 999; background: #1a0d0e; border: 1px solid var(--red-dim); color: var(--red); font-family: var(--mono); font-size: 0.72rem; padding: 12px 20px; border-radius: 2px; max-width: 360px; transform: translateY(20px); opacity: 0; transition: all .2s; pointer-events: none; letter-spacing: 0.04em; }
#toast.show { transform: translateY(0); opacity: 1; }

/* Session panel */
#session-panel { display: none; position: absolute; top: calc(100% + 8px); right: 0; width: 340px; background: var(--bg2); border: 1px solid var(--border2); border-radius: 4px; z-index: 200; box-shadow: 0 8px 32px rgba(0,0,0,.6); overflow: hidden; }
#session-panel.open { display: block; }
.sp-head { padding: 14px 16px; border-bottom: 1px solid var(--border); font-family: var(--mono); font-size: 0.68rem; letter-spacing: 0.1em; text-transform: uppercase; color: var(--muted); }
.sp-save { display: flex; gap: 8px; padding: 12px 16px; border-bottom: 1px solid var(--border); }
.sp-input { flex: 1; background: var(--bg3); border: 1px solid var(--border2); border-radius: 2px; padding: 7px 10px; font-family: var(--mono); font-size: 0.72rem; color: var(--cream); outline: none; }
.sp-input:focus { border-color: var(--muted2); }
.sp-input::placeholder { color: var(--muted); }
.sp-save-btn { padding: 7px 14px; background: var(--red); color: #fff; border: none; border-radius: 2px; font-family: var(--mono); font-size: 0.7rem; cursor: pointer; white-space: nowrap; transition: background .15s; }
.sp-save-btn:hover { background: #b82232; }
.sp-list { max-height: 260px; overflow-y: auto; }
.sp-empty { padding: 24px 16px; font-family: var(--mono); font-size: 0.7rem; color: var(--muted); text-align: center; }
.sp-item { display: flex; align-items: center; gap: 8px; padding: 10px 16px; border-bottom: 1px solid var(--border); transition: background .12s; }
.sp-item:hover { background: var(--bg3); }
.sp-item-name { flex: 1; font-family: var(--mono); font-size: 0.75rem; color: var(--cream); overflow: hidden; text-overflow: ellipsis; white-space: nowrap; }
.sp-item-meta { font-family: var(--mono); font-size: 0.62rem; color: var(--muted); white-space: nowrap; }
.sp-load-btn { padding: 4px 10px; background: none; border: 1px solid var(--border2); border-radius: 2px; font-family: var(--mono); font-size: 0.65rem; color: var(--muted2); cursor: pointer; transition: all .12s; white-space: nowrap; }
.sp-load-btn:hover { border-color: var(--cream); color: var(--cream); }
.sp-del-btn { padding: 4px 8px; background: none; border: 1px solid transparent; border-radius: 2px; font-family: var(--mono); font-size: 0.65rem; color: var(--muted); cursor: pointer; transition: all .12s; }
.sp-del-btn:hover { border-color: var(--red-dim); color: var(--red); }
.session-toggle-btn { position: relative; }
</style>
</head>
<body>
<div id="app">
  <header>
    <div>
      <div class="logo">Wort<span>schatz</span></div>
      <div class="logo-sub">Deutsche Frequenzanalyse</div>
    </div>
    <div class="header-right">
      {% if model_ok %}
        <div class="model-badge model-ok">de_core_news_sm ✓</div>
      {% else %}
        <div class="model-badge model-err" title="{{ model_error }}">Modell fehlt ✗</div>
      {% endif %}
      <select id="file-selector" class="file-select" style="display:none;" onchange="switchFile(this.value)"></select>
      <button class="new-btn" id="upload-more-btn" style="display:none" onclick="show('upload-screen')">+ Dateien</button>
      <div class="session-toggle-btn">
        <button class="new-btn" onclick="toggleSessionPanel()" id="session-btn">💾 Sessions</button>
        <div id="session-panel">
          <div class="sp-head">Session speichern</div>
          <div class="sp-save">
            <input class="sp-input" id="sp-name-input" placeholder="Name…" maxlength="80"
              onkeydown="if(event.key==='Enter') saveSession()">
            <button class="sp-save-btn" onclick="saveSession()">Speichern</button>
          </div>
          <div class="sp-head" style="margin-top:0">Gespeicherte Sessions</div>
          <div class="sp-list" id="sp-list"><div class="sp-empty">Keine Sessions vorhanden.</div></div>
        </div>
      </div>
    </div>
  </header>

  <section id="upload-screen">
    <div class="upload-hero">
      <h1>Analysiere deinen<br><em>deutschen</em> Text</h1>
      <p>TXT · SRT · Alle Wortarten · Deklinationsformen</p>
    </div>
    <div class="drop-zone" id="drop-zone">
      <div class="drop-icon">📄</div>
      <h3>Datei(en) hier ablegen</h3>
      <p>Mehrere .txt oder .srt möglich</p>
      <input type="file" id="file-input" accept=".txt,.srt" multiple>
    </div>
    <div class="selected-file" id="selected-file-label"></div>
    <button class="upload-btn" id="upload-btn" disabled onclick="doUpload()">Analysieren →</button>
    <button class="new-btn" id="cancel-upload-btn" style="margin-top:16px; display:none;" onclick="show('results-screen')">Zurück</button>
  </section>

  <section id="loading-screen">
    <div class="loading-label" id="loading-label">Analysiere…</div>
    <div class="pulse-bar"></div>
  </section>

  <section id="results-screen">
    <div class="stats-bar" id="stats-bar"></div>
    <div class="controls">
      <div class="mode-toggle">
        <button class="mode-btn active" id="btn-raw" onclick="setMode('raw')">Wortformen</button>
        <button class="mode-btn" id="btn-lemma" onclick="setMode('lemma')">Grundformen</button>
      </div>
      <div class="pos-filter" id="pos-filter">
        <button class="pos-btn active" data-pos="ALL" onclick="setPOS(this)">Alle</button>
        <button class="pos-btn" data-pos="NOUN" onclick="setPOS(this)">Nomen</button>
        <button class="pos-btn" data-pos="PROPN" onclick="setPOS(this)">Eigenname</button>
        <button class="pos-btn" data-pos="VERB" onclick="setPOS(this)">Verb</button>
        <button class="pos-btn" data-pos="AUX" onclick="setPOS(this)">Hilfsverb</button>
        <button class="pos-btn" data-pos="ADJ" onclick="setPOS(this)">Adjektiv</button>
        <button class="pos-btn" data-pos="ADV" onclick="setPOS(this)">Adverb</button>
        <button class="pos-btn" data-pos="OTHER" onclick="setPOS(this)">Sonstige</button>
      </div>
      <div class="search-wrap">
        <span class="search-icon">⌕</span>
        <input class="search-input" type="text" id="search-input" placeholder="Suchen…" oninput="renderTable()">
      </div>
      <button class="export-btn" onclick="exportCSV()">↓ CSV</button>
      <span class="tagged-badge" id="tagged-badge"></span>
      <button class="anki-btn" onclick="exportAnki(false)" title="Aktuelle Ansicht als Anki-Deck exportieren">↓ Anki</button>
      <button class="anki-btn" id="anki-tagged-btn" onclick="exportAnki(true)" disabled title="Nur markierte Einträge exportieren">↓ Anki (Markiert)</button>
    </div>
    <div class="table-wrap">
      <table><thead id="table-head"></thead><tbody id="table-body"></tbody></table>
    </div>
  </section>
</div>
<div id="toast"></div>

<script>
let DATA = null; let mode = 'raw'; let posFilter = 'ALL'; let sortCol = 'count'; let sortDir = -1;
let TAGGED = new Set();
const OTHER_POS = new Set(['DET','PRON','ADP','CCONJ','SCONJ','PART','NUM','INTJ','X']);
const dropZone = document.getElementById('drop-zone');
const fileInput = document.getElementById('file-input');
const uploadBtn = document.getElementById('upload-btn');
const fileLabel = document.getElementById('selected-file-label');
const INITIAL_FILES = {{ initial_files | safe }};

let selectedFiles = [];

window.onload = () => {
    if (INITIAL_FILES && INITIAL_FILES.length > 0) {
        updateFileSelector(INITIAL_FILES);
        switchFile(INITIAL_FILES[0]);
    }
};

dropZone.addEventListener('dragover', e => { e.preventDefault(); dropZone.classList.add('dragging'); });
dropZone.addEventListener('dragleave', () => dropZone.classList.remove('dragging'));
dropZone.addEventListener('drop', e => { e.preventDefault(); dropZone.classList.remove('dragging'); if(e.dataTransfer.files.length) setFiles(e.dataTransfer.files); });
fileInput.addEventListener('change', () => { if(fileInput.files.length) setFiles(fileInput.files); });

function setFiles(files) {
  const valid = Array.from(files).filter(f => f.name.match(/\.(txt|srt)$/i));
  if (valid.length === 0) { showToast('Nur .txt und .srt Dateien.'); return; }
  selectedFiles = valid;
  fileLabel.textContent = valid.length === 1 ? `📎 ${valid[0].name} (${(valid[0].size/1024).toFixed(1)} KB)` : `📎 ${valid.length} Dateien ausgewählt`;
  fileLabel.style.display = 'block'; uploadBtn.disabled = false;
}

function doUpload() {
  if (selectedFiles.length === 0) return;
  show('loading-screen'); document.getElementById('loading-label').textContent = `Analysiere Dateien…`;
  const fd = new FormData();
  selectedFiles.forEach(f => fd.append('files', f));

  fetch('/api/upload', { method: 'POST', body: fd })
    .then(r => r.json())
    .then(data => {
      if (data.error) { show('upload-screen'); showToast(data.error); return; }
      selectedFiles = []; fileInput.value = ''; fileLabel.style.display = 'none'; uploadBtn.disabled = true;
      updateFileSelector(data.files);
      switchFile(data.newest);
    }).catch(() => { show('upload-screen'); showToast('Verbindungsfehler.'); });
}

function updateFileSelector(files) {
  const sel = document.getElementById('file-selector');
  sel.innerHTML = files.map(f => `<option value="${esc(f)}">${esc(f)}</option>`).join('');
  sel.style.display = files.length > 0 ? 'inline-block' : 'none';
  document.getElementById('upload-more-btn').style.display = 'inline-block';
}

function switchFile(filename) {
  if (!filename) return;
  show('loading-screen'); document.getElementById('loading-label').textContent = `Lade „${filename}“…`;
  fetch(`/api/files/${encodeURIComponent(filename)}`).then(r => r.json()).then(data => {
    if(data.error) { showToast(data.error); show('upload-screen'); return; }
    DATA = data; renderStats();
    mode = 'raw'; posFilter = 'ALL'; sortCol = 'count'; sortDir = -1;
    document.getElementById('search-input').value = '';
    document.getElementById('btn-raw').classList.add('active'); document.getElementById('btn-lemma').classList.remove('active');
    document.querySelectorAll('.pos-btn').forEach(b => b.classList.toggle('active', b.dataset.pos === 'ALL'));
    renderHead(); renderTable(); show('results-screen');
    document.getElementById('file-selector').value = filename;
  }).catch(() => { showToast('Fehler.'); show('upload-screen'); });
}

function show(id) {
  ['upload-screen','loading-screen','results-screen'].forEach(s => {
    document.getElementById(s).style.display = s === id ? 'flex' : 'none';
  });
  if (id === 'upload-screen') {
     const hasFiles = document.getElementById('file-selector').options.length > 0;
     document.getElementById('cancel-upload-btn').style.display = hasFiles ? 'inline-block' : 'none';
  }
}

function renderStats() {
  document.getElementById('stats-bar').innerHTML = `
    <div class="stat"><div class="stat-val">${DATA.total_tokens.toLocaleString('de')}</div><div class="stat-key">Tokens gesamt</div></div>
    <div class="stat"><div class="stat-val">${DATA.unique_forms.toLocaleString('de')}</div><div class="stat-key">Wortformen</div></div>
    <div class="stat"><div class="stat-val">${DATA.unique_lemmas.toLocaleString('de')}</div><div class="stat-key">Grundformen</div></div>
    <div class="stat stat-file"><div class="stat-val">${esc(DATA.filename)}</div><div class="stat-key">Datei</div></div>`;
}

function setMode(m) { mode = m; document.getElementById('btn-raw').classList.toggle('active', m === 'raw'); document.getElementById('btn-lemma').classList.toggle('active', m === 'lemma'); sortCol = 'count'; sortDir = -1; renderHead(); renderTable(); }
function setPOS(btn) { posFilter = btn.dataset.pos; document.querySelectorAll('.pos-btn').forEach(b => b.classList.toggle('active', b === btn)); renderTable(); }

function renderHead() {
  const cols = mode === 'raw'
    ? [{key:'index',label:'#',sort:false},{key:'_tag',label:'',sort:false},{key:'word',label:'Wortform',sort:true},{key:'count',label:'Häufigkeit',sort:true},{key:'pos_label',label:'Wortart',sort:true},{key:'lemma',label:'Grundform',sort:true}]
    : [{key:'index',label:'#',sort:false},{key:'_tag',label:'',sort:false},{key:'lemma',label:'Grundform',sort:true},{key:'count',label:'Häufigkeit',sort:true},{key:'pos_label',label:'Wortart',sort:true},{key:'forms',label:'Formen',sort:false}];
  document.getElementById('table-head').innerHTML = '<tr>' + cols.map(c => {
    let cls = c.sort ? (sortCol === c.key ? (sortDir === -1 ? 'sorted-desc' : 'sorted-asc') : '') : 'no-sort';
    return `<th class="${cls}" ${c.sort ? `onclick="sortBy('${c.key}')"` : ''}>${c.label}</th>`;
  }).join('') + '</tr>';
}

function filterRows(rows) {
  const q = document.getElementById('search-input').value.toLowerCase().trim();
  return rows.filter(r => {
    if (posFilter !== 'ALL' && (posFilter === 'OTHER' ? !OTHER_POS.has(r.pos) : r.pos !== posFilter)) return false;
    if (q) {
      const word = (mode === 'raw' ? r.word : r.lemma).toLowerCase();
      const lm = (r.lemma || '').toLowerCase();
      if (!word.includes(q) && !lm.includes(q)) return false;
    } return true;
  });
}
function sortRows(rows) {
  return [...rows].sort((a, b) => {
    let va = typeof a[sortCol] === 'string' ? a[sortCol].toLowerCase() : a[sortCol];
    let vb = typeof b[sortCol] === 'string' ? b[sortCol].toLowerCase() : b[sortCol];
    if (va < vb) return -sortDir; if (va > vb) return sortDir; return 0;
  });
}
function posClass(pos) { return ['NOUN','PROPN'].includes(pos) ? 'pos-NOUN' : ['VERB','AUX'].includes(pos) ? 'pos-VERB' : pos === 'ADJ' ? 'pos-ADJ' : pos === 'ADV' ? 'pos-ADV' : 'pos-OTHER'; }
function toggleRow(id) { const el = document.getElementById(id); if (el) el.style.display = el.style.display === 'none' ? 'table-row' : 'none'; }

function highlightWord(sentence, item, mode) {
  let terms = mode === 'raw' ? [item.word] : [item.lemma, ...(item.forms?.map(f=>f.form)||[])];
  terms = [...new Set(terms)].filter(Boolean).sort((a,b) => b.length - a.length);
  if(!terms.length) return sentence;
  const pattern = terms.map(t => t.replace(/[-\/\\^$*+?.()|[\]{}]/g, '\\$&')).join('|');
  return sentence.replace(new RegExp(`(?<![a-zA-ZäöüÄÖÜß])(${pattern})(?![a-zA-ZäöüÄÖÜß])`, 'gi'), '<b style="color:var(--gold); background:#261f0d; padding:0 4px; border-radius:2px; font-weight:500;">$1</b>');
}

function renderTable() {
  const sorted = sortRows(filterRows(mode === 'raw' ? DATA.raw : DATA.lemma));
  const maxCount = sorted.length ? sorted[0].count : 1;
  const tbody = document.getElementById('table-body');
  if (!sorted.length) return tbody.innerHTML = `<tr><td colspan="6" style="text-align:center; padding:40px; color:var(--muted);">Keine Einträge.</td></tr>`;

  tbody.innerHTML = sorted.map((r, i) => {
    const barW = Math.round((r.count / maxCount) * 80); const rowId = `ex-${i}`;
    const key = mode === 'raw' ? r.word : r.lemma;
    const isTagged = TAGGED.has(key);
    const tagBtn = `<td class="tag-cell" onclick="event.stopPropagation()"><button class="tag-btn ${isTagged ? 'tagged' : ''}" onclick="toggleTag('${esc(key)}')" title="${isTagged ? 'Markierung entfernen' : 'Für Anki markieren'}">${isTagged ? '★' : '☆'}</button></td>`;
    const exHtml = (r.examples||[]).length ? `<ol style="padding-left:18px; margin:0; line-height:1.6; font-family:var(--sans);">${r.examples.map(ex => `<li>${highlightWord(esc(ex), r, mode)}</li>`).join('')}</ol>` : `<span style="color:var(--muted); font-style:italic;">Keine Beispielsätze gefunden.</span>`;

    if (mode === 'raw') {
      return `<tr class="clickable-row" style="cursor:pointer;" onclick="toggleRow('${rowId}')">
        <td class="index-cell">${i + 1}</td>${tagBtn}<td class="word-cell">${esc(r.word)}</td><td class="count-cell">${r.count.toLocaleString('de')}<span class="rank-bar" style="width:${barW}px"></span></td>
        <td><span class="pos-pill ${posClass(r.pos)}">${esc(r.pos_short || r.pos_label)}</span></td><td class="lemma-cell">${esc(r.lemma)}</td>
      </tr>
      <tr id="${rowId}" style="display:none; background:#111 !important;"><td colspan="6" style="padding:14px 20px; border-bottom:1px solid var(--border);"><div style="font-family:var(--mono); font-size:0.65rem; color:var(--muted); margin-bottom:8px; text-transform:uppercase;">Kontext:</div>${exHtml}</td></tr>`;
    } else {
      const formsHtml = r.forms.map(f => `<span class="form-tag">${esc(f.form)}<span class="fc">${f.count}</span></span>`).join('');
      return `<tr class="clickable-row" style="cursor:pointer;" onclick="toggleRow('${rowId}')">
        <td class="index-cell">${i + 1}</td>${tagBtn}<td class="word-cell">${esc(r.lemma)}</td><td class="count-cell">${r.count.toLocaleString('de')}<span class="rank-bar" style="width:${barW}px"></span></td>
        <td><span class="pos-pill ${posClass(r.pos)}">${esc(r.pos_short || r.pos_label)}</span></td><td><div class="forms-cell">${formsHtml}</div></td>
      </tr>
      <tr id="${rowId}" style="display:none; background:#111 !important;"><td colspan="6" style="padding:14px 20px; border-bottom:1px solid var(--border);"><div style="font-family:var(--mono); font-size:0.65rem; color:var(--muted); margin-bottom:8px; text-transform:uppercase;">Kontext:</div>${exHtml}</td></tr>`;
    }
  }).join('');
}

function sortBy(col) { if (sortCol === col) sortDir *= -1; else { sortCol = col; sortDir = col === 'count' ? -1 : 1; } renderHead(); renderTable(); }
function toggleTag(key) {
  if (TAGGED.has(key)) { TAGGED.delete(key); } else { TAGGED.add(key); }
  updateTaggedCount();
  renderTable();
}

function updateTaggedCount() {
  const badge = document.getElementById('tagged-badge');
  const btn = document.getElementById('anki-tagged-btn');
  if (TAGGED.size > 0) {
    badge.textContent = `★ ${TAGGED.size} markiert`;
    badge.style.display = 'inline-block';
    btn.disabled = false;
  } else {
    badge.style.display = 'none';
    btn.disabled = true;
  }
}

function buildAnkiBack(r) {
  const posStr = r.pos_label || r.pos || '';
  const freqStr = `Häufigkeit: ${r.count}`;
  const lemmaStr = mode === 'raw' && r.lemma ? `Grundform: <b>${r.lemma}</b>` : '';
  const formsStr = mode === 'lemma' && r.forms && r.forms.length
    ? `Formen: ${r.forms.slice(0,8).map(f => f.form).join(', ')}`
    : '';
  const examples = (r.examples || []).slice(0, 3);
  const exHtml = examples.length
    ? '<hr style="margin:6px 0; border-color:#555">' + examples.map(ex => `<div style="font-size:0.9em; color:#bbb; margin:3px 0">· ${ex}</div>`).join('')
    : '';
  return [posStr, freqStr, lemmaStr, formsStr].filter(Boolean).join('<br>') + exHtml;
}

function exportAnki(taggedOnly) {
  const rows = mode === 'raw' ? DATA.raw : DATA.lemma;
  let items = taggedOnly
    ? rows.filter(r => TAGGED.has(mode === 'raw' ? r.word : r.lemma))
    : filterRows(rows);
  if (!items.length) { showToast('Keine Einträge zum Exportieren.'); return; }

  const srcTag = DATA.filename.replace(/\.[^.]+$/, '').replace(/\s+/g, '_').toLowerCase();
  const lines = ['#separator:tab', '#html:true', '#notetype column:3', 'Vorderseite\tRückseite\tNotiztyp\tTags'];
  items.forEach(r => {
    const front = esc(mode === 'raw' ? r.word : r.lemma);
    const back = buildAnkiBack(r);
    const tags = `wortschatz ${srcTag} ${(r.pos_label || '').toLowerCase().replace(/\s+/g,'_')}`;
    lines.push(`${front}\t${back}\tBasic\t${tags}`);
  });

  const blob = new Blob([lines.join('\n')], { type: 'text/plain;charset=utf-8' });
  const a = document.createElement('a');
  const suffix = taggedOnly ? '_markiert' : `_${mode}`;
  a.href = URL.createObjectURL(blob);
  a.download = `wortschatz_anki${suffix}_${srcTag}.txt`;
  a.click();
  showToast(`${items.length} Karten exportiert.`);
}

// ── Session management ────────────────────────────────────────────────────────
function toggleSessionPanel() {
  const panel = document.getElementById('session-panel');
  const isOpen = panel.classList.toggle('open');
  if (isOpen) { loadSessionList(); document.getElementById('sp-name-input').focus(); }
}

document.addEventListener('click', e => {
  const panel = document.getElementById('session-panel');
  const btn = document.getElementById('session-btn');
  if (panel.classList.contains('open') && !panel.contains(e.target) && !btn.contains(e.target))
    panel.classList.remove('open');
});

function loadSessionList() {
  fetch('/api/sessions').then(r => r.json()).then(sessions => {
    const list = document.getElementById('sp-list');
    if (!sessions.length) { list.innerHTML = '<div class="sp-empty">Keine Sessions vorhanden.</div>'; return; }
    list.innerHTML = sessions.map(s => {
      const date = new Date(s.mtime * 1000).toLocaleDateString('de', {day:'2-digit', month:'2-digit', year:'2-digit'});
      return `<div class="sp-item">
        <span class="sp-item-name" title="${esc(s.name)}">${esc(s.name)}</span>
        <span class="sp-item-meta">${s.size_kb} KB · ${date}</span>
        <button class="sp-load-btn" onclick="loadSession('${esc(s.name)}')">Laden</button>
        <button class="sp-del-btn" onclick="deleteSession('${esc(s.name)}', this)" title="Löschen">✕</button>
      </div>`;
    }).join('');
  });
}

function saveSession() {
  const name = document.getElementById('sp-name-input').value.trim();
  if (!name) { showToast('Bitte einen Namen eingeben.'); return; }
  fetch('/api/sessions/save', { method: 'POST', headers: {'Content-Type':'application/json'}, body: JSON.stringify({name}) })
    .then(r => r.json()).then(data => {
      if (data.error) { showToast(data.error); return; }
      document.getElementById('sp-name-input').value = '';
      showToast(`Session „${data.name}" gespeichert.`);
      loadSessionList();
    });
}

function loadSession(name) {
  fetch('/api/sessions/load', { method: 'POST', headers: {'Content-Type':'application/json'}, body: JSON.stringify({name}) })
    .then(r => r.json()).then(data => {
      if (data.error) { showToast(data.error); return; }
      document.getElementById('session-panel').classList.remove('open');
      updateFileSelector(data.files);
      if (data.files.length) switchFile(data.files[0]);
      showToast(`Session „${name}" geladen (${data.files.length} Datei(en)).`);
    });
}

function deleteSession(name, btn) {
  btn.textContent = '…';
  fetch(`/api/sessions/${encodeURIComponent(name)}`, { method: 'DELETE' })
    .then(r => r.json()).then(data => {
      if (data.error) { showToast(data.error); return; }
      loadSessionList();
    });
}

function exportCSV() {
  const filtered = filterRows(mode === 'raw' ? DATA.raw : DATA.lemma);
  const csv = mode === 'raw'
    ? '#,Wortform,Häufigkeit,Wortart,Grundform\n' + filtered.map((r, i) => `${i + 1},"${r.word}",${r.count},"${r.pos_label}","${r.lemma}"`).join('\n')
    : '#,Grundform,Häufigkeit,Wortart,Formen\n' + filtered.map((r, i) => `${i + 1},"${r.lemma}",${r.count},"${r.pos_label}","${r.forms.map(f=>f.form+'('+f.count+')').join(' ')}"`).join('\n');
  const a = document.createElement('a'); a.href = URL.createObjectURL(new Blob(['\uFEFF' + csv], { type: 'text/csv;charset=utf-8' }));
  a.download = `wortschatz_${mode}_${DATA.filename.replace(/\.[^.]+$/,'')}.csv`; a.click();
}
function esc(s) { return String(s ?? '').replace(/&/g,'&amp;').replace(/</g,'&lt;').replace(/>/g,'&gt;'); }
function showToast(msg) { const t = document.getElementById('toast'); t.textContent = msg; t.classList.add('show'); setTimeout(() => t.classList.remove('show'), 4000); }
</script>
</body>
</html>
"""
