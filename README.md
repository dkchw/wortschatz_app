# Wortschatz – German Vocabulary Frequency Analyzer

**Wortschatz** is a lightweight web tool for analyzing German texts (plain text or `.srt` subtitles). It extracts every word, determines its part of speech (POS), shows inflection forms, and presents frequency lists – both for raw word forms and lemmas (base forms). The tool runs locally, no data leaves your machine.

## ✨ Features

- **Upload** `.txt` or `.srt` files (multiple, up to 50 MB total per request).
- **Automatic language analysis** using spaCy’s German model `de_core_news_sm`.
- **Two analysis modes**:
  - *Word forms* – surface forms with frequency, POS, and lemma.
  - *Lemmas* – base forms aggregated, showing all inflected variants.
- **POS filtering** – focus on nouns, verbs, adjectives, adverbs, etc.
- **Search** across words/lemmas.
- **Context examples** – each word appears together with the sentence it was found in, with the word/lemma highlighted.
- **Export** current view as CSV.
- **CLI pre‑loader** – pre‑analyse whole folders before starting the web server.
- **In‑memory session** – all uploaded files are kept until you stop the server.

## 🚀 Installation

Make sure you have **Python 3.9+** and a virtual environment (recommended).

```bash
# Clone or download the project, then inside the project folder:
pip install .
```

This installs the required dependencies: `flask` and `spacy`.  
After installation, download the German spaCy model:

```bash
python -m spacy download de_core_news_sm
```

## 🖥️ Usage

### Start the web interface

```bash
wortschatz
```

Then open your browser at `http://localhost:5000`.  
You can change the port with `--port 8080`.

### Pre‑load files from the command line

```bash
wortschatz ./mein_text.txt ./untertitel_folder/
```

All recognised `.txt` and `.srt` files will be analysed immediately and appear in the file dropdown when you open the browser.

### Run without installing globally

```bash
python -m wortschatz.cli [paths...] --port 5000
```

## 📁 File support

- **Encoding** – tries UTF‑8, falls back to latin‑1.
- **SRT files** – auto‑removes timestamps, cue numbers and HTML tags, merges all subtitle lines into a single text.
- **Minimum length** – files with less than 10 readable characters are ignored.

## 🔧 How it works (in brief)

1. The uploaded file is parsed (special handling for `.srt`).
2. spaCy tokenises the text, lemmatises, and assigns POS tags.
3. Punctuation, spaces, symbols, numbers and very short tokens (length < 2) are skipped.
4. For each word form and each lemma, the frequency, POS, and up to 10 example sentences are stored.
5. The frontend (pure HTML/CSS/JS) displays the data, allows sorting, filtering and exporting.

## 📡 API endpoints

| Endpoint | Method | Description |
|----------|--------|-------------|
| `/` | GET | Main web UI |
| `/api/upload` | POST | Upload one or more files (`multipart/form-data`, field `files`). Returns updated file list and the name of the first newly analysed file. |
| `/api/files/<fname>` | GET | Retrieve the full analysis result for a given filename. |

All analysis results are kept **only in memory** – they disappear when the server stops.

## 🧪 Example

```bash
wortschatz --port 5000
# Upload a German short story
# Switch between "Wortformen" and "Grundformen"
# Click on any row to see the original sentences
```

## 📦 Development

If you want to modify the code:

```bash
git clone ...
cd wortschatz
pip install -e .
```

The project is a single‑package Flask app with two modules:  
- `app.py` – contains the server logic, the analysis engine, and the HTML template.  
- `cli.py` – the command‑line entry point.

## ⚠️ Notes / Limitations

- **In‑memory only** – no persistent database. Once you restart the server, all uploaded files are gone.
- **Performance** – very large texts (>> 1M tokens) may be slow. spaCy’s `max_length` is set to 5 million characters, but your RAM is the real limit.
- **POS groups** – “Sonstige” includes determiners, pronouns, prepositions, conjunctions, particles, interjections.
- **No authentication** – intended for local use only. Do not expose to the internet.

## 📄 License

[Apache License 2.0](LICENSE)
