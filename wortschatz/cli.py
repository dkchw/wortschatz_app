import argparse
import os
import sys
import logging
from .app import app, load_model, pre_load_path, IN_MEMORY_DB

def main():
    parser = argparse.ArgumentParser(description="Wortschatz: Deutsche Frequenzanalyse")
    parser.add_argument("paths", nargs="*", help="Dateien oder Ordner zum Vorladen (.txt, .srt)")
    parser.add_argument("--port", type=int, default=5000, help="Port für den lokalen Server")
    args = parser.parse_args()

    print("=" * 60)
    print("  Wortschatz · Deutsche Frequenzanalyse")
    print("=" * 60)

    ok, err = load_model()
    if ok:
        print("  ✓ spaCy Modell geladen: de_core_news_sm")
    else:
        print(f"  ✗ Fehler: {err}")
        sys.exit(1)

    if args.paths:
        print("\n  Lade Dateien...")
        for path in args.paths:
            pre_load_path(path)
        print(f"  → {len(IN_MEMORY_DB)} Datei(en) für diese Session geladen.")

    print(f"\n  → http://localhost:{args.port}")
    print("=" * 60)

    # Hide standard Flask routing logs for a cleaner CLI
    log = logging.getLogger('werkzeug')
    log.setLevel(logging.ERROR)

    app.run(debug=False, port=args.port)
