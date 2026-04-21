#!/usr/bin/env python3
"""
Build script for static Vercel deployment.
Bundles sessions.json into the ui/ directory for static hosting.

Usage:
    python build.py                        # Auto-detect sessions
    python build.py inference-engineering  # Specific book slug
"""
import json
import shutil
import sys
from pathlib import Path


def find_sessions(book_slug=None):
    if book_slug:
        path = Path(f"data/{book_slug}/output/sessions.json")
        if path.exists():
            return path, book_slug
        print(f"Error: data/{book_slug}/output/sessions.json not found")
        sys.exit(1)

    # Auto-detect: prefer data/ books, fall back to legacy output/
    data_dir = Path("data")
    if data_dir.exists():
        for book_dir in sorted(data_dir.iterdir()):
            candidate = book_dir / "output" / "sessions.json"
            if candidate.exists():
                return candidate, book_dir.name

    legacy = Path("output/sessions.json")
    if legacy.exists():
        return legacy, None

    print("Error: No sessions.json found. Run the pipeline first.")
    sys.exit(1)


def build(book_slug=None):
    sessions_src, detected_slug = find_sessions(book_slug)
    slug_label = detected_slug or "legacy"
    print(f"Building from: {sessions_src}  (book: {slug_label})")

    with open(sessions_src, encoding='utf-8') as f:
        sessions = json.load(f)

    # Strip pipeline-only fields to reduce payload size
    for session in sessions.values():
        session.pop('evaluation', None)
        session.pop('needs_review', None)
        session.pop('tension_excerpt', None)

    dist = Path("dist")
    if dist.exists():
        shutil.rmtree(dist)
    dist.mkdir()

    # Copy all ui/ files
    for f in Path("ui").iterdir():
        if f.is_file() and f.name != 'sessions.json':
            shutil.copy2(f, dist / f.name)

    # Write stripped sessions.json
    out_path = dist / "sessions.json"
    with open(out_path, 'w', encoding='utf-8') as f:
        json.dump(sessions, f, ensure_ascii=False, separators=(',', ':'))

    size_kb = out_path.stat().st_size // 1024
    print(f"  dist/sessions.json — {len(sessions)} sessions, {size_kb} KB")
    print(f"  dist/ ready.\n")
    print("Deploy:")
    print("  Vercel (CLI):  vercel dist/ --prod")
    print("  Netlify (CLI): netlify deploy --dir=dist --prod")
    print("  GitHub Pages:  push dist/ contents to gh-pages branch")


if __name__ == '__main__':
    book = sys.argv[1] if len(sys.argv) > 1 else None
    build(book)
