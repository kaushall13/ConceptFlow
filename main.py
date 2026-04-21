#!/usr/bin/env python3
"""
Micro-Learning Curriculum Builder - Main Entry Point

Transforms technical PDF books into structured 15-minute reading sessions.
"""

import sys
# Force UTF-8 output on Windows (prevents UnicodeEncodeError for β, α, etc.)
if sys.stdout.encoding and sys.stdout.encoding.lower() != 'utf-8':
    sys.stdout.reconfigure(encoding='utf-8', errors='replace')
if sys.stderr.encoding and sys.stderr.encoding.lower() != 'utf-8':
    sys.stderr.reconfigure(encoding='utf-8', errors='replace')

import argparse
import json
import os
import sys
from datetime import datetime
from pathlib import Path
from flask import Flask, jsonify, request, send_from_directory

sys.path.insert(0, str(Path(__file__).parent))

from utils.error_handler import (
    validate_pdf_path,
    validate_config,
    print_error,
    print_success,
    print_info,
    handle_pipeline_errors
)

app = Flask(__name__)

CONFIG_FILE = "config.json"
STATE_DIR = "state"
OUTPUT_DIR = "output"
UI_DIR = "ui"
DATA_DIR = "data"


def get_book_slug(pdf_path: str) -> str:
    import re
    stem = Path(pdf_path).stem
    slug = re.sub(r'(?<!^)(?=[A-Z])', '-', stem).lower()
    slug = re.sub(r'[^a-z0-9-]', '-', slug)
    slug = re.sub(r'-+', '-', slug).strip('-')
    return slug or "book"


def get_book_paths(book_slug: str = None):
    """Return (state_dir, output_dir) for a book slug, or legacy paths if None."""
    if book_slug:
        base = Path(DATA_DIR) / book_slug
        return str(base / "state"), str(base / "output")
    return STATE_DIR, OUTPUT_DIR


def load_config():
    if not os.path.exists(CONFIG_FILE):
        return {}
    with open(CONFIG_FILE, 'r', encoding='utf-8') as f:
        return json.load(f)


def save_config(config):
    with open(CONFIG_FILE, 'w', encoding='utf-8') as f:
        json.dump(config, f, indent=2)


def ensure_api_key():
    config = load_config()

    # Accept either new dual-key fields or legacy single-key field
    has_key = (
        config.get('cerebras_api_key_1') or
        config.get('cerebras_api_key_2') or
        config.get('cerebras_api_key')  # legacy fallback
    )

    if not has_key:
        print("Cerebras API key not found in config.json")
        api_key = input("Please enter your Cerebras API key: ").strip()
        if not api_key:
            print("Error: API key is required")
            sys.exit(1)
        if len(api_key) < 10:
            print("Warning: API key seems unusually short")
        config['cerebras_api_key_1'] = api_key
        save_config(config)
        print("API key saved to config.json as cerebras_api_key_1")

    if 'ollama_host' not in config:
        config['ollama_host'] = 'http://localhost:11434'
    if 'ollama_model' not in config:
        config['ollama_model'] = 'llama3.2:3b'
    if 'cerebras_model' not in config:
        config['cerebras_model'] = 'llama-3.3-70b'
    if 'cerebras_daily_token_limit' not in config:
        config['cerebras_daily_token_limit'] = 900000

    try:
        validate_config(config)
    except Exception as e:
        print_error(f"Configuration error: {e}")
        sys.exit(1)

    return config


def run_pipeline(pdf_path):
    try:
        pdf_path = validate_pdf_path(pdf_path)
        print(f"Starting curriculum generation pipeline for: {pdf_path}")
        print("This will process the PDF through 9 stages and may take 15-30 minutes.")

        config = ensure_api_key()

        try:
            from pipeline.orchestrator import run_orchestrator
        except ImportError as e:
            print_error(f"Error importing pipeline components: {e}")
            print_info("Please install dependencies: pip install -r requirements.txt")
            sys.exit(1)

        book_slug = get_book_slug(pdf_path)
        run_orchestrator(pdf_path, config, auto_continue=True, book_slug=book_slug)
        _, out_dir = get_book_paths(book_slug)
        print("\nSUCCESS: Pipeline completed successfully!")
        print(f"Sessions saved to {out_dir}/sessions.json")
        print("Start the web server with: python main.py --serve")

    except KeyboardInterrupt:
        print("\n\nPipeline interrupted by user.")
        print(f"Progress has been saved. Resume with: python main.py --pipeline {pdf_path}")
        sys.exit(0)
    except Exception as e:
        print_error(f"Pipeline failed: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)


def serve_ui():
    try:
        print("Starting Micro-Learning Curriculum Builder web interface...")
        print("Open http://localhost:5000 in your browser")
        print("Press Ctrl+C to stop the server")

        os.makedirs(STATE_DIR, exist_ok=True)
        os.makedirs(OUTPUT_DIR, exist_ok=True)

        app.run(host='0.0.0.0', port=5000, debug=False)

    except KeyboardInterrupt:
        print("\n\nServer stopped by user.")
    except Exception as e:
        print_error(f"Failed to start server: {e}")
        sys.exit(1)


def export_sessions():
    try:
        print("Exporting sessions and progress...")
        os.makedirs(OUTPUT_DIR, exist_ok=True)

        sessions_file = os.path.join(OUTPUT_DIR, "sessions.json")
        progress_file = os.path.join(OUTPUT_DIR, "progress.json")

        if not os.path.exists(sessions_file):
            print_error("No sessions found to export")
            print_info("Complete the pipeline first: python main.py --pipeline <pdf>")
            sys.exit(1)

        with open(sessions_file, 'r', encoding='utf-8') as f:
            sessions = json.load(f)

        progress = {}
        if os.path.exists(progress_file):
            with open(progress_file, 'r', encoding='utf-8') as f:
                progress = json.load(f)

        export_data = {
            "timestamp": datetime.now().isoformat(),
            "sessions": sessions,
            "progress": progress
        }

        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        export_file = f"curriculum_export_{timestamp}.json"

        with open(export_file, 'w', encoding='utf-8') as f:
            json.dump(export_data, f, indent=2)

        print_success(f"Exported to {export_file}")
        print_info(f"  Sessions: {len(sessions)}")
        print_info(f"  Progress records: {len(progress)}")

    except Exception as e:
        print_error(f"Export failed: {e}")
        sys.exit(1)


def import_sessions(import_file):
    print(f"Importing from {import_file}...")

    if not os.path.exists(import_file):
        print(f"Error: Import file not found: {import_file}")
        sys.exit(1)

    with open(import_file, 'r', encoding='utf-8') as f:
        export_data = json.load(f)

    if "sessions" not in export_data:
        print("Error: Invalid export file format")
        sys.exit(1)

    os.makedirs(OUTPUT_DIR, exist_ok=True)

    sessions_file = os.path.join(OUTPUT_DIR, "sessions.json")
    with open(sessions_file, 'w', encoding='utf-8') as f:
        json.dump(export_data["sessions"], f, indent=2)

    if "progress" in export_data:
        progress_file = os.path.join(OUTPUT_DIR, "progress.json")
        with open(progress_file, 'w', encoding='utf-8') as f:
            json.dump(export_data["progress"], f, indent=2)

    print(f"SUCCESS: Imported from {import_file}")
    print(f"  Timestamp: {export_data.get('timestamp', 'unknown')}")
    print(f"  Sessions: {len(export_data['sessions'])}")


def reset_state():
    print("WARNING: This will delete all pipeline state and generated sessions!")
    print("This action cannot be undone.")

    confirmation = input("Type 'yes' to confirm: ").strip().lower()

    if confirmation != 'yes':
        print("Reset cancelled")
        return

    files_to_delete = [
        os.path.join(STATE_DIR, "pipeline_state.json"),
        os.path.join(OUTPUT_DIR, "sessions.json"),
        os.path.join(OUTPUT_DIR, "progress.json")
    ]

    deleted_count = 0
    for file_path in files_to_delete:
        if os.path.exists(file_path):
            os.remove(file_path)
            deleted_count += 1
            print(f"SUCCESS: Deleted: {file_path}")

    print(f"\nSUCCESS: Reset complete. Deleted {deleted_count} file(s)")


# Flask Routes

@app.route('/')
def index():
    return send_from_directory(UI_DIR, 'index.html')


@app.route('/<path:filename>')
def serve_static(filename):
    return send_from_directory(UI_DIR, filename)


@app.route('/api/books')
def list_books():
    books = []
    data_dir = Path(DATA_DIR)
    if data_dir.exists():
        for book_dir in sorted(data_dir.iterdir()):
            if not book_dir.is_dir():
                continue
            info = {"slug": book_dir.name}
            sessions_f = book_dir / "output" / "sessions.json"
            state_f = book_dir / "state" / "pipeline_state.json"
            if sessions_f.exists():
                with open(sessions_f, encoding='utf-8') as f:
                    info["session_count"] = len(json.load(f))
            if state_f.exists():
                with open(state_f, encoding='utf-8') as f:
                    info["stage"] = json.load(f).get("current_stage", "Unknown")
            books.append(info)
    # Legacy output/ dir
    legacy = Path(OUTPUT_DIR) / "sessions.json"
    if legacy.exists():
        with open(legacy, encoding='utf-8') as f:
            books.insert(0, {"slug": "_legacy", "session_count": len(json.load(f)), "stage": "Complete"})
    return jsonify({"books": books})


@app.route('/api/state')
@app.route('/api/status')
def get_state():
    book = request.args.get('book')
    state_dir, _ = get_book_paths(book)
    state_file = os.path.join(state_dir, "pipeline_state.json")
    if not os.path.exists(state_file):
        return jsonify({"current_stage": "Not started", "outputs": {}, "session_results": {}})
    with open(state_file, 'r', encoding='utf-8') as f:
        return jsonify(json.load(f))


@app.route('/api/index', methods=['GET', 'POST'])
def handle_index():
    book = request.args.get('book')
    state_dir, _ = get_book_paths(book)
    state_file = os.path.join(state_dir, "pipeline_state.json")

    if not os.path.exists(state_file):
        return jsonify({"error": "No pipeline state found"}), 404

    with open(state_file, 'r', encoding='utf-8') as f:
        state = json.load(f)

    if request.method == 'GET':
        index_data = state.get('outputs', {}).get('IndexGen', {}).get('index')
        if index_data:
            return jsonify({
                "index": index_data,
                "approved": state.get('outputs', {}).get('IndexGen', {}).get('approved', False)
            })
        return jsonify({"error": "Index not generated yet"}), 404

    elif request.method == 'POST':
        data = request.json
        action = data.get('action')

        if action == 'approve':
            if 'IndexGen' not in state['outputs']:
                return jsonify({"error": "No index to approve"}), 400
            state['outputs']['IndexGen']['approved'] = True
            with open(state_file, 'w', encoding='utf-8') as f:
                json.dump(state, f, indent=2)
            return jsonify({"status": "approved"})

        elif action == 'edit':
            instructions = data.get('instructions', '')
            if not instructions:
                return jsonify({"error": "Edit instructions required"}), 400
            if 'IndexGen' not in state.get('outputs', {}):
                return jsonify({"error": "No index to edit"}), 400
            state['outputs']['IndexGen']['edit_instructions'] = instructions
            state['outputs']['IndexGen']['approved'] = False
            with open(state_file, 'w', encoding='utf-8') as f:
                json.dump(state, f, indent=2)
            return jsonify({"status": "edit_submitted"})

        return jsonify({"error": "Invalid action"}), 400


@app.route('/api/config', methods=['GET'])
def get_config():
    config = load_config()
    has_key_1 = bool(config.get('cerebras_api_key_1') or config.get('cerebras_api_key'))
    has_key_2 = bool(config.get('cerebras_api_key_2'))
    safe_config = {
        'ollama_host': config.get('ollama_host'),
        'ollama_model': config.get('ollama_model'),
        'cerebras_model': config.get('cerebras_model', 'llama-3.3-70b'),
        'cerebras_daily_token_limit': config.get('cerebras_daily_token_limit', 900000),
        'groq_model': config.get('groq_model', 'llama-3.3-70b-versatile'),
        'has_cerebras_key': has_key_1,
        'has_cerebras_key_2': has_key_2,
        'has_groq_key': bool(config.get('groq_api_key')),
    }
    return jsonify(safe_config)


@app.route('/api/config', methods=['POST'])
def update_config():
    try:
        data = request.json
        if 'ollama_host' in data:
            host = data['ollama_host']
            if not host.startswith('http://') and not host.startswith('https://'):
                return jsonify({"error": "Ollama host must start with http:// or https://"}), 400
        config = load_config()
        config.update(data)
        save_config(config)
        return jsonify({"status": "updated"})
    except Exception as e:
        return jsonify({"error": str(e)}), 500


@app.route('/api/sessions')
def get_sessions():
    book = request.args.get('book')
    _, output_dir = get_book_paths(book)
    sessions_file = os.path.join(output_dir, "sessions.json")
    if not os.path.exists(sessions_file):
        return jsonify({"sessions": {}})
    with open(sessions_file, 'r', encoding='utf-8') as f:
        sessions = json.load(f)
    # Strip full content for list endpoint — return metadata only for performance
    stripped = {}
    for sid, session in sessions.items():
        stripped[sid] = {k: v for k, v in session.items() if k != 'content'}
    return jsonify({"sessions": stripped})


@app.route('/api/sessions/<session_id>')
def get_session(session_id):
    book = request.args.get('book')
    _, output_dir = get_book_paths(book)
    sessions_file = os.path.join(output_dir, "sessions.json")
    if not os.path.exists(sessions_file):
        return jsonify({"error": "No sessions found"}), 404
    with open(sessions_file, 'r', encoding='utf-8') as f:
        sessions = json.load(f)
    if session_id not in sessions:
        return jsonify({"error": "Session not found"}), 404
    return jsonify(sessions[session_id])


@app.route('/api/progress', methods=['GET', 'POST'])
def handle_progress():
    book = request.args.get('book')
    _, output_dir = get_book_paths(book)
    progress_file = os.path.join(output_dir, "progress.json")

    progress = {}
    if os.path.exists(progress_file):
        with open(progress_file, 'r', encoding='utf-8') as f:
            progress = json.load(f)

    if request.method == 'GET':
        return jsonify(progress)

    elif request.method == 'POST':
        data = request.json
        session_id = data.get('session_id')

        if not session_id:
            return jsonify({"error": "session_id required"}), 400

        if session_id not in progress:
            progress[session_id] = {}

        # Support both envelope shape {session_id, updates: {...}} and flat shape
        # {session_id, completed, bookmark_word_index, ...}
        updates = data.get('updates')
        if updates is not None:
            progress[session_id].update(updates)
        else:
            # Flat shape — merge all keys except session_id
            for k, v in data.items():
                if k == 'session_id':
                    continue
                if k == 'bookmark_word_index':
                    # Normalise to 'bookmark' key used internally
                    if v is None:
                        progress[session_id].pop('bookmark', None)
                    else:
                        progress[session_id]['bookmark'] = v
                elif v is None and k in progress[session_id]:
                    progress[session_id].pop(k, None)
                else:
                    progress[session_id][k] = v

        os.makedirs(output_dir, exist_ok=True)
        with open(progress_file, 'w', encoding='utf-8') as f:
            json.dump(progress, f, indent=2)

        return jsonify({"status": "updated", "progress": progress[session_id]})


@app.route('/api/bookmark', methods=['POST'])
def set_bookmark():
    data = request.json
    session_id = data.get('session_id')
    word_index = data.get('word_index')

    if not session_id or word_index is None:
        return jsonify({"error": "session_id and word_index required"}), 400

    book = request.args.get('book')
    _, output_dir = get_book_paths(book)
    progress_file = os.path.join(output_dir, "progress.json")
    progress = {}
    if os.path.exists(progress_file):
        with open(progress_file, 'r', encoding='utf-8') as f:
            progress = json.load(f)

    if session_id not in progress:
        progress[session_id] = {}

    progress[session_id]['bookmark'] = word_index

    os.makedirs(output_dir, exist_ok=True)
    with open(progress_file, 'w', encoding='utf-8') as f:
        json.dump(progress, f, indent=2)

    return jsonify({"status": "bookmarked"})


@handle_pipeline_errors
def main():
    parser = argparse.ArgumentParser(
        description='Micro-Learning Curriculum Builder - Transform PDFs into structured reading sessions',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python main.py --pipeline book.pdf           # Run full pipeline
  python main.py --serve                      # Start web interface
  python main.py --export                     # Export sessions
  python main.py --import export.json         # Import sessions
  python main.py --reset                      # Reset all state
        """
    )

    parser.add_argument('--pipeline', metavar='PDF_PATH',
                        help='Run full curriculum generation pipeline on PDF file')
    parser.add_argument('--serve', action='store_true',
                        help='Start Flask web server')
    parser.add_argument('--export', action='store_true',
                        help='Export sessions and progress to JSON file')
    parser.add_argument('--import', metavar='FILE', dest='import_file',
                        help='Import sessions and progress from JSON file')
    parser.add_argument('--reset', action='store_true',
                        help='Reset all pipeline state and sessions')
    parser.add_argument('--continue', action='store_true', dest='continue_flag',
                        help='Continue pipeline without prompting')

    args = parser.parse_args()

    if not any([args.pipeline, args.serve, args.export, args.import_file, args.reset]):
        parser.print_help()
        sys.exit(1)

    if args.pipeline:
        if not os.path.exists(args.pipeline):
            print(f"Error: PDF file not found: {args.pipeline}")
            sys.exit(1)
        run_pipeline(args.pipeline)

    elif args.serve:
        serve_ui()

    elif args.export:
        export_sessions()

    elif args.import_file:
        import_sessions(args.import_file)

    elif args.reset:
        reset_state()


if __name__ == '__main__':
    main()
