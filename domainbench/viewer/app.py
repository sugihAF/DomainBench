"""
DomainBench Result Viewer - Flask application
"""

import json
import os
import re
from pathlib import Path
from typing import Optional

from flask import Flask, jsonify, render_template, request, send_from_directory, abort


# Capability detection by filename prefix
CAPABILITY_PREFIXES = {
    "chat_completion": ["chat_", "chat-", "chatcompletion", "chat_completion"],
    "ocr": ["ocr_", "ocr-", "ocr"],
    "function_calling": ["func_", "func-", "function_", "function-", "funccall", "func_call"],
    "voice": ["voice_", "voice-", "voice"],
}


def detect_capability(filename: str) -> Optional[str]:
    """Detect capability from filename prefix."""
    filename_lower = filename.lower()
    for capability, prefixes in CAPABILITY_PREFIXES.items():
        for prefix in prefixes:
            if filename_lower.startswith(prefix):
                return capability
    return None


def find_results_dir() -> str:
    """Find the results directory, checking multiple locations."""
    # Check common locations
    candidates = [
        os.path.join(os.getcwd(), "results"),  # Current directory
        os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(__file__))), "results"),  # Package root
    ]

    for path in candidates:
        if os.path.isdir(path) and any(f.endswith('.json') for f in os.listdir(path)):
            return path

    # Default to current directory
    return os.path.join(os.getcwd(), "results")


def create_app(results_dir: Optional[str] = None) -> Flask:
    """Create and configure the Flask application."""
    app = Flask(__name__, template_folder="templates")

    # Default results directory - try to find it automatically
    if results_dir is None:
        results_dir = find_results_dir()

    app.config["RESULTS_DIR"] = results_dir

    @app.route("/")
    def index():
        """Serve the main dashboard page."""
        return render_template("index.html")

    @app.route("/api/capabilities")
    def list_capabilities():
        """List available capabilities that have results."""
        results_path = Path(app.config["RESULTS_DIR"])
        found_capabilities = set()

        if not results_path.exists():
            return jsonify({"capabilities": []})

        # Scan all JSON files and detect capabilities by prefix
        for f in results_path.glob("*.json"):
            capability = detect_capability(f.name)
            if capability:
                found_capabilities.add(capability)

        # Return in consistent order
        ordered = []
        for cap in ["chat_completion", "ocr", "function_calling", "voice"]:
            if cap in found_capabilities:
                ordered.append(cap)

        return jsonify({"capabilities": ordered})

    @app.route("/api/results/<capability>")
    def list_results(capability: str):
        """List available result files for a capability."""
        results_path = Path(app.config["RESULTS_DIR"])

        if not results_path.exists():
            return jsonify({"results": []})

        results = []
        # Get all JSON files that match this capability
        for f in sorted(results_path.glob("*.json"), key=lambda x: x.stat().st_mtime, reverse=True):
            if detect_capability(f.name) != capability:
                continue

            try:
                with open(f, "r", encoding="utf-8") as file:
                    data = json.load(file)

                    # Voice results use a different structure
                    if capability == "voice":
                        models_list = data.get("config", {}).get("models", [])
                        total_runs = sum(
                            len(m.get("runs", []))
                            for m in data.get("results", {}).values()
                        )
                        summary = {
                            "filename": f.name,
                            "path": str(f),
                            "benchmark_name": data.get("config", {}).get("dataset", f.stem),
                            "timestamp": data.get("timestamp", "Unknown"),
                            "total_test_cases": total_runs,
                            "models": models_list,
                        }
                    else:
                        # Extract summary info for preview
                        summary = {
                            "filename": f.name,
                            "path": str(f),
                            "benchmark_name": data.get("benchmark_name", f.stem),
                            "timestamp": data.get("timestamp", "Unknown"),
                            "total_test_cases": data.get("summary", {}).get("total_test_cases", 0),
                            "models": list(data.get("summary", {}).get("models", {}).keys()),
                        }
                    results.append(summary)
            except (json.JSONDecodeError, KeyError):
                # Skip invalid files
                continue

        return jsonify({"results": results})

    @app.route("/api/result/<capability>/<filename>")
    def get_result(capability: str, filename: str):
        """Get a specific result file."""
        # File is directly in results directory, not in subdirectory
        results_path = Path(app.config["RESULTS_DIR"]) / filename

        if not results_path.exists():
            return jsonify({"error": "Result not found"}), 404

        # Verify the file matches the requested capability
        if detect_capability(filename) != capability:
            return jsonify({"error": "Capability mismatch"}), 400

        try:
            with open(results_path, "r", encoding="utf-8") as f:
                data = json.load(f)
            return jsonify(data)
        except json.JSONDecodeError:
            return jsonify({"error": "Invalid JSON file"}), 400

    @app.route("/api/audio/<path:filepath>")
    def serve_audio(filepath: str):
        """Serve audio files from the audio subdirectory within results."""
        # Audio files live under {results_dir}/audio/...
        audio_base = Path(app.config["RESULTS_DIR"]) / "audio"
        full_path = (audio_base / filepath).resolve()

        # Security: ensure the resolved path is inside the audio directory
        try:
            full_path.relative_to(audio_base.resolve())
        except ValueError:
            abort(403)

        if not full_path.is_file():
            abort(404)

        return send_from_directory(
            str(full_path.parent), full_path.name,
        )

    return app


def run_viewer(
    results_dir: Optional[str] = None,
    host: str = "127.0.0.1",
    port: int = 5000,
    debug: bool = False
):
    """Run the viewer web server."""
    app = create_app(results_dir)

    print(f"\n  DomainBench Result Viewer")
    print(f"  ========================")
    print(f"  Results directory: {app.config['RESULTS_DIR']}")
    print(f"  Open in browser: http://{host}:{port}")
    print(f"\n  Press Ctrl+C to stop\n")

    app.run(host=host, port=port, debug=debug)


if __name__ == "__main__":
    run_viewer(debug=True)
