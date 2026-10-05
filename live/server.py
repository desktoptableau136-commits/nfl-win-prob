"""Serves the dashboard (the docs/ folder) on your home network.

    .venv\\Scripts\\python -m live.server

Then open http://localhost:8000, or http://<this PC's IP>:8000 from a phone on the same Wi-Fi.
This is the same page GitHub Pages hosts. Everything (ESPN data + the model)
runs in the browser, so this server only hands out files.
"""
import socket
from functools import partial
from http.server import SimpleHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path

SITE = Path(__file__).resolve().parent.parent / "docs"
PORT = 8000


class Handler(SimpleHTTPRequestHandler):
    # Windows can map .js to text/plain, which browsers refuse for modules
    extensions_map = {**SimpleHTTPRequestHandler.extensions_map,
                      ".js": "text/javascript", ".json": "application/json", ".html": "text/html",
                      ".webmanifest": "application/manifest+json"}

    def end_headers(self):
        self.send_header("Cache-Control", "no-cache")  # always pick up a retrained model
        super().end_headers()

    def log_message(self, *args):
        pass


def lan_ip() -> str:
    with socket.socket(socket.AF_INET, socket.SOCK_DGRAM) as s:
        try:
            s.connect(("8.8.8.8", 80))
            return s.getsockname()[0]
        except OSError:
            return "localhost"


if __name__ == "__main__":
    server = ThreadingHTTPServer(("0.0.0.0", PORT), partial(Handler, directory=str(SITE)))
    print(f"\n  Dashboard:  http://localhost:{PORT}\n  From your phone (same Wi-Fi):  http://{lan_ip()}:{PORT}\n"
          "  Press Ctrl+C to stop.\n", flush=True)
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        pass
