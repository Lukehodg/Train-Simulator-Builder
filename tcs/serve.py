"""Local static server for the viewer.

The bundles are Arrow/JSON files fetched by the page, so the app cannot run from file:// - it needs an HTTP
origin. This is a dependency-free server (stdlib only) used by `tcs serve` and by the packaged distribution.
"""
from __future__ import annotations

import http.server
import socket
import socketserver
import threading
import webbrowser
from functools import partial
from pathlib import Path

MIME = {
    ".arrow": "application/vnd.apache.arrow.file", ".json": "application/json", ".js": "text/javascript",
    ".mjs": "text/javascript", ".css": "text/css", ".html": "text/html; charset=utf-8", ".svg": "image/svg+xml",
    ".png": "image/png", ".jpg": "image/jpeg", ".webp": "image/webp", ".woff2": "font/woff2", ".parquet": "application/vnd.apache.parquet",
    ".geojson": "application/geo+json", ".csv": "text/csv", ".xlsx": "application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
    ".docx": "application/vnd.openxmlformats-officedocument.wordprocessingml.document",
}


class Handler(http.server.SimpleHTTPRequestHandler):
    def guess_type(self, path):  # noqa: A003 - stdlib signature
        ext = Path(path).suffix.lower()
        return MIME.get(ext) or super().guess_type(path)

    def end_headers(self):
        # Local-only tool: no caching, so a rebuilt bundle shows up on refresh.
        self.send_header("Cache-Control", "no-store")
        super().end_headers()

    def log_message(self, fmt, *args):
        if "404" in (fmt % args):
            super().log_message(fmt, *args)


def free_port(preferred: int) -> int:
    for port in [preferred, 0]:
        with socket.socket() as s:
            try:
                s.bind(("127.0.0.1", port))
                return s.getsockname()[1]
            except OSError:
                continue
    raise OSError("no free port")


def serve(root: Path, port: int = 8000, open_browser: bool = True, block: bool = True) -> tuple[str, socketserver.TCPServer]:
    port = free_port(port)
    handler = partial(Handler, directory=str(root))
    httpd = socketserver.ThreadingTCPServer(("127.0.0.1", port), handler)
    httpd.daemon_threads = True
    url = f"http://localhost:{port}/"
    thread = threading.Thread(target=httpd.serve_forever, daemon=True)
    thread.start()
    if open_browser:
        webbrowser.open(url)
    if block:
        try:
            thread.join()
        except KeyboardInterrupt:
            httpd.shutdown()
    return url, httpd
