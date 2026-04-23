#!/usr/bin/env python3
"""
Robotics Visualizer Gallery Server

Serves all datasets under a data directory. Datasets are discovered automatically
by scanning for the pattern: v2d_{name}_retarget*/*/recordings/*.viser

Usage:
  python serve.py                                  # http://0.0.0.0:8080, data in ./
  python serve.py --data-dir ~/ROBOTICS_VISUALIZER
  python serve.py --data-dir /data/vis --port 9000
  python serve.py --host 127.0.0.1 --port 8080
"""

import argparse
import json
import re
import mimetypes
from http.server import HTTPServer, BaseHTTPRequestHandler
from pathlib import Path
from socketserver import ThreadingMixIn
from urllib.parse import urlparse, unquote

CHUNK = 1024 * 1024  # 1 MB read chunks

ROOT = Path(__file__).parent
DATASETS_DIR = ROOT / "datasets"

# Set by main() from --data-dir; defaults to DATASETS_DIR
DATA_DIR: Path = DATASETS_DIR

# Extra html dirs added via --html-dir; mounted at _live/<dirname>/ in URLs.
# Maps virtual URL prefix → real filesystem path.
_LIVE_MOUNTS: dict[str, Path] = {}


# ── Dataset discovery ─────────────────────────────────────────────────────────

def _vc_bundle_key(vc_dir: Path):
    """(filename, size) of the main JS bundle — identifies identical viser-client builds."""
    bundles = sorted(vc_dir.glob("assets/index-*.js"))
    if not bundles:
        return None
    b = bundles[0]
    return (b.name, b.stat().st_size)


def discover_datasets():
    """Scan for datasets under DATA_DIR and any --html-dir mounts."""
    datasets = []
    # bundle_key -> canonical vc path; identical builds share one browser-cache entry
    canonical_vc: dict = {}

    # ── standard layout: DATA_DIR/v2d_*/*/recordings ──────────────────────────
    for recordings_dir in sorted(DATA_DIR.glob("v2d_*/*/recordings")):
        top = recordings_dir.parts[-3]          # e.g. v2d_h2o_retarget_exp_200
        m = re.match(r"v2d_(.+?)_retarget", top)
        if not m:
            continue
        name = m.group(1)

        viser_files = sorted(recordings_dir.glob("*.viser"))
        if not viser_files:
            continue

        viser_client_dir = recordings_dir.parent / "viser-client"
        rec_rel = recordings_dir.relative_to(DATA_DIR).as_posix()

        vc_rel = None
        if viser_client_dir.exists():
            key = _vc_bundle_key(viser_client_dir)
            if key and key in canonical_vc:
                vc_rel = canonical_vc[key]
            else:
                vc_rel = viser_client_dir.relative_to(DATA_DIR).as_posix()
                if key:
                    canonical_vc[key] = vc_rel

        recordings = []
        for f in viser_files:
            stem = f.stem
            mp4 = recordings_dir / f"{stem}.mp4"
            recordings.append({
                "stem": stem,
                "viser": f"{rec_rel}/{f.name}",
                "mp4": f"{rec_rel}/{stem}.mp4" if mp4.exists() else None,
            })

        datasets.append({
            "name": name,
            "viser_client": vc_rel,
            "count": len(viser_files),
            "recordings": recordings,
        })

    # ── --html-dir mounts: each dir has recordings/ directly inside ───────────
    for prefix, html_dir in sorted(_LIVE_MOUNTS.items()):
        recordings_dir = html_dir / "recordings"
        if not recordings_dir.exists():
            continue
        viser_files = sorted(recordings_dir.glob("*.viser"))
        if not viser_files:
            continue

        m = re.match(r"v2d_(.+?)_retarget", html_dir.name)
        name = m.group(1) if m else html_dir.name

        rec_rel = f"{prefix}/recordings"
        viser_client_dir = html_dir / "viser-client"
        vc_rel = None
        if viser_client_dir.exists():
            key = _vc_bundle_key(viser_client_dir)
            if key and key in canonical_vc:
                vc_rel = canonical_vc[key]
            else:
                vc_rel = f"{prefix}/viser-client"
                if key:
                    canonical_vc[key] = vc_rel

        recordings = []
        for f in viser_files:
            stem = f.stem
            mp4 = recordings_dir / f"{stem}.mp4"
            recordings.append({
                "stem": stem,
                "viser": f"{rec_rel}/{f.name}",
                "mp4": f"{rec_rel}/{stem}.mp4" if mp4.exists() else None,
            })

        datasets.append({
            "name": name,
            "viser_client": vc_rel,
            "count": len(viser_files),
            "recordings": recordings,
        })

    return datasets


# ── Gallery HTML ──────────────────────────────────────────────────────────────

GALLERY_HTML = """\
<!DOCTYPE html>
<html lang="en">
<head>
  <meta charset="UTF-8">
  <meta name="viewport" content="width=device-width, initial-scale=1.0">
  <title>Robotics Visualizer</title>
  <style>
    :root {
      --bg:        #0d1117;
      --surface:   #161b22;
      --surface2:  #1c2128;
      --border:    #30363d;
      --accent:    #58a6ff;
      --accent-bg: #1a2d45;
      --text:      #e6edf3;
      --muted:     #7d8590;
      --r:         8px;
    }
    *, *::before, *::after { box-sizing: border-box; margin: 0; padding: 0; }
    html, body {
      height: 100%;
      background: var(--bg);
      color: var(--text);
      font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', system-ui, sans-serif;
      font-size: 14px;
      overflow: hidden;
    }

    /* ── Layout ─────────────────────────────── */
    #app { display: flex; flex-direction: column; height: 100vh; }

    #header {
      display: flex;
      align-items: center;
      gap: 20px;
      padding: 12px 20px;
      background: var(--surface);
      border-bottom: 1px solid var(--border);
      flex-shrink: 0;
    }
    #header h1 { font-size: 16px; font-weight: 600; color: var(--accent); }
    #stats { font-size: 13px; color: var(--muted); }
    #stats strong { color: var(--text); }

    #body { display: flex; flex: 1; overflow: hidden; }

    /* ── Sidebar ─────────────────────────────── */
    #sidebar {
      width: 300px;
      min-width: 260px;
      flex-shrink: 0;
      border-right: 1px solid var(--border);
      overflow-y: auto;
      padding: 10px;
    }
    #sidebar::-webkit-scrollbar { width: 4px; }
    #sidebar::-webkit-scrollbar-thumb { background: var(--border); border-radius: 2px; }

    .ds-card {
      border: 1px solid var(--border);
      border-radius: var(--r);
      margin-bottom: 8px;
      overflow: hidden;
    }
    .ds-header {
      display: flex;
      align-items: center;
      gap: 8px;
      padding: 10px 14px;
      background: var(--surface);
      cursor: pointer;
      user-select: none;
      transition: background .12s;
    }
    .ds-header:hover { background: var(--surface2); }
    .ds-name { font-weight: 600; flex: 1; font-size: 14px; }
    .ds-badge {
      font-size: 11px;
      color: var(--muted);
      background: var(--bg);
      border: 1px solid var(--border);
      padding: 1px 8px;
      border-radius: 10px;
      white-space: nowrap;
    }
    .chevron { color: var(--muted); flex-shrink: 0; transition: transform .2s; }
    .ds-card.open .chevron { transform: rotate(180deg); }

    .ds-body { display: none; padding: 6px; }
    .ds-card.open .ds-body { display: block; }

    .group-label {
      font-size: 11px;
      font-weight: 500;
      color: var(--muted);
      text-transform: uppercase;
      letter-spacing: .07em;
      padding: 8px 8px 3px;
    }
    .group-label:first-child { padding-top: 4px; }

    .seq-btn {
      display: flex;
      align-items: center;
      gap: 7px;
      width: 100%;
      padding: 6px 10px;
      background: transparent;
      border: none;
      border-radius: 5px;
      color: var(--text);
      font-size: 13px;
      text-align: left;
      cursor: pointer;
      transition: background .1s;
    }
    .seq-btn:hover { background: var(--surface2); }
    .seq-btn.active {
      background: var(--accent-bg);
      color: var(--accent);
    }
    .seq-dot {
      width: 6px; height: 6px;
      border-radius: 50%;
      background: var(--border);
      flex-shrink: 0;
    }
    .seq-btn.active .seq-dot { background: var(--accent); }

    /* ── Viewer ──────────────────────────────── */
    #viewer {
      flex: 1;
      display: flex;
      flex-direction: column;
      overflow: hidden;
      min-width: 0;
    }

    #viewer-bar {
      flex-shrink: 0;
      padding: 7px 16px;
      background: var(--surface);
      border-bottom: 1px solid var(--border);
      font-size: 12px;
      color: var(--muted);
      display: none;
    }
    #viewer-bar.show { display: block; }
    #viewer-bar strong { color: var(--text); }

    #viewer-empty {
      flex: 1;
      display: flex;
      flex-direction: column;
      align-items: center;
      justify-content: center;
      gap: 12px;
      color: var(--muted);
    }
    #viewer-empty svg { opacity: .3; }

    #viser-wrap { flex: 1; overflow: hidden; min-height: 0; }
    #viser-frame { width: 100%; height: 100%; border: none; display: block; }

    #video-pane {
      height: 240px;
      flex-shrink: 0;
      border-top: 1px solid var(--border);
      background: #000;
      display: flex;
      align-items: center;
      justify-content: center;
      position: relative;
    }
    #video-pane video {
      max-width: 100%;
      height: 100%;
      object-fit: contain;
    }
    #video-label {
      position: absolute;
      top: 7px; left: 12px;
      font-size: 11px;
      color: rgba(255,255,255,.35);
      pointer-events: none;
      letter-spacing: .04em;
    }
    .no-video { color: var(--muted); font-size: 13px; }
  </style>
</head>
<body>
<div id="app">
  <div id="header">
    <h1>Robotics Visualizer</h1>
    <span id="stats">Loading&hellip;</span>
  </div>
  <div id="body">
    <div id="sidebar">
      <div id="ds-list"><span style="color:var(--muted)">Loading&hellip;</span></div>
    </div>
    <div id="viewer">
      <div id="viewer-bar"></div>
      <div id="viewer-empty">
        <svg width="56" height="56" viewBox="0 0 24 24" fill="none"
             stroke="currentColor" stroke-width="1">
          <path d="M21 16V8a2 2 0 00-1-1.73l-7-4a2 2 0 00-2 0l-7 4A2 2 0 003 8v8a2 2 0 001 1.73l7 4a2 2 0 002 0l7-4A2 2 0 0021 16z"/>
          <polyline points="3.27 6.96 12 12.01 20.73 6.96"/>
          <line x1="12" y1="22.08" x2="12" y2="12"/>
        </svg>
        <p>Select a sequence to visualize</p>
      </div>
    </div>
  </div>
</div>

<script>
let datasets = [];
let activeViewerSrc = null;
let loopWatcher = null;
let loopTimer   = null;
let loopBlocked = false;
let lastMax     = null;

function getOverlay() {
  const wrap = document.getElementById('viser-wrap');
  if (!wrap) return null;
  let ov = document.getElementById('loop-overlay');
  if (!ov) {
    ov = document.createElement('div');
    ov.id = 'loop-overlay';
    ov.style.cssText =
      'position:absolute;inset:0;background:#0d1117;z-index:9;' +
      'pointer-events:none;opacity:0;transition:opacity .05s';
    wrap.style.position = 'relative';
    wrap.appendChild(ov);
  }
  return ov;
}

function getBestSlider(frame) {
  const sliders = frame.contentDocument.querySelectorAll('[role="slider"]');
  if (!sliders.length) return null;
  let best = null;
  for (const s of sliders) {
    const m = parseFloat(s.getAttribute('aria-valuemax'));
    if (!best || m > parseFloat(best.getAttribute('aria-valuemax'))) best = s;
  }
  return best;
}

function triggerLoop() {
  if (loopBlocked) return;
  if (loopTimer) { clearTimeout(loopTimer); loopTimer = null; }

  // Guard: if user seeked away, reschedule rather than fire
  const frame = document.getElementById('viser-frame');
  try {
    const best = getBestSlider(frame);
    if (best) {
      const max = parseFloat(best.getAttribute('aria-valuemax'));
      const cur = parseFloat(best.getAttribute('aria-valuenow'));
      if (!isNaN(cur) && cur < max - 0.3) { lastMax = null; return; }
    }
  } catch(e) {}

  loopBlocked = true;
  lastMax = null;

  // Viser auto-loops internally — we only need a brief overlay to hide its
  // ~30 ms blank flash (h() sets scene objects invisible before frame 0 runs)
  const ov = getOverlay();
  if (ov) ov.style.opacity = '1';
  setTimeout(() => {
    if (ov) ov.style.opacity = '0';
    loopBlocked = false;
  }, 200);
}

function startLoopWatcher() {
  if (loopWatcher) clearInterval(loopWatcher);
  if (loopTimer)  { clearTimeout(loopTimer); loopTimer = null; }
  loopBlocked = false;
  lastMax = null;

  loopWatcher = setInterval(() => {
    if (loopBlocked) return;
    const frame = document.getElementById('viser-frame');
    if (!frame || !activeViewerSrc) return;
    try {
      const best = getBestSlider(frame);
      if (!best) return;
      const max     = parseFloat(best.getAttribute('aria-valuemax'));
      const current = parseFloat(best.getAttribute('aria-valuenow'));
      if (!(max > 5) || isNaN(current)) return;

      // Fallback: already at end
      if (current >= max - 0.1) { triggerLoop(); return; }

      // Precise timer: fires just before viser's internal loop resets the scene
      if (max !== lastMax) {
        lastMax = max;
        if (loopTimer) clearTimeout(loopTimer);
        loopTimer = setTimeout(triggerLoop, Math.max((max - current - 0.05) * 1000, 0));
      }
    } catch(e) {}
  }, 200);
}

function buildGallery() {
  const total = datasets.reduce((s, d) => s + d.count, 0);
  document.getElementById('stats').innerHTML =
    `<strong>${datasets.length}</strong> dataset${datasets.length !== 1 ? 's' : ''} &nbsp;&middot;&nbsp; ` +
    `<strong>${total}</strong> total sequences`;

  const list = document.getElementById('ds-list');
  list.innerHTML = '';

  for (const ds of datasets) {
    const card = document.createElement('div');
    card.className = 'ds-card';
    card.innerHTML = `
      <div class="ds-header">
        <span class="ds-name">${ds.name}</span>
        <span class="ds-badge">${ds.count} sequences</span>
        <svg class="chevron" width="14" height="14" viewBox="0 0 24 24"
             fill="none" stroke="currentColor" stroke-width="2.5">
          <polyline points="6 9 12 15 18 9"/>
        </svg>
      </div>
      <div class="ds-body"></div>`;

    card.querySelector('.ds-header').addEventListener('click', () => card.classList.toggle('open'));

    const body = card.querySelector('.ds-body');
    for (const rec of ds.recordings) {
      const btn = document.createElement('button');
      btn.className = 'seq-btn';
      btn.dataset.stem = rec.stem;
      btn.innerHTML = `<span class="seq-dot"></span>${rec.stem}`;
      btn.addEventListener('click', () => activate(rec, ds, btn));
      body.appendChild(btn);
    }

    list.appendChild(card);
  }
}

function activate(rec, ds, btn) {
  document.querySelectorAll('.seq-btn.active').forEach(b => b.classList.remove('active'));
  btn.classList.add('active');

  const origin = window.location.origin;

  // Pre-fetch the .viser file immediately — the iframe won't request it for
  // ~300-500ms (while its JS loads), so this gives the download a head start.
  // The browser coalesces the iframe's later request with this in-flight fetch.
  fetch(`${origin}/${rec.viser}`).catch(() => {});
  const playbackUrl  = `${origin}/${rec.viser}`;
  const viewerSrc    = `${origin}/${ds.viser_client}/?playbackPath=${encodeURIComponent(playbackUrl)}`;

  // Title bar
  const bar = document.getElementById('viewer-bar');
  bar.className = 'show';
  bar.innerHTML = `<strong>${ds.name}</strong> &nbsp;/&nbsp; ${rec.stem}`;

  const viewer = document.getElementById('viewer');

  // Remove empty state once
  const empty = document.getElementById('viewer-empty');
  if (empty) empty.remove();

  // Viser iframe — reuse wrapper, just swap src
  activeViewerSrc = viewerSrc;
  let wrap = document.getElementById('viser-wrap');
  if (!wrap) {
    wrap = document.createElement('div');
    wrap.id = 'viser-wrap';
    wrap.innerHTML = `<iframe id="viser-frame" src="" allow="fullscreen"></iframe>`;
    viewer.insertBefore(wrap, document.getElementById('video-pane'));
  }
  document.getElementById('viser-frame').src = viewerSrc;
  startLoopWatcher();

  // Video pane — reuse, swap src
  let vpane = document.getElementById('video-pane');
  if (!vpane) {
    vpane = document.createElement('div');
    vpane.id = 'video-pane';
    viewer.appendChild(vpane);
  }
  if (rec.mp4) {
    // Replace video element to force reload
    vpane.innerHTML = `
      <div id="video-label">Camera feed &mdash; ${rec.stem}</div>
      <video src="/${rec.mp4}" autoplay loop muted playsinline controls></video>`;
  } else {
    vpane.innerHTML = `<span class="no-video">No video available</span>`;
  }
}

fetch('/api/datasets')
  .then(r => r.json())
  .then(data => { datasets = data; buildGallery(); })
  .catch(err => {
    document.getElementById('ds-list').innerHTML =
      `<span style="color:var(--muted)">Error: ${err.message}</span>`;
  });
</script>
</body>
</html>
"""


# ── HTTP Handler ──────────────────────────────────────────────────────────────

class Handler(BaseHTTPRequestHandler):

    def do_GET(self):
        raw_path = unquote(urlparse(self.path).path).lstrip("/")

        # Gallery root
        if raw_path in ("", "index.html"):
            self._send_text(GALLERY_HTML, "text/html; charset=utf-8")
            return

        # Dataset API
        if raw_path == "api/datasets":
            self._send_json(discover_datasets())
            return

        # Static files — check _live/ mounts first, then DATA_DIR
        file_path = None
        for prefix, html_dir in _LIVE_MOUNTS.items():
            if raw_path == prefix or raw_path.startswith(prefix + "/"):
                rel = raw_path[len(prefix):].lstrip("/")
                file_path = html_dir / rel if rel else html_dir
                break
        if file_path is None:
            file_path = DATA_DIR / raw_path
        if file_path.is_dir():
            file_path = file_path / "index.html"
        if not file_path.is_file():
            self.send_error(404, f"Not found: {raw_path}")
            return

        self._send_file(file_path)

    # ── helpers ──

    def _send_text(self, body: str, ctype: str):
        data = body.encode("utf-8")
        self.send_response(200)
        self.send_header("Content-Type", ctype)
        self.send_header("Content-Length", str(len(data)))
        self.end_headers()
        self.wfile.write(data)

    def _send_json(self, obj):
        data = json.dumps(obj, indent=2).encode("utf-8")
        self.send_response(200)
        self.send_header("Content-Type", "application/json")
        self.send_header("Access-Control-Allow-Origin", "*")
        self.send_header("Content-Length", str(len(data)))
        self.end_headers()
        self.wfile.write(data)

    def _send_file(self, file_path: Path):
        mime = mimetypes.guess_type(str(file_path))[0] or "application/octet-stream"
        size = file_path.stat().st_size

        # Immutable assets (hash in name) get a long cache lifetime
        name = file_path.name
        if re.search(r"-[A-Za-z0-9]{8,}\.(js|css|wasm)$", name):
            cache = "public, max-age=31536000, immutable"
        elif file_path.suffix in (".viser", ".mp4", ".hdr", ".ttf"):
            cache = "public, max-age=604800"  # recordings don't change; 7-day cache
        else:
            cache = "no-cache"

        # Range request — required for video seeking
        range_hdr = self.headers.get("Range")
        if range_hdr:
            m = re.match(r"bytes=(\d+)-(\d*)", range_hdr)
            if m:
                start  = int(m.group(1))
                end    = int(m.group(2)) if m.group(2) else size - 1
                end    = min(end, size - 1)
                length = end - start + 1
                self.send_response(206)
                self.send_header("Content-Type", mime)
                self.send_header("Content-Range", f"bytes {start}-{end}/{size}")
                self.send_header("Content-Length", str(length))
                self.send_header("Accept-Ranges", "bytes")
                self.send_header("Cache-Control", cache)
                self.end_headers()
                self._stream(file_path, start, length)
                return

        self.send_response(200)
        self.send_header("Content-Type", mime)
        self.send_header("Content-Length", str(size))
        self.send_header("Accept-Ranges", "bytes")
        self.send_header("Cache-Control", cache)
        self.end_headers()
        self._stream(file_path, 0, size)

    def _stream(self, file_path: Path, offset: int, length: int):
        try:
            with open(file_path, "rb") as f:
                f.seek(offset)
                remaining = length
                while remaining > 0:
                    chunk = f.read(min(CHUNK, remaining))
                    if not chunk:
                        break
                    self.wfile.write(chunk)
                    remaining -= len(chunk)
        except (BrokenPipeError, ConnectionResetError):
            pass  # client navigated away mid-transfer

    def log_message(self, fmt, *args):
        pass  # suppress per-request noise


# ── Entry point ───────────────────────────────────────────────────────────────

def main():
    global DATA_DIR

    parser = argparse.ArgumentParser(description="Robotics Visualizer Gallery Server")
    parser.add_argument("--host", default="0.0.0.0",
                        help="Bind address (default: 0.0.0.0 — all interfaces)")
    parser.add_argument("--port", type=int, default=8080,
                        help="Port (default: 8080)")
    parser.add_argument("--data-dir", type=Path, default=DATASETS_DIR,
                        help="Directory containing v2d_* dataset folders (default: visualizer/datasets/)")
    parser.add_argument("--html-dir", type=Path, action="append", default=[],
                        metavar="DIR",
                        help="Extra html output dir (has recordings/ directly inside). "
                             "Can be repeated. E.g. the --html_dir passed to vis_retargeted.py.")
    args = parser.parse_args()

    DATA_DIR = args.data_dir.expanduser().resolve()
    if not DATA_DIR.is_dir():
        raise SystemExit(f"ERROR: --data-dir does not exist: {DATA_DIR}")

    for raw in args.html_dir:
        p = raw.expanduser().resolve()
        if not p.is_dir():
            raise SystemExit(f"ERROR: --html-dir does not exist: {p}")
        _LIVE_MOUNTS[f"_live/{p.name}"] = p

    datasets = discover_datasets()
    total    = sum(d["count"] for d in datasets)

    print(f"\n  Robotics Visualizer  →  http://{args.host}:{args.port}")
    print(f"  data dir: {DATA_DIR}")
    print(f"  {len(datasets)} dataset(s)  ·  {total} sequences total")
    for d in datasets:
        print(f"    {d['name']:20s}  {d['count']} sequences")
    print("\n  Press Ctrl+C to stop.\n")

    class ThreadedHTTPServer(ThreadingMixIn, HTTPServer):
        daemon_threads = True

    server = ThreadedHTTPServer((args.host, args.port), Handler)
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        print("\nStopped.")


if __name__ == "__main__":
    main()
