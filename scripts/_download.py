"""Shared download helpers (resumable HTTP + sha256), used by the two download scripts."""
from __future__ import annotations

import hashlib
import os
import sys
import time
import urllib.error
import urllib.parse
import urllib.request


def human(n: float) -> str:
    n = n or 0
    return f"{n/2**30:.2f} GB" if n >= 2**30 else f"{n/2**20:.0f} MB"


def sha256_file(path: str, chunk: int = 1 << 20) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for b in iter(lambda: f.read(chunk), b""):
            h.update(b)
    return h.hexdigest()


def _hf_token() -> str | None:
    """A Hugging Face token, if the user has one: $HF_TOKEN, else the file `hf auth login`
    writes. Only needed for private or gated repos; public downloads work without it."""
    tok = os.environ.get("HF_TOKEN")
    if tok:
        return tok.strip()
    home = os.environ.get("HF_HOME") or os.path.join(os.path.expanduser("~"), ".cache", "huggingface")
    path = os.path.join(home, "token")
    if os.path.exists(path):
        with open(path) as f:
            return f.read().strip() or None
    return None


def _authorize(req: urllib.request.Request, url: str) -> None:
    """Attach the HF token for huggingface.co URLs only.

    `add_unredirected_header` matters: huggingface.co answers with a redirect to a
    pre-signed CDN URL on another host, and a normal header would be forwarded there
    too. The token must never leave huggingface.co.
    """
    host = urllib.parse.urlparse(url).hostname or ""
    if host == "huggingface.co" or host.endswith(".huggingface.co"):
        tok = _hf_token()
        if tok:
            req.add_unredirected_header("Authorization", f"Bearer {tok}")


def download(url: str, dest: str, expect_size: int | None = None, indent: str = "    ") -> None:
    """Stream `url` to `dest`, resuming a partial file when the server supports ranges.

    Servers that ignore `Range` (some CDNs do) answer 200 instead of 206; the partial
    file is then discarded and the transfer restarts, rather than being appended to --
    appending to a full-body response is how you silently produce a corrupt archive.
    """
    have = os.path.getsize(dest) if os.path.exists(dest) else 0
    if expect_size and have == expect_size:
        print(f"{indent}already complete ({human(have)})")
        return
    req = urllib.request.Request(url, headers={"User-Agent": "snap-downloader"})
    _authorize(req, url)
    if have:
        req.add_header("Range", f"bytes={have}-")
    try:
        r = urllib.request.urlopen(req)
    except urllib.error.HTTPError as e:
        host = urllib.parse.urlparse(url).hostname or ""
        if e.code in (401, 403) and host.endswith("huggingface.co"):
            raise SystemExit(
                f"{indent}HTTP {e.code} from Hugging Face for {url}\n"
                f"{indent}The weights repository is private (or gated) and you are not authorised.\n"
                f"{indent}If you have access, log in once with `hf auth login` (or set HF_TOKEN)\n"
                f"{indent}and re-run; otherwise the weights are not public yet.")
        if e.code == 404:
            raise SystemExit(f"{indent}HTTP 404: nothing at {url}\n"
                             f"{indent}Check `base_url` in checkpoints/manifest.json.")
        raise SystemExit(f"{indent}HTTP {e.code} ({e.reason}) while downloading {url}")
    except urllib.error.URLError as e:
        raise SystemExit(f"{indent}could not reach {url}: {e.reason}")
    with r:
        resuming = (r.status == 206)
        if have and not resuming:
            print(f"{indent}server ignored the resume request; restarting the download")
            have = 0
        elif have:
            print(f"{indent}resuming at {human(have)}")
        total = int(r.headers.get("Content-Length") or 0) + have
        t0, last, got = time.time(), 0.0, have
        with open(dest, "ab" if resuming else "wb") as f:
            while True:
                chunk = r.read(1 << 20)
                if not chunk:
                    break
                f.write(chunk)
                got += len(chunk)
                if time.time() - last > 0.5:
                    last = time.time()
                    pct = f"{100*got/total:5.1f}%" if total else "  ?  "
                    rate = got / max(time.time() - t0, 1e-9) / 2**20
                    sys.stdout.write(f"\r{indent}{pct}  {human(got)}  {rate:5.1f} MB/s")
                    sys.stdout.flush()
    sys.stdout.write("\r" + " " * 64 + "\r")
    size = os.path.getsize(dest)
    if expect_size and size != expect_size:
        raise SystemExit(f"{dest}: got {size} bytes, expected {expect_size}. "
                         f"Delete the file and retry.")
    print(f"{indent}downloaded {human(size)}")
