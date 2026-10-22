"""End-to-end API test: upload a drawing over HTTP and fetch the results."""

from __future__ import annotations

import json
import subprocess
import sys
import time
import urllib.error
import urllib.request
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
for _s in (sys.stdout, sys.stderr):
    try:
        _s.reconfigure(encoding="utf-8", errors="replace")
    except Exception:
        pass

PORT = int(sys.argv[1]) if len(sys.argv) > 1 else 8791
BASE = f"http://127.0.0.1:{PORT}"


def get(path: str):
    with urllib.request.urlopen(BASE + path, timeout=60) as r:
        return r.status, r.read()


def post_multipart(path: str, file_path: Path, fields: dict[str, str]):
    boundary = "----ConstructionAITestBoundary"
    body = b""
    for key, value in fields.items():
        body += f"--{boundary}\r\n".encode()
        body += f'Content-Disposition: form-data; name="{key}"\r\n\r\n'.encode()
        body += f"{value}\r\n".encode()
    body += f"--{boundary}\r\n".encode()
    body += f'Content-Disposition: form-data; name="pdf"; filename="{file_path.name}"\r\n'.encode()
    body += b"Content-Type: application/pdf\r\n\r\n"
    body += file_path.read_bytes()
    body += f"\r\n--{boundary}--\r\n".encode()

    req = urllib.request.Request(
        BASE + path, data=body,
        headers={"Content-Type": f"multipart/form-data; boundary={boundary}"},
    )
    with urllib.request.urlopen(req, timeout=120) as r:
        return r.status, r.read()


def main() -> int:
    sample = ROOT / "output" / "selftest" / "selftest_drawing.pdf"
    if not sample.exists():
        subprocess.run(
            [sys.executable, str(ROOT / "tools" / "make_sample_drawing.py"), "--out", str(sample)],
            check=True, cwd=ROOT,
        )

    from src.config import get_config
    from src.deployment import app

    import threading

    threading.Thread(
        target=lambda: app.run(host="127.0.0.1", port=PORT, debug=False, threaded=True),
        daemon=True,
    ).start()
    time.sleep(2.5)

    failures = 0

    def check(name: str, ok: bool, detail: str = "") -> None:
        nonlocal failures
        if not ok:
            failures += 1
        print(f"  [{'PASS' if ok else 'FAIL'}] {name}{('  -> ' + detail) if detail else ''}")

    # --- static UI ---------------------------------------------------- #
    try:
        status, body = get("/")
        check("GET / serves the UI", status == 200 and b"Construction AI Estimation" in body,
              f"status={status} bytes={len(body)}")
    except urllib.error.HTTPError as e:
        check("GET / serves the UI", False, f"HTTP {e.code}")

    for asset in ("/static/css/app.css", "/static/js/app.js"):
        try:
            status, body = get(asset)
            check(f"GET {asset}", status == 200 and len(body) > 500, f"status={status} bytes={len(body)}")
        except urllib.error.HTTPError as e:
            check(f"GET {asset}", False, f"HTTP {e.code}")

    status, body = get("/health")
    check("GET /health", status == 200 and json.loads(body)["status"] == "ok")

    # --- upload ------------------------------------------------------- #
    try:
        status, body = post_multipart(
            "/api/upload", sample,
            {"project_name": "API Test Block", "market": "gcc"},
        )
        payload = json.loads(body)
        check("POST /api/upload accepted", status == 202 and "job_id" in payload, str(payload)[:120])
        job_id = payload["job_id"]
    except urllib.error.HTTPError as e:
        check("POST /api/upload accepted", False, f"HTTP {e.code}")
        return 1

    # --- poll --------------------------------------------------------- #
    deadline = time.time() + 180
    final = None
    while time.time() < deadline:
        _, body = get(f"/api/job/{job_id}")
        state = json.loads(body)
        if state["status"] in ("done", "error"):
            final = state
            break
        time.sleep(0.7)
    check("job completes", final is not None and final["status"] == "done",
          json.dumps(final)[:200] if final else "timeout")

    # --- results ------------------------------------------------------ #
    status, body = get(f"/api/result/{job_id}")
    data = json.loads(body)
    result = data.get("result") or {}
    check("result payload has a total", result.get("grand_total", 0) > 0,
          f"{result.get('grand_total')}")
    check("result has BOQ lines", len(result.get("boq", [])) >= 3, f"{len(result.get('boq', []))} lines")
    check("result has rebar rows", len(result.get("rebar", [])) >= 3, f"{len(result.get('rebar', []))}")
    check("result has pages", len(data.get("pages", [])) >= 1)
    check("project name echoed", result.get("project_name") == "API Test Block")
    check("market applied", result.get("market") == "gcc", str(result.get("market")))
    check("gcc multiplier applied", result.get("grand_total", 0) > 0)

    # --- downloads ---------------------------------------------------- #
    for fmt in ("csv", "pdf"):
        try:
            status, body = get(f"/api/download/{job_id}/{fmt}")
            check(f"download {fmt.upper()}", status == 200 and len(body) > 500,
                  f"status={status} bytes={len(body)}")
        except urllib.error.HTTPError as e:
            check(f"download {fmt.upper()}", False, f"HTTP {e.code}")

    # --- annotated page ------------------------------------------------ #
    try:
        status, body = get(f"/api/page/{job_id}/0")
        check("annotated page image", status == 200 and body[:2] == b"\xff\xd8",
              f"status={status} bytes={len(body)}")
    except urllib.error.HTTPError as e:
        check("annotated page image", False, f"HTTP {e.code}")

    print(f"\n  {'ALL API CHECKS PASSED' if not failures else f'{failures} CHECK(S) FAILED'}\n")
    return 1 if failures else 0


if __name__ == "__main__":
    sys.exit(main())
