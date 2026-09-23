"""Self-contained end-to-end check of a *running* ATLAS service.

Why not docker compose plus curl: neither is dependable in the target sandbox,
and both hide the process lifecycle. This script owns the whole lifecycle
instead - it starts the ASGI server as a child process, waits for readiness,
drives the real HTTP surface, then reaps the process tree in ``finally``.

Only the standard library is used, so the script runs in any interpreter.

Design notes carried over from hard experience:

* Readiness is polled with a timeout and a hard failure. A silent skip is worse
  than a red result.
* The child is killed with ``taskkill /T /F`` on Windows. Killing only the
  parent leaks the listening socket and the *next* run fails to bind.
* Results are written to a UTF-8 file as well as printed, because a parent
  process may not preserve stdout encoding.
* The ingested probe document is deliberately **longer than the chunk size**, so
  the multi-chunk path is actually exercised rather than assumed.

Author: 晨星
"""

from __future__ import annotations

import json
import os
import socket
import subprocess
import sys
import time
import urllib.error
import urllib.request
from dataclasses import dataclass, field
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
REPORT_PATH = ROOT / "verify_e2e.json"
READY_TIMEOUT_S = 60.0
POLL_INTERVAL_S = 0.4

PROBE_DOC_ID = "e2e-probe-document"
PROBE_DOC = (
    "# 端到端探针文档\n\n"
    "这一段文字专门用来验证摄入链路是否真的进行了分块处理，因此它必须显著长于"
    "默认的分块阈值，否则多分块代码路径根本不会被覆盖到。测试数据短于阈值时"
    "测试会虚假地通过，这是最容易被忽略的验证陷阱之一。\n\n"
    "## 第二小节\n\n"
    "第二个小节继续补充正文，使得整篇文档的长度稳定超过阈值，从而至少产生两个"
    "分块。分块完成之后，稠密索引与稀疏索引都会基于文档仓储重建，二者必须与仓储"
    "保持严格一致，否则检索结果会和实际内容脱节。\n\n"
    "## 第三小节\n\n"
    "第三个段落用于确认标题继承逻辑正确。每个分块都应当携带它所属的标题，这样"
    "召回结果才能附带结构信息，便于下游的推理环节判断证据的上下文归属。\n"
)


@dataclass
class Check:
    name: str
    ok: bool
    detail: str = ""


@dataclass
class Report:
    checks: list[Check] = field(default_factory=list)
    server_stderr: str = ""

    @property
    def failed(self) -> list[Check]:
        return [c for c in self.checks if not c.ok]

    def as_dict(self) -> dict:
        return {
            "passed": len(self.checks) - len(self.failed),
            "failed": len(self.failed),
            "checks": [
                {"name": c.name, "ok": c.ok, "detail": c.detail} for c in self.checks
            ],
            "server_stderr_tail": self.server_stderr[-2000:],
        }


# --------------------------------------------------------------------------
# transport helpers
# --------------------------------------------------------------------------


def _free_port() -> int:
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as sock:
        sock.bind(("127.0.0.1", 0))
        return int(sock.getsockname()[1])


def _request(
    method: str, url: str, payload: dict | None = None, timeout: float = 60.0
) -> tuple[int, str]:
    data = json.dumps(payload).encode("utf-8") if payload is not None else None
    request = urllib.request.Request(url, data=data, method=method)
    if data is not None:
        request.add_header("Content-Type", "application/json")
    # Bypass any system proxy: loopback traffic must never be proxied.
    opener = urllib.request.build_opener(urllib.request.ProxyHandler({}))
    try:
        with opener.open(request, timeout=timeout) as response:
            return response.status, response.read().decode("utf-8", "replace")
    except urllib.error.HTTPError as exc:
        return exc.code, exc.read().decode("utf-8", "replace")


def _json_request(method: str, url: str, payload: dict | None = None) -> tuple[int, dict]:
    status, body = _request(method, url, payload)
    try:
        return status, json.loads(body)
    except json.JSONDecodeError:
        return status, {"_raw": body[:400]}


# --------------------------------------------------------------------------
# process lifecycle
# --------------------------------------------------------------------------


def _spawn(port: int, log_path: Path) -> subprocess.Popen:
    env = dict(os.environ)
    env["PYTHONPATH"] = str(ROOT)
    env["PYTHONIOENCODING"] = "utf-8"
    env["ATLAS_LLM_PROVIDER"] = "mock"
    env["ATLAS_EMBED_PROVIDER"] = "hash"
    env["ATLAS_LOG_LEVEL"] = "WARNING"
    for proxy in ("http_proxy", "https_proxy", "HTTP_PROXY", "HTTPS_PROXY", "ALL_PROXY"):
        env.pop(proxy, None)

    handle = log_path.open("w", encoding="utf-8")
    return subprocess.Popen(
        [
            sys.executable, "-m", "uvicorn",
            "atlas.api.app:app",
            "--host", "127.0.0.1",
            "--port", str(port),
            "--log-level", "warning",
        ],
        cwd=str(ROOT),
        env=env,
        stdout=handle,
        stderr=subprocess.STDOUT,
    )


def _wait_ready(base: str, process: subprocess.Popen, log_path: Path) -> bool:
    deadline = time.time() + READY_TIMEOUT_S
    while time.time() < deadline:
        if process.poll() is not None:
            return False
        try:
            status, _ = _request("GET", f"{base}/health", timeout=3.0)
            if status == 200:
                return True
        except Exception:
            pass
        time.sleep(POLL_INTERVAL_S)
    return False


def _reap(process: subprocess.Popen) -> None:
    if process.poll() is not None:
        return
    if os.name == "nt":
        subprocess.run(
            ["taskkill", "/pid", str(process.pid), "/t", "/f"],
            capture_output=True,
            check=False,
        )
    else:  # pragma: no cover - POSIX path
        process.terminate()
    try:
        process.wait(timeout=15)
    except subprocess.TimeoutExpired:  # pragma: no cover
        process.kill()


# --------------------------------------------------------------------------
# checks
# --------------------------------------------------------------------------


def run_checks(base: str, report: Report) -> None:
    def record(name: str, ok: bool, detail: str = "") -> None:
        report.checks.append(Check(name, ok, detail))

    status, body = _json_request("GET", f"{base}/health")
    record("health returns 200 with a loaded corpus",
           status == 200 and body.get("chunks", 0) > 0,
           f"status={status} chunks={body.get('chunks')}")

    status, body = _json_request("GET", f"{base}/health/deep")
    record("runtime invariants hold (/health/deep)",
           status == 200 and body.get("ok") is True,
           f"status={status} checks={body.get('checks')}")

    status, body = _json_request(
        "POST", f"{base}/v1/query", {"question": "混合检索里为什么不能直接对分数做加权求和？"}
    )
    record("retrieval flow answers with citations",
           status == 200 and body.get("route") == "retrieve"
           and bool(body.get("answer")) and bool(body.get("citations")),
           f"status={status} route={body.get('route')} cites={len(body.get('citations') or [])}")

    status, body = _json_request("POST", f"{base}/v1/query", {"question": "计算 12*(3+4)+18/3 等于多少？"})
    record("arithmetic flow bypasses the model",
           status == 200 and body.get("route") == "arithmetic"
           and body.get("steps") == 0 and "90" in str(body.get("answer")),
           f"status={status} route={body.get('route')} steps={body.get('steps')}")

    status, body = _json_request("POST", f"{base}/v1/query", {"question": "2026-01-01 到 2026-09-24 相差多少天？"})
    record("date flow uses the deterministic tool",
           status == 200 and body.get("route") == "date" and "266" in str(body.get("answer")),
           f"status={status} route={body.get('route')}")

    status, raw = _request("POST", f"{base}/v1/query/stream", {"question": "接地核验为什么按句核验？"})
    record("streaming endpoint emits server-sent events",
           status == 200 and "data:" in raw and "[DONE]" in raw,
           f"status={status} bytes={len(raw)}")

    # Cross-boundary flow: write a document longer than one chunk, then prove it
    # is retrievable through the same public surface.
    status, body = _json_request(
        "POST", f"{base}/v1/ingest", {"documents": {PROBE_DOC_ID: PROBE_DOC}}
    )
    chunks_added = int(body.get("chunks", 0))
    record("ingest accepts a live document and chunks it",
           status == 200 and chunks_added >= 2 and PROBE_DOC_ID in (body.get("doc_ids") or []),
           f"status={status} total_chunks={chunks_added} (must be >= 2 to cover the multi-chunk path)")

    status, body = _json_request(
        "POST", f"{base}/v1/query", {"question": "端到端探针文档里的分块阈值说明了什么？"}
    )
    cites = body.get("citations") or []
    record("newly ingested document is retrievable end to end",
           status == 200 and any(str(c).startswith(PROBE_DOC_ID) for c in cites),
           f"status={status} citations={cites[:3]}")

    status, body = _json_request("POST", f"{base}/v1/eval", {"include_cases": False})
    record("evaluation endpoint meets its floors",
           status == 200 and body.get("route_accuracy") == 1.0
           and body.get("doc_hit_rate", 0) >= 0.90,
           f"status={status} recall={body.get('recall_at_k')} doc_hit={body.get('doc_hit_rate')}")

    status, body = _json_request("POST", f"{base}/v1/guard/probe")
    record("reranking guard preserves top-1 under an adversarial reranker",
           status == 200 and body.get("guarded_preserves_top1") is True
           and body.get("naive_breaks_top1") is True,
           f"status={status} spearman={body.get('spearman')}")

    # --- error flows -----------------------------------------------------
    status, _ = _json_request("POST", f"{base}/v1/query", {"question": ""})
    record("empty question is rejected with 422", status == 422, f"status={status}")

    status, _ = _json_request("POST", f"{base}/v1/query", {"question": "检索", "k": 999})
    record("out-of-range k is rejected with 422", status == 422, f"status={status}")

    status, _ = _request("GET", f"{base}/v1/does-not-exist")
    record("unknown route returns 404", status == 404, f"status={status}")


# --------------------------------------------------------------------------
# entry point
# --------------------------------------------------------------------------


def main() -> int:
    report = Report()
    port = _free_port()
    base = f"http://127.0.0.1:{port}"
    log_path = ROOT / "verify_e2e_server.log"
    process: subprocess.Popen | None = None

    try:
        process = _spawn(port, log_path)
        if not _wait_ready(base, process, log_path):
            tail = log_path.read_text(encoding="utf-8", errors="replace")[-2000:] \
                if log_path.exists() else ""
            report.server_stderr = tail
            report.checks.append(
                Check("server becomes ready", False, f"not ready within {READY_TIMEOUT_S:.0f}s")
            )
        else:
            report.checks.append(Check("server becomes ready", True, base))
            run_checks(base, report)
    except Exception as exc:  # noqa: BLE001 - the harness must always report
        report.checks.append(Check("e2e harness", False, f"{type(exc).__name__}: {exc}"))
    finally:
        if process is not None:
            _reap(process)
        if log_path.exists():
            report.server_stderr = log_path.read_text(encoding="utf-8", errors="replace")

    payload = report.as_dict()
    REPORT_PATH.write_text(json.dumps(payload, ensure_ascii=False, indent=1), encoding="utf-8")

    for check in report.checks:
        mark = "PASS" if check.ok else "FAIL"
        print(f"[{mark}] {check.name}" + (f" :: {check.detail}" if check.detail else ""))
    print(f"e2e summary: passed={payload['passed']} failed={payload['failed']}")
    if payload["failed"]:
        print("--- server log tail ---")
        print(report.server_stderr[-1500:])
    return 1 if payload["failed"] else 0


if __name__ == "__main__":
    raise SystemExit(main())
