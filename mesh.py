"""Neuro: one dashboard for Windows, Linux, and Mac workers.

Workers report memory and can start the local llama.cpp RPC server.
They do not accept shell commands.
"""

from __future__ import annotations

import ast
import ctypes
import hashlib
import hmac
import ipaddress
import json
import os
import platform
import re
import secrets
import shutil
import socket
import stat
import subprocess
import sys
import tempfile
import threading
import time
import unicodedata
import urllib.error
import urllib.request
from collections import deque
from concurrent.futures import ThreadPoolExecutor
from fractions import Fraction
from html.parser import HTMLParser
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path

_agent_home = Path(__file__).resolve().parent.parent / "neuro-agents"
if (_agent_home / "agents.py").is_file():
    sys.path.insert(0, str(_agent_home))
from urllib.parse import parse_qs, urlparse

class _NoRedirect(urllib.request.HTTPRedirectHandler):
    def redirect_request(self, req, fp, code, msg, headers, newurl):
        raise urllib.error.URLError("redirect")


_NO_REDIRECT = urllib.request.build_opener(_NoRedirect)


def file_state(path: Path) -> str:
    try:
        info = path.stat()
    except FileNotFoundError:
        return "missing"
    except OSError:
        return "busy"
    if stat.S_ISREG(info.st_mode) and info.st_size > 0:
        return "file"
    return "missing"


def open_limited(request, timeout: float):
    return _NO_REDIRECT.open(request, timeout=timeout)


ROOT = Path(__file__).resolve().parent
TOKEN_PATH = ROOT / "cluster.token"
NODES_PATH = ROOT / "nodes.json"
SPECS_PATH = ROOT / "specs.json"
PID_PATH = ROOT / "rpc.pid"
DASHBOARD = ROOT / "dashboard.html"
# Third-party files the page loads from this PC (no CDN at runtime). Only the
# names listed here are ever served from vendor/, each with a fixed type.
VENDOR_DIR = ROOT / "vendor"
VENDOR_FILES = {
    "three.module.min.js": "text/javascript; charset=utf-8",
    "LICENSE": "text/plain; charset=utf-8",
}
RPC_PORT = 50052
DASHBOARD_PORT = 8780
WORKER_PORT = 8781
MAX_REPLY = 2 * 1024 * 1024

# Optional, gitignored: where this PC's training run writes its log. See
# training.example.json. Every key is optional; NEURO_* variables win.
TRAINING_CONFIG = ROOT / "training.json"


def training_config(path: Path | None = None) -> dict:
    """Read training.json (small, local, owner-written). Missing or broken -> {}."""
    target = path or TRAINING_CONFIG
    try:
        if target.stat().st_size > 64_000:
            return {}
        data = json.loads(target.read_text(encoding="utf-8-sig"))
    except (OSError, ValueError):
        return {}
    if not isinstance(data, dict):
        return {}
    return {key: str(value).strip() for key, value in data.items()
            if key in {"log", "run_dir", "gpu_lock", "data_root", "name", "process_match"}
            and isinstance(value, str) and value.strip()}


def config_path(env: str, key: str, default: Path) -> Path:
    raw = os.environ.get(env, "").strip() or training_config().get(key, "")
    return Path(raw) if raw else default


def data_root() -> Path:
    """Training data folder. Default: a neuro-ai-data folder next to this repo."""
    return config_path("NEURO_AI_DATA", "data_root", ROOT.parent / "neuro-ai-data")


# The trainer holds this file while it owns the GPU. Default: train_local/gpu.lock
# next to this repo folder.
GPU_LOCK = config_path("NEURO_GPU_LOCK", "gpu_lock", ROOT.parent / "train_local" / "gpu.lock")

# Never scanned and never allowed to talk to the dashboard or the worker:
# link-local, the VirtualBox host-only net, and the 26.x VPN range.
_BLOCKED_NETS = tuple(
    ipaddress.ip_network(net) for net in ("169.254.0.0/16", "192.168.56.0/24", "26.0.0.0/8")
)
_PRIVATE_NETS = tuple(
    ipaddress.ip_network(net) for net in ("10.0.0.0/8", "172.16.0.0/12", "192.168.0.0/16")
)


def private_lan(ip: str) -> bool:
    """True for a home-LAN IPv4 address that may be scanned or bound."""
    try:
        addr = ipaddress.ip_address(str(ip).strip())
    except ValueError:
        return False
    if addr.version != 4:
        return False
    if any(addr in net for net in _BLOCKED_NETS):
        return False
    return any(addr in net for net in _PRIVATE_NETS)


def lan_client(ip: str) -> bool:
    """True when a request may be answered: loopback or the private LAN."""
    raw = str(ip or "").split("%", 1)[0]
    try:
        addr = ipaddress.ip_address(raw)
    except ValueError:
        return False
    if addr.version == 6 and addr.ipv4_mapped:
        addr = addr.ipv4_mapped
    if addr.is_loopback:
        return True
    return private_lan(str(addr))


def browser_host_ok(host_header: str) -> bool:
    """Reject DNS-rebinding names: allow IP literals, localhost and LAN-style names."""
    host = str(host_header or "").strip().lower()
    if not host:
        return True
    if host.startswith("["):
        host = host[1:].split("]", 1)[0]
    elif host.count(":") == 1:
        host = host.split(":", 1)[0]
    try:
        ipaddress.ip_address(host)
        return True
    except ValueError:
        pass
    if host == "localhost" or "." not in host:
        return True
    return host.endswith((".local", ".lan", ".home", ".internal", ".home.arpa", ".localhost"))


def origin_ok(origin: str | None, host_header: str | None) -> bool:
    """A browser POST must come from this page itself, not from another site."""
    if origin is None:
        return True
    value = origin.strip()
    if not value or value == "null":
        return False
    parsed = urlparse(value)
    netloc = (parsed.netloc or "").lower()
    return bool(netloc) and netloc == str(host_header or "").strip().lower()


def machine_key(status: dict) -> tuple[str, str]:
    """One key per real computer. A Mac on two addresses, or a name that
    flips between NAME and NAME.local, is still one computer."""
    name = str(status.get("name") or "").strip().lower()
    name = name.split(".", 1)[0]
    return (name, str(status.get("os") or "").strip().lower())


# "serve" (the main PC) may make cluster.token on first launch. A worker
# ("work") never does: it pairs with the main PC or uses a hand-copied file.
_ROLE = {"name": "serve"}


def token() -> str:
    try:
        TOKEN_PATH.stat()
    except FileNotFoundError:
        if _ROLE["name"] == "work":
            raise RuntimeError("This worker has no cluster.token yet.") from None
        try:
            with TOKEN_PATH.open("x", encoding="utf-8") as handle:
                handle.write(os.urandom(24).hex())
        except FileExistsError:
            pass
    except OSError as exc:
        raise RuntimeError("The cluster token cannot be read.") from exc
    return TOKEN_PATH.read_text(encoding="utf-8").strip()


def token_id() -> str:
    return hashlib.sha256(token().encode("utf-8")).hexdigest()[:8]


_found_lock = threading.Lock()
_found: list[dict] = []
_heavy_cache = {"at": 0.0, "rows": []}


def authorized(header: str | None, secret: str | None = None) -> bool:
    """Constant-time check of an `Authorization: Bearer <cluster.token>`
    header. Compared as bytes: compare_digest refuses non-ASCII str, and a
    header is whatever the client sent."""
    secret = token() if secret is None else secret
    if not secret:
        return False
    expected = ("Bearer " + secret).encode("utf-8")
    got = (header or "").encode("utf-8", errors="replace")
    return hmac.compare_digest(got, expected)


NO_TOKEN_NOTE = (
    "This computer has no cluster.token, so it refuses remote controls. "
    "It pairs with the main PC by itself: approve its code on the main PC's page (or copy cluster.token by hand)."
)


def saved_token() -> str:
    """This computer's cluster.token, or "" when there is none. Unlike
    token() it never makes a new one: a worker must hold the main PC's
    token, and a random one made here would only fail later as a mismatch."""
    if file_state(TOKEN_PATH) == "missing":
        return ""
    try:
        return token()
    except (RuntimeError, OSError, UnicodeDecodeError):
        return ""


# --- proof before the token leaves this PC ---------------------------------
# Any device on the LAN can answer /hello. Before serve sends cluster.token
# to a computer, that computer must prove it already holds the same token:
# serve sends a fresh random nonce, the worker answers
# HMAC-SHA256(token, "neuro-proof:" + nonce + "|" + its own ip:port). The
# address is the local end of the TCP connection as the worker sees it, so a
# rogue device cannot relay a real worker's answer as its own. A good proof
# is cached per URL for a few minutes.
PROOF_SECONDS = 300.0
PROOF_RATE = (30, 60.0)  # answers per client address per window (seconds)
_NONCE = re.compile(r"^[0-9a-f]{32,128}$")
_HEX64_PROOF = re.compile(r"^[0-9a-f]{64}$")
_proof_lock = threading.Lock()
_proofs: dict[str, float] = {}
_proof_hits: dict[str, list[float]] = {}
PROOF_MISSING_NOTE = "That computer has no cluster.token, so this PC did not send its token. Approve its pairing code on this page (or copy cluster.token there by hand)."
PROOF_MISMATCH_NOTE = "That computer has no matching cluster.token, so this PC did not send its token. Restart mesh.py work there and approve its pairing code on this page (or copy cluster.token by hand)."
PROOF_OLD_NOTE = "That computer is running an older worker, so this PC did not send its token. Pull and start it again."


class WorkerProofError(RuntimeError):
    """The worker did not prove it holds this PC's token. No token was sent.
    `answered` is False when the computer did not answer at all."""

    def __init__(self, note: str, answered: bool = True) -> None:
        super().__init__(note)
        self.answered = answered


def proof_digest(secret: str, nonce: str, address: str) -> str:
    message = b"neuro-proof:" + nonce.encode("ascii") + b"|" + address.encode("ascii")
    return hmac.new(secret.encode("utf-8"), message, hashlib.sha256).hexdigest()


def proof_ok(secret: str, nonce: str, address: str, answer: object) -> bool:
    if not secret or not isinstance(answer, str) or not _HEX64_PROOF.match(answer):
        return False
    return hmac.compare_digest(proof_digest(secret, nonce, address).encode("ascii"), answer.encode("ascii"))


def _rate_ok(hits_by_client: dict, client: str, rule: tuple[int, float], now: float | None = None) -> bool:
    """Sliding-window limit: at most rule[0] hits per rule[1] seconds."""
    now = time.time() if now is None else now
    limit, window = rule
    with _proof_lock:
        hits = [at for at in hits_by_client.get(client, []) if now - at < window]
        allowed = len(hits) < limit
        if allowed:
            hits.append(now)
        hits_by_client[client] = hits
        if len(hits_by_client) > 1024:
            for key in [key for key, value in hits_by_client.items() if not value]:
                hits_by_client.pop(key, None)
    return allowed


def proof_rate_ok(client: str, now: float | None = None) -> bool:
    return _rate_ok(_proof_hits, client, PROOF_RATE, now)


def forget_proof(url: str) -> None:
    with _proof_lock:
        _proofs.pop(url, None)


def prove_worker(base: str) -> None:
    """Raise WorkerProofError unless the worker at `base` (a clean_url)
    proves it holds this PC's cluster.token. Never sends the token."""
    with _proof_lock:
        if _proofs.get(base, 0.0) > time.time():
            return
    parsed = urlparse(base)
    try:
        address = f"{socket.gethostbyname(parsed.hostname or '')}:{parsed.port}"
    except OSError as exc:
        raise WorkerProofError("That computer's address could not be resolved, so this PC did not send its token.", answered=False) from exc
    nonce = secrets.token_hex(16)
    request = urllib.request.Request(base + "/hello/proof?nonce=" + nonce)
    try:
        with _DIRECT.open(request, timeout=4) as response:
            body = read_reply(response)
    except urllib.error.HTTPError as exc:
        raise WorkerProofError(PROOF_OLD_NOTE if exc.code == 404 else PROOF_MISMATCH_NOTE) from None
    except (OSError, urllib.error.URLError, ValueError) as exc:
        raise WorkerProofError("That computer did not answer the token check, so this PC did not send its token.", answered=False) from exc
    if not isinstance(body, dict):
        raise WorkerProofError(PROOF_MISMATCH_NOTE)
    if body.get("has_token") is False:
        raise WorkerProofError(PROOF_MISSING_NOTE)
    if not proof_ok(token(), nonce, address, body.get("proof")):
        raise WorkerProofError(PROOF_MISMATCH_NOTE)
    with _proof_lock:
        _proofs[base] = time.time() + PROOF_SECONDS


def worker_call(url: str, path: str, body: dict | None = None, timeout: float = 5.0) -> object:
    """serve -> worker request that carries cluster.token. The token is only
    added here, after the worker proved it holds the same token; the page
    never sees it. POST with a JSON body, or GET when body is None. Raises
    WorkerProofError (nothing sent), HTTPError, URLError, OSError, ValueError."""
    base = clean_url(url)
    prove_worker(base)
    headers = {"Authorization": "Bearer " + token()}
    if body is not None:
        headers["Content-Type"] = "application/json"
    request = urllib.request.Request(
        base + path,
        data=None if body is None else json.dumps(body, ensure_ascii=False).encode("utf-8"),
        method="GET" if body is None else "POST",
        headers=headers,
    )
    try:
        # No proxy and no redirect: the request carries the cluster token.
        with _DIRECT.open(request, timeout=timeout) as response:
            if response.status == 204:
                return {}
            return read_reply(response)
    except urllib.error.HTTPError as exc:
        if exc.code in (401, 403):
            forget_proof(base)
        raise


# --- pairing: a new computer joins without copying cluster.token -------------
# Like Bluetooth or Syncthing approval. The worker makes a random pair_secret
# and a 6-digit code, shows the code on its own screen, and sends serve only
# the code and sha256(pair_secret). The person approves on the main PC's own
# page after checking that the codes match. The worker then fetches the token
# once with the pair_secret, from the same LAN address it asked from.
PAIR_TTL = 900.0
PAIR_MAX_PENDING = 8
PAIR_REQUEST_RATE = (5, 600.0)
PAIR_STATUS_RATE = (120, 60.0)
PAIR_POLL_SECONDS = 4.0
PAIR_RETRY_SECONDS = 120.0
PAIR_DENIED_RETRY_SECONDS = 1800.0
_PAIR_ID = re.compile(r"^[0-9a-f]{16}$")
_PAIR_CODE = re.compile(r"^[0-9]{6}$")
_TOKEN_TEXT = re.compile(r"^[A-Za-z0-9_-]{16,256}$")
_pair_lock = threading.Lock()
_pairs: dict[str, dict] = {}
_pair_hits: dict[str, list[float]] = {}
PAIR_NOTE_UNKNOWN = "No such pairing request. It expired or was already used. The worker starts a new one."


def _pair_prune(now: float) -> None:
    for key in [key for key, item in _pairs.items() if item["expires"] <= now]:
        _pairs.pop(key, None)


def pair_request(client: str, body: dict, now: float | None = None) -> tuple[int, dict]:
    """POST /pair/request from a worker. Stores only sha256(pair_secret)."""
    now = time.time() if now is None else now
    if not _rate_ok(_pair_hits, "request:" + client, PAIR_REQUEST_RATE, now):
        return 429, {"error": "Too many pairing requests from this computer. Wait a few minutes."}
    code = str(body.get("code") or "")
    digest = str(body.get("secret_sha256") or "")
    if not _PAIR_CODE.match(code) or not _HEX64_PROOF.match(digest):
        return 400, {"error": "Send code (6 digits) and secret_sha256 (64 hex characters)."}
    # Shown on the page and in the console: printable characters only.
    name = "".join(ch for ch in clip(body.get("name"), 64) if ch.isprintable()).strip() or "computer at " + client
    key = "".join(ch for ch in clip(body.get("machine_key"), 80) if ch.isprintable()).strip()
    with _pair_lock:
        _pair_prune(now)
        # A new code from the same computer replaces its older request.
        for old in [old for old, item in _pairs.items() if item["ip"] == client]:
            _pairs.pop(old, None)
        if sum(1 for item in _pairs.values() if item["state"] == "pending") >= PAIR_MAX_PENDING:
            return 429, {"error": "Too many computers are waiting. Approve or deny them on the main PC first."}
        pair_id = secrets.token_hex(8)
        _pairs[pair_id] = {
            "id": pair_id, "ip": client, "name": name, "machine_key": key, "code": code,
            "hash": digest, "state": "pending", "created": now, "expires": now + PAIR_TTL,
        }
    return 200, {"id": pair_id, "expires_in": int(PAIR_TTL), "state": "pending"}


def pair_status(client: str, body: dict, now: float | None = None) -> tuple[int, dict]:
    """POST /pair/status. The token leaves only for an approved request, the
    matching pair_secret, and the same client address; then the request is gone."""
    now = time.time() if now is None else now
    if not _rate_ok(_pair_hits, "status:" + client, PAIR_STATUS_RATE, now):
        return 429, {"state": "wait", "error": "Asking too often. Slow down."}
    pair_id = str(body.get("id") or "")
    secret = str(body.get("secret") or "")
    if not _PAIR_ID.match(pair_id) or not _HEX64_PROOF.match(secret):
        return 400, {"state": "bad", "error": "Send id and secret."}
    with _pair_lock:
        item = _pairs.get(pair_id)
        if item is None:
            return 404, {"state": "unknown", "error": PAIR_NOTE_UNKNOWN}
        if item["expires"] <= now:
            _pairs.pop(pair_id, None)
            return 410, {"state": "expired", "error": "This pairing request expired. The worker starts a new one."}
        given = hashlib.sha256(secret.encode("ascii")).hexdigest().encode("ascii")
        if item["ip"] != client or not hmac.compare_digest(given, item["hash"].encode("ascii")):
            return 403, {"state": "refused", "error": "This pairing request belongs to another computer."}
        if item["state"] == "pending":
            return 200, {"state": "pending"}
        _pairs.pop(pair_id, None)
        if item["state"] == "denied":
            return 200, {"state": "denied", "error": "The main PC denied this computer."}
    secret_token = token()
    print(f"Paired {item['name']} at {client}.", flush=True)
    return 200, {"state": "approved", "token": secret_token}


def pair_list(now: float | None = None) -> list[dict]:
    """Pending requests for the main PC's own page (never the hash)."""
    now = time.time() if now is None else now
    with _pair_lock:
        _pair_prune(now)
        rows = [item for item in _pairs.values() if item["state"] == "pending"]
        return [
            {"id": item["id"], "name": item["name"], "ip": item["ip"], "code": item["code"],
             "machine_key": item["machine_key"], "expires_in": int(item["expires"] - now)}
            for item in sorted(rows, key=lambda row: row["created"])
        ]


def pair_decide(pair_id: str, approve: bool, now: float | None = None) -> tuple[int, dict]:
    now = time.time() if now is None else now
    with _pair_lock:
        _pair_prune(now)
        item = _pairs.get(str(pair_id or ""))
        if item is None or item["state"] != "pending":
            return 404, {"error": "That request is gone. It expired or was already answered."}
        item["state"] = "approved" if approve else "denied"
        # Leave the worker time to fetch the answer.
        item["expires"] = max(item["expires"], now + 120.0)
        name = item["name"]
    return 200, {"ok": True, "state": item["state"], "name": name, "pending": pair_list(now)}


def write_token(value: str) -> None:
    """Replace cluster.token atomically; the old one is kept as
    cluster.token.old. Never prints the token."""
    if not isinstance(value, str) or not _TOKEN_TEXT.match(value):
        raise ValueError("The main PC sent a token in an unexpected form.")
    folder = TOKEN_PATH.parent
    temp = folder / f"{TOKEN_PATH.name}.tmp-{secrets.token_hex(4)}"
    try:
        with open(temp, "x", encoding="utf-8") as handle:
            handle.write(value)
            handle.flush()
            os.fsync(handle.fileno())
        if os.name != "nt":
            os.chmod(temp, 0o600)
        if file_state(TOKEN_PATH) == "file":
            old = folder / (TOKEN_PATH.name + ".old")
            shutil.copy2(TOKEN_PATH, old)
        os.replace(temp, TOKEN_PATH)
    finally:
        temp.unlink(missing_ok=True)


_pair_wanted = threading.Event()
_token_changed = threading.Event()
_pair_view: dict = {"code": "", "state": ""}


def want_pairing() -> None:
    _pair_wanted.set()


def _pair_post(base: str, path: str, body: dict, timeout: float = 10.0) -> tuple[int, dict]:
    request = urllib.request.Request(
        base + path,
        data=json.dumps(body).encode("utf-8"),
        method="POST",
        headers={"Content-Type": "application/json"},
    )
    try:
        with _DIRECT.open(request, timeout=timeout) as response:
            reply = read_reply(response)
            return response.status, reply if isinstance(reply, dict) else {}
    except urllib.error.HTTPError as exc:
        try:
            reply = json.loads(exc.read(4096).decode("utf-8", errors="replace"))
        except (OSError, ValueError):
            reply = {}
        return exc.code, reply if isinstance(reply, dict) else {}


def pair_once(base: str, post=_pair_post, sleep=time.sleep, clock=time.time) -> str:
    """One pairing attempt from a worker. Returns "paired", "denied" or
    "retry". The token is written to cluster.token, never printed."""
    secret = secrets.token_hex(32)
    code = f"{secrets.randbelow(1_000_000):06d}"
    key = local_key()
    body = {
        "name": socket.gethostname()[:64],
        "machine_key": "/".join(key),
        "code": code,
        "secret_sha256": hashlib.sha256(secret.encode("ascii")).hexdigest(),
    }
    try:
        status, reply = post(base, "/pair/request", body)
    except (OSError, urllib.error.URLError, ValueError):
        print("Pairing: the main PC did not answer. Trying again later.", flush=True)
        return "retry"
    pair_id = str(reply.get("id") or "")
    if status != 200 or not _PAIR_ID.match(pair_id):
        print("Pairing: " + clip(reply.get("error") or f"the main PC said http {status}", 200), flush=True)
        return "retry"
    _pair_view.update({"code": code, "state": "waiting"})
    print(
        f"\nPairing code: {code[:3]} {code[3:]} - approve this computer on the main PC's Neuro page ({base}).\n"
        "Approve it only if the page shows the same code.\n",
        flush=True,
    )
    deadline = clock() + PAIR_TTL
    try:
        while clock() < deadline:
            sleep(PAIR_POLL_SECONDS)
            try:
                status, reply = post(base, "/pair/status", {"id": pair_id, "secret": secret})
            except (OSError, urllib.error.URLError, ValueError):
                continue
            state = reply.get("state")
            if status == 200 and state == "approved":
                try:
                    write_token(reply.get("token"))
                except (ValueError, OSError) as exc:
                    print(f"Pairing: approved, but cluster.token could not be saved ({type(exc).__name__}).", flush=True)
                    return "retry"
                print(f"Paired with the main PC. token id {token_id()}. Jobs resume now.", flush=True)
                return "paired"
            if state == "denied":
                print("Pairing: the main PC denied this computer.", flush=True)
                return "denied"
            if state in ("expired", "unknown", "refused", "bad"):
                print("Pairing: " + clip(reply.get("error") or str(state), 200), flush=True)
                return "retry"
        print("Pairing: nobody approved the code in 15 minutes. A new code comes later.", flush=True)
        return "retry"
    finally:
        _pair_view.update({"code": "", "state": ""})


def pair_loop(base: str) -> None:
    """Worker thread: pairs whenever want_pairing() was called."""
    while True:
        _pair_wanted.wait()
        result = pair_once(base)
        if result == "paired":
            _pair_wanted.clear()
            _token_changed.set()
            continue
        time.sleep(PAIR_DENIED_RETRY_SECONDS if result == "denied" else PAIR_RETRY_SECONDS)


def worker_refusal(exc: urllib.error.HTTPError) -> str:
    """Plain words for a worker that refused a control request."""
    detail = error_detail(exc)
    if exc.code == 404:
        return "That computer is running an older worker. Pull and start it again."
    if exc.code == 401:
        return (
            "That computer did not accept this PC's cluster.token. Approve its pairing code on this page "
            "(or copy cluster.token by hand, or pull and restart an older worker)."
        )
    if exc.code == 403 and "cluster.token" in detail:
        return "That computer has no cluster.token yet. Approve its pairing code on this page (or copy cluster.token by hand)."
    return detail or f"http {exc.code}"


def ram_mib() -> dict:
    if os.name == "nt":
        class Status(ctypes.Structure):
            _fields_ = [
                ("dwLength", ctypes.c_ulong),
                ("dwMemoryLoad", ctypes.c_ulong),
                ("ullTotalPhys", ctypes.c_ulonglong),
                ("ullAvailPhys", ctypes.c_ulonglong),
                ("ullTotalPageFile", ctypes.c_ulonglong),
                ("ullAvailPageFile", ctypes.c_ulonglong),
                ("ullTotalVirtual", ctypes.c_ulonglong),
                ("ullAvailVirtual", ctypes.c_ulonglong),
                ("ullAvailExtendedVirtual", ctypes.c_ulonglong),
            ]

        status = Status()
        status.dwLength = ctypes.sizeof(Status)
        ctypes.windll.kernel32.GlobalMemoryStatusEx(ctypes.byref(status))
        return {
            "total_mib": int(status.ullTotalPhys // (1024 * 1024)),
            "free_mib": int(status.ullAvailPhys // (1024 * 1024)),
        }
    if sys.platform == "darwin":
        return darwin_ram()
    # Linux: MemAvailable, not MemFree. The kernel keeps "free" memory as page cache (a
    # memory-mapped model file lives there), so MemFree looks near zero on a busy worker even
    # when gigabytes can be reclaimed. That paused every job on a Steam Deck as "low memory".
    try:
        with open("/proc/meminfo", encoding="ascii") as handle:
            info = {}
            for line in handle:
                key, _, rest = line.partition(":")
                parts = rest.split()
                if parts and parts[0].isdigit():
                    info[key] = int(parts[0])
        if "MemTotal" in info and "MemAvailable" in info:
            return {"total_mib": info["MemTotal"] // 1024, "free_mib": info["MemAvailable"] // 1024}
    except (OSError, ValueError):
        pass
    try:
        page = os.sysconf("SC_PAGE_SIZE")
        total = os.sysconf("SC_PHYS_PAGES") * page // (1024 * 1024)
        free = os.sysconf("SC_AVPHYS_PAGES") * page // (1024 * 1024)
        return {"total_mib": int(total), "free_mib": int(free)}
    except (ValueError, OSError, AttributeError):
        return {"total_mib": 0, "free_mib": 0}


def darwin_ram() -> dict:
    total = 0
    try:
        out = subprocess.check_output(["sysctl", "-n", "hw.memsize"], text=True, timeout=3)
        total = int(out.strip()) // (1024 * 1024)
    except (OSError, subprocess.SubprocessError, ValueError):
        total = 0
    free_pages = 0
    page = 4096
    try:
        out = subprocess.check_output(["vm_stat"], text=True, timeout=3)
    except (OSError, subprocess.SubprocessError):
        return {"total_mib": total, "free_mib": 0}
    for line in out.splitlines():
        if "page size of" in line:
            for word in line.replace("(", " ").replace(")", " ").split():
                if word.isdigit():
                    page = int(word)
        for label in ("Pages free", "Pages inactive", "Pages speculative"):
            if line.startswith(label):
                number = line.split(":")[-1].strip().strip(".")
                if number.isdigit():
                    free_pages += int(number)
    return {"total_mib": total, "free_mib": free_pages * page // (1024 * 1024)}


_apple_cache: dict = {"at": 0.0, "cards": []}


def darwin_temperature() -> int | None:
    try:
        out = subprocess.check_output(
            ["osx-cpu-temp"],
            text=True,
            timeout=3,
            stderr=subprocess.DEVNULL,
        )
    except (OSError, subprocess.SubprocessError):
        return None
    number = ""
    for char in out:
        if char.isdigit() or (char == "." and number and "." not in number):
            number += char
        elif number:
            break
    if not number:
        return None
    try:
        return int(float(number))
    except ValueError:
        return None


def apple_chip() -> str:
    try:
        out = subprocess.check_output(["system_profiler", "SPHardwareDataType"], text=True, timeout=15)
    except (OSError, subprocess.SubprocessError):
        return "Apple GPU"
    for line in out.splitlines():
        if "Chip:" in line or "Processor Name:" in line:
            name = line.split(":", 1)[-1].strip()
            if name:
                return name
    return "Apple GPU"


def apple_cards() -> list[dict]:
    now = time.time()
    cached = _apple_cache.get("cards") or []
    if cached and now - float(_apple_cache.get("at") or 0) < 20:
        return cached
    ram = darwin_ram()
    cards = [
        {
            "name": apple_chip(),
            "memory_used_mib": max(0, int(ram.get("total_mib") or 0) - int(ram.get("free_mib") or 0)),
            "memory_total_mib": int(ram.get("total_mib") or 0),
            "temperature_c": darwin_temperature(),
            "unified": True,
        }
    ]
    _apple_cache["at"] = now
    _apple_cache["cards"] = cards
    return cards


def machine_cards() -> list[dict]:
    cards = nvidia()
    if cards:
        return cards
    if sys.platform == "darwin":
        return apple_cards()
    if sys.platform.startswith("linux"):
        return linux_cards()
    return []


# --- AMD / Intel GPUs on Linux, read from sysfs (no extra packages) ---------
GPU_VENDORS = {0x1002: "amd", 0x8086: "intel", 0x10DE: "nvidia"}
STEAM_DECK_GPU_IDS = {0x163F, 0x1435}
DRM_ROOT = Path("/sys/class/drm")
_lspci_cache: dict = {"names": None}


def _sys_text(path: Path) -> str:
    try:
        return path.read_text(encoding="ascii", errors="replace").strip()
    except OSError:
        return ""


def _sys_int(path: Path, base: int = 10) -> int | None:
    text = _sys_text(path)
    if not text:
        return None
    try:
        return int(text, base)
    except ValueError:
        return None


def parse_lspci(out: str) -> dict[tuple[int, int], str]:
    """`lspci -mm -nn` lines -> {(vendor, device): device name}. Only display
    controllers (class 03xx)."""
    names: dict[tuple[int, int], str] = {}
    for line in out.splitlines():
        fields = re.findall(r'"([^"]*)"', line)
        if len(fields) < 3:
            continue
        klass, vendor, device = fields[0], fields[1], fields[2]
        if not re.search(r"\[03[0-9a-f]{2}\]\s*$", klass, re.I):
            continue
        vid = re.search(r"\[([0-9a-f]{4})\]\s*$", vendor, re.I)
        did = re.search(r"\[([0-9a-f]{4})\]\s*$", device, re.I)
        if not vid or not did:
            continue
        name = re.sub(r"\s*\[[0-9a-f]{4}\]\s*$", "", device, flags=re.I).strip()
        if name:
            names[(int(vid.group(1), 16), int(did.group(1), 16))] = clip(name, 60)
    return names


def lspci_names() -> dict[tuple[int, int], str]:
    cached = _lspci_cache.get("names")
    if cached is not None:
        return cached
    names: dict[tuple[int, int], str] = {}
    if shutil.which("lspci"):
        try:
            out = subprocess.check_output(["lspci", "-mm", "-nn"], text=True, timeout=5, stderr=subprocess.DEVNULL)
            names = parse_lspci(out)
        except (OSError, subprocess.SubprocessError):
            names = {}
    _lspci_cache["names"] = names
    return names


def gpu_name(vendor_id: int, device_id: int, names: dict[tuple[int, int], str] | None = None) -> str:
    vendor = GPU_VENDORS.get(vendor_id, "")
    if vendor == "amd" and device_id in STEAM_DECK_GPU_IDS:
        return "AMD Radeon (Steam Deck APU)"
    label = {"amd": "AMD", "intel": "Intel", "nvidia": "NVIDIA"}.get(vendor, "")
    if not label:
        return "GPU"
    found = (names or {}).get((vendor_id, device_id), "")
    return f"{label} {found}" if found else f"{label} GPU"


def linux_cards(root: Path | None = None, names: dict[tuple[int, int], str] | None = None) -> list[dict]:
    """GPUs from /sys/class/drm/card*/device. Read-only. VRAM and GTT come
    from amdgpu's mem_info_* files where present; an APU (shared memory) is
    marked unified and reports VRAM carve-out plus GTT as its memory."""
    root = DRM_ROOT if root is None else root
    try:
        entries = sorted(root.glob("card*"))
    except OSError:
        return []
    cards: list[dict] = []
    seen: set[str] = set()
    for entry in entries:
        if not re.fullmatch(r"card\d+", entry.name):
            continue  # card0-DP-1 and similar are connectors, not GPUs
        device = entry / "device"
        vendor_id = _sys_int(device / "vendor", 16)
        device_id = _sys_int(device / "device", 16)
        if vendor_id is None or device_id is None:
            continue
        try:
            key = str(device.resolve())
        except OSError:
            key = str(device)
        if key in seen:
            continue
        seen.add(key)
        vendor = GPU_VENDORS.get(vendor_id, "other")
        mib = 1024 * 1024
        vram_total = (_sys_int(device / "mem_info_vram_total") or 0) // mib
        vram_used = (_sys_int(device / "mem_info_vram_used") or 0) // mib
        gtt_total = (_sys_int(device / "mem_info_gtt_total") or 0) // mib
        gtt_used = (_sys_int(device / "mem_info_gtt_used") or 0) // mib
        unified = (vendor == "amd" and (device_id in STEAM_DECK_GPU_IDS or (0 < vram_total <= 2048 and gtt_total > 0))) or (
            vendor == "intel" and not vram_total
        )
        temperature = None
        try:
            for sensor in sorted((device / "hwmon").glob("hwmon*/temp1_input")):
                value = _sys_int(sensor)
                if value is not None:
                    temperature = value // 1000
                    break
        except OSError:
            temperature = None
        card = {
            "name": gpu_name(vendor_id, device_id, lspci_names() if names is None else names),
            "vendor": vendor,
            "pci_id": f"{vendor_id:04x}:{device_id:04x}",
            "memory_used_mib": vram_used + (gtt_used if unified else 0),
            "memory_total_mib": vram_total + (gtt_total if unified else 0),
            "vram_mib": vram_total,
            "gtt_mib": gtt_total,
            "temperature_c": temperature,
        }
        if unified:
            card["unified"] = True
        cards.append(card)
    return cards


def nvidia() -> list[dict]:
    try:
        out = subprocess.check_output(
            [
                "nvidia-smi",
                "--query-gpu=name,memory.used,memory.total,temperature.gpu",
                "--format=csv,noheader,nounits",
            ],
            text=True,
            timeout=8,
            stderr=subprocess.DEVNULL,
        )
    except (OSError, subprocess.SubprocessError):
        return []
    return parse_nvidia(out)


def _number(text: str) -> int | None:
    try:
        return int(float(text))
    except (TypeError, ValueError):
        return None


def parse_nvidia(out: str) -> list[dict]:
    # A laptop GPU in low power can print "[N/A]" for a field. That must not
    # take the whole status reply down, or the machine vanishes from the page.
    cards = []
    for line in out.splitlines():
        parts = [part.strip() for part in line.split(",")]
        if len(parts) < 4 or not parts[0]:
            continue
        cards.append(
            {
                "name": parts[0],
                "memory_used_mib": _number(parts[1]) or 0,
                "memory_total_mib": _number(parts[2]) or 0,
                "temperature_c": _number(parts[3]),
            }
        )
    return cards


# GPU telemetry history for this PC (serve only): one fixed nvidia-smi query
# every 15 s into a 24 h ring buffer, plus, on Windows, the shared (system)
# memory the GPU spills into, read once a minute from a performance counter.
TELEMETRY_INTERVAL_S = 15
TELEMETRY_WINDOW_S = 24 * 3600
TELEMETRY_SHARED_EVERY_S = 60
TELEMETRY_POINTS = 400
TELEMETRY_ARGS = [
    "nvidia-smi",
    "--query-gpu=index,name,utilization.gpu,power.draw,power.limit,temperature.gpu,memory.used,memory.total",
    "--format=csv,noheader,nounits",
]
SHARED_COUNTER_ARGS = [
    "typeperf",
    r"\GPU Adapter Memory(*)\Dedicated Usage",
    r"\GPU Adapter Memory(*)\Shared Usage",
    "-sc",
    "1",
]


def _float_or_none(text: str) -> float | None:
    try:
        value = float(str(text).strip())
    except (TypeError, ValueError):
        return None  # "[N/A]", "[Not Supported]", ""
    return value if value == value and abs(value) != float("inf") else None


def parse_nvidia_telemetry(out: str) -> list[dict]:
    """One dict per GPU line of TELEMETRY_ARGS. A field a GPU cannot report
    ("[N/A]") becomes None instead of dropping the line."""
    gpus = []
    for line in str(out or "").splitlines():
        parts = [part.strip() for part in line.split(",")]
        if len(parts) < 8:
            continue
        index = _float_or_none(parts[0])
        if index is None:
            continue
        gpus.append({
            "index": int(index),
            "name": clip(parts[1], 80),
            "util": _float_or_none(parts[2]),
            "power_w": _float_or_none(parts[3]),
            "power_limit_w": _float_or_none(parts[4]),
            "temp_c": _float_or_none(parts[5]),
            "mem_used_mib": _float_or_none(parts[6]),
            "mem_total_mib": _float_or_none(parts[7]),
        })
    return gpus[:8]


def parse_shared_counters(out: str) -> float | None:
    """typeperf CSV with Dedicated Usage and Shared Usage columns per GPU
    adapter (bytes). Returns the shared MiB of the adapter that holds the
    most dedicated memory (the discrete GPU), or None."""
    rows = [line for line in str(out or "").splitlines() if line.startswith('"')]
    if len(rows) < 2:
        return None
    header = [cell.strip('"') for cell in rows[0].split('","')]
    values = [cell.strip('"') for cell in rows[-1].split('","')]
    dedicated: dict[str, float] = {}
    shared: dict[str, float] = {}
    for name, raw in zip(header[1:], values[1:]):
        match = re.search(r"GPU Adapter Memory\((.*?)\)\\(Dedicated|Shared) Usage", name)
        number = _float_or_none(raw)
        if not match or number is None:
            continue
        (dedicated if match.group(2) == "Dedicated" else shared)[match.group(1)] = number
    if not dedicated or not shared:
        return None
    adapter = max(dedicated, key=lambda key: dedicated[key])
    if adapter not in shared:
        return None
    return round(shared[adapter] / (1024 * 1024), 1)


def _run_quiet(run, argv: list[str], timeout: float) -> str:
    result = run(argv, capture_output=True, text=True, timeout=timeout, errors="replace",
                 creationflags=getattr(subprocess, "CREATE_NO_WINDOW", 0))
    return str(getattr(result, "stdout", "") or "")


def sample_telemetry(run=subprocess.run) -> tuple[list[dict], str]:
    """(gpus, error). The command is fixed; nothing from a request reaches it."""
    try:
        gpus = parse_nvidia_telemetry(_run_quiet(run, TELEMETRY_ARGS, 5))
    except FileNotFoundError:
        return [], "nvidia-smi was not found on this PC."
    except (OSError, subprocess.SubprocessError, ValueError) as exc:
        return [], f"nvidia-smi failed: {type(exc).__name__}"
    return gpus, "" if gpus else "nvidia-smi reported no GPU."


def sample_shared(run=subprocess.run) -> float | None:
    if os.name != "nt":
        return None
    try:
        return parse_shared_counters(_run_quiet(run, SHARED_COUNTER_ARGS, 8))
    except (OSError, subprocess.SubprocessError, ValueError):
        return None


def downsample_rows(rows: list, limit: int) -> list:
    """At most `limit` rows [x, a, b, ...]: each bucket becomes its column
    means (None ignored; x rounded), the newest row stays exact."""
    if len(rows) <= limit:
        return [list(row) for row in rows]
    size = len(rows) / (limit - 1)
    out = []
    for index in range(limit - 1):
        bucket = rows[int(index * size): max(int((index + 1) * size), int(index * size) + 1)] or [rows[-2]]
        merged = [round(sum(row[0] for row in bucket) / len(bucket))]
        for column in range(1, len(bucket[0])):
            seen = [row[column] for row in bucket if row[column] is not None]
            merged.append(round(sum(seen) / len(seen), 4) if seen else None)
        out.append(merged)
    out.append(list(rows[-1]))
    return out


class Telemetry:
    """24 h ring buffer of GPU samples. Thread-safe."""

    def __init__(self, interval_s: int = TELEMETRY_INTERVAL_S, window_s: int = TELEMETRY_WINDOW_S):
        self.interval_s = interval_s
        self.window_s = window_s
        self.samples: deque = deque(maxlen=window_s // interval_s + 8)
        self.error = ""
        self.lock = threading.Lock()

    def add(self, at: float, gpus: list[dict], shared_mib: float | None = None, error: str = "") -> None:
        with self.lock:
            self.error = error
            if gpus:
                self.samples.append((float(at), [dict(gpu) for gpu in gpus], shared_mib))
            while self.samples and self.samples[0][0] < at - self.window_s:
                self.samples.popleft()

    def view(self, window_s: int = TELEMETRY_WINDOW_S, limit: int = TELEMETRY_POINTS, now: float | None = None) -> dict:
        window_s = max(300, min(int(window_s or self.window_s), self.window_s))
        with self.lock:
            samples = list(self.samples)
            error = self.error
        end = now if now is not None else (samples[-1][0] if samples else time.time())
        samples = [item for item in samples if item[0] >= end - window_s]
        gpus: dict[int, dict] = {}
        for at, cards, shared in samples:
            for card in cards:
                entry = gpus.setdefault(card["index"], {"index": card["index"], "name": card.get("name") or "",
                                                        "mem_total_mib": None, "power_limit_w": None, "rows": []})
                entry["name"] = card.get("name") or entry["name"]
                entry["mem_total_mib"] = card.get("mem_total_mib") or entry["mem_total_mib"]
                entry["power_limit_w"] = card.get("power_limit_w") or entry["power_limit_w"]
                entry["rows"].append([int(at), card.get("util"), card.get("power_w"), card.get("temp_c"),
                                      card.get("mem_used_mib"), shared if card["index"] == 0 else None])
        out = []
        for index in sorted(gpus):
            entry = gpus[index]
            rows = entry.pop("rows")
            entry["points"] = downsample_rows(rows, limit)
            out.append(entry)
        return {
            "interval_s": self.interval_s,
            "window_s": window_s,
            "available": bool(out),
            "error": error,
            "count": len(samples),
            "columns": ["t", "util_pct", "power_w", "temp_c", "mem_used_mib", "shared_mib"],
            "gpus": out,
        }


TELEMETRY = Telemetry()


def telemetry_tick(store: Telemetry, run=subprocess.run, now: float | None = None,
                   state: dict | None = None) -> None:
    """One sample. The shared-memory counter is read at most once a minute."""
    now = time.time() if now is None else now
    state = state if state is not None else {}
    gpus, error = sample_telemetry(run)
    if gpus and now - float(state.get("shared_at") or 0) >= TELEMETRY_SHARED_EVERY_S:
        state["shared_at"] = now
        state["shared"] = sample_shared(run)
    store.add(now, gpus, state.get("shared") if gpus else None, error)


def telemetry_loop(store: Telemetry, run=subprocess.run) -> None:
    state: dict = {}
    while True:
        try:
            telemetry_tick(store, run, state=state)
        except Exception as exc:  # the page keeps the samples it has
            print(f"telemetry failed: {type(exc).__name__}: {exc}", flush=True)
        time.sleep(TELEMETRY_INTERVAL_S)


# Stopping these turns the computer off or closes the login session.
PROTECTED = {
    "system",
    "registry",
    "secure system",
    "memory compression",
    "smss.exe",
    "csrss.exe",
    "wininit.exe",
    "services.exe",
    "lsass.exe",
    "winlogon.exe",
    "fontdrvhost.exe",
    "svchost.exe",
    "dwm.exe",
    "explorer.exe",
    "sihost.exe",
    "taskhostw.exe",
    "spoolsv.exe",
    "lsaiso.exe",
    "msmpeng.exe",
    "audiodg.exe",
    # Console hosts: stopping one also ends the training run or this page
    # that was started inside it.
    "conhost.exe",
    "openconsole.exe",
    "windowsterminal.exe",
    "cmd.exe",
    "powershell.exe",
    "pwsh.exe",
    "kernel_task",
    "launchd",
    "windowserver",
    "loginwindow",
    "finder",
    "dock",
    "systemuiserver",
    "terminal",
    "iterm2",
    "systemd",
    "xorg",
    "gnome-shell",
    "gnome-terminal-server",
    "sshd",
    # Desktop sessions: stopping one ends the login, or the window the worker runs in.
    "searchhost.exe",
    "startmenuexperiencehost.exe",
    "shellexperiencehost.exe",
    "textinputhost.exe",
    "runtimebroker.exe",
    "ctfmon.exe",
    "wudfhost.exe",
    "controlcenter",
    "coreaudiod",
    "plasmashell",
    "kwin_wayland",
    "kwin_x11",
    "kwin_wayland_wrapper",
    "xwayland",
    "ksmserver",
    "kded5",
    "kded6",
    "konsole",
    "gamescope",
    "gamescope-wl",
    "steamos-session",
    "gnome-session-binary",
    "mutter",
    "xfwm4",
    "xfce4-session",
    "sddm",
    "gdm",
    "lightdm",
    "pipewire",
    "wireplumber",
    "pulseaudio",
    "dbus-daemon",
    "xdg-desktop-portal",
    "networkmanager",
    "polkitd",
    "systemd-logind",
}


def process_running(image: str) -> bool:
    if os.name == "nt" and not image.lower().endswith(".exe"):
        return False
    try:
        if os.name == "nt":
            out = subprocess.check_output(
                ["tasklist", "/FI", f"IMAGENAME eq {image}", "/NH"],
                text=True,
                timeout=8,
                stderr=subprocess.DEVNULL,
            )
            return image.lower() in out.lower()
        out = subprocess.check_output(["ps", "-ax", "-o", "comm="], text=True, timeout=8)
    except (OSError, subprocess.SubprocessError):
        return False
    return image in out


def port_open(port: int) -> bool:
    try:
        with socket.create_connection(("127.0.0.1", port), 0.4):
            return True
    except OSError:
        return False


def rpc_binary() -> str | None:
    override = os.environ.get("NEURO_RPC_BIN", "").strip()
    if override:
        state = file_state(Path(override))
        if state == "busy":
            return None
        if state == "file":
            return override
    names = ["ggml-rpc-server.exe", "ggml-rpc-server"]
    candidates = [
        ROOT.parent / "tools" / "llama-cpp-cuda" / "ggml-rpc-server.exe",
        ROOT / "ggml-rpc-server.exe",
        ROOT / "ggml-rpc-server",
    ]
    for candidate in candidates:
        state = file_state(candidate)
        if state == "busy":
            return None
        if state == "file":
            return str(candidate)
    for name in names:
        found = shutil.which(name)
        if found:
            return found
    return None


def heavy_processes() -> list[dict]:
    now = time.time()
    if now - _heavy_cache["at"] < 4:
        return list(_heavy_cache["rows"])
    rows: list[dict] = []
    kept = keep_pids()
    # While the GPU lock is held (or cannot be read) the training run is a
    # Python process whose PID may not be readable, so no Python process on
    # this computer gets a Stop button.
    guard_python = lock_held(GPU_LOCK)
    try:
        if os.name == "nt":
            out = subprocess.check_output(["tasklist", "/FO", "CSV", "/NH"], text=True, timeout=8, stderr=subprocess.DEVNULL, errors="replace")
            for line in out.splitlines():
                parts = [part.strip().strip('"') for part in line.split('","')]
                if len(parts) < 5:
                    continue
                pid_text = parts[1].strip()
                mem = re.sub(r"[^0-9]", "", parts[-1])
                if not pid_text.isdigit() or not mem.isdigit():
                    continue
                rows_add(rows, parts[0].strip('"'), int(pid_text), int(mem) // 1024, kept, guard_python)
        else:
            out = subprocess.check_output(["ps", "-axo", "pid=,rss=,comm="], text=True, timeout=4, errors="replace")
            for line in out.splitlines():
                bits = line.strip().split(None, 2)
                if len(bits) < 3 or not bits[0].isdigit() or not bits[1].isdigit():
                    continue
                name = bits[2].replace("\\", "/").split("/")[-1][:40]
                rows_add(rows, name, int(bits[0]), int(bits[1]) // 1024, kept, guard_python)
    except (OSError, subprocess.SubprocessError, ValueError):
        return []
    rows.sort(key=lambda item: item["mb"], reverse=True)
    kept = rows[:40]
    _heavy_cache["at"] = time.time()
    _heavy_cache["rows"] = kept
    return list(kept)


def process_role(name: str, pid: int, kept: set[int], guard_python: bool) -> tuple[str, bool]:
    """Return (role, stoppable) for one process row. Pure, so it can be tested."""
    lowered = name.lower()
    if pid == os.getpid():
        return "page", False
    if pid in kept:
        return "training", False
    if lowered.startswith("llama-server"):
        return "model", True
    if "rpc-server" in lowered:
        return "share", True
    if lowered.startswith("python"):
        return "python", not guard_python and lowered not in PROTECTED
    return "other", lowered not in PROTECTED


def rows_add(rows: list[dict], name: str, pid: int, mb: int, kept: set[int], guard_python: bool) -> None:
    if mb < 12 or pid <= 0 or not name:
        return
    role, stoppable = process_role(name, pid, kept, guard_python)
    rows.append({"name": name, "pid": pid, "mb": mb, "role": role, "stoppable": stoppable})


def keep_pids() -> set[int]:
    pids = {os.getpid()}
    try:
        text = GPU_LOCK.read_text(encoding="utf-8", errors="replace").strip()
    except OSError:
        text = ""
    if text.isdigit():
        pids.add(int(text))
    else:
        # The lock may hold JSON such as {"pid": 1234}.
        try:
            data = json.loads(text) if text.startswith("{") else {}
        except ValueError:
            data = {}
        pid = data.get("pid") if isinstance(data, dict) else None
        if isinstance(pid, int) and pid > 0:
            pids.add(pid)
    return pids


def lock_held(lock: Path) -> bool:
    try:
        lock.stat()
    except FileNotFoundError:
        return False
    except OSError:
        return True
    return True


def dataset_index() -> list[dict]:
    root = data_root() / "training"
    if lock_held(GPU_LOCK):
        return [{"name": "held while training runs", "mb": 0}]
    rows: list[dict] = []
    try:
        for path in root.rglob("*.jsonl"):
            try:
                size = path.stat().st_size
                name = path.relative_to(root).as_posix()
            except (OSError, ValueError):
                continue
            rows.append({"name": name, "mb": round(size / (1024 * 1024), 1)})
            if len(rows) >= 16:
                break
    except OSError:
        # A folder that cannot be read ends the walk; what was listed stays.
        return rows
    return rows


_training_cache: dict = {"at": 0.0, "view": None}
# A log that changed within this many seconds counts as a live run.
TRAINING_FRESH_S = 300
LOG_HEAD_BYTES = 16_000   # the runner's first lines (start time, run folder)
LOG_TAIL_BYTES = 96_000   # the newest progress lines; the file itself keeps growing


def newest_log(log_dir: Path, pattern: str = "") -> tuple[Path | None, float, bool]:
    """Return (newest *.log whose name starts with `pattern`, its mtime,
    readable). readable is False when the folder exists but cannot be
    listed, so the caller can say "unknown" instead of "stopped"."""
    newest: Path | None = None
    newest_at = 0.0
    try:
        with os.scandir(log_dir) as entries:
            for entry in entries:
                if not (entry.name.startswith(pattern) and entry.name.endswith(".log")):
                    continue
                try:
                    at = entry.stat().st_mtime
                except OSError:
                    continue
                if newest is None or at > newest_at:
                    newest, newest_at = Path(entry.path), at
    except FileNotFoundError:
        return None, 0.0, True
    except OSError:
        return None, 0.0, False
    return newest, newest_at, True


def configured_log() -> tuple[Path | None, float, bool] | None:
    """The log named by NEURO_TRAINING_LOG or training.json "log", as
    (path or None, mtime, readable). None when neither is set."""
    raw = os.environ.get("NEURO_TRAINING_LOG", "").strip() or training_config().get("log", "")
    if not raw:
        return None
    path = Path(raw)
    try:
        return path, path.stat().st_mtime, True
    except FileNotFoundError:
        return None, 0.0, True
    except OSError:
        return None, 0.0, False


def _decode_log(raw: bytes) -> str:
    if raw[:2] in (b"\xff\xfe", b"\xfe\xff") or b"\x00" in raw[:80]:
        return raw.decode("utf-16-le", errors="replace").lstrip("﻿")
    return raw.decode("utf-8", errors="replace").lstrip("﻿")


def read_log(path: Path, head_bytes: int = LOG_HEAD_BYTES, tail_bytes: int = LOG_TAIL_BYTES) -> tuple[str, str]:
    """(head, tail) of a growing log. Only these two slices are read, never
    the whole file. A small file comes back whole as the tail."""
    try:
        with path.open("rb") as handle:
            size = os.fstat(handle.fileno()).st_size
            if size <= head_bytes + tail_bytes:
                return "", _decode_log(handle.read(head_bytes + tail_bytes))
            head = handle.read(head_bytes)
            start = size - tail_bytes
            if start % 2:
                start += 1  # keeps UTF-16 aligned; one lost byte of UTF-8 is harmless
            handle.seek(start)
            return _decode_log(head), _decode_log(handle.read(tail_bytes))
    except OSError:
        return "", ""


_RUNNER = re.compile(r"^(\d{4}-\d{2}-\d{2}T\d{2}:\d{2}:\d{2})\S*\s+(.*)$")
# tqdm: "Loading weights:  2%|▏   | 14/760 [00:47<24:46,  1.99s/it]" or a bare "  4500/6933 [..]".
_BAR = re.compile(
    r"(?:(?P<pct>\d{1,3})%\|[^|]*\|)?\s*(?P<n>\d+)/(?P<total>\d+)\s*\["
    r"(?P<elapsed>[0-9:]+)?(?:<(?P<eta>[0-9:?]+))?(?:,\s*(?P<rate>[0-9.]+)\s*(?P<unit>s/it|it/s))?"
)
_NUM = r"(-?[0-9]+(?:\.[0-9]+)?(?:[eE][-+]?[0-9]+)?)"
_LOSS = re.compile(r"[\"']loss[\"']\s*:\s*[\"']?" + _NUM)
# "eval_loss", or "eval_all_loss" when the trainer also reports per-domain eval sets.
_EVAL_LOSS = re.compile(r"[\"']eval_(?:all_)?loss[\"']\s*:\s*[\"']?" + _NUM)
_EPOCH = re.compile(r"[\"']epoch[\"']\s*:\s*[\"']?" + _NUM)
_LR = re.compile(r"[\"']learning_rate[\"']\s*:\s*[\"']?" + _NUM)
_EXC = re.compile(r"^([A-Za-z_][\w.]*(?:Error|Exception|Interrupt|Exit))\b:?(.*)$")
_LOADING_DESC = re.compile(r"load|shard|weight|checkpoint|map|token|generat|split|download|pack", re.I)


def clock_seconds(text: str | None) -> int | None:
    """"20:31:00" -> 73860, "10:05" -> 605. "?" or junk -> None."""
    if not text or "?" in text:
        return None
    try:
        parts = [int(part) for part in text.split(":")]
    except ValueError:
        return None
    seconds = 0
    for part in parts:
        seconds = seconds * 60 + part
    return seconds


def _num(pattern: re.Pattern, line: str) -> float | None:
    found = pattern.search(line)
    if not found:
        return None
    try:
        return float(found.group(1))
    except ValueError:
        return None


def parse_training_log(head: str, tail: str = "", max_steps_hint: int = 0) -> dict:
    """Read a Hugging Face Trainer log (tqdm bars split by \\r, loss dicts,
    eval dicts, timestamped runner lines, tracebacks). Pure, so it is tested."""
    info: dict = {
        "step": None, "max_steps": None, "loss": None, "history": [], "eval_loss": None, "eval_history": [],
        "epoch": None, "learning_rate": None, "s_per_it": None, "eta_s": None, "elapsed_s": None,
        "phase": None, "load": None, "run_dir": "", "started": "", "last_event": "", "last_event_at": "",
        "error": "", "finished": False,
    }
    bars: list[tuple[int, str, re.Match]] = []
    events: list[tuple[int, str, object]] = []
    segments = [seg for seg in re.split(r"[\r\n]+", head + "\n" + tail) if seg.strip()]
    for index, segment in enumerate(segments):
        line = segment.strip()
        runner = _RUNNER.match(line)
        if runner:
            stamp, message = runner.group(1), runner.group(2).strip()
            if not info["started"]:
                info["started"] = stamp
            if "Run folder:" in message:
                info["run_dir"] = message.split("Run folder:", 1)[1].strip()[:300]
            events.append((index, "runner", (stamp, message)))
            continue
        matches = list(_BAR.finditer(segment))
        if matches:
            match = matches[-1]
            desc = segment[: match.start()].strip().rstrip(":").strip()
            if match.group("pct") is not None or match.group("elapsed") is not None or not desc:
                bars.append((index, desc, match))
        if "Traceback (most recent call last)" in line:
            events.append((index, "traceback", None))
        exc = _EXC.match(line)
        if exc:
            events.append((index, "exception", line[:240]))
        if "'train_runtime'" in line or '"train_runtime"' in line:
            events.append((index, "finished", None))
        loss = _num(_LOSS, line)
        if loss is not None:
            events.append((index, "loss", (loss, _num(_EPOCH, line), _num(_LR, line))))
        eval_loss = _num(_EVAL_LOSS, line)
        if eval_loss is not None:
            events.append((index, "eval", (eval_loss, _num(_EPOCH, line))))
    # A staged runner writes every stage into one log. Only the newest run counts: an older stage
    # with more steps made the new stage's train bar look like an eval bar (stuck on "evaluating").
    folders = [i for i, kind, data in events if kind == "runner" and "Run folder:" in data[1]]
    if folders:
        boundary = folders[-1]
        previous = folders[-2] if len(folders) > 1 else -1
        bars = [bar for bar in bars if bar[0] > boundary]
        # Runner lines of this stage (mix, probe, start) stay; numbers come only after its Run folder line.
        events = [event for event in events
                  if event[0] >= boundary or (event[1] == "runner" and event[0] > previous)]
        info["started"] = next(data[0] for _, kind, data in events if kind == "runner")
    # The train bar has no description and the largest total (or the known
    # max_steps); an eval bar shares the log with a smaller total.
    plain = [(i, m) for i, desc, m in bars if not desc]
    totals = [int(m.group("total")) for _, m in plain if int(m.group("total")) > 0]
    top = max(totals, default=0)
    train_total = max_steps_hint if max_steps_hint and (max_steps_hint in totals or top < max_steps_hint // 10) else top
    for index, desc, match in bars:
        total = int(match.group("total"))
        if total <= 0:
            continue
        if desc and _LOADING_DESC.search(desc):
            events.append((index, "loadbar", (desc[:60], int(match.group("n")), total)))
        elif not desc and total == train_total:
            events.append((index, "train", match))
        elif not desc:
            events.append((index, "evalbar", match))
    events.sort(key=lambda item: item[0])
    in_traceback = False
    for _, kind, data in events:
        if kind == "runner":
            stamp, message = data
            info["last_event"], info["last_event_at"] = message[:200], stamp
            lowered = message.lower()
            if lowered.startswith("start") or "preflight" in lowered:
                info.update(error="", finished=False, phase="loading")
                in_traceback = False
            elif re.search(r"\b(failed|crashed|exited with)\b", lowered):
                info["error"] = message[:240]
            elif re.search(r"\b(finished|training complete)\b", lowered):
                info.update(finished=True, phase=None)
        elif kind == "traceback":
            in_traceback = True
            info["error"] = info["error"] or "Traceback in the log"
        elif kind == "exception" and in_traceback:
            info["error"] = data
        elif kind == "finished":
            info.update(finished=True, phase=None)
        elif kind == "loadbar":
            label, n, total = data
            info.update(phase="loading", load={"label": label, "step": n, "total": total}, error="", finished=False)
            in_traceback = False
        elif kind in {"train", "evalbar"}:
            match = data
            in_traceback = False
            info["error"] = ""
            info["finished"] = False
            if kind == "evalbar":
                info["phase"] = "evaluating"
                continue
            n, total = int(match.group("n")), int(match.group("total"))
            info.update(step=n, max_steps=total, phase="training", load=None)
            rate = match.group("rate")
            if rate:
                try:
                    value = float(rate)
                    info["s_per_it"] = round(value if match.group("unit") == "s/it" else (1 / value if value else 0), 3) or None
                except ValueError:
                    pass
            info["elapsed_s"] = clock_seconds(match.group("elapsed"))
            eta = clock_seconds(match.group("eta"))
            if eta is None and info["s_per_it"]:
                eta = int((total - n) * info["s_per_it"])
            info["eta_s"] = eta
        elif kind == "loss":
            loss, epoch, rate = data
            info["history"].append(round(loss, 4))
            info["loss"] = round(loss, 4)
            if epoch is not None:
                info["epoch"] = round(epoch, 3)
            if rate is not None:
                info["learning_rate"] = rate
        elif kind == "eval":
            eval_loss, epoch = data
            info["eval_history"].append(round(eval_loss, 4))
            info["eval_loss"] = round(eval_loss, 4)
            if info["phase"] == "evaluating":
                info["phase"] = "training"
    if info["step"] is not None and info["max_steps"] and info["step"] >= info["max_steps"] and not info["error"]:
        info["finished"] = info["finished"] or info["phase"] == "training"
    info["history"] = info["history"][-40:]
    info["eval_history"] = info["eval_history"][-20:]
    return info


# Full-run loss curve. The log is read incrementally: the cache remembers the
# file, the byte offset already parsed and the points found so far, so each
# call reads only the bytes added since the last one.
LOSS_POINTS_CAP = 5000
LOSS_SERIES_POINTS = 400
LOSS_READ_CHUNK = 4_000_000
LOSS_READ_BUDGET = 32_000_000
_STEPS_PER_EPOCH = re.compile(r"[\"']optimizer_steps_per_epoch[\"']\s*:\s*" + _NUM)
_GRAD_NORM = re.compile(r"[\"']grad_norm[\"']\s*:\s*[\"']?" + _NUM)
# Per-domain eval losses: 'eval_cpp_loss', 'eval_extra/luau_loss', 'eval_all_loss' (the overall one).
_EVAL_DOMAIN = re.compile(r"[\"']eval_([A-Za-z0-9_./-]{1,60}?)_loss[\"']\s*:\s*[\"']?" + _NUM)
EVAL_DOMAINS_CAP = 24
_loss_cache: dict = {}
_loss_lock = threading.Lock()


def _loss_state(path: str, fingerprint: bytes = b"") -> dict:
    return {"path": path, "offset": 0, "encoding": None, "fingerprint": fingerprint, "loss": [], "eval": [],
            "grad": [], "lr": [], "domains": {}, "speed": [], "speed_step": None,
            "bar_step": None, "bar_total": 0, "steps_per_epoch": None}


def _new_run(state: dict) -> None:
    state.update(loss=[], eval=[], grad=[], lr=[], domains={}, speed=[], speed_step=None,
                 bar_step=None, bar_total=0, steps_per_epoch=None)


def _keep(points: list, row: list, cap: int) -> None:
    points.append(row)
    if len(points) > cap:
        _thin(points, cap)


def _last_break(raw: bytes, encoding: str) -> int:
    """Index just past the last line break in raw, or 0 when there is none."""
    if encoding == "utf-16-le":
        best = -1
        for mark in (b"\n\x00", b"\r\x00"):
            at = raw.rfind(mark)
            while at >= 0 and at % 2:
                at = raw.rfind(mark, 0, at)
            best = max(best, at)
        return best + 2 if best >= 0 else 0
    best = max(raw.rfind(b"\n"), raw.rfind(b"\r"))
    return best + 1 if best >= 0 else 0


def _thin(points: list, cap: int) -> None:
    """Halve a point list in place (keeping the first and the newest point) until it fits the cap."""
    while len(points) > cap:
        points[:] = points[:-1:2] + [points[-1]]


def _parse_loss_text(state: dict, text: str, max_steps_hint: int = 0, cap: int = LOSS_POINTS_CAP) -> None:
    for segment in re.split(r"[\r\n]+", text):
        line = segment.strip()
        if not line:
            continue
        runner = _RUNNER.match(line)
        if runner:
            message = runner.group(2).strip()
            if message.lower().startswith("start") or "Run folder:" in message:
                # A new run (or a new stage of a staged runner) in the same log: its curve starts over.
                _new_run(state)
            continue
        spe = _num(_STEPS_PER_EPOCH, line)
        if spe and spe > 0:
            state["steps_per_epoch"] = spe
        matches = list(_BAR.finditer(segment))
        if matches:
            match = matches[-1]
            if not segment[: match.start()].strip():
                total = int(match.group("total"))
                if total > 0 and (total == max_steps_hint or total >= state["bar_total"]):
                    state["bar_total"] = max(state["bar_total"], total)
                    step = int(match.group("n"))
                    state["bar_step"] = step
                    # One speed sample per new train step: [step, s/it, elapsed s].
                    rate = match.group("rate")
                    if rate and step != state["speed_step"]:
                        try:
                            value = float(rate)
                            seconds = value if match.group("unit") == "s/it" else (1 / value if value else 0)
                        except ValueError:
                            seconds = 0
                        if seconds > 0:
                            state["speed_step"] = step
                            _keep(state["speed"], [step, round(seconds, 3), clock_seconds(match.group("elapsed"))], cap)
        epoch = None
        loss = _num(_LOSS, line)
        if loss is not None:
            epoch = _num(_EPOCH, line)
            _keep(state["loss"], [epoch, state["bar_step"], round(loss, 4)], cap)
        grad = _num(_GRAD_NORM, line)
        if grad is not None:
            epoch = _num(_EPOCH, line) if epoch is None else epoch
            _keep(state["grad"], [epoch, state["bar_step"], round(grad, 4)], cap)
        rate_value = _num(_LR, line)
        if rate_value is not None:
            epoch = _num(_EPOCH, line) if epoch is None else epoch
            _keep(state["lr"], [epoch, state["bar_step"], float(f"{rate_value:.4g}")], cap)
        eval_loss = _num(_EVAL_LOSS, line)
        if eval_loss is not None:
            _keep(state["eval"], [_num(_EPOCH, line), state["bar_step"], round(eval_loss, 4)], cap)
        if "eval_" in line:
            for found in _EVAL_DOMAIN.finditer(line):
                name = found.group(1)
                if name == "all":
                    continue  # eval_all_loss is the overall curve above
                try:
                    value = float(found.group(2))
                except ValueError:
                    continue
                domains = state["domains"]
                if name not in domains and len(domains) >= EVAL_DOMAINS_CAP:
                    continue
                _keep(domains.setdefault(name, []), [_num(_EPOCH, line), state["bar_step"], round(value, 4)], cap)


def update_loss_cache(path: Path, max_steps_hint: int = 0, cache: dict | None = None,
                      cap: int = LOSS_POINTS_CAP, budget: int = LOSS_READ_BUDGET) -> dict:
    """Parse only the bytes added to `path` since the last call. Starts over
    when the path changes, the file shrank, or its first bytes changed."""
    cache = _loss_cache if cache is None else cache
    key = str(path)
    try:
        with path.open("rb") as handle:
            size = os.fstat(handle.fileno()).st_size
            fingerprint = handle.read(64)
            state = cache.get("state")
            if (not state or state["path"] != key or size < state["offset"]
                    or fingerprint[: len(state["fingerprint"])] != state["fingerprint"]):
                state = _loss_state(key, fingerprint)
                cache["state"] = state
            if len(state["fingerprint"]) < 64 and len(fingerprint) > len(state["fingerprint"]):
                state["fingerprint"] = fingerprint
            read = 0
            while state["offset"] < size and read < budget:
                handle.seek(state["offset"])
                raw = handle.read(min(LOSS_READ_CHUNK, size - state["offset"]))
                if not raw:
                    break
                if state["encoding"] is None:
                    state["encoding"] = "utf-16-le" if raw[:2] in (b"\xff\xfe", b"\xfe\xff") or b"\x00" in raw[:80] else "utf-8"
                cut = _last_break(raw, state["encoding"])
                if not cut:
                    if len(raw) < LOSS_READ_CHUNK:
                        break  # an unfinished line: wait for its end
                    cut = len(raw) - (len(raw) % 2 if state["encoding"] == "utf-16-le" else 0)
                text = raw[:cut].decode(state["encoding"], errors="replace").lstrip("﻿")
                _parse_loss_text(state, text, max_steps_hint, cap)
                state["offset"] += cut
                read += cut
            return state
    except OSError:
        return cache.get("state") or _loss_state(key)


def downsample(points: list, limit: int = LOSS_SERIES_POINTS, sig: int = 0) -> list:
    """At most `limit` [x, y] points: each bucket becomes its mean, the newest point stays exact.
    y is rounded to 4 decimals, or to `sig` significant digits (tiny values like a learning rate)."""
    if len(points) <= limit:
        return [list(point) for point in points]
    size = len(points) / (limit - 1)
    out = []
    for index in range(limit - 1):
        bucket = points[int(index * size): max(int((index + 1) * size), int(index * size) + 1)]
        bucket = bucket or [points[-2]]
        mean = sum(p[1] for p in bucket) / len(bucket)
        out.append([round(sum(p[0] for p in bucket) / len(bucket)), float(f"{mean:.{sig}g}") if sig else round(mean, 4)])
    out.append(list(points[-1]))
    return out


def _series_steps(points: list, steps_per_epoch: float | None, max_steps: int, fallback_spe: float | None = None) -> list:
    out = []
    for index, (epoch, bar_step, value) in enumerate(points):
        if epoch is not None and steps_per_epoch:
            x = round(epoch * steps_per_epoch)
        elif bar_step is not None:
            x = bar_step
        elif epoch is not None and fallback_spe:
            x = round(epoch * fallback_spe)
        else:
            x = index + 1
        if max_steps:
            x = min(x, max_steps)
        out.append([x, value])
    return out


def loss_series(state: dict, max_steps: int = 0, step: int | None = None, epoch: float | None = None,
                limit: int = LOSS_SERIES_POINTS) -> dict:
    """Downsampled [[step, loss], ...] and [[step, eval_loss], ...] for the page.
    Epochs become steps with the log's optimizer_steps_per_epoch, else with the
    training bar step seen next to each loss, else with step / epoch now."""
    spe = state.get("steps_per_epoch")
    pairs = [(e, s) for e, s, _ in state.get("loss", []) + state.get("eval", []) if e and s]
    if pairs:
        fallback = pairs[-1][1] / pairs[-1][0]
    elif step and epoch:
        fallback = step / epoch
    else:
        fallback = None
    steps = lambda key: _series_steps(state.get(key) or [], spe, max_steps, fallback)  # noqa: E731
    domains = state.get("domains") or {}
    speed = []
    for row in state.get("speed") or []:
        x = min(row[0], max_steps) if max_steps else row[0]
        speed.append([x, row[1], row[2]])
    return {"loss_series": downsample(steps("loss"), limit),
            "eval_series": downsample(steps("eval"), limit),
            "grad_norm_series": downsample(steps("grad"), limit),
            "lr_series": downsample(steps("lr"), limit, sig=4),
            "eval_domain_series": {name: downsample(_series_steps(domains[name], spe, max_steps, fallback), limit)
                                   for name in sorted(domains)[:EVAL_DOMAINS_CAP]},
            # [[step, seconds per step, elapsed seconds or None], ...] from the tqdm bar
            "step_time_series": downsample_rows(speed, limit)}


EMPTY_SERIES = {"loss_series": [], "eval_series": [], "grad_norm_series": [], "lr_series": [],
                "eval_domain_series": {}, "step_time_series": []}


def full_loss_series(log: str, max_steps: int = 0, step: int | None = None, epoch: float | None = None) -> dict:
    if not log:
        return {key: (dict(value) if isinstance(value, dict) else list(value)) for key, value in EMPTY_SERIES.items()}
    with _loss_lock:
        state = update_loss_cache(Path(log), max_steps)
        return loss_series(state, max_steps, step, epoch)


def parse_log_tail(tail: str) -> dict:
    """Older, smaller view of parse_training_log: step, max_steps, loss."""
    info = parse_training_log("", tail)
    return {"step": info["step"], "max_steps": info["max_steps"], "loss": info["loss"]}


def run_state(run_dir: Path) -> dict:
    """Step, max_steps and loss history from the newest trainer_state.json in
    a run folder (the folder itself, or its newest checkpoint-N)."""
    candidates: list[tuple[int, Path]] = []
    try:
        if (run_dir / "trainer_state.json").is_file():
            candidates.append((1 << 30, run_dir / "trainer_state.json"))
        with os.scandir(run_dir) as entries:
            for entry in entries:
                match = re.fullmatch(r"checkpoint-(\d+)", entry.name)
                if match and entry.is_dir():
                    candidates.append((int(match.group(1)), Path(entry.path) / "trainer_state.json"))
    except OSError:
        return {}
    for _, path in sorted(candidates, reverse=True):
        try:
            if path.stat().st_size > 5_000_000:
                continue
            state = json.loads(path.read_text(encoding="utf-8"))
        except (OSError, ValueError):
            continue
        if not isinstance(state, dict):
            continue
        history = []
        for row in state.get("log_history") or []:
            if isinstance(row, dict) and isinstance(row.get("loss"), (int, float)):
                history.append(round(float(row["loss"]), 4))
        try:
            return {"step": int(state.get("global_step") or 0), "max_steps": int(state.get("max_steps") or 0),
                    "history": history[-40:], "checkpoint": path.parent.name}
        except (TypeError, ValueError):
            continue
    return {}


TRAINER_PATTERN = re.compile(r"ml[\\/]+training[\\/]+train(?:_next)?\.py", re.I)


def trainer_matches(command: str) -> bool:
    if TRAINER_PATTERN.search(command):
        return True
    extra = training_config().get("process_match", "")
    return bool(extra) and extra.lower() in command.lower()


def trainer_alive() -> bool | None:
    """True if a trainer process exists on this machine, False if none, None
    if the process list (or a Python command line in it) could not be read:
    a trainer started as another user or as SYSTEM hides its command line."""
    try:
        if os.name == "nt":
            out = subprocess.check_output(
                ["powershell", "-NoProfile", "-NonInteractive", "-Command",
                 "Get-CimInstance Win32_Process -Filter \"Name like 'python%'\" | "
                 "ForEach-Object { 'P ' + $_.CommandLine }"],
                text=True, timeout=10, errors="replace", creationflags=getattr(subprocess, "CREATE_NO_WINDOW", 0),
            )
        else:
            out = subprocess.check_output(["ps", "-axo", "command="], text=True, timeout=5, errors="replace")
    except (OSError, subprocess.SubprocessError):
        return None
    lines = out.splitlines()
    if any(trainer_matches(line) for line in lines):
        return True
    if os.name == "nt" and any(line.strip() == "P" for line in lines):
        return None
    return False


def live_training(data_root: Path, found: tuple[Path | None, float, bool] | None = None,
                  max_steps_hint: int = 0) -> dict:
    """Live state of the run from its log and the GPU lock. `found` is the
    configured log; without one the newest log in <data root>/training/logs."""
    held = lock_held(GPU_LOCK)
    newest, newest_at, readable = found if found is not None else newest_log(data_root / "training" / "logs")
    info = parse_training_log("", "", max_steps_hint)
    age = None
    if newest is not None:
        head, tail = read_log(newest)
        info = parse_training_log(head, tail, max_steps_hint)
        age = max(0, int(time.time() - newest_at))
    fresh = age is not None and age < TRAINING_FRESH_S
    running: bool | None
    stale_lock = False
    quiet = False
    if newest is not None and info["error"]:
        running, state = False, "error"
    elif newest is not None and info["finished"]:
        running, state = False, "finished"
    elif fresh:
        running = True
        state = "loading" if info["phase"] == "loading" or info["step"] is None else "running"
    elif held:
        if trainer_alive() is False:
            # The lock is held but no trainer process exists: the run stopped or crashed and
            # left its lock behind. The card stays reserved (lock_held stays True everywhere
            # else), but the page must not claim training is going.
            running, state, stale_lock = False, "stopped_lock_held", True
        else:
            running, quiet = True, newest is not None
            state = "loading" if newest is not None and info["phase"] == "loading" else "running"
    elif newest is None:
        # No log and a free lock. A folder that could not be read is not proof
        # the run stopped, so the page shows "unknown".
        running = False if readable else None
        state = "stopped" if readable else "unknown"
    else:
        alive = trainer_alive()
        running, quiet = alive is True, alive is True
        state = "running" if alive else "stopped"
    return {**info, "running": running, "state": state, "log_readable": readable, "stale_lock": stale_lock,
            "quiet": quiet, "log": str(newest) if newest else "", "updated_s": age}


def _ago_words(seconds: int | None) -> str:
    if seconds is None:
        return "an unknown time"
    if seconds < 90:
        return f"{seconds} s"
    if seconds < 5400:
        return f"{round(seconds / 60)} min"
    if seconds < 172800:
        return f"{round(seconds / 3600)} h"
    return f"{round(seconds / 86400)} days"


def training_view() -> dict:
    now = time.time()
    cached = _training_cache.get("view")
    if cached and now - float(_training_cache.get("at") or 0) < 10:
        return cached
    config = training_config()
    root = data_root()
    versions_path = root / "versions.json"
    active = ""
    if file_state(versions_path) == "busy":
        active = "The version list could not be read."
    elif file_state(versions_path) == "file":
        try:
            if versions_path.stat().st_size > 1_000_000:
                raise OSError
            versions = json.loads(versions_path.read_text(encoding="utf-8"))
            active = str(versions.get("activePeft") or "")[:80] if isinstance(versions, dict) else ""
        except (OSError, ValueError):
            active = "The version list could not be read."
    found = configured_log()
    run_dir_raw = os.environ.get("NEURO_TRAINING_RUN_DIR", "").strip() or config.get("run_dir", "")
    saved = run_state(Path(run_dir_raw)) if run_dir_raw else {}
    live = live_training(root, found, int(saved.get("max_steps") or 0))
    if not run_dir_raw and live.get("run_dir"):
        run_dir_raw = live["run_dir"]
        saved = run_state(Path(run_dir_raw))
    step = live["step"] if live["step"] is not None else saved.get("step") or 0
    max_steps = live["max_steps"] or saved.get("max_steps") or 0
    history = live["history"] if len(live["history"]) >= 2 else (saved.get("history") or live["history"])
    loss = live["loss"] if live["loss"] is not None else (history[-1] if history else None)
    name = config.get("name") or (Path(run_dir_raw).name if run_dir_raw else "")
    state = live["state"]
    if state == "stopped_lock_held":
        reason = ("No trainer process is running, but the GPU lock file is still there: the run stopped "
                  "or crashed. Check the lock and remove it by hand. Nothing was restarted.")
    elif state == "error":
        reason = "The log ends with an error (below). Nothing was restarted."
    elif state == "finished":
        reason = "The run finished. Its checkpoints are in the run folder."
    elif state == "loading":
        reason = "Loading the model and the data. Steps start after that."
    elif state == "running" and live["quiet"]:
        reason = (f"The trainer is still there, but the log has not changed for {_ago_words(live['updated_s'])} "
                  "(saving a checkpoint, or stuck).")
    elif state == "running":
        reason = "Training on this PC. One run is not split across computers."
    elif state == "unknown":
        reason = ("The training log cannot be read by this page, so the state is unknown. "
                  "Point NEURO_TRAINING_LOG or training.json at a log this page can read.")
    elif not live["log"]:
        reason = ("No training log found. To show a run here, copy training.example.json to training.json "
                  "next to mesh.py and set \"log\" to the run's log file, or set NEURO_TRAINING_LOG.")
    else:
        reason = f"Not training. The log last changed {_ago_words(live['updated_s'])} ago."
    view = {
        # True, False, or None when the log folder cannot be read.
        "running": live["running"] if live["running"] is None else bool(live["running"]),
        "state": state,
        "phase": live["phase"],
        "stale_lock": bool(live["stale_lock"]),
        "log_readable": bool(live["log_readable"]),
        "configured": found is not None,
        "run": name[:80],
        "run_dir": run_dir_raw[:300],
        "log": live["log"][:300],
        "updated_s": live["updated_s"],
        "quiet": bool(live["quiet"]),
        "step": step,
        "max_steps": max_steps,
        "loss": loss,
        "history": history,
        "eval_loss": live["eval_loss"],
        "eval_history": live["eval_history"],
        **full_loss_series(live["log"], int(max_steps or 0), step, live["epoch"]),
        "epoch": live["epoch"],
        "learning_rate": live["learning_rate"],
        "s_per_it": live["s_per_it"],
        "eta_s": live["eta_s"] if state in {"running", "loading"} else None,
        "elapsed_s": live["elapsed_s"],
        "load": live["load"] if state == "loading" else None,
        "started": live["started"],
        "last_event": live["last_event"],
        "error": str(live["error"])[:240] if state == "error" else "",
        "checkpoint": saved.get("checkpoint") or "",
        "served": "Neuro",
        "active_adapter": active,
        "reason": reason,
        "datasets": dataset_index(),
    }
    _training_cache["at"] = now
    _training_cache["view"] = view
    return view


def luau_binary() -> str | None:
    override = os.environ.get("NEURO_LUAU_BIN", "").strip()
    if override:
        state = file_state(Path(override))
        if state == "busy":
            return None
        if state == "file":
            return override
    for candidate in (
        ROOT.parent / "tools" / "luau" / "luau.exe",
        ROOT / "tools" / "luau.exe",
        ROOT / "tools" / "luau",
        ROOT / "luau.exe",
        ROOT / "luau",
    ):
        state = file_state(candidate)
        if state == "busy":
            return None
        if state == "file":
            return str(candidate)
    found = shutil.which("luau")
    if found:
        return found
    # Where start-worker.ps1, platforms/windows-worker.ps1 and
    # platforms/steamdeck.sh put the official release binary.
    home = user_luau_path()
    return str(home) if home and file_state(home) == "file" else None


def user_luau_path() -> Path | None:
    """%LOCALAPPDATA%\\Neuro\\luau\\luau.exe on Windows,
    ~/.local/share/neuro/luau/luau elsewhere."""
    if os.name == "nt":
        base = os.environ.get("LOCALAPPDATA", "").strip()
        return Path(base) / "Neuro" / "luau" / "luau.exe" if base else None
    try:
        return Path.home() / ".local" / "share" / "neuro" / "luau" / "luau"
    except (RuntimeError, KeyError):
        return None


_cpu_sample: dict = {"at": 0.0, "idle": 0, "total": 0, "pct": None}
_cpu_lock = threading.Lock()


def cpu_busy_pct() -> int | None:
    """Whole-machine CPU use in percent, or None when it is not known yet."""
    if os.name == "nt":
        idle, kernel, user = ctypes.c_ulonglong(), ctypes.c_ulonglong(), ctypes.c_ulonglong()
        try:
            ok = ctypes.windll.kernel32.GetSystemTimes(ctypes.byref(idle), ctypes.byref(kernel), ctypes.byref(user))
        except (AttributeError, OSError):
            return None
        if not ok:
            return None
        # The worker's job slots ask from several threads at once.
        with _cpu_lock:
            now = time.time()
            if now - float(_cpu_sample["at"]) < 1.0:
                return _cpu_sample["pct"]
            # Kernel time includes idle time on Windows.
            total = kernel.value + user.value
            now_idle = idle.value
            last_total, last_idle = int(_cpu_sample["total"]), int(_cpu_sample["idle"])
            _cpu_sample.update({"at": now, "total": total, "idle": now_idle})
            span = total - last_total
            if last_total and span > 0:
                _cpu_sample["pct"] = max(0, min(100, round(100 * (1 - (now_idle - last_idle) / span))))
            return _cpu_sample["pct"]
    try:
        load = os.getloadavg()[0]
    except (OSError, AttributeError):
        return None
    cores = os.cpu_count() or 1
    return max(0, min(100, round(100 * load / cores)))


def is_cuda_card(card: dict) -> bool:
    """An NVIDIA card with its own memory. Older workers send no "vendor";
    their non-unified cards always came from nvidia-smi."""
    return not card.get("unified") and str(card.get("vendor") or "nvidia") == "nvidia"


_model_cache: dict = {"at": 0.0, "up": False}


def local_model_cached(ttl: float = 15.0) -> bool:
    """local_model_ready() at most every `ttl` seconds: /hello is asked often."""
    now = time.time()
    if now - float(_model_cache["at"]) < ttl:
        return bool(_model_cache["up"])
    up = local_model_ready()
    _model_cache.update({"at": now, "up": up})
    return up


def pick_accelerator(cards: list[dict], metal: bool, model_up: bool) -> str:
    if any(is_cuda_card(card) for card in cards):
        return "cuda"
    if metal:
        return "metal"
    # llama.cpp on AMD/Intel under Linux runs on Vulkan. Only claimed while a
    # local llama-server is really up.
    if model_up and any(str(card.get("vendor") or "") in {"amd", "intel"} for card in cards):
        return "vulkan"
    return "cpu"


def capabilities(cards: list[dict], ram: dict, chat: bool, rpc_ready: bool) -> dict:
    """What this computer can do for side work. Small and read-only."""
    busy = cpu_busy_pct()
    free_mib = int(ram.get("free_mib") or 0)
    gpu_busy = lock_held(GPU_LOCK) if os.name == "nt" else False
    cuda = [card for card in cards if is_cuda_card(card)]
    metal = sys.platform == "darwin" and platform.machine() == "arm64"
    model_up = local_model_cached()
    vram_total = max((int(card.get("memory_total_mib") or 0) for card in cuda), default=0)
    vram_free = max(
        (max(0, int(card.get("memory_total_mib") or 0) - int(card.get("memory_used_mib") or 0)) for card in cuda),
        default=0,
    )
    return {
        "version": 1,
        "cpu_cores": os.cpu_count() or 0,
        "cpu_busy_pct": busy,
        "free_ram_mib": free_mib,
        "accelerator": pick_accelerator(cards, metal, bool(chat or model_up)),
        # The local llama-server that `generate` jobs use (NEURO_LOCAL_LLM,
        # default 127.0.0.1:8080). False means this computer takes no
        # generate jobs.
        "local_model": model_up,
        "vram_total_mib": vram_total,
        "vram_free_mib": vram_free,
        "rpc_server": rpc_ready,
        "luau": bool(luau_binary()),
        "python": platform.python_version(),
        "gpu_lock": gpu_busy,
        "idle": bool(not chat and not gpu_busy and (busy is None or busy < 35) and free_mib >= 2048),
        # Data-factory job kinds this worker really runs. Empty unless the
        # job thread of `mesh.py work` is on.
        "job_kinds": worker_kinds() if _job_state.get("on") else [],
        # Whether and how this worker runs model-written tests (verify_tests,
        # cross_tests). Network access is not blocked for them.
        "tests": ({"langs": sorted(sandbox_status()["langs"]), "note": sandbox_status()["note"]}
                  if _job_state.get("on") else None),
        # done/failed/paused/current as before, plus "running" (every kind
        # the job slots run now) and "slots" (busy/total per slot sort).
        "jobs": job_state_view(),
    }


def local_status() -> dict:
    cards = machine_cards()
    binary = rpc_binary()
    chat = process_running("llama-server.exe") or process_running("llama-server")
    ram = ram_mib()
    return {
        "name": socket.gethostname(),
        "os": platform.system().lower(),
        "machine": platform.machine(),
        "ram": ram,
        "gpus": cards,
        "heavy": heavy_processes(),
        "chat_server": chat,
        "rpc_binary": binary,
        "rpc_listening": port_open(RPC_PORT),
        "temperature_c": cards[0]["temperature_c"] if cards else None,
        "caps": capabilities(cards, ram, chat, bool(binary)),
    }


# The discovery thread and the page's request threads both read and rewrite
# nodes.json and specs.json. One lock plus an atomic replace keeps a reader
# from seeing a half-written file, reading it as empty, and then writing the
# empty list back over every saved computer.
_files_lock = threading.RLock()


def read_json_list(path: Path, strict: bool = False) -> list[dict]:
    """Read a saved list. With strict=True an unreadable or damaged file
    raises RuntimeError, so a read-modify-write never replaces it with []."""
    with _files_lock:
        try:
            raw = path.read_text(encoding="utf-8")
        except FileNotFoundError:
            return []
        except (OSError, ValueError) as exc:
            if strict:
                raise RuntimeError(path.name + " cannot be read. Nothing was changed.") from exc
            return []
    try:
        data = json.loads(raw)
    except ValueError as exc:
        if strict:
            raise RuntimeError(path.name + " is damaged. Nothing was changed.") from exc
        return []
    if not isinstance(data, list):
        if strict:
            raise RuntimeError(path.name + " is not a list. Nothing was changed.")
        return []
    return [item for item in data if isinstance(item, dict)]


def write_json_list(path: Path, rows: list[dict]) -> None:
    raw = json.dumps(rows, ensure_ascii=False, indent=2)
    with _files_lock:
        temp = path.with_name(path.name + ".tmp")
        temp.write_text(raw, encoding="utf-8")
        for attempt in range(5):
            try:
                os.replace(temp, path)
                return
            except PermissionError:
                # Windows refuses a replace while another program has the file open.
                time.sleep(0.05 * (attempt + 1))
        os.replace(temp, path)


def read_nodes(strict: bool = False) -> list[dict]:
    return [item for item in read_json_list(NODES_PATH, strict) if isinstance(item.get("url"), str) and item.get("url")]


def write_nodes(nodes: list[dict]) -> None:
    write_json_list(NODES_PATH, nodes)


def read_specs(strict: bool = False) -> list[dict]:
    return read_json_list(SPECS_PATH, strict)


def write_specs(specs: list[dict]) -> None:
    write_json_list(SPECS_PATH, specs)


def add_node(url: str) -> None:
    with _files_lock:
        nodes = read_nodes(strict=True)
        if not any(node["url"] == url for node in nodes):
            nodes.append({"url": url})
            write_nodes(nodes)


def clip(value: object, limit: int = 80) -> str:
    return str(value or "").strip()[:limit]


def save_spec(raw: dict) -> dict:
    system = clip(raw.get("os"), 16)
    if system not in {"windows", "mac", "linux"}:
        raise ValueError("Choose windows, mac, or linux")
    name = clip(raw.get("name"), 40)
    if not name:
        raise ValueError("Name the computer")
    kind = clip(raw.get("kind"), 16)
    if kind not in {"laptop", "mini", "desktop", "penguin"}:
        kind = {"windows": "laptop", "mac": "mini", "linux": "penguin"}[system]
    url = clip(raw.get("url"), 200)
    item = {
        "os": system,
        "kind": kind,
        "name": name,
        "url": clean_url(url) if url else "",
        "cpu": clip(raw.get("cpu")),
        "ram_gb": clip(raw.get("ram_gb"), 16),
        "gpu": clip(raw.get("gpu")),
        "vram_gb": clip(raw.get("vram_gb"), 16),
        "role": "agent-check",
    }
    with _files_lock:
        specs = [spec for spec in read_specs(strict=True) if spec.get("name") != name]
        if len(specs) >= 6:
            raise ValueError("Six computers are already saved")
        specs.append(item)
        write_specs(specs)
        if item["url"]:
            add_node(item["url"])
    return item


def clean_url(value: str) -> str:
    parsed = urlparse(value.strip())
    if parsed.scheme not in {"http", "https"} or not parsed.hostname or not parsed.port:
        raise ValueError("Address must look like http://192.168.1.20:8781")
    return f"{parsed.scheme}://{parsed.hostname}:{parsed.port}"


def fetch_worker(url: str) -> dict:
    body = worker_call(url, "/status", None, 4)
    if not isinstance(body, dict):
        raise ValueError("Bad worker response")
    return body


def read_reply(response) -> object:
    raw = response.read(MAX_REPLY + 1)
    if len(raw) > MAX_REPLY:
        raise ValueError("Worker reply is too large")
    return json.loads(raw.decode("utf-8"))


def fetch_public(url: str, path: str) -> dict:
    request = urllib.request.Request(url.rstrip("/") + path)
    with open_limited(request, 4) as response:
        body = read_reply(response)
    if not isinstance(body, dict):
        raise ValueError("Bad worker response")
    return body


def pull_legacy(url: str) -> dict:
    try:
        return {"url": url, "ok": True, "status": fetch_worker(url)}
    except WorkerProofError as exc:
        if not exc.answered:
            return {"url": url, "ok": False, "error": "no reply"}
        # It answered but did not prove the token: shown by address only,
        # and this PC's token was not sent to it.
        host = urlparse(url).hostname or url
        return {"url": url, "ok": True, "limited": True, "status": {"name": "worker at " + host, "os": "", "ram": {}}}
    except urllib.error.HTTPError as exc:
        if exc.code == 401:
            # An old worker with its own token. It is running, but it says
            # nothing about itself, so it is named by its address, not
            # guessed as "Mac".
            host = urlparse(url).hostname or url
            return {
                "url": url,
                "ok": True,
                "limited": True,
                "status": {"name": "worker at " + host, "os": "", "ram": {}},
            }
        return {"url": url, "ok": False, "error": f"http {exc.code}"}
    except (OSError, urllib.error.URLError, ValueError):
        return {"url": url, "ok": False, "error": "no reply"}


def pull_worker(url: str, legacy: bool = True) -> dict:
    """Ask one worker for /hello. legacy=False (used by the LAN sweep) never
    falls back to /status, so this PC's token is only ever sent to a saved
    address, not to any device that happens to answer on port 8781."""
    try:
        status = fetch_public(url, "/hello")
    except urllib.error.HTTPError as exc:
        if exc.code == 404 and legacy:
            return pull_legacy(url)
        return {"url": url, "ok": False, "error": f"http {exc.code}"}
    except (OSError, urllib.error.URLError, ValueError):
        return {"url": url, "ok": False, "error": "no reply"}
    if status.get("app") != "neuroai-mesh":
        return {"url": url, "ok": False, "error": "not a Neuro worker"}
    status["name"] = clip(status.get("name"), 64)
    status["os"] = clip(status.get("os"), 16).lower()
    return {"url": url, "ok": True, "status": status}


def ipv4_prefixes() -> tuple[list[str], set[str]]:
    prefixes: list[str] = []
    own: set[str] = set()
    if os.name != "nt":
        try:
            ip = socket.gethostbyname(socket.gethostname())
        except OSError:
            return [], set()
        if private_lan(ip):
            return [ip.rsplit(".", 1)[0]], {ip}
        return [], set()
    try:
        out = subprocess.check_output(["ipconfig"], text=True, errors="replace", timeout=10, stderr=subprocess.DEVNULL)
    except (OSError, subprocess.SubprocessError):
        return [], set()
    return parse_ipconfig(out)


def parse_ipconfig(out: str) -> tuple[list[str], set[str]]:
    prefixes: list[str] = []
    own: set[str] = set()
    current = ""
    for line in out.splitlines():
        if "IPv4" in line and ":" in line:
            current = re.sub(r"\(.*\)", "", line.split(":")[-1]).strip()
            if current.count(".") == 3:
                own.add(current)
        elif current and ("Subnet Mask" in line or "Mask" in line) and ":" in line:
            mask = line.split(":")[-1].strip()
            if mask == "255.255.255.0" and private_lan(current):
                prefix = current.rsplit(".", 1)[0]
                if prefix not in prefixes:
                    prefixes.append(prefix)
            current = ""
    return prefixes, own


def probe_host(ip: str) -> dict | None:
    if not private_lan(ip):
        return None
    sock = socket.socket()
    sock.settimeout(0.6)
    try:
        sock.connect((ip, WORKER_PORT))
    except OSError:
        return None
    finally:
        sock.close()
    return pull_worker(f"http://{ip}:{WORKER_PORT}", legacy=False)


def remember_peer(item: dict) -> None:
    if not item.get("ok") or item.get("limited"):
        return
    status = item.get("status") or {}
    url = str(item.get("url") or "")
    if not url:
        return
    try:
        add_node(url)
    except (RuntimeError, OSError):
        return
    system = str(status.get("os") or "").lower()
    bay = {"darwin": "mac", "linux": "linux", "windows": "windows"}.get(system)
    if not bay:
        return
    name = clip(status.get("name") or "", 40)
    if not name:
        return
    short = name.lower().split(".", 1)[0]
    if any(
        spec.get("url") == url or str(spec.get("name") or "").lower().split(".", 1)[0] == short
        for spec in read_specs()
    ):
        return
    total = int((status.get("ram") or {}).get("total_mib") or 0)
    kind = {"windows": "laptop", "mac": "mini", "linux": "penguin"}[bay]
    if bay == "mac" and "mini" not in name.lower():
        kind = "desktop"
    try:
        save_spec(
            {
                "os": bay,
                "kind": kind,
                "name": name,
                "url": url,
                "ram_gb": str(max(1, round(total / 1024))) if total else "",
                "cpu": status.get("machine") or "",
                "gpu": "",
            }
        )
    except (ValueError, RuntimeError, OSError):
        return


def sweep_hosts(prefixes: list[str], own: set[str]) -> list[str]:
    """Every address to probe. Only private /24 nets; never 26.x, 169.254.x,
    or the 192.168.56.x host-only net; never this computer's own address."""
    hosts: list[str] = []
    for prefix in prefixes:
        for n in range(1, 255):
            ip = f"{prefix}.{n}"
            if ip not in own and private_lan(ip):
                hosts.append(ip)
    return hosts


def scan_lan() -> list[dict]:
    prefixes, own = ipv4_prefixes()
    lan = lan_host()
    if lan:
        own.add(lan)
        prefixes = [lan.rsplit(".", 1)[0]]
    hosts = sweep_hosts(prefixes, own)
    if not hosts:
        return []
    with ThreadPoolExecutor(max_workers=16) as pool:
        return [item for item in pool.map(probe_host, hosts) if item and item.get("ok")]


_seen_peers: set[str] = set()
_arrivals: list[dict] = []
_checks: dict[str, dict] = {}
_arrival_lock = threading.Lock()


def note_arrival(item: dict) -> None:
    url = str(item.get("url") or "")
    if not url:
        return
    seen_key = "|".join(peer_identity(item))
    with _arrival_lock:
        if seen_key in _seen_peers:
            return
        _seen_peers.add(seen_key)
    result = ask_check(url)
    status = item.get("status") or {}
    result["name"] = str(status.get("name") or url)
    with _arrival_lock:
        _arrivals.append(result)
        del _arrivals[:-8]


_hold: dict[str, dict] = {}
_sweep_at = 0.0


def peer_identity(item: dict) -> tuple[str, str]:
    status = item.get("status") or {}
    if status.get("os") and status.get("name") and not item.get("limited"):
        return machine_key(status)
    return ("url", str(item.get("url") or ""))


def local_key() -> tuple[str, str]:
    return machine_key({"name": socket.gethostname(), "os": platform.system().lower()})


def settle_peers(
    items: list[dict],
    prefer: set[str],
    misses: dict[str, int] | None = None,
    skip: tuple[str, str] | None = None,
) -> list[dict]:
    """One row per real computer. Among the addresses of one computer, the
    one that answered most recently wins; on a tie the address already shown
    stays, so the row does not jump between .247 and .248. A worker on this
    same computer is skipped, because this PC is already the first row."""
    misses = misses or {}
    ranked: dict[tuple[str, str], dict] = {}
    for item in items:
        key = peer_identity(item)
        if skip is not None and key == skip:
            continue
        current = ranked.get(key)
        if current is None:
            ranked[key] = item
            continue
        url = str(item.get("url") or "")
        current_url = str(current.get("url") or "")
        mine = (misses.get(url, 0), 0 if url in prefer else 1)
        theirs = (misses.get(current_url, 0), 0 if current_url in prefer else 1)
        if mine < theirs:
            ranked[key] = item
    return list(ranked.values())


def count_misses(hold: dict[str, dict], answered: list[dict], now: float, limit: int = 3) -> None:
    """Update the held rows in place. A row that did not answer gains a miss;
    after more than `limit` misses in a row it is dropped. A row that
    answered is reset to zero misses."""
    answered_urls = {str(item.get("url") or "") for item in answered}
    for url, held in list(hold.items()):
        if url in answered_urls:
            continue
        held["misses"] = int(held.get("misses") or 0) + 1
        if held["misses"] > limit:
            hold.pop(url, None)
    for item in answered:
        url = str(item.get("url") or "")
        if url:
            hold[url] = {"item": item, "misses": 0, "at": now}


def known_worker_urls() -> list[str]:
    urls: list[str] = []
    seen: set[str] = set()
    with _found_lock:
        extra = [str(item.get("url") or "") for item in _found]
    for url in [str(node.get("url") or "") for node in read_nodes()] + extra + list(_hold.keys()):
        if url and url not in seen:
            seen.add(url)
            urls.append(url)
    return urls


def discovery_loop() -> None:
    global _found, _sweep_at
    while True:
        try:
            with _found_lock:
                previous = {str(item.get("url") or "") for item in _found}
                previous_keys = {peer_identity(item) for item in _found}
            saved = {str(node.get("url") or "") for node in read_nodes()}
            urls = known_worker_urls()
            alive: list[dict] = []
            if urls:
                with ThreadPoolExecutor(max_workers=8) as pool:
                    pulled = list(pool.map(lambda url: pull_worker(url, legacy=url in saved), urls))
                alive = [item for item in pulled if item.get("ok")]
            now = time.time()
            if now - _sweep_at >= 45:
                _sweep_at = now
                asked = {str(item.get("url") or "") for item in alive}
                alive.extend(item for item in scan_lan() if str(item.get("url") or "") not in asked)
            count_misses(_hold, alive, now)
            misses = {url: int(held.get("misses") or 0) for url, held in _hold.items()}
            found = settle_peers([held["item"] for held in _hold.values()], previous, misses, local_key())
            shown: list[dict] = []
            fresh: list[dict] = []
            for item in found:
                remember_peer(item)
                url = str(item.get("url") or "")
                held = _hold.get(url) or {}
                # A copy that tells the page how old this row is. A row that
                # missed a check is still shown, but marked, not faked live.
                row = dict(item)
                row["misses"] = int(held.get("misses") or 0)
                row["seen_s"] = max(0, int(now - float(held.get("at") or now)))
                shown.append(row)
                if peer_identity(item) not in previous_keys:
                    fresh.append(item)
            with _found_lock:
                _found = shown
            for item in fresh:
                threading.Thread(target=note_arrival, args=(item,), name="neuro-arrival", daemon=True).start()
        except Exception as exc:
            print(f"discovery failed: {type(exc).__name__}: {exc}", flush=True)
        time.sleep(10)


def start_discovery() -> None:
    threading.Thread(target=discovery_loop, name="neuro-discovery", daemon=True).start()


_served_cache: dict = {"at": 0.0, "ids": set()}


def served_model_ids(ttl: float = 15.0) -> set[str]:
    """File names the chat server on 127.0.0.1:8080 says it serves (/v1/models)."""
    now = time.time()
    if now - float(_served_cache["at"]) < ttl:
        return set(_served_cache["ids"])
    ids: set[str] = set()
    try:
        with _DIRECT.open("http://127.0.0.1:8080/v1/models", timeout=1.5) as response:
            body = read_reply(response)
        rows = body.get("data") if isinstance(body, dict) else None
        for row in rows if isinstance(rows, list) else []:
            if isinstance(row, dict) and row.get("id"):
                ids.add(str(row["id"]).replace("\\", "/").split("/")[-1].lower()[:200])
    except (OSError, urllib.error.URLError, ValueError):
        pass
    _served_cache.update({"at": now, "ids": ids})
    return set(ids)


def known_models() -> list[dict]:
    models = []
    served = served_model_ids()
    folder = data_root() / "models" / "gguf"
    paths: list[Path] = []
    try:
        if folder.is_dir():
            for path in folder.rglob("*.gguf"):
                paths.append(path)
                if len(paths) >= 64:
                    break
    except OSError:
        pass
    extra = ROOT / "models.json"
    try:
        saved = json.loads(extra.read_text(encoding="utf-8"))
        if isinstance(saved, list):
            paths.extend(
                Path(item["path"]) for item in saved if isinstance(item, dict) and isinstance(item.get("path"), str) and item["path"]
            )
    except (OSError, ValueError):
        pass
    seen = set()
    for path in paths:
        try:
            if not path.is_file() or path.suffix.lower() != ".gguf" or "mmproj" in path.name.lower():
                continue
            key = str(path.resolve()).lower()
            size = path.stat().st_size
        except OSError:
            continue
        if key in seen:
            continue
        seen.add(key)
        models.append(
            {
                "name": path.stem,
                "path": str(path),
                "mb": size // (1024 * 1024),
                "loaded": path.name.lower() in served or path.stem.lower() in served,
            }
        )
        if len(models) >= 8:
            break
    return models


def save_model(raw_path: str) -> dict:
    path = Path(raw_path.strip().strip('"'))
    if path.suffix.lower() != ".gguf" or not path.is_file():
        raise ValueError("Choose a GGUF file that is already on this PC.")
    if "mmproj" in path.name.lower():
        raise ValueError("That file is a projector, not the model.")
    extra = ROOT / "models.json"
    item = {"name": path.stem, "path": str(path)}
    with _files_lock:
        saved = read_json_list(extra, strict=True)
        if not any(entry.get("path") == item["path"] for entry in saved):
            saved.append(item)
            write_json_list(extra, saved)
    return item


def side_jobs(status: dict, is_home: bool, training_running: bool | None, chat_on_home: bool) -> list[dict]:
    """Which side job fits this computer. Pure: built only from the status a
    worker already reports, so an old worker without "caps" still gets a
    best guess from its RAM and GPU list. Nothing here starts a job."""
    caps = status.get("caps") if isinstance(status.get("caps"), dict) else {}
    ram = status.get("ram") if isinstance(status.get("ram"), dict) else {}
    gpus = [card for card in (status.get("gpus") or []) if isinstance(card, dict)]
    cuda = [card for card in gpus if is_cuda_card(card)]
    os_name = str(status.get("os") or "").lower()
    free_mib = int(caps.get("free_ram_mib") or ram.get("free_mib") or 0)
    total_mib = int(ram.get("total_mib") or 0)
    cores = int(caps.get("cpu_cores") or 0)
    vram_total = int(caps.get("vram_total_mib") or max((int(card.get("memory_total_mib") or 0) for card in cuda), default=0))
    vram_free = int(
        caps.get("vram_free_mib")
        or max((int(card.get("memory_total_mib") or 0) - int(card.get("memory_used_mib") or 0) for card in cuda), default=0)
    )
    busy_pct = caps.get("cpu_busy_pct")
    chat_here = bool(status.get("chat_server"))
    has_caps = bool(caps)
    cpu_free = not chat_here and (busy_pct is None or int(busy_pct) < 60)
    jobs: list[dict] = []

    def add(job: str, label: str, ready: bool, why: str) -> None:
        jobs.append({"job": job, "label": label, "ready": bool(ready), "why": why})

    if is_home:
        add("train", "Training", bool(cuda), "The one training run stays on this NVIDIA PC.")
        return jobs
    if cuda and vram_total >= 4000:
        add(
            "eval",
            "Eval on CUDA",
            vram_free >= 3500 and not chat_here,
            f"{round(vram_total / 1024)} GB NVIDIA memory, {round(max(0, vram_free) / 1024, 1)} GB free. Fits a small quantized model or a saved checkpoint eval, not training.",
        )
    if cores >= 4 or (not has_caps and total_mib >= 8000):
        if caps.get("luau"):
            add("verify", "Verified-code checks", cpu_free, "Runs the Luau checks on its own processor.")
        else:
            add("verify", "Agent code checks", cpu_free, "Runs the agent drill on its own processor." + ("" if has_caps else " Older worker: the core count is not reported."))
    if free_mib >= 4096 or (not has_caps and total_mib >= 12000):
        add("dedup", "Data dedup and filtering", cpu_free and free_mib >= 4096, "CPU and RAM work on a copy of a dataset shard. The training files are not touched.")
    rpc_ok = bool(caps.get("rpc_server") or status.get("rpc_binary"))
    if rpc_ok or os_name == "darwin":
        ready = rpc_ok and not chat_on_home and free_mib >= 4096
        why = "Lends memory to one llama.cpp inference run over the LAN, only after chat on this PC is off."
        if not rpc_ok:
            why = "Needs ggml-rpc-server built first. " + why
        elif chat_on_home:
            why = "Waits: chat is still on this PC. " + why
        add("rpc", "llama.cpp RPC memory", ready, why)
    return jobs


def mesh_snapshot() -> dict:
    local = local_status()
    # The page used to ask every saved worker again here, one by one, on the
    # request thread. An asleep laptop made each refresh wait. The discovery
    # thread already asks them every 10 seconds, so its result is used.
    with _found_lock:
        found = [dict(item) for item in _found]
    training = training_view()
    chat_home = bool(local.get("chat_server"))
    nodes = [{"url": "local", "ok": True, "status": local}, *found]
    for index, item in enumerate(nodes):
        status = item.get("status") if isinstance(item.get("status"), dict) else {}
        try:
            item["fit"] = side_jobs(status, index == 0, training.get("running"), chat_home)
        except (TypeError, ValueError):
            item["fit"] = []
    lan = lan_host()
    with _arrival_lock:
        arrivals = list(_arrivals)
        checks = list(_checks.values())
    try:
        jobs = job_queue().summary()
    except (OSError, RuntimeError, ValueError):
        jobs = {"kinds": {}, "nodes": {}, "verified_rows": 0, "total": 0, "verifiers": {}, "rate": {}}
    return {
        "jobs": jobs,
        "nodes": nodes,
        "found": found,
        "token_id": token_id(),
        "rpc_port": RPC_PORT,
        "chat": "http://127.0.0.1:8080/v1",
        "limits": {
            "remote_shell": False,
            "joint_training": False,
            "reason": "Extra machines check the agent code on their own CPU and can lend memory to one inference run. One training run stays on the NVIDIA PC.",
        },
        "specs": read_specs(),
        "training": training,
        "models": known_models(),
        "lan": ("http://" + lan + ":" + str(DASHBOARD_PORT)) if lan else "",
        "arrivals": arrivals,
        "checks": checks,
        "share": {
            "model_on_this_pc": bool(local.get("chat_server")),
            "training": "this PC only",
            "peers": [
                {
                    "name": (item.get("status") or {}).get("name") or "",
                    "listening": bool((item.get("status") or {}).get("rpc_listening")),
                }
                for item in found
                if item.get("ok")
            ],
        },
    }


def lan_host() -> str:
    """This computer's private LAN address. A UDP connect sends no packet; it
    only asks the OS which interface would carry the route. The default route
    (TEST-NET address) comes first, then the usual private ranges."""
    for probe in ("192.0.2.1", "192.168.0.1", "192.168.1.1", "10.0.0.1", "172.16.0.1"):
        try:
            with socket.socket(socket.AF_INET, socket.SOCK_DGRAM) as sock:
                sock.connect((probe, 9))
                host = sock.getsockname()[0]
        except OSError:
            continue
        if private_lan(host):
            return host
    return ""


def ask_check(url: str) -> dict:
    try:
        body = worker_call(url, "/check", {"rounds": 3}, 20)
    except WorkerProofError as exc:
        return {"url": url, "ok": False, "error": str(exc)}
    except urllib.error.HTTPError as exc:
        return {"url": url, "ok": False, "error": worker_refusal(exc)}
    except (OSError, urllib.error.URLError, json.JSONDecodeError, ValueError):
        return {"url": url, "ok": False, "error": "no reply"}
    if not isinstance(body, dict):
        return {"url": url, "ok": False, "error": "bad reply"}
    return {"url": url, "ok": bool(body.get("ok")), "passed": body.get("passed"), "rounds": body.get("rounds"), "host": body.get("host") or ""}


def check_helpers() -> dict:
    with _found_lock:
        peers = [item for item in _found if item.get("ok") and item.get("url")]
    rows = [ask_check(str(item["url"])) for item in peers]
    with _arrival_lock:
        for row in rows:
            url = str(row.get("url") or "")
            if url:
                _checks[url] = row
    return {"checked": rows, "training": "unchanged"}


def helper_loop() -> None:
    time.sleep(15)
    while True:
        try:
            check_helpers()
        except Exception as exc:
            print(f"agent check failed: {exc}", flush=True)
        time.sleep(300)


def start_rpc() -> dict:
    status = local_status()
    if status["chat_server"]:
        raise RuntimeError("The GPU is serving Neuro. Stop the chat server before splitting weights.")
    if status["rpc_listening"]:
        return {"started": False, "already": True, "port": RPC_PORT}
    binary = status["rpc_binary"]
    if not binary:
        raise RuntimeError("ggml-rpc-server was not found on this machine.")
    host = lan_host()
    if not host:
        raise RuntimeError("RPC stays off. This machine has no private LAN address.")
    process = subprocess.Popen(
        [binary, "--host", host, "--port", str(RPC_PORT)],
        cwd=str(Path(binary).parent),
        stdout=subprocess.DEVNULL,
        stderr=subprocess.DEVNULL,
    )
    PID_PATH.write_text(str(process.pid), encoding="utf-8")
    return {"started": True, "pid": process.pid, "port": RPC_PORT}


def ask_local(message: str) -> dict:
    text = message if isinstance(message, str) else ""
    text = text.replace("\0", "")[:4_000]
    payload = json.dumps(
        {
            "messages": [
                {
                    "role": "system",
                    "content": "You are Neuro, running on this computer. Answer the user.",
                },
                {"role": "user", "content": text},
            ],
            "temperature": 0.7,
            "max_tokens": 2048,
            "chat_template_kwargs": {"enable_thinking": False},
        }
    ).encode("utf-8")
    request = urllib.request.Request(
        "http://127.0.0.1:8080/v1/chat/completions",
        data=payload,
        headers={"Content-Type": "application/json"},
    )
    try:
        with urllib.request.urlopen(request, timeout=180) as response:
            raw = response.read(1_000_000)
        body = json.loads(raw.decode("utf-8"))
    except (urllib.error.URLError, TimeoutError, OSError, UnicodeDecodeError, json.JSONDecodeError, ValueError):
        return {"reply": "", "reasoning": "", "tokens": {}, "error": "Neuro did not answer."}
    if not isinstance(body, dict):
        return {"reply": "", "reasoning": "", "tokens": {}, "error": "Neuro did not answer."}
    choices = body.get("choices")
    message_body = choices[0].get("message") if isinstance(choices, list) and choices and isinstance(choices[0], dict) else None
    if not isinstance(message_body, dict):
        return {"reply": "", "reasoning": "", "tokens": {}, "error": "Neuro did not answer."}
    reply = message_body.get("content") or ""
    reasoning = message_body.get("reasoning_content") or ""
    return {
        "reply": reply[:12_000] if isinstance(reply, str) else "",
        "reasoning": reasoning[:4_000] if isinstance(reasoning, str) else "",
        "tokens": body.get("usage") if isinstance(body.get("usage"), dict) else {},
    }


def offer_share() -> dict:
    local = local_status()
    with _found_lock:
        peers = [item for item in _found if item.get("ok") and item.get("url")]
    results = []
    for item in peers:
        url = str(item["url"]).rstrip("/")
        name = str((item.get("status") or {}).get("name") or url)
        try:
            body = worker_call(url, "/rpc/start", {}, 8)
            body = body if isinstance(body, dict) else {}
            listening = bool(body.get("started") or body.get("already"))
            results.append({"name": name, "ok": listening, "detail": "listening" if listening else "answered"})
        except WorkerProofError as exc:
            results.append({"name": name, "ok": False, "detail": str(exc)})
        except urllib.error.HTTPError as exc:
            results.append({"name": name, "ok": False, "detail": worker_refusal(exc)})
        except (OSError, urllib.error.URLError, ValueError) as exc:
            results.append({"name": name, "ok": False, "detail": str(exc)[:180]})
    return {
        "moved": False,
        "model_on_this_pc": bool(local.get("chat_server")),
        "peers": results,
        "note": "The loaded model stays on this PC. A computer that is listening joins the next start. Splitting a model that already fits this GPU makes the run slower.",
    }


def error_detail(exc: urllib.error.HTTPError) -> str:
    """The worker's own error text, never the raw HTTP body dump."""
    try:
        raw = exc.read(4096).decode("utf-8", errors="replace")
    except OSError:
        return f"http {exc.code}"
    try:
        body = json.loads(raw)
    except ValueError:
        return raw[:180]
    if isinstance(body, dict):
        return clip(body.get("error") or body.get("note") or f"http {exc.code}", 180)
    return raw[:180]


def process_name(pid: int) -> str:
    if pid <= 0:
        return ""
    try:
        if os.name == "nt":
            out = subprocess.check_output(
                ["tasklist", "/FI", f"PID eq {pid}", "/FO", "CSV", "/NH"],
                text=True,
                timeout=8,
                stderr=subprocess.DEVNULL,
            )
            for line in out.splitlines():
                if "No tasks" in line or not line.strip():
                    continue
                parts = [part.strip().strip('"') for part in line.split('","')]
                if parts:
                    return parts[0].strip('"')
            return ""
        out = subprocess.check_output(["ps", "-p", str(pid), "-o", "comm="], text=True, timeout=4)
    except (OSError, subprocess.SubprocessError):
        return ""
    # Same 40-character cut as the process list, so a long Mac name matches.
    return out.strip().replace("\\", "/").split("/")[-1][:40]


def stop_owned_process(pid: int, expected: str = "") -> dict:
    if pid <= 0:
        return {"stopped": False, "note": "Missing process."}
    name = process_name(pid)
    if not name:
        return {"stopped": False, "note": "That process is already gone."}
    wanted = expected.strip()
    if not wanted or name.lower() != wanted.lower():
        return {"stopped": False, "note": "That process changed. Nothing was stopped."}
    if name.lower() in PROTECTED or pid in keep_pids() or pid == os.getpid():
        return {"stopped": False, "note": name + " keeps the training run or the computer up, so it was left on."}
    # Only a program that the page lists with a Stop button can be stopped.
    # This also covers Python while the GPU lock is held, and anything small.
    _heavy_cache["at"] = 0.0
    listed = next((row for row in heavy_processes() if row.get("pid") == pid), None)
    if listed is None or not listed.get("stoppable") or str(listed.get("name") or "").lower() != name.lower():
        return {"stopped": False, "note": name + " is not on the list of programs that can be stopped, so it was left on."}
    if os.name == "nt":
        try:
            done = subprocess.run(["taskkill", "/PID", str(pid), "/F"], capture_output=True, text=True, timeout=15)
        except (OSError, subprocess.SubprocessError):
            return {"stopped": False, "note": "Could not stop " + name + "."}
        stopped = done.returncode == 0
    else:
        try:
            os.kill(pid, 15)
            stopped = True
        except OSError:
            stopped = False
    if not stopped:
        return {"stopped": False, "note": "Could not stop " + name + "."}
    _heavy_cache["at"] = 0.0
    return {"stopped": True, "name": name, "pid": pid, "note": "Stopped " + name + ". That memory is being released."}


def peer_known(url: str) -> bool:
    try:
        clean = clean_url(url)
    except ValueError:
        return False
    with _found_lock:
        found = list(_found)
    known: list[str] = []
    for item in list(found) + read_nodes():
        raw = str(item.get("url") or "")
        if not raw:
            continue
        try:
            known.append(clean_url(raw))
        except ValueError:
            continue
    return clean in known


def stop_at(url: str, pid: int, name: str = "") -> dict:
    if pid <= 0:
        return {"stopped": False, "note": "Missing process."}
    if not url:
        return stop_owned_process(pid, name)
    try:
        clean = clean_url(url)
    except ValueError:
        return {"stopped": False, "note": "That address is not a joined computer."}
    if not peer_known(clean):
        return {"stopped": False, "note": "That computer is not connected."}
    try:
        body = worker_call(clean, "/process/stop", {"pid": pid, "name": name}, 20)
        if isinstance(body, dict):
            return {"stopped": bool(body.get("stopped")), "note": clip(body.get("note") or "", 240)}
    except WorkerProofError as exc:
        return {"stopped": False, "note": str(exc)}
    except urllib.error.HTTPError as exc:
        return {"stopped": False, "note": worker_refusal(exc) or "Could not stop that process."}
    except (OSError, urllib.error.URLError, ValueError):
        return {"stopped": False, "note": "That computer did not answer."}
    return {"stopped": False, "note": "Could not stop that process."}


def stop_model() -> dict:
    try:
        if os.name == "nt":
            done = subprocess.run(["taskkill", "/IM", "llama-server.exe", "/F"], capture_output=True, text=True, timeout=15)
        else:
            done = subprocess.run(["pkill", "-x", "llama-server"], capture_output=True, text=True, timeout=15)
        stopped = done.returncode == 0
    except (OSError, subprocess.SubprocessError):
        stopped = False
    _heavy_cache["at"] = 0.0
    return {
        "stopped": stopped,
        "note": "Stopped the loaded model on this PC. Code workers and other programs were left running.",
    }


def stop_rpc() -> dict:
    try:
        pid_text = PID_PATH.read_text(encoding="utf-8").strip()
    except OSError:
        return {"stopped": False}
    PID_PATH.unlink(missing_ok=True)
    if not pid_text.isdigit():
        return {"stopped": False}
    pid = int(pid_text)
    # The saved PID may have been reused by another program since.
    if "rpc-server" not in process_name(pid).lower():
        return {"stopped": False, "note": "The share listener was already gone."}
    if os.name == "nt":
        try:
            subprocess.run(["taskkill", "/PID", str(pid), "/F"], check=False, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL, timeout=15)
        except (OSError, subprocess.SubprocessError):
            return {"stopped": False}
    else:
        try:
            os.kill(pid, 15)
        except OSError:
            return {"stopped": False}
    return {"stopped": True, "pid": pid}


# ---------------------------------------------------------------------------
# Data factory: a small job queue on the main PC. Idle workers pull a job,
# run it with a built-in handler, and post the result. A job is data for one
# of the handlers below. No job can name a program, a shell line, a host, or
# a file path. Training itself stays on the NVIDIA PC.
# ---------------------------------------------------------------------------

JOBS_DIR = ROOT / "jobs"
JOB_KINDS = (
    "generate", "verify_luau", "verify_python_syntax", "verify_js", "verify_ts",
    "verify_html_css", "verify_sql", "verify_math", "dedup_hash",
    "verify_tests", "cross_tests",
)
VERIFY_KINDS = (
    "verify_luau", "verify_python_syntax", "verify_js", "verify_ts",
    "verify_html_css", "verify_sql", "verify_math", "verify_tests",
)
# The verifiers added for the coding set never run generated code:
#   verify_js        node --check (parse only), one temp file per code block
#   verify_ts        node's built-in stripTypeScriptTypes (a parser), then node --check
#   verify_html_css  html.parser structure checks plus a CSS brace/string scanner
#   verify_sql       sqlite3 in memory: DDL builds a throwaway catalog, every other
#                    statement is only compiled with EXPLAIN, never stepped
# A verifier that cannot run here (no node, an old node) reports
# "unavailable" as a failed job, never as a pass or as a rejected answer.
CODE_KINDS = {"verify_python_syntax": "python", "verify_js": "js", "verify_ts": "ts", "verify_html_css": "html", "verify_sql": "sql"}
MAX_SAMPLES = 16
MAX_CODE_BLOCKS = 16
CODE_LIMIT = 200000
NODE_TIMEOUT = 10
NODE_OUTPUT_CAP = 256 * 1024
JOB_LINE_LIMIT = 256 * 1024
JOB_RESULT_LIMIT = 512 * 1024
JOB_OUTPUT_LIMIT = 256 * 1024
GENERATE_MAX_TOKENS = 4096
# A generate result cut off by max_tokens (finish_reason "length") is not
# checked: the server queues one replacement with LONG_RETRY_TOKENS instead.
# A long retry that is still cut off is kept as truncated: never verified,
# never exported, never a "rejected" answer in pairs.
LONG_RETRY_TOKENS = 4096
# Workers that do not say how many tokens they allow ran older code that
# capped generate at 2048; they only get generate jobs up to that size.
LEGACY_GEN_MAX = 2048
GENERATE_TIMEOUT = 600
LUAU_TIMEOUT = 10
LUAU_OUTPUT_CAP = 1024 * 1024
MINHASH_PERM = 32
MAX_ATTEMPTS = 3
# verify_tests / cross_tests RUN model-written code, so they get one lease
# retry at most (a worker that vanished) and never a second verdict.
KIND_ATTEMPTS = {"verify_tests": 2, "cross_tests": 2}
# A generate job that failed because the worker's model was briefly unable to answer (HTTP 500 when
# parallel long replies filled the KV cache, a restart, a timeout) goes back to the queue instead of
# being lost: 846 of the hardest jobs (long retries) were lost that way. At most MAX_ATTEMPTS leases.
TRANSIENT_MODEL_ERROR = re.compile(
    r"HTTP Error 50[0234]|HTTP 50[0234]|Connection refused|Errno 111|Errno 61|10061|RemoteDisconnected"
    r"|timed out|TimeoutError|Connection reset|returned no text",
    re.IGNORECASE,
)
# Verifiers whose answers were cut off at 2048 tokens most of the time (luau 37%, tests 22%): each
# cut-off reply was thrown away and generated again from scratch. These start with room to finish.
LONG_FROM_START = frozenset({"verify_luau", "verify_tests"})
RATE_WINDOW = 3600  # seconds of finished jobs behind the rates and the ETA
LEASE_SECONDS = {
    "generate": 900, "verify_luau": 180, "verify_python_syntax": 120, "verify_js": 180, "verify_ts": 240,
    "verify_html_css": 120, "verify_sql": 120, "verify_math": 120, "dedup_hash": 300,
    "verify_tests": 120, "cross_tests": 400,
}
# verify_tests: the answer holds the solution plus assert tests in one fenced
# block; it runs in a sandbox (see run_sandboxed) and passes on exit code 0
# with every top-level statement finished and at least MIN_ASSERTS real
# asserts. cross_tests (CodeT-lite) runs each passing sample of a prompt
# group against the other samples' asserts and scores the agreement.
TEST_LANGS = {"python": "python", "javascript": "js"}  # verify.lang -> fence language
TEST_JOB_KINDS = ("verify_tests", "cross_tests")
TEST_TIMEOUT = 10            # wall seconds per run
TEST_CPU_SECONDS = 10        # CPU seconds per run (Job Object / RLIMIT_CPU)
TEST_MEMORY = 1024 ** 3      # Job Object memory / RLIMIT_AS
TEST_FILE_LIMIT = 16 * 1024 * 1024
TEST_OUTPUT_CAP = 64 * 1024  # bytes kept per stream
TEST_OUTPUT_STOP = 4 * 1024 * 1024  # a run that prints more is stopped
MIN_ASSERTS = 3
MAX_CROSS_CHECKS = 200       # other samples' asserts run against one solution
AGREEMENT_MIN = 0.5
_HEX64 = re.compile(r"^[0-9a-f]{64}$")
_HEX16 = re.compile(r"^[0-9a-f]{16}$")
SKILLS = {"general", "code", "math", "luau"}
# Worker job slots: "g0".."g7" lease generate, "c0".."c7" lease checks.
_SLOT_ID = re.compile(r"[gc][0-7]")
SLOT_CAP = 8
GEN_SLOT_DEFAULT_CAP = 4


class JobError(ValueError):
    pass


def _job_text(value: object, limit: int, field: str) -> str:
    if isinstance(value, (int, float)) and not isinstance(value, bool):
        value = str(value)
    if not isinstance(value, str):
        raise JobError(field + " must be text")
    if len(value) > limit:
        raise JobError(f"{field} is longer than {limit} characters")
    return value


def _clamp(value: object, low: float, high: float, default: float) -> float:
    try:
        number = float(value)  # type: ignore[arg-type]
    except (TypeError, ValueError):
        return default
    if number != number:  # NaN
        return default
    return max(low, min(high, number))


def validate_verify_spec(spec: object) -> dict:
    if not isinstance(spec, dict):
        raise JobError("verify must be an object")
    kind = spec.get("kind")
    if kind not in VERIFY_KINDS:
        raise JobError("verify.kind must be one of " + ", ".join(VERIFY_KINDS))
    out: dict = {"kind": kind}
    if kind == "verify_math":
        out["expected"] = _job_text(spec.get("expected"), 200, "verify.expected")
        if spec.get("tol") is not None:
            out["tol"] = _clamp(spec.get("tol"), 0.0, 1.0, 1e-6)
    if kind == "verify_luau" and spec.get("expect_stdout") is not None:
        out["expect_stdout"] = _job_text(spec.get("expect_stdout"), 4000, "verify.expect_stdout")
    if kind == "verify_sql" and spec.get("schema") is not None:
        out["schema"] = _job_text(spec.get("schema"), 64000, "verify.schema")
    if kind == "verify_html_css" and spec.get("require_tags") is not None:
        out["require_tags"] = _tag_list(spec.get("require_tags"), "verify.require_tags")
    if kind == "verify_tests":
        if not isinstance(spec.get("lang"), str) or spec["lang"] not in TEST_LANGS:
            raise JobError("verify.lang must be python or javascript")
        out["lang"] = spec["lang"]
    return out


def _tag_list(value: object, field: str) -> list[str]:
    if not isinstance(value, list) or len(value) > 32:
        raise JobError(field + " must be a list of up to 32 tag names")
    tags = []
    for item in value:
        tag = _job_text(item, 40, field).strip().lower()
        if not re.fullmatch(r"[a-z][a-z0-9-]*", tag):
            raise JobError(field + " holds a bad tag name")
        tags.append(tag)
    return tags


def _code_blocks_field(raw: dict, lang: str) -> list[str]:
    """Code for verify_js / verify_ts: `blocks` (a list), `code` (one
    block) or `answer` (a markdown reply; its fenced blocks are taken)."""
    if raw.get("blocks") is not None:
        blocks = raw.get("blocks")
        if not isinstance(blocks, list):
            raise JobError("blocks must be a list of code strings")
        blocks = [_job_text(block, CODE_LIMIT, "block") for block in blocks]
    elif raw.get("code") is not None:
        blocks = [_job_text(raw.get("code"), CODE_LIMIT, "code")]
    elif raw.get("answer") is not None:
        blocks = pick_blocks(_job_text(raw.get("answer"), CODE_LIMIT, "answer"), lang)
    else:
        raise JobError("needs code, blocks or answer")
    if len(blocks) > MAX_CODE_BLOCKS:
        raise JobError(f"more than {MAX_CODE_BLOCKS} code blocks")
    if sum(len(block) for block in blocks) > CODE_LIMIT:
        raise JobError(f"the code is longer than {CODE_LIMIT} characters")
    return blocks


def validate_job(kind: str, raw: object) -> dict:
    """Check one job and return the clean payload that is stored and sent.
    Unknown fields are dropped, sizes are capped."""
    if kind not in JOB_KINDS:
        raise JobError("Unknown job kind. Use one of " + ", ".join(JOB_KINDS))
    if not isinstance(raw, dict):
        raise JobError("A job must be a JSON object")
    payload: dict
    if kind == "generate":
        messages = raw.get("messages")
        if messages is None and isinstance(raw.get("prompt"), str):
            messages = []
            if isinstance(raw.get("system"), str) and raw["system"].strip():
                messages.append({"role": "system", "content": raw["system"]})
            messages.append({"role": "user", "content": raw["prompt"]})
        if not isinstance(messages, list) or not 1 <= len(messages) <= 32:
            raise JobError("generate needs messages (1 to 32) or a prompt")
        clean = []
        total = 0
        for message in messages:
            if not isinstance(message, dict) or message.get("role") not in {"system", "user", "assistant"}:
                raise JobError("Each message needs role system, user or assistant")
            content = _job_text(message.get("content"), 32000, "message content")
            total += len(content)
            clean.append({"role": message["role"], "content": content})
        if total > 64000:
            raise JobError("The messages are longer than 64000 characters")
        if clean[-1]["role"] != "user":
            raise JobError("The last message must be from the user")
        payload = {
            "messages": clean,
            "max_tokens": int(_clamp(raw.get("max_tokens"), 1, GENERATE_MAX_TOKENS, 1024)),
            "temperature": _clamp(raw.get("temperature"), 0.0, 2.0, 0.7),
            "top_p": _clamp(raw.get("top_p"), 0.01, 1.0, 0.95),
        }
        if raw.get("verify") is not None:
            payload["verify"] = validate_verify_spec(raw.get("verify"))
        if raw.get("long_retry") is True:
            payload["long_retry"] = True
        # Rejection sampling: `samples` copies of one prompt share a `group`
        # (JobQueue.add_many makes the copies). `seed` keeps them different.
        samples = int(_clamp(raw.get("samples"), 1, MAX_SAMPLES, 1))
        if samples > 1:
            payload["samples"] = samples
        group = raw.get("group")
        if isinstance(group, str) and _HEX16.match(group):
            payload["group"] = group
            payload["sample"] = int(_clamp(raw.get("sample"), 0, MAX_SAMPLES - 1, 0))
        seed = raw.get("seed")
        if isinstance(seed, int) and not isinstance(seed, bool) and 0 <= seed < 2 ** 31:
            payload["seed"] = seed
    elif kind in ("verify_js", "verify_ts"):
        payload = {"blocks": _code_blocks_field(raw, CODE_KINDS[kind])}
    elif kind == "verify_tests":
        if not isinstance(raw.get("lang"), str) or raw["lang"] not in TEST_LANGS:
            raise JobError("verify_tests needs lang python or javascript")
        payload = {"lang": raw["lang"], "blocks": _code_blocks_field(raw, TEST_LANGS[raw["lang"]])}
    elif kind == "cross_tests":
        if not isinstance(raw.get("lang"), str) or raw["lang"] not in TEST_LANGS:
            raise JobError("cross_tests needs lang python or javascript")
        group = raw.get("group")
        if not isinstance(group, str) or not _HEX16.match(group):
            raise JobError("cross_tests needs the group id")
        samples = raw.get("samples")
        if not isinstance(samples, list) or not 2 <= len(samples) <= MAX_SAMPLES:
            raise JobError(f"cross_tests needs 2 to {MAX_SAMPLES} samples")
        clean_samples = []
        for sample in samples:
            if not isinstance(sample, dict) or not isinstance(sample.get("id"), str) or not _HEX16.match(sample["id"]):
                raise JobError("each cross_tests sample needs a job id and code")
            clean_samples.append({"id": sample["id"], "code": _job_text(sample.get("code"), CODE_LIMIT, "code")})
        if len({sample["id"] for sample in clean_samples}) != len(clean_samples):
            raise JobError("cross_tests samples repeat a job id")
        if sum(len(sample["code"]) for sample in clean_samples) > CODE_LIMIT:
            raise JobError(f"the code is longer than {CODE_LIMIT} characters")
        payload = {"lang": raw["lang"], "group": group, "samples": clean_samples}
    elif kind == "verify_html_css":
        if raw.get("html") is None and raw.get("css") is None and raw.get("answer") is not None:
            html, css = split_web(_job_text(raw.get("answer"), CODE_LIMIT, "answer"))
        else:
            html = _job_text(raw.get("html") if raw.get("html") is not None else raw.get("code") or "", CODE_LIMIT, "html")
            css = _job_text(raw.get("css") or "", CODE_LIMIT, "css")
        if len(html) + len(css) > CODE_LIMIT:
            raise JobError(f"the code is longer than {CODE_LIMIT} characters")
        payload = {"html": html, "css": css}
        if raw.get("require_tags") is not None:
            payload["require_tags"] = _tag_list(raw.get("require_tags"), "require_tags")
    elif kind == "verify_sql":
        if raw.get("code") is None and raw.get("answer") is not None:
            code = join_sql(pick_blocks(_job_text(raw.get("answer"), 64000, "answer"), "sql"))
        else:
            code = raw.get("code")
        payload = {"code": _job_text(code, 64000, "code")}
        if raw.get("schema") is not None:
            payload["schema"] = _job_text(raw.get("schema"), 64000, "schema")
    elif kind == "verify_luau":
        payload = {"code": _job_text(raw.get("code"), 64000, "code")}
        if raw.get("expect_stdout") is not None:
            payload["expect_stdout"] = _job_text(raw.get("expect_stdout"), 4000, "expect_stdout")
    elif kind == "verify_python_syntax":
        payload = {"code": _job_text(raw.get("code"), 200000, "code")}
    elif kind == "verify_math":
        payload = {
            "answer": _job_text(raw.get("answer"), 64000, "answer"),
            "expected": _job_text(raw.get("expected"), 200, "expected"),
        }
        if raw.get("tol") is not None:
            payload["tol"] = _clamp(raw.get("tol"), 0.0, 1.0, 1e-6)
    else:  # dedup_hash
        texts = raw.get("texts")
        if not isinstance(texts, list) or not 1 <= len(texts) <= 500:
            raise JobError("dedup_hash needs texts: a list of 1 to 500 strings")
        clean_texts = [_job_text(text, 50000, "text") for text in texts]
        if sum(len(text) for text in clean_texts) > 1_000_000:
            raise JobError("The texts are longer than 1,000,000 characters in total")
        payload = {"texts": clean_texts}
    meta = raw.get("meta")
    if isinstance(meta, dict):
        packed = json.dumps(meta, ensure_ascii=False)
        if len(packed) > 2000:
            raise JobError("meta is larger than 2000 characters")
        payload["meta"] = json.loads(packed)
    return payload


def strip_thinking(text: str) -> str:
    return re.sub(r"<think>.*?</think>", "", text, flags=re.DOTALL).strip()


def extract_code(text: str, languages: tuple[str, ...]) -> str:
    """The last fenced code block, preferring one tagged with a wanted
    language. Without a fence the whole reply is the code."""
    text = strip_thinking(text)
    blocks = re.findall(r"```([A-Za-z0-9_+-]*)[ \t]*\r?\n(.*?)```", text, flags=re.DOTALL)
    if not blocks:
        return text
    tagged = [body for tag, body in blocks if tag.lower() in languages]
    return (tagged or [body for _, body in blocks])[-1]


# Fence info string -> language. jsx/tsx/vue are left out on purpose: no
# verifier here parses them, so they never count as JavaScript.
LANG_ALIASES = {
    "javascript": "js", "js": "js", "mjs": "js", "cjs": "js", "node": "js", "nodejs": "js", "ecmascript": "js",
    "typescript": "ts", "ts": "ts", "mts": "ts", "cts": "ts",
    "html": "html", "htm": "html", "xhtml": "html", "html5": "html",
    "css": "css",
    "sql": "sql", "sqlite": "sql", "sqlite3": "sql",
    "python": "python", "py": "python", "python3": "python",
    "lua": "luau", "luau": "luau",
}
_FENCE_OPEN = re.compile(r"^ {0,3}(`{3,}|~{3,})[ \t]*([^\s`{]*)")


def code_blocks(text: str) -> list[dict]:
    """Fenced code blocks of a markdown reply, in order:
    [{"lang": canonical or "", "tag": raw info word, "code": body, "closed": bool}].
    A fence that is never closed (a cut-off reply) runs to the end, so a
    truncated block is still checked and fails instead of being skipped."""
    lines = strip_thinking(text).splitlines()
    blocks: list[dict] = []
    index = 0
    while index < len(lines):
        match = _FENCE_OPEN.match(lines[index])
        if not match or (match.group(1)[0] == "`" and "`" in lines[index][match.end():]):
            index += 1
            continue
        fence, tag = match.group(1), match.group(2).lower().lstrip(".")
        closing = re.compile(r"^ {0,3}" + re.escape(fence[0]) + "{" + str(len(fence)) + r",}[ \t]*$")
        body: list[str] = []
        closed = False
        index += 1
        while index < len(lines):
            if closing.match(lines[index]):
                closed = True
                index += 1
                break
            body.append(lines[index])
            index += 1
        blocks.append({"lang": LANG_ALIASES.get(tag, ""), "tag": tag, "code": "\n".join(body) + "\n", "closed": closed})
    return blocks


def pick_blocks(text: str, lang: str) -> list[str]:
    """Blocks tagged with `lang`; else the untagged blocks; else, when the
    reply has no fence at all, the whole reply. A reply whose only blocks are
    in other languages gives [] (the verifier then fails it)."""
    blocks = code_blocks(text)
    tagged = [block["code"] for block in blocks if block["lang"] == lang]
    if tagged:
        return tagged
    untagged = [block["code"] for block in blocks if not block["tag"]]
    if untagged:
        return untagged
    return [] if blocks else [strip_thinking(text)]


def split_web(text: str) -> tuple[str, str]:
    """HTML and CSS of a markdown reply. Untagged blocks (or a reply with no
    fence) count as HTML when they start with '<', else as CSS."""
    blocks = code_blocks(text)
    html = [block["code"] for block in blocks if block["lang"] == "html"]
    css = [block["code"] for block in blocks if block["lang"] == "css"]
    if not html and not css:
        loose = [block["code"] for block in blocks if not block["tag"]] if blocks else [strip_thinking(text)]
        for part in loose:
            (html if part.lstrip().startswith("<") else css).append(part)
    return "\n".join(html), "\n".join(css)


def join_sql(blocks: list[str]) -> str:
    """SQL blocks in order, each ended with ';' so blocks cannot run together."""
    import sqlite3

    parts = []
    for block in blocks:
        part = block.strip()
        if part:
            parts.append(part if sqlite3.complete_statement(part) else part + "\n;")
    return "\n".join(parts)


def verify_payload(spec: dict, text: str) -> dict:
    """Build the verify job that checks one generated reply."""
    kind = spec["kind"]
    if kind == "verify_math":
        payload: dict = {"answer": strip_thinking(text)[-64000:], "expected": spec["expected"]}
        if "tol" in spec:
            payload["tol"] = spec["tol"]
        return payload
    if kind == "verify_python_syntax":
        return {"code": extract_code(text, ("python", "py", "python3"))}
    if kind in ("verify_js", "verify_ts"):
        return {"blocks": pick_blocks(text, CODE_KINDS[kind])}
    if kind == "verify_tests":
        return {"lang": spec["lang"], "blocks": pick_blocks(text, TEST_LANGS[spec["lang"]])}
    if kind == "verify_html_css":
        html, css = split_web(text)
        payload = {"html": html, "css": css}
        if "require_tags" in spec:
            payload["require_tags"] = spec["require_tags"]
        return payload
    if kind == "verify_sql":
        payload = {"code": join_sql(pick_blocks(text, "sql"))}
        if "schema" in spec:
            payload["schema"] = spec["schema"]
        return payload
    payload = {"code": extract_code(text, ("lua", "luau"))}
    if "expect_stdout" in spec:
        payload["expect_stdout"] = spec["expect_stdout"]
    return payload


def is_truncated(output: object) -> bool:
    """A generate result the model stopped because it hit max_tokens."""
    return isinstance(output, dict) and output.get("finish_reason") == "length"


def looping_answer(text: str) -> bool:
    """A reply stuck repeating itself, so it never becomes training data. Two signs, both strict
    enough for code (test lines look alike but are not the same): half of all 12-word runs are
    repeats, or the reply ends in the same 20-200 characters four times over."""
    words = (text or "").split()
    if len(words) >= 120:
        grams = [tuple(words[i:i + 12]) for i in range(len(words) - 11)]
        if 1 - len(set(grams)) / len(grams) >= 0.5:
            return True
    tail = (text or "").rstrip()[-900:]
    for period in range(20, 201):
        if len(tail) < 4 * period:
            break
        piece = tail[-period:]
        if tail[-2 * period:-period] == piece and tail[-3 * period:-2 * period] == piece and tail[-4 * period:-3 * period] == piece:
            return True
    return False


def check_output(kind: str, payload: dict, ok: bool, output: object) -> tuple[bool, bool | None, dict]:
    """Validate a worker's result. Returns (ok, passed, output to store)."""
    if not isinstance(output, dict):
        return False, None, {"error": "output must be a JSON object"}
    if len(json.dumps(output, ensure_ascii=False)) > JOB_OUTPUT_LIMIT:
        return False, None, {"error": "output is too large"}
    if not ok:
        return False, None, {"error": clip(output.get("error") or "failed", 300)}
    if kind == "generate":
        text = output.get("text")
        if not isinstance(text, str) or not text.strip():
            return False, None, {"error": "generate returned no text"}
        usage = output.get("usage") if isinstance(output.get("usage"), dict) else {}
        return True, None, {
            "text": text,
            "finish_reason": clip(output.get("finish_reason"), 40),
            "model": clip(output.get("model"), 120),
            "usage": {key: int(usage[key]) for key in ("prompt_tokens", "completion_tokens") if isinstance(usage.get(key), int)},
        }
    if kind in VERIFY_KINDS:
        passed = output.get("passed")
        if not isinstance(passed, bool):
            return False, None, {"error": "verify result has no passed flag"}
        clean = {"passed": passed}
        for key in ("error", "detail", "stderr_tail", "stdout_tail", "found", "returncode", "line", "asserts", "stderr"):
            if key in output:
                value = output[key]
                clean[key] = value if isinstance(value, int) and not isinstance(value, bool) else clip(value, 4000)
        seconds = output.get("seconds")
        if isinstance(seconds, (int, float)) and not isinstance(seconds, bool) and 0 <= seconds < 1e6:
            clean["seconds"] = round(float(seconds), 3)
        return True, passed, clean
    if kind == "cross_tests":
        scores = output.get("scores")
        ids = {sample["id"] for sample in payload.get("samples") or []}
        if not isinstance(scores, dict) or not scores or not set(scores) <= ids:
            return False, None, {"error": "cross_tests returned no scores for its samples"}
        clean_scores = {}
        for jid, row in scores.items():
            if not isinstance(row, dict):
                return False, None, {"error": "cross_tests returned a bad score"}
            agreement = row.get("agreement")
            good, total = row.get("passed"), row.get("total")
            if not all(isinstance(v, int) and not isinstance(v, bool) and 0 <= v <= 100000 for v in (good, total)) or good > total:
                return False, None, {"error": "cross_tests returned bad counts"}
            if agreement is not None:
                if not isinstance(agreement, (int, float)) or isinstance(agreement, bool) or not 0 <= agreement <= 1:
                    return False, None, {"error": "cross_tests returned a bad agreement"}
                agreement = round(float(agreement), 4)
            clean_scores[jid] = {"agreement": agreement, "passed": good, "total": total}
        return True, None, {"scores": clean_scores, "detail": clip(output.get("detail") or "", 300)}
    hashes = output.get("hashes")
    texts = payload.get("texts") or []
    if not isinstance(hashes, list) or len(hashes) != len(texts):
        return False, None, {"error": "dedup_hash returned the wrong number of rows"}
    for row in hashes:
        if not isinstance(row, dict) or not isinstance(row.get("sha256"), str) or not _HEX64.match(row["sha256"]):
            return False, None, {"error": "dedup_hash returned a bad sha256"}
        signature = row.get("minhash")
        if not isinstance(signature, list) or len(signature) != MINHASH_PERM or not all(isinstance(v, int) for v in signature):
            return False, None, {"error": "dedup_hash returned a bad minhash"}
    return True, None, {"hashes": [{"sha256": row["sha256"], "minhash": row["minhash"]} for row in hashes], "num_perm": MINHASH_PERM}


class JobQueue:
    """Append-only queue in jobs/queue.jsonl plus jobs/results.jsonl.

    queue.jsonl holds "add" and "finish" events. Both `mesh.py serve` and the
    `mesh.py jobs add` command append to it, so the server reads new lines
    before each lease. Leases are kept in memory only: after a restart every
    leased job is simply queued again.
    """

    def __init__(self, folder: Path = JOBS_DIR):
        self.folder = folder
        self.queue_path = folder / "queue.jsonl"
        self.results_path = folder / "results.jsonl"
        self.lock = threading.RLock()
        self.jobs: dict[str, dict] = {}
        self.order: list[str] = []
        self.leases: dict[str, dict] = {}
        self.by_node: dict[str, str] = {}
        self.node_done: dict[str, dict[str, int]] = {}
        # (at, node, kind, ok) of every finish event that carries a time,
        # newest last. Rates and the queue ETA are counted from these.
        self.finished: deque[tuple[int, str, str, bool]] = deque(maxlen=200000)
        self.offset = 0
        # Generate job ids whose result was cut off (finish_reason "length"),
        # read from results.jsonl, so older log lines count too.
        self.truncated: set[str] = set()
        self.results_offset = 0
        # verify_tests prompt groups: group id -> its generate job ids, every
        # job's children, the groups that already have a cross_tests job (so
        # one is queued once per group, also after a restart), and the
        # cross-check agreement per generate job (from cross_tests finishes).
        self.groups: dict[str, list[str]] = {}
        self.children: dict[str, list[str]] = {}
        self.crossed: set[str] = set()
        self.agreement: dict[str, float | None] = {}
        # Generate leases handed out per source since serve started (fair turns between seed files).
        self.served: dict[str, int] = {}

    # -- file side ---------------------------------------------------------
    def _file_lock(self) -> int | None:
        self.folder.mkdir(parents=True, exist_ok=True)
        lock = self.folder / "queue.lock"
        deadline = time.time() + 10
        while True:
            try:
                return os.open(str(lock), os.O_CREAT | os.O_EXCL | os.O_WRONLY)
            except FileExistsError:
                try:
                    if time.time() - lock.stat().st_mtime > 30:
                        lock.unlink(missing_ok=True)
                        continue
                except OSError:
                    pass
                if time.time() > deadline:
                    raise RuntimeError("The job queue is locked by another command.")
                time.sleep(0.02)

    def _file_unlock(self, handle: int | None) -> None:
        if handle is None:
            return
        os.close(handle)
        (self.folder / "queue.lock").unlink(missing_ok=True)

    def _append(self, path: Path, rows: list[dict]) -> None:
        data = b"".join(
            (json.dumps(row, ensure_ascii=False, separators=(",", ":")) + "\n").encode("utf-8") for row in rows
        )
        handle = self._file_lock()
        try:
            with open(path, "ab", buffering=0) as out:
                out.write(data)
        finally:
            self._file_unlock(handle)

    def sync(self) -> None:
        with self.lock:
            try:
                size = self.queue_path.stat().st_size
            except FileNotFoundError:
                return
            if size < self.offset:
                # The file was replaced. Read it again from the start.
                self.jobs.clear()
                self.order.clear()
                self.leases.clear()
                self.by_node.clear()
                self.node_done.clear()
                self.finished.clear()
                self.groups.clear()
                self.children.clear()
                self.crossed.clear()
                self.agreement.clear()
                self.offset = 0
            if size == self.offset:
                return
            with open(self.queue_path, "rb") as handle:
                handle.seek(self.offset)
                chunk = handle.read(size - self.offset)
            end = chunk.rfind(b"\n")
            if end < 0:
                return
            self.offset += end + 1
            for raw in chunk[:end].split(b"\n"):
                try:
                    event = json.loads(raw.decode("utf-8"))
                except ValueError:
                    continue
                if isinstance(event, dict):
                    self._apply(event)

    def _sync_truncated(self) -> None:
        """Read new results.jsonl lines and note cut-off generations."""
        try:
            size = self.results_path.stat().st_size
        except FileNotFoundError:
            return
        if size < self.results_offset:
            self.truncated.clear()
            self.results_offset = 0
        if size == self.results_offset:
            return
        with open(self.results_path, "rb") as handle:
            handle.seek(self.results_offset)
            chunk = handle.read(size - self.results_offset)
        end = chunk.rfind(b"\n")
        if end < 0:
            return
        self.results_offset += end + 1
        for raw in chunk[:end].split(b"\n"):
            if b'"length"' not in raw:
                continue
            try:
                row = json.loads(raw.decode("utf-8"))
            except ValueError:
                continue
            if isinstance(row, dict) and row.get("kind") == "generate" and row.get("ok") and is_truncated(row.get("output")):
                self.truncated.add(str(row.get("id")))

    def _apply(self, event: dict) -> None:
        op = event.get("op")
        jid = str(event.get("id") or "")
        if op == "add":
            kind = event.get("kind")
            if not jid or jid in self.jobs or kind not in JOB_KINDS or not isinstance(event.get("payload"), dict):
                return
            self.jobs[jid] = {
                "id": jid,
                "kind": kind,
                "payload": event["payload"],
                "parent": str(event.get("parent") or ""),
                "state": "queued",
                "attempts": 0,
                "node": "",
                "passed": None,
                "verified": None,
            }
            self.order.append(jid)
            payload = event["payload"]
            parent = self.jobs[jid]["parent"]
            if parent:
                self.children.setdefault(parent, []).append(jid)
            spec = payload.get("verify")
            if kind == "generate" and isinstance(payload.get("group"), str) and isinstance(spec, dict) and spec.get("kind") == "verify_tests":
                self.groups.setdefault(payload["group"], []).append(jid)
            if kind == "cross_tests" and isinstance(payload.get("group"), str):
                self.crossed.add(payload["group"])
        elif op == "finish":
            job = self.jobs.get(jid)
            if job is None or job["state"] in {"done", "failed"}:
                return
            job["state"] = "done" if event.get("ok") else "failed"
            job["passed"] = event.get("passed") if isinstance(event.get("passed"), bool) else None
            job["node"] = str(event.get("node") or "")
            held = self.leases.pop(jid, None)
            if held and self.by_node.get(held.get("key", held["node"])) == jid:
                self.by_node.pop(held.get("key", held["node"]), None)
            if job["node"]:
                bucket = self.node_done.setdefault(job["node"], {"done": 0, "failed": 0})
                bucket["done" if job["state"] == "done" else "failed"] += 1
            at = event.get("at")
            if isinstance(at, (int, float)) and not isinstance(at, bool) and at > 0:
                self.finished.append((int(at), job["node"], job["kind"], job["state"] == "done"))
            parent = self.jobs.get(job["parent"]) if job["parent"] else None
            if parent is not None and job["kind"] in VERIFY_KINDS:
                parent["verified"] = bool(job["state"] == "done" and job["passed"])
            scores = event.get("scores")
            if job["kind"] == "cross_tests" and job["state"] == "done" and isinstance(scores, dict):
                members = {sample.get("id") for sample in job["payload"].get("samples") or [] if isinstance(sample, dict)}
                for gid, value in scores.items():
                    if gid not in members or gid not in self.jobs:
                        continue
                    if value is None:
                        self.agreement[gid] = None
                    elif isinstance(value, (int, float)) and not isinstance(value, bool) and 0 <= value <= 1:
                        self.agreement[gid] = float(value)

    def _expire(self, now: float) -> None:
        for jid, held in list(self.leases.items()):
            if held["until"] > now:
                continue
            self.leases.pop(jid, None)
            if self.by_node.get(held.get("key", held["node"])) == jid:
                self.by_node.pop(held.get("key", held["node"]), None)
            job = self.jobs.get(jid)
            if job is None or job["state"] != "leased":
                continue
            if job["attempts"] >= KIND_ATTEMPTS.get(job["kind"], MAX_ATTEMPTS):
                self._append(self.queue_path, [{"op": "finish", "id": jid, "ok": False, "node": "", "note": "lease expired", "at": int(now)}])
                self.sync()
                self._maybe_cross(job)
            else:
                job["state"] = "queued"

    # -- public ------------------------------------------------------------
    def add(self, kind: str, raw: dict, parent: str = "") -> str:
        return self.add_many([(kind, raw)], parent)[0]

    def add_many(self, rows: list[tuple[str, dict]], parent: str = "") -> list[str]:
        events = []
        ids = []
        now = int(time.time())
        expanded = [(kind, copy) for kind, raw in rows for copy in expand_samples(kind, raw)]
        for kind, raw in expanded:
            payload = validate_job(kind, raw)
            jid = secrets.token_hex(8)
            event = {"op": "add", "id": jid, "kind": kind, "payload": payload, "parent": parent, "at": now}
            if len(json.dumps(event, ensure_ascii=False)) > JOB_LINE_LIMIT:
                raise JobError("One job is larger than 256 KB")
            events.append(event)
            ids.append(jid)
        if events:
            with self.lock:
                self._append(self.queue_path, events)
                self.sync()
        return ids

    def lease(self, node: str, kinds: object, now: float | None = None, gen_max: object = None,
              test_langs: object = None, slot: object = None) -> dict | None:
        node = machine_key({"name": node, "os": ""})[0]
        if not node or not isinstance(kinds, list):
            return None
        # A worker with job slots names the slot ("g0".."g7" generate,
        # "c0".."c7" checks), so it holds at most one job per slot and at most
        # 16 at once. A worker that sends no slot keeps one job at a time.
        key = node + "/" + slot if isinstance(slot, str) and _SLOT_ID.fullmatch(slot) else node
        wanted = {kind for kind in kinds if kind in JOB_KINDS}
        if not wanted:
            return None
        # The most tokens this worker's generate handler allows. Older
        # workers do not send it and cap at 2048, so a longer job would only
        # come back cut off again.
        allowed = int(_clamp(gen_max, LEGACY_GEN_MAX, GENERATE_MAX_TOKENS, LEGACY_GEN_MAX))
        # Languages this worker's sandbox runs. A worker may run Python
        # tests but have no node that works under its limits.
        langs = {str(lang) for lang in test_langs} if isinstance(test_langs, list) else None
        now = time.time() if now is None else now
        with self.lock:
            self.sync()
            self._expire(now)
            current = self.by_node.get(key)
            if current and current in self.leases:
                # One job per worker slot at a time: a worker that restarted
                # gets the same job back instead of a second one.
                return self._view(current)
            # Checks go first come, first served. Generate jobs take turns by source (seed file):
            # in plain queue order, 4800 math tasks waited a day behind 10000 others.
            first_gen: dict[str, str] = {}
            picked = ""
            for jid in self.order:
                job = self.jobs[jid]
                if job["state"] != "queued" or job["kind"] not in wanted:
                    continue
                if job["kind"] == "generate" and int(job["payload"].get("max_tokens") or 0) > allowed:
                    continue
                if job["kind"] in TEST_JOB_KINDS and langs is not None and job["payload"].get("lang") not in langs:
                    continue
                if job["kind"] != "generate":
                    picked = jid
                    break
                first_gen.setdefault(job_source(job["payload"]), jid)
            if not picked and first_gen:
                # Fewest leases so far wins; a tie goes to the source whose job is oldest (dict order).
                source = min(first_gen, key=lambda name: self.served.get(name, 0))
                self.served[source] = self.served.get(source, 0) + 1
                picked = first_gen[source]
            if picked:
                jid = picked
                job = self.jobs[jid]
                payload = job["payload"]
                spec = payload.get("verify") if isinstance(payload.get("verify"), dict) else {}
                if (job["kind"] == "generate" and spec.get("kind") in LONG_FROM_START and allowed >= LONG_RETRY_TOKENS
                        and int(payload.get("max_tokens") or 0) < LONG_RETRY_TOKENS):
                    job["payload"] = {**payload, "max_tokens": LONG_RETRY_TOKENS, "long_retry": True}
                job["state"] = "leased"
                job["attempts"] += 1
                job["node"] = node
                self.leases[jid] = {"lease": secrets.token_hex(12), "node": node, "key": key, "until": now + LEASE_SECONDS[job["kind"]]}
                self.by_node[key] = jid
                return self._view(jid)
        return None

    def _view(self, jid: str) -> dict:
        job = self.jobs[jid]
        held = self.leases[jid]
        return {
            "id": jid,
            "kind": job["kind"],
            "payload": job["payload"],
            "lease": held["lease"],
            "lease_s": int(LEASE_SECONDS[job["kind"]]),
        }

    def finish(self, body: dict, now: float | None = None) -> tuple[bool, str]:
        jid = str(body.get("id") or "")
        lease = str(body.get("lease") or "")
        node = machine_key({"name": str(body.get("node") or ""), "os": ""})[0]
        now = time.time() if now is None else now
        with self.lock:
            self.sync()
            self._expire(now)
            held = self.leases.get(jid)
            if held is None or not lease or not hmac.compare_digest(held["lease"], lease):
                return False, "No open lease for that job. It may have expired and gone back to the queue."
            if node != held["node"]:
                return False, "That job is leased to another computer."
            job = self.jobs[jid]
            ok, passed, output = check_output(job["kind"], job["payload"], body.get("ok") is True, body.get("output"))
            self._append(
                self.results_path,
                [{"id": jid, "kind": job["kind"], "parent": job["parent"], "node": node, "ok": ok, "passed": passed, "output": output, "at": int(now)}],
            )
            if (job["kind"] == "generate" and not ok and job["attempts"] < MAX_ATTEMPTS
                    and TRANSIENT_MODEL_ERROR.search(str(output.get("error") or ""))):
                # The worker's model could not answer just then. Back to the queue, like an expired
                # lease but at once; no finish event, so after a restart it is simply queued.
                self.leases.pop(jid, None)
                if self.by_node.get(held.get("key", node)) == jid:
                    self.by_node.pop(held.get("key", node), None)
                job["state"] = "queued"
                return True, "the model could not answer; the job went back to the queue"
            event = {"op": "finish", "id": jid, "ok": ok, "passed": passed, "node": node, "at": int(now)}
            if job["kind"] == "cross_tests" and ok:
                event["scores"] = {gid: row["agreement"] for gid, row in output["scores"].items()}
            self._append(self.queue_path, [event])
            self.sync()
            note = "saved"
            spec = job["payload"].get("verify")
            if job["kind"] == "generate" and ok and is_truncated(output):
                self.truncated.add(jid)
                payload = job["payload"]
                if not payload.get("long_retry") and int(payload.get("max_tokens") or 0) < LONG_RETRY_TOKENS:
                    # Cut off by max_tokens: ask once more with room to finish
                    # instead of checking half an answer.
                    retry = {key: value for key, value in payload.items() if key != "samples"}
                    retry.update({"max_tokens": LONG_RETRY_TOKENS, "long_retry": True})
                    try:
                        self.add("generate", retry, parent=jid)
                        note = f"saved; truncated at {payload.get('max_tokens')} tokens, long retry queued"
                    except JobError as exc:
                        note = "saved; truncated, no retry: " + str(exc)[:120]
                else:
                    note = "saved; truncated, not verified"
                job["verified"] = False
            elif job["kind"] == "generate" and ok and looping_answer(str(output.get("text") or "")):
                # A reply stuck repeating itself is never checked nor trained on: the model would
                # learn the loop. It still counts as done, so the prompt is not asked again.
                job["verified"] = False
                note = "saved; the reply repeats itself, not verified"
            elif job["kind"] == "generate" and ok and isinstance(spec, dict):
                try:
                    self.add(spec["kind"], verify_payload(spec, output["text"]), parent=jid)
                    note = "saved; " + spec["kind"] + " queued"
                except (JobError, KeyError) as exc:
                    job["verified"] = False
                    note = "saved; not verified: " + str(exc)[:120]
            crossed = self._maybe_cross(job)
            if crossed:
                note += "; " + crossed
            return True, note

    def _maybe_cross(self, job: dict) -> str:
        """Queue one cross_tests job for a verify_tests prompt group once
        every sample of it is settled (generated, retried, checked or
        failed) and at least two passed their own tests. Never twice for a
        group: `crossed` is rebuilt from the queue file, so a restart does
        not queue it again either. cross_tests itself never queues anything."""
        gen = job if job["kind"] == "generate" else self.jobs.get(job["parent"]) if job["parent"] else None
        if gen is None or gen["kind"] != "generate":
            return ""
        payload = gen["payload"]
        group, spec = payload.get("group"), payload.get("verify")
        if not isinstance(group, str) or not isinstance(spec, dict) or spec.get("kind") != "verify_tests" or group in self.crossed:
            return ""
        members = [self.jobs[gid] for gid in self.groups.get(group, []) if gid in self.jobs]
        originals = [member for member in members if not member["payload"].get("long_retry")]
        if not originals or len(originals) < int(originals[0]["payload"].get("samples") or 1):
            return ""
        candidates = []
        for member in members:
            if member["state"] in {"queued", "leased"}:
                return ""
            kids = [self.jobs[cid] for cid in self.children.get(member["id"], []) if cid in self.jobs]
            if any(kid["state"] in {"queued", "leased"} for kid in kids):
                return ""
            if member["state"] != "done" or not member["verified"] or member["id"] in self.truncated:
                continue
            checks = [kid for kid in kids if kid["kind"] == "verify_tests" and kid["state"] == "done" and kid["passed"]]
            blocks = [block for block in checks[-1]["payload"].get("blocks") or [] if block.strip()] if checks else []
            if len(blocks) == 1:
                candidates.append({"id": member["id"], "code": blocks[0]})
        if len(candidates) < 2:
            return ""
        try:
            self.add("cross_tests", {"lang": spec["lang"], "group": group, "samples": candidates[:MAX_SAMPLES]})
        except JobError as exc:
            return "no cross-check: " + str(exc)[:120]
        return f"cross_tests queued for {len(candidates)} samples"

    def summary(self, now: float | None = None) -> dict:
        now = time.time() if now is None else now
        with self.lock:
            try:
                self.sync()
                self._expire(now)
            except (OSError, RuntimeError):
                pass
            try:
                self._sync_truncated()
            except OSError:
                pass
            kinds = {kind: {"queued": 0, "leased": 0, "done": 0, "failed": 0, "passed": 0} for kind in JOB_KINDS}
            kinds["generate"]["truncated"] = 0
            # Per verify kind: generations cut off, how many got a long
            # retry, and how many stay cut off for good.
            cut = {kind: {"truncated": 0, "retried": 0, "truncated_final": 0} for kind in VERIFY_KINDS}
            retried = {job["parent"] for job in self.jobs.values() if job["kind"] == "generate" and job["payload"].get("long_retry") and job["parent"]}
            exportable = 0
            # Verify jobs that queued generate jobs will add when they finish.
            coming = {kind: 0 for kind in JOB_KINDS}
            for job in self.jobs.values():
                row = kinds[job["kind"]]
                row[job["state"]] += 1
                if job["passed"]:
                    row["passed"] += 1
                if job["kind"] == "generate" and job["verified"] and job["id"] not in self.truncated:
                    exportable += 1
                if job["kind"] == "generate" and job["id"] in self.truncated:
                    kinds["generate"]["truncated"] += 1
                    spec = job["payload"].get("verify")
                    bucket = cut.get(spec.get("kind")) if isinstance(spec, dict) else None
                    if bucket is not None:
                        bucket["truncated"] += 1
                        bucket["retried" if job["id"] in retried else "truncated_final"] += 1
                if job["kind"] == "generate" and job["state"] in {"queued", "leased"}:
                    spec = job["payload"].get("verify")
                    if isinstance(spec, dict) and spec.get("kind") in coming:
                        coming[spec["kind"]] += 1
            empty = {"done": 0, "failed": 0, "now": "", "running": [], "hour_done": 0, "hour_failed": 0, "last_s": None}
            nodes = {}
            for name, bucket in self.node_done.items():
                nodes[name] = dict(empty, done=bucket["done"], failed=bucket["failed"], running=[])
            for jid, held in self.leases.items():
                row = nodes.setdefault(held["node"], dict(empty, running=[]))
                row["running"].append(self.jobs[jid]["kind"])
            for row in nodes.values():
                # "now" stays one kind for older pages; "running" has every
                # job a computer's slots hold now.
                row["running"].sort(key=lambda kind: (kind != "generate", kind))
                row["now"] = row["running"][0] if row["running"] else ""
            hour = {kind: 0 for kind in JOB_KINDS}
            for at, node, kind, good in reversed(self.finished):
                age = now - at
                if node in nodes and nodes[node]["last_s"] is None and age >= -60:
                    nodes[node]["last_s"] = max(0, int(age))
                if not -60 <= age < RATE_WINDOW:
                    continue
                hour[kind] = hour.get(kind, 0) + 1
                if node in nodes:
                    nodes[node]["hour_done" if good else "hour_failed"] += 1
            verifiers = {}
            for kind in VERIFY_KINDS:
                row = kinds[kind]
                checked = row["done"] + row["failed"]
                verifiers[kind] = {
                    "checked": checked,
                    "passed": row["passed"],
                    "pct": round(100 * row["passed"] / checked) if checked else None,
                    **cut[kind],
                }
            scored = [value for value in self.agreement.values() if value is not None]
            verifiers["verify_tests"]["agreement"] = {
                "groups": len(self.crossed),
                "scored": len(scored),
                "avg": round(sum(scored) / len(scored), 3) if scored else None,
                "agree": sum(1 for value in scored if value >= AGREEMENT_MIN),
                "min": AGREEMENT_MIN,
            }
            return {
                "kinds": kinds,
                "nodes": nodes,
                "verified_rows": exportable,
                "total": len(self.jobs),
                "verifiers": verifiers,
                "rate": queue_rate(kinds, coming, hour),
                "at": int(now),
            }

    def _results(self) -> tuple[dict[str, str], dict[str, dict]]:
        """Generated texts by job id, and the last verifier verdict by the
        generate job it checked. A generation cut off by max_tokens
        (finish_reason "length") is left out of the texts, also for older
        log lines, so it is never exported, neither as chosen nor as
        rejected."""
        texts: dict[str, str] = {}
        verdicts: dict[str, dict] = {}
        try:
            with open(self.results_path, "rb") as handle:
                for raw in handle:
                    try:
                        row = json.loads(raw.decode("utf-8"))
                    except ValueError:
                        continue
                    if not isinstance(row, dict):
                        continue
                    output = row.get("output") if isinstance(row.get("output"), dict) else {}
                    if row.get("kind") == "generate" and row.get("ok"):
                        text = output.get("text")
                        if is_truncated(output):
                            texts.pop(str(row.get("id")), None)
                        elif isinstance(text, str):
                            texts[str(row.get("id"))] = text
                    elif row.get("kind") in VERIFY_KINDS and row.get("parent"):
                        verdicts[str(row["parent"])] = {
                            "ok": row.get("ok") is True,
                            "passed": row.get("passed") if isinstance(row.get("passed"), bool) else None,
                            "error": clip(output.get("error") or "", 300),
                        }
        except FileNotFoundError:
            pass
        return texts, verdicts

    def _ranked(self, jids: list[str], texts: dict[str, str]) -> list[str]:
        """Candidates of one prompt, best first: with a cross-check, the
        highest agreement, then the shortest answer; otherwise queue order."""
        if not any(jid in self.agreement for jid in jids):
            return list(jids)
        def key(jid: str) -> tuple[float, int]:
            value = self.agreement.get(jid)
            return (-(value if value is not None else -1.0), len(strip_thinking(texts.get(jid, ""))))
        return sorted(jids, key=key)

    def export(self, name: str, per_prompt: int = 1, require_agreement: bool = False) -> tuple[Path, int]:
        """Write only verified generate rows, as chat training rows, into
        the jobs folder. With rejection sampling, at most `per_prompt`
        verified candidates of one prompt group are kept (0 keeps all);
        identical answers are written once. verify_tests groups with a
        cross-check give the highest agreement first (ties: shortest).
        require_agreement keeps only candidates whose cross-check agreement
        is at least AGREEMENT_MIN (rows without a cross-check are left out)."""
        base = Path(str(name)).name
        if not base.endswith(".jsonl") or base in {"queue.jsonl", "results.jsonl"}:
            raise JobError("Name the export file like verified.jsonl")
        with self.lock:
            self.sync()
            texts, _ = self._results()
            rows = []
            kept: dict[str, list[str]] = {}
            by_group: dict[str, list[str]] = {}
            for jid in self.order:
                job = self.jobs[jid]
                if job["kind"] != "generate" or not job["verified"] or jid not in texts:
                    continue
                if require_agreement and (self.agreement.get(jid) is None or self.agreement[jid] < AGREEMENT_MIN):
                    continue
                by_group.setdefault(str(job["payload"].get("group") or jid), []).append(jid)
            for key, members in by_group.items():
                for jid in self._ranked(members, texts):
                    job = self.jobs[jid]
                    payload = job["payload"]
                    answer = strip_thinking(texts[jid])
                    group = kept.setdefault(key, [])
                    if answer in group or (per_prompt > 0 and len(group) >= per_prompt) or looping_answer(answer):
                        continue
                    group.append(answer)
                    rows.append(
                        {
                            "messages": payload["messages"] + [{"role": "assistant", "content": answer}],
                            "skill": job_skill(payload),
                            "verified_by": (payload.get("verify") or {}).get("kind", ""),
                            "node": job["node"],
                            "job": jid,
                            **({"group": payload["group"]} if "group" in payload else {}),
                            **({"agreement": self.agreement[jid]} if self.agreement.get(jid) is not None else {}),
                            **({"meta": payload["meta"]} if "meta" in payload else {}),
                        }
                    )
        self.folder.mkdir(parents=True, exist_ok=True)
        out = self.folder / base
        _write_jsonl_atomic(out, rows)
        return out, len(rows)

    def export_pairs(self, target: str) -> tuple[Path, int]:
        """Preference rows for DPO/KTO: for one prompt, a candidate the
        verifier passed (chosen) against one it ran and failed (rejected).
        A candidate whose verifier could not run (unavailable, sandbox
        error) or never ran is neither, and a cut-off reply is never used.
        With a cross-check, chosen goes best agreement first, and a sample
        that passed its own tests but agrees below AGREEMENT_MIN with the
        others is not chosen. Candidates pair up once each. The file goes
        where the user names it, but never into the live training folder."""
        out = pairs_target(target, self.folder)
        with self.lock:
            self.sync()
            texts, verdicts = self._results()
            groups: dict[str, dict] = {}
            for jid in self.order:
                job = self.jobs[jid]
                verdict = verdicts.get(jid)
                if job["kind"] != "generate" or jid not in texts or not verdict or not verdict["ok"] or verdict["passed"] is None:
                    continue
                payload = job["payload"]
                key = str(payload.get("group") or hashlib.sha256(
                    json.dumps([payload["messages"], payload.get("verify")], sort_keys=True, ensure_ascii=False).encode("utf-8")
                ).hexdigest()[:16])
                bucket = groups.setdefault(key, {"chosen": [], "rejected": []})
                answer = strip_thinking(texts[jid])
                if verdict["passed"] and self.agreement.get(jid) is not None and self.agreement[jid] < AGREEMENT_MIN:
                    continue
                if verdict["passed"] and looping_answer(answer):
                    continue  # never "chosen"; a loop is a fine example of what not to do
                side = bucket["chosen" if verdict["passed"] else "rejected"]
                if answer and all(answer != seen for _, seen in bucket["chosen"] + bucket["rejected"]):
                    side.append((jid, answer))
            rows = []
            for key, bucket in groups.items():
                order = self._ranked([jid for jid, _ in bucket["chosen"]], texts)
                bucket["chosen"].sort(key=lambda item: order.index(item[0]))
                for (good, good_text), (bad, bad_text) in zip(bucket["chosen"], bucket["rejected"]):
                    payload = self.jobs[good]["payload"]
                    rows.append(
                        {
                            "prompt": payload["messages"],
                            "chosen": [{"role": "assistant", "content": good_text}],
                            "rejected": [{"role": "assistant", "content": bad_text}],
                            "skill": job_skill(payload),
                            "verified_by": (payload.get("verify") or {}).get("kind", ""),
                            "rejected_error": verdicts[bad]["error"],
                            "group": key,
                            "chosen_job": good,
                            "rejected_job": bad,
                            **({"chosen_agreement": self.agreement[good]} if self.agreement.get(good) is not None else {}),
                            **({"meta": payload["meta"]} if "meta" in payload else {}),
                        }
                    )
        _write_jsonl_atomic(out, rows)
        return out, len(rows)


def queue_rate(kinds: dict, coming: dict, hour: dict) -> dict:
    """Jobs per hour over the last RATE_WINDOW, and how long the open jobs
    take at that rate. Each kind runs in parallel on its own workers, so the
    queue ETA is the slowest kind. A verify job that a queued generate job
    still has to make can only finish after that generate job, so such a kind
    takes at least as long as generate. A kind with open jobs but nothing
    finished in the window is "stalled" (no computer takes it now) and has no
    ETA."""
    def own_eta(left: int, done: int) -> int | None:
        return int(left * RATE_WINDOW / done) if left and done else 0 if not left else None

    gen = kinds.get("generate") or {}
    gen_eta = own_eta(int(gen.get("queued") or 0) + int(gen.get("leased") or 0), int(hour.get("generate") or 0))
    by_kind = {}
    stalled = []
    etas = []
    left_total = 0
    for kind, row in kinds.items():
        open_now = int(row["queued"]) + int(row["leased"])
        later = int(coming.get(kind) or 0)
        done = int(hour.get(kind) or 0)
        if not open_now and not later and not done:
            continue
        left_total += open_now + later
        kind_eta = own_eta(open_now, done)
        if open_now and not done:
            stalled.append(kind)
        if later:
            # Waits on generate. Unknown when generate itself has no rate.
            kind_eta = None if gen_eta is None or kind_eta is None else max(kind_eta, gen_eta)
        if kind_eta is not None and (open_now or later):
            etas.append(kind_eta)
        # later: verify jobs that queued generate jobs still have to make.
        by_kind[kind] = {
            "left": open_now + later,
            "later": later,
            "per_hour": round(done * 3600 / RATE_WINDOW, 1),
            "eta_s": kind_eta if open_now or later else 0,
        }
    return {
        "window_s": RATE_WINDOW,
        "per_hour": round(sum(hour.values()) * 3600 / RATE_WINDOW, 1),
        "left": left_total,
        "eta_s": max(etas) if etas else (0 if not left_total else None),
        "stalled": stalled,
        "by_kind": by_kind,
    }


def _write_jsonl_atomic(out: Path, rows: list[dict]) -> None:
    temp = out.with_name(out.name + ".tmp")
    try:
        with open(temp, "w", encoding="utf-8", newline="\n") as handle:
            for row in rows:
                handle.write(json.dumps(row, ensure_ascii=False) + "\n")
        os.replace(temp, out)
    finally:
        temp.unlink(missing_ok=True)


def job_skill(payload: dict) -> str:
    """Training domain of a generate row (prepare.py reads `skill`)."""
    meta = payload.get("meta") if isinstance(payload.get("meta"), dict) else {}
    if meta.get("skill") in SKILLS:
        return str(meta["skill"])
    kind = (payload.get("verify") or {}).get("kind", "")
    if kind == "verify_luau":
        return "luau"
    if kind == "verify_math":
        return "math"
    return "code" if kind in CODE_KINDS or kind == "verify_tests" else "general"


def job_source(payload: dict) -> str:
    """Where a generate job came from (its seed file), for fair turns between sources."""
    meta = payload.get("meta") if isinstance(payload.get("meta"), dict) else {}
    source = meta.get("source")
    if isinstance(source, str) and source.strip():
        return source.strip()[:120]
    return "verify:" + str((payload.get("verify") or {}).get("kind") or "none")


def expand_samples(kind: str, raw: object) -> list[object]:
    """A generate row with "samples": N becomes N jobs that share a group id
    and differ by seed (and by a slightly warmer default temperature)."""
    if kind != "generate" or not isinstance(raw, dict) or isinstance(raw.get("group"), str):
        return [raw]
    samples = int(_clamp(raw.get("samples"), 1, MAX_SAMPLES, 1))
    if samples <= 1:
        return [raw]
    group = secrets.token_hex(8)
    copies = []
    for index in range(samples):
        copy = dict(raw)
        copy.update({"samples": samples, "group": group, "sample": index, "seed": secrets.randbelow(2 ** 31)})
        if raw.get("temperature") is None:
            copy["temperature"] = 0.8
        copies.append(copy)
    return copies


def training_roots() -> list[Path]:
    # The default folder is always refused, also when NEURO_AI_DATA or
    # training.json points somewhere else.
    roots = [ROOT.parent / "neuro-ai-data" / "training", data_root() / "training"]
    env = os.environ.get("NEURO_AI_DATA", "").strip()
    if env:
        roots.append(Path(env) / "training")
    return list(dict.fromkeys(roots))


def pairs_target(name: str, jobs_folder: Path) -> Path:
    """Where `jobs export --pairs` may write. A bare file name goes into the
    jobs folder. Any path inside a training folder is refused, also through
    a link, so preference rows can never land in the SFT data by accident."""
    raw = str(name or "").strip()
    if not raw:
        raise JobError("Name the pairs file, like pairs.jsonl")
    path = Path(raw)
    if path.suffix.lower() != ".jsonl":
        raise JobError("Name the pairs file like pairs.jsonl")
    if not path.is_absolute() and len(path.parts) == 1:
        jobs_folder.mkdir(parents=True, exist_ok=True)
        path = jobs_folder / path
    try:
        resolved = path.resolve()
        roots = [root.resolve() for root in training_roots()]
        reserved = {(jobs_folder / item).resolve() for item in ("queue.jsonl", "results.jsonl")}
    except (OSError, RuntimeError):
        raise JobError("The pairs path could not be checked. Nothing was written.")
    folded = os.path.normcase(str(resolved))
    for root in roots:
        top = os.path.normcase(str(root))
        if folded == top or folded.startswith(top.rstrip("\\/") + os.sep):
            raise JobError(f"Refused: {resolved} is inside the training folder {root}. Write pairs somewhere else.")
    if resolved in reserved:
        raise JobError("Refused: that is the job queue itself.")
    if resolved.is_dir():
        raise JobError("The pairs path is a folder. Name a .jsonl file.")
    if not resolved.parent.is_dir():
        raise JobError(f"The folder {resolved.parent} does not exist.")
    return resolved


_queue_lock = threading.Lock()
_queue: JobQueue | None = None


def job_queue() -> JobQueue:
    global _queue
    with _queue_lock:
        if _queue is None:
            _queue = JobQueue()
        return _queue


# -- worker-side handlers ----------------------------------------------------


class _NoRedirect(urllib.request.HTTPRedirectHandler):
    def redirect_request(self, *args, **kwargs):  # noqa: D401 - urllib hook
        return None


# No proxy and no redirects: a job reply or a token can only go to the exact
# address named here.
_DIRECT = urllib.request.build_opener(urllib.request.ProxyHandler({}), _NoRedirect)


def local_llm_base() -> str:
    raw = os.environ.get("NEURO_LOCAL_LLM", "").strip()
    port = 8080
    if raw.isdigit():
        port = int(raw)
    elif raw:
        parsed = urlparse(raw if "://" in raw else "http://" + raw)
        try:
            if parsed.hostname in {"127.0.0.1", "localhost"} and parsed.port:
                port = parsed.port
        except ValueError:
            port = 8080
    if not 1 <= port <= 65535:
        port = 8080
    return f"http://127.0.0.1:{port}"


def local_model_ready() -> bool:
    try:
        with _DIRECT.open(local_llm_base() + "/v1/models", timeout=2) as response:
            return response.status == 200 and isinstance(read_reply(response), dict)
    except (OSError, urllib.error.URLError, ValueError):
        return False


def luau_path() -> str | None:
    override = os.environ.get("NEURO_LUAU", "").strip()
    if override:
        try:
            if Path(override).is_file():
                return override
        except OSError:
            pass
        found = shutil.which(override)
        if found:
            return found
    return luau_binary()


_kinds_cache: dict = {"at": 0.0, "kinds": []}
# The worker's job slots run in parallel threads. Everything below is read by
# the /hello thread too, so it is only touched under _job_lock.
_job_lock = threading.Lock()
_job_state: dict = {"on": False, "done": 0, "failed": 0, "paused": "", "current": "", "told": ""}
_job_running: dict[str, str] = {}   # slot id -> kind it runs now
_job_held: set[str] = set()         # job ids the slots run now
_job_paused: dict[str, str] = {}    # slot id -> why it does not lease now
_slot_cache: dict = {"at": 0.0, "gen": 0, "cpu": 0}


def job_state_view() -> dict:
    """The worker's job state for /hello: counters, the kinds its slots run
    now, and how many slots of each sort are busy."""
    with _job_lock:
        running = sorted(_job_running.items())
        gen_busy = sum(1 for slot, _ in running if slot.startswith("g"))
        kinds = sorted((kind for _, kind in running), key=lambda kind: (kind != "generate", kind))
        reasons = sorted({why for slot, why in _job_paused.items() if why and slot not in _job_running})
        return {
            "on": bool(_job_state.get("on")),
            "done": int(_job_state.get("done") or 0),
            "failed": int(_job_state.get("failed") or 0),
            "paused": "; ".join(reasons) if reasons else str(_job_state.get("paused") or ""),
            "current": kinds[0] if kinds else "",
            "running": kinds,
            "slots": {
                "generate": {"busy": gen_busy, "total": max(gen_busy, int(_slot_cache["gen"]))},
                "check": {"busy": len(running) - gen_busy, "total": max(len(running) - gen_busy, int(_slot_cache["cpu"]))},
            },
        }


def _slot_env(name: str) -> int | None:
    raw = os.environ.get(name, "").strip()
    return min(int(raw), SLOT_CAP) if raw.isdigit() else None


def _int_field(value: object) -> int | None:
    return value if isinstance(value, int) and not isinstance(value, bool) and value > 0 else None


def model_parallel_slots() -> int | None:
    """How many requests the local llama-server serves at once: GET /props
    (total_slots), else the length of GET /slots. None when it does not say."""
    base = local_llm_base()
    try:
        with _DIRECT.open(base + "/props", timeout=2) as response:
            props = read_reply(response)
        if isinstance(props, dict):
            settings = props.get("default_generation_settings")
            settings = settings if isinstance(settings, dict) else {}
            for value in (props.get("total_slots"), settings.get("total_slots"), settings.get("n_slots")):
                if _int_field(value):
                    return value
    except (OSError, urllib.error.URLError, ValueError):
        pass
    try:
        with _DIRECT.open(base + "/slots", timeout=2) as response:
            slots = read_reply(response)
        if isinstance(slots, list) and slots:
            return len(slots)
    except (OSError, urllib.error.URLError, ValueError):
        pass
    return None


def gen_slot_count(model_slots: int | None) -> int:
    """NEURO_GEN_SLOTS, else the model server's own slot count capped at 4, else 1."""
    override = _slot_env("NEURO_GEN_SLOTS")
    if override is not None:
        return override
    return max(1, min(GEN_SLOT_DEFAULT_CAP, model_slots)) if model_slots else 1


def cpu_slot_count(cpus: int | None = None) -> int:
    """NEURO_CPU_SLOTS, else one check slot per 4 cores, 1 to 4."""
    override = _slot_env("NEURO_CPU_SLOTS")
    if override is not None:
        return override
    return max(1, min(4, int(cpus or os.cpu_count() or 1) // 4))


def slot_plan(refresh: bool = False) -> dict:
    """How many generate and check slots lease jobs now. Asked again every
    minute, so a model started later gets its generate slots."""
    now = time.time()
    with _job_lock:
        if not refresh and now - float(_slot_cache["at"]) < 60:
            return {"gen": int(_slot_cache["gen"]), "cpu": int(_slot_cache["cpu"])}
    kinds = worker_kinds()
    gen = gen_slot_count(model_parallel_slots()) if "generate" in kinds else 0
    cpu = cpu_slot_count() if any(kind != "generate" for kind in kinds) else 0
    with _job_lock:
        _slot_cache.update({"at": now, "gen": gen, "cpu": cpu})
    return {"gen": gen, "cpu": cpu}


def worker_kinds() -> list[str]:
    now = time.time()
    if now - float(_kinds_cache["at"]) < 60:
        return list(_kinds_cache["kinds"])
    kinds = ["verify_python_syntax", "verify_html_css", "verify_math", "dedup_hash"]
    if sqlite_ready():
        kinds.insert(2, "verify_sql")
    node = node_path()
    if node:
        kinds.insert(1, "verify_js")
        if node_strips_types(node):
            kinds.insert(2, "verify_ts")
    if luau_path():
        kinds.insert(0, "verify_luau")
    if local_model_ready():
        kinds.insert(0, "generate")
    # Only a computer that can sandbox model-written code runs its tests
    # (Windows Job Object, or NEURO_RUN_TESTS=1 with rlimits elsewhere).
    if sandbox_status()["langs"]:
        kinds.extend(TEST_JOB_KINDS)
    _kinds_cache.update({"at": now, "kinds": kinds})
    return list(kinds)


def job_generate(payload: dict) -> tuple[bool, dict]:
    body = {
        "messages": payload["messages"],
        "max_tokens": min(int(payload.get("max_tokens") or 1024), GENERATE_MAX_TOKENS),
        "temperature": payload.get("temperature", 0.7),
        "top_p": payload.get("top_p", 0.95),
        "stream": False,
        "chat_template_kwargs": {"enable_thinking": False},
    }
    if isinstance(payload.get("seed"), int):
        body["seed"] = payload["seed"]
    request = urllib.request.Request(
        local_llm_base() + "/v1/chat/completions",
        data=json.dumps(body).encode("utf-8"),
        headers={"Content-Type": "application/json"},
        method="POST",
    )
    try:
        with _DIRECT.open(request, timeout=GENERATE_TIMEOUT) as response:
            reply = read_reply(response)
    except urllib.error.HTTPError as exc:
        # Keep llama-server's own reason ("context size exceeded", "out of memory" ...): a bare
        # "HTTP Error 500" hid why 846 jobs failed.
        return False, {"error": clip(f"HTTP Error {exc.code}: {error_detail(exc)}", 300)}
    choices = reply.get("choices") if isinstance(reply, dict) else None
    first = choices[0] if isinstance(choices, list) and choices and isinstance(choices[0], dict) else {}
    message = first.get("message") if isinstance(first.get("message"), dict) else {}
    text = message.get("content")
    if not isinstance(text, str) or not text.strip():
        return False, {"error": "The local model returned no text."}
    usage = reply.get("usage") if isinstance(reply.get("usage"), dict) else {}
    return True, {
        "text": text[:120000],
        "finish_reason": clip(first.get("finish_reason"), 40),
        "model": clip(reply.get("model"), 120),
        "usage": {key: usage[key] for key in ("prompt_tokens", "completion_tokens") if isinstance(usage.get(key), int)},
    }


def _tail(path: Path, limit: int) -> str:
    try:
        size = path.stat().st_size
        with open(path, "rb") as handle:
            handle.seek(max(0, size - limit))
            return handle.read().decode("utf-8", errors="replace")
    except OSError:
        return ""


def job_verify_luau(payload: dict) -> tuple[bool, dict]:
    binary = luau_path()
    if not binary:
        return False, {"error": "luau is not installed on this computer."}
    folder = Path(tempfile.mkdtemp(prefix="neuro-luau-"))
    try:
        script = folder / "job.luau"
        script.write_text(payload["code"], encoding="utf-8")
        out_path, err_path = folder / "out.txt", folder / "err.txt"
        reason = ""
        flags = getattr(subprocess, "CREATE_NO_WINDOW", 0) if os.name == "nt" else 0
        with open(out_path, "wb") as out, open(err_path, "wb") as err:
            process = subprocess.Popen(
                [binary, str(script)],
                cwd=str(folder),
                stdin=subprocess.DEVNULL,
                stdout=out,
                stderr=err,
                creationflags=flags,
            )
            deadline = time.time() + LUAU_TIMEOUT
            while process.poll() is None:
                if time.time() > deadline:
                    reason = f"timed out after {LUAU_TIMEOUT} s"
                    break
                try:
                    if out_path.stat().st_size + err_path.stat().st_size > LUAU_OUTPUT_CAP:
                        reason = "printed more than 1 MB"
                        break
                except OSError:
                    pass
                time.sleep(0.05)
            if reason:
                process.kill()
            try:
                process.wait(5)
            except subprocess.TimeoutExpired:
                pass
        stdout = _tail(out_path, 4000)
        stderr = _tail(err_path, 2000)
        code = process.returncode if process.returncode is not None else -1
        expected = payload.get("expect_stdout")
        passed = not reason and code == 0 and (expected is None or stdout.strip() == str(expected).strip())
        result = {"passed": passed, "returncode": int(code), "stdout_tail": stdout, "stderr_tail": stderr}
        if reason:
            result["error"] = reason
        return True, result
    finally:
        shutil.rmtree(folder, ignore_errors=True)


def job_verify_python_syntax(payload: dict) -> tuple[bool, dict]:
    # compile() only builds a code object. It is never run.
    try:
        compile(payload["code"], "<job>", "exec", dont_inherit=True)
    except SyntaxError as exc:
        return True, {"passed": False, "error": clip(f"{type(exc).__name__}: {exc.msg}", 300), "line": int(exc.lineno or 0)}
    except (ValueError, RecursionError, MemoryError, OverflowError) as exc:
        return True, {"passed": False, "error": clip(f"{type(exc).__name__}: {exc}", 300)}
    return True, {"passed": True}


# -- coding verifiers: parse only, never run the generated code --------------


def node_path() -> str | None:
    override = os.environ.get("NEURO_NODE", "").strip()
    if override:
        try:
            if Path(override).is_file():
                return override
        except OSError:
            pass
        return shutil.which(override)
    return shutil.which("node")


def _node_env(folder: Path) -> dict:
    """A scrubbed environment: no PATH, no NODE_OPTIONS / NODE_PATH, temp and
    home inside the job folder."""
    env = {"TEMP": str(folder), "TMP": str(folder), "TMPDIR": str(folder), "HOME": str(folder),
           "USERPROFILE": str(folder), "NODE_NO_WARNINGS": "1"}
    if os.name == "nt":
        env["SYSTEMROOT"] = os.environ.get("SYSTEMROOT", r"C:\Windows")
    return env


def _run_fixed(argv: list[str], folder: Path, timeout: float, output_cap: int) -> dict:
    """Run a fixed argument list (never a shell) with a time limit and an
    output cap. Output goes to files in `folder`."""
    out_path, err_path = folder / "out.txt", folder / "err.txt"
    reason = ""
    flags = getattr(subprocess, "CREATE_NO_WINDOW", 0) if os.name == "nt" else 0
    with open(out_path, "wb") as out, open(err_path, "wb") as err:
        process = subprocess.Popen(
            argv, cwd=str(folder), stdin=subprocess.DEVNULL, stdout=out, stderr=err,
            env=_node_env(folder), shell=False, creationflags=flags,
        )
        deadline = time.time() + timeout
        while process.poll() is None:
            if time.time() > deadline:
                reason = f"timed out after {timeout:g} s"
                break
            try:
                if out_path.stat().st_size + err_path.stat().st_size > output_cap:
                    reason = "printed too much"
                    break
            except OSError:
                pass
            time.sleep(0.02)
        if reason:
            process.kill()
        try:
            process.wait(5)
        except subprocess.TimeoutExpired:
            pass
    code = process.returncode if process.returncode is not None else -1
    stderr = _tail(err_path, 2000)
    for spelled in {str(folder), str(folder).replace("\\", "/")}:
        stderr = stderr.replace(spelled, "<job>")
    return {"returncode": int(code), "stdout": _tail(out_path, 2000), "stderr": stderr, "error": reason}


_node_probe: dict[str, bool] = {}


def node_strips_types(binary: str) -> bool:
    """True when this node has module.stripTypeScriptTypes (22.13+ / 23.2+).
    The probe runs a fixed expression, never job code."""
    if binary in _node_probe:
        return _node_probe[binary]
    folder = Path(tempfile.mkdtemp(prefix="neuro-node-"))
    try:
        ran = _run_fixed([binary, "-p", "typeof require('node:module').stripTypeScriptTypes"], folder, NODE_TIMEOUT, 4096)
        found = not ran["error"] and ran["returncode"] == 0 and ran["stdout"].strip() == "function"
    except OSError:
        found = False
    finally:
        shutil.rmtree(folder, ignore_errors=True)
    _node_probe[binary] = found
    return found


def _node_check(binary: str, script: Path, folder: Path, label: str) -> tuple[bool | None, str]:
    """node --check on one file. (True, "") parsed; (False, why) a syntax
    error in the code; (None, why) node itself could not give a verdict."""
    ran = _run_fixed([binary, "--check", str(script)], folder, NODE_TIMEOUT, NODE_OUTPUT_CAP)
    if ran["error"]:
        return None, ran["error"]
    if ran["returncode"] == 0:
        return True, ""
    if "SyntaxError" in ran["stderr"]:
        return False, label + ": " + clip(ran["stderr"].strip(), 1500)
    if ran["returncode"] == 9:
        return None, "unavailable: this node does not accept --check"
    return None, clip(f"node exited {ran['returncode']}: {ran['stderr'].strip()}", 400)


# Fixed helper for verify_ts. It reads one .ts file and writes the same code
# with the types blanked out; it never evaluates the input.
TS_STRIP_HELPER = r"""'use strict';
const fs = require('node:fs');
const mod = require('node:module');
if (typeof mod.stripTypeScriptTypes !== 'function') process.exit(3);
let out;
try {
  out = mod.stripTypeScriptTypes(fs.readFileSync(process.argv[2], 'utf8'));
} catch (e) {
  process.stderr.write(String((e && e.message) || e).slice(0, 1500));
  process.exit(e && e.code === 'ERR_INVALID_TYPESCRIPT_SYNTAX' ? 2 : e && e.code === 'ERR_UNSUPPORTED_TYPESCRIPT_SYNTAX' ? 4 : 5);
}
fs.writeFileSync(process.argv[3], out);
"""


def _verify_node_blocks(payload: dict, typescript: bool) -> tuple[bool, dict]:
    name = "TypeScript" if typescript else "JavaScript"
    blocks = [block for block in payload.get("blocks") or [] if block.strip()]
    if not blocks:
        return True, {"passed": False, "error": f"no {name} code found"}
    binary = node_path()
    if not binary:
        return False, {"error": "unavailable: node is not installed on this computer"}
    if typescript and not node_strips_types(binary):
        return False, {"error": "unavailable: this node has no TypeScript parser (needs 22.13+ or 23.2+)"}
    folder = Path(tempfile.mkdtemp(prefix="neuro-ts-" if typescript else "neuro-js-"))
    try:
        # An empty package.json stops node from reading one in a parent
        # folder; .js files then get node's CommonJS/ES module detection.
        (folder / "package.json").write_text("{}\n", encoding="utf-8")
        if typescript:
            (folder / "strip.cjs").write_text(TS_STRIP_HELPER, encoding="utf-8")
        for index, code in enumerate(blocks):
            label = f"block {index + 1}"
            script = folder / f"block{index}.js"
            if typescript:
                source = folder / f"block{index}.ts"
                source.write_text(code, encoding="utf-8")
                ran = _run_fixed([binary, str(folder / "strip.cjs"), str(source), str(script)], folder, NODE_TIMEOUT, NODE_OUTPUT_CAP)
                if ran["error"]:
                    return False, {"error": ran["error"]}
                if ran["returncode"] == 2:
                    return True, {"passed": False, "error": label + ": " + clip(ran["stderr"].strip(), 1500)}
                if ran["returncode"] == 3:
                    return False, {"error": "unavailable: this node has no TypeScript parser"}
                if ran["returncode"] == 4:
                    # enum / namespace / parameter properties: valid TypeScript
                    # that node's strip-only parser cannot erase. No verdict.
                    return False, {"error": "inconclusive: " + clip(ran["stderr"].strip(), 300)}
                if ran["returncode"] != 0:
                    return False, {"error": clip(f"TypeScript parser exited {ran['returncode']}: {ran['stderr'].strip()}", 400)}
            else:
                script.write_text(code, encoding="utf-8")
            verdict, why = _node_check(binary, script, folder, label)
            if verdict is None:
                return False, {"error": why}
            if verdict is False:
                return True, {"passed": False, "error": why}
        return True, {"passed": True, "detail": f"{len(blocks)} block(s) parsed"}
    finally:
        shutil.rmtree(folder, ignore_errors=True)


def job_verify_js(payload: dict) -> tuple[bool, dict]:
    return _verify_node_blocks(payload, typescript=False)


def job_verify_ts(payload: dict) -> tuple[bool, dict]:
    return _verify_node_blocks(payload, typescript=True)


VOID_TAGS = frozenset(
    "area base br col embed hr img input link meta param source track wbr keygen command basefont frame isindex".split()
)
# Elements whose end tag HTML lets you leave out.
OPTIONAL_END_TAGS = frozenset(
    "html head body p li dt dd option optgroup tr td th thead tbody tfoot colgroup caption rb rt rtc rp".split()
)


class _HtmlCheck(HTMLParser):
    def __init__(self) -> None:
        super().__init__(convert_charrefs=True)
        self.stack: list[tuple[str, int]] = []
        self.seen: set[str] = set()
        self.styles: list[str] = []
        self.problem = ""

    def handle_starttag(self, tag: str, attrs: list) -> None:
        self.seen.add(tag)
        if tag not in VOID_TAGS:
            self.stack.append((tag, self.getpos()[0]))

    def handle_startendtag(self, tag: str, attrs: list) -> None:
        self.seen.add(tag)  # <x/>: nothing left open (void or SVG/MathML)

    def handle_endtag(self, tag: str) -> None:
        if self.problem or tag in VOID_TAGS:
            return
        line = self.getpos()[0]
        for depth in range(len(self.stack) - 1, -1, -1):
            if self.stack[depth][0] == tag:
                open_inside = [(name, at) for name, at in self.stack[depth + 1:] if name not in OPTIONAL_END_TAGS]
                if open_inside:
                    name, at = open_inside[-1]
                    self.problem = f"</{tag}> on line {line} closes <{tag}> while <{name}> from line {at} is still open"
                del self.stack[depth:]
                return
        self.problem = f"</{tag}> on line {line} has no open <{tag}>"

    def handle_data(self, data: str) -> None:
        if self.stack and self.stack[-1][0] == "style":
            self.styles.append(data)


def css_problem(css: str) -> str:
    """Structure of a style sheet: comments and strings end, braces and
    parentheses balance, and there is at least one rule. "" when fine."""
    depth = parens = 0
    rules = 0
    index, size = 0, len(css)
    while index < size:
        char = css[index]
        if css.startswith("/*", index):
            end = css.find("*/", index + 2)
            if end < 0:
                return "a /* comment is never closed"
            index = end + 2
            continue
        if char in "\"'":
            index += 1
            while index < size and css[index] != char:
                if css[index] == "\\":
                    index += 1
                elif css[index] == "\n":
                    return "a string runs past the end of its line"
                index += 1
            if index >= size:
                return "a string is never closed"
        elif char == "{":
            depth += 1
            rules += 1
        elif char == "}":
            depth -= 1
            if depth < 0:
                return "a } has no matching {"
        elif char == "(":
            parens += 1
        elif char == ")":
            parens -= 1
            if parens < 0:
                return "a ) has no matching ("
        index += 1
    if depth:
        return f"{depth} {{ never closed"
    if parens:
        return "a ( is never closed"
    if not rules:
        return "no CSS rule found"
    return ""


def job_verify_html_css(payload: dict) -> tuple[bool, dict]:
    """Structure only (html.parser plus a CSS scanner). Nothing is rendered,
    fetched or run."""
    html, css = payload.get("html") or "", payload.get("css") or ""
    if not html.strip() and not css.strip():
        return True, {"passed": False, "error": "no HTML or CSS code found"}
    sheets = [css] if css.strip() else []
    seen: set[str] = set()
    if html.strip():
        parser = _HtmlCheck()
        try:
            parser.feed(html)
            raw_text = str(getattr(parser, "rawdata", "") or "")
            cdata = getattr(parser, "cdata_elem", None)
            parser.close()
        except Exception as exc:  # html.parser may raise on hostile markup
            return True, {"passed": False, "error": clip(f"HTML parser: {type(exc).__name__}: {exc}", 300)}
        if parser.problem:
            return True, {"passed": False, "error": parser.problem}
        unclosed_raw = [(name, at) for name, at in parser.stack if name in ("script", "style")]
        if cdata or unclosed_raw:
            name, at = unclosed_raw[0] if unclosed_raw else (str(cdata), 0)
            return True, {"passed": False, "error": f"<{name}> from line {at} is never closed"}
        if raw_text.lstrip().startswith("<"):
            return True, {"passed": False, "error": "the HTML ends inside an unfinished tag or comment"}
        still_open = [(name, at) for name, at in parser.stack if name not in OPTIONAL_END_TAGS]
        if still_open:
            name, at = still_open[-1]
            return True, {"passed": False, "error": f"<{name}> from line {at} is never closed"}
        if not parser.seen:
            return True, {"passed": False, "error": "no HTML tags found"}
        seen = parser.seen
        sheets.extend(style for style in parser.styles if style.strip())
    for number, sheet in enumerate(sheets, 1):
        problem = css_problem(sheet)
        if problem:
            return True, {"passed": False, "error": f"CSS {number}: {problem}"}
    missing = [tag for tag in payload.get("require_tags") or [] if tag not in seen]
    if missing:
        return True, {"passed": False, "error": "missing tags: " + ", ".join(missing)}
    return True, {"passed": True, "detail": f"{len(seen)} tag names, {len(sheets)} style sheet(s)"}


def sqlite_ready() -> bool:
    try:
        import sqlite3  # noqa: F401
    except ImportError:
        return False
    return True


# Statement heads verify_sql accepts. ATTACH, DETACH, PRAGMA, VACUUM (INTO
# writes a file), ANALYZE, REINDEX and EXPLAIN itself are refused.
SQL_HEADS = frozenset("select with values insert replace update delete create drop alter begin commit end rollback savepoint release".split())
SQL_DDL = frozenset(("create", "drop", "alter"))
SQL_MAX_STATEMENTS = 200
_SQL_NAME = r'(?:"[^"]*"|`[^`]*`|\[[^\]]*\]|[\w$]+)'
_SQL_CTAS = re.compile(r"^\s*create\s+(?:temp\s+|temporary\s+)?table\s+(?:if\s+not\s+exists\s+)?" + _SQL_NAME + r"(?:\s*\.\s*" + _SQL_NAME + r")?\s+as\b", re.I)
_SQL_BLOCKED_FUNCTIONS = frozenset(("load_extension", "fts3_tokenizer", "readfile", "writefile", "edit"))


def _sql_strip_comments(sql: str) -> str:
    out, index, size = [], 0, len(sql)
    while index < size:
        if sql.startswith("--", index):
            end = sql.find("\n", index)
            index = size if end < 0 else end + 1
            out.append(" ")
        elif sql.startswith("/*", index):
            end = sql.find("*/", index + 2)
            index = size if end < 0 else end + 2
            out.append(" ")
        elif sql[index] in "'\"`[":
            close = "]" if sql[index] == "[" else sql[index]
            end = sql.find(close, index + 1)
            end = size - 1 if end < 0 else end
            out.append(sql[index:end + 1])
            index = end + 1
        else:
            out.append(sql[index])
            index += 1
    return "".join(out)


def sql_statements(code: str) -> tuple[list[str], str]:
    """Split SQL into complete statements with sqlite3.complete_statement
    (it knows strings, comments and trigger bodies). Returns (statements,
    problem); problem is "" or why the text does not split cleanly."""
    import sqlite3

    statements: list[str] = []
    buffer = ""
    pieces = code.split(";")
    for number, piece in enumerate(pieces):
        buffer += piece + (";" if number < len(pieces) - 1 else "")
        if number < len(pieces) - 1 and sqlite3.complete_statement(buffer):
            if _sql_strip_comments(buffer).strip(" \t\r\n;"):
                statements.append(buffer.strip())
            buffer = ""
    if _sql_strip_comments(buffer).strip(" \t\r\n;"):
        if sqlite3.complete_statement(buffer + "\n;"):
            statements.append(buffer.strip() + "\n;")
        else:
            return statements, "the SQL ends inside an unfinished statement, string or comment"
    return statements, ""


def job_verify_sql(payload: dict) -> tuple[bool, dict]:
    """Parse and name-check SQL against SQLite without running queries.
    DDL (CREATE / DROP / ALTER) builds a throwaway in-memory catalog so later
    statements can resolve table and column names; every other statement is
    only compiled with EXPLAIN, which lists the program and never steps it.
    Write the prompt for SQLite: other dialects can fail here."""
    try:
        import sqlite3
    except ImportError:
        return False, {"error": "unavailable: this Python has no sqlite3"}
    statements, problem = sql_statements(payload.get("code") or "")
    if problem:
        return True, {"passed": False, "error": problem}
    if not statements:
        return True, {"passed": False, "error": "no SQL statement found"}
    if len(statements) > SQL_MAX_STATEMENTS:
        return True, {"passed": False, "error": f"more than {SQL_MAX_STATEMENTS} statements"}

    def authorize(action, arg1, arg2, dbname, source):
        if action in (sqlite3.SQLITE_ATTACH, sqlite3.SQLITE_DETACH, sqlite3.SQLITE_PRAGMA):
            return sqlite3.SQLITE_DENY
        if action == sqlite3.SQLITE_FUNCTION and str(arg2 or "").lower() in _SQL_BLOCKED_FUNCTIONS:
            return sqlite3.SQLITE_DENY
        return sqlite3.SQLITE_OK

    steps = {"n": 0}

    def budget():
        steps["n"] += 1
        return 1 if steps["n"] > 2000 else 0  # abort after ~2M VM steps

    db = sqlite3.connect(":memory:", isolation_level=None)
    try:
        db.set_authorizer(authorize)
        db.set_progress_handler(budget, 1000)
        if payload.get("schema"):
            schema, schema_problem = sql_statements(payload["schema"])
            if schema_problem:
                return False, {"error": "the job's schema does not parse: " + schema_problem}
            for statement in schema:
                head = (re.match(r"\s*([A-Za-z]+)", _sql_strip_comments(statement)) or [None, ""])[1].lower()
                if head in SQL_DDL:
                    try:
                        db.execute(statement)
                    except sqlite3.Error as exc:
                        return False, {"error": clip(f"the job's schema failed: {exc}", 300)}
        for number, statement in enumerate(statements, 1):
            plain = _sql_strip_comments(statement)
            found = re.match(r"\s*([A-Za-z]+)", plain)
            head = found.group(1).lower() if found else ""
            if head not in SQL_HEADS:
                return True, {"passed": False, "error": f"statement {number}: {head.upper() or 'this'} is not allowed here"}
            try:
                if head in SQL_DDL and not _SQL_CTAS.match(plain):
                    db.execute(statement)
                else:
                    db.execute("EXPLAIN " + statement).fetchall()
            except sqlite3.Warning as exc:
                return True, {"passed": False, "error": clip(f"statement {number}: {exc}", 300)}
            except sqlite3.Error as exc:
                return True, {"passed": False, "error": clip(f"statement {number}: {type(exc).__name__}: {exc}", 300)}
    finally:
        db.close()
    return True, {"passed": True, "detail": f"{len(statements)} statement(s) compiled"}


def _boxed(text: str) -> str | None:
    start = text.rfind("\\boxed{")
    if start < 0:
        return None
    depth = 0
    for index in range(start + len("\\boxed{") - 1, len(text)):
        char = text[index]
        if char == "{":
            depth += 1
        elif char == "}":
            depth -= 1
            if depth == 0:
                return text[start + len("\\boxed{"):index]
    return None


def final_answer(text: str) -> str:
    text = strip_thinking(text)
    boxed = _boxed(text)
    if boxed is not None:
        return boxed.strip()
    marked = re.findall(r"(?:final answer|answer)\s*(?:is)?\s*[:=]?\s*\$?([^\n$]+)", text, flags=re.IGNORECASE)
    if marked:
        return marked[-1].strip().rstrip(".")
    numbers = re.findall(r"-?\d[\d,]*(?:\.\d+)?(?:/\d+)?", text)
    return numbers[-1] if numbers else text.strip()[-200:]


def parse_number(text: str) -> Fraction | None:
    value = str(text).strip().strip("$").strip()
    value = re.sub(r"\\text\{[^}]*\}", "", value)
    value = re.sub(r"\\[dt]?frac\{([^{}]+)\}\{([^{}]+)\}", r"(\1)/(\2)", value)
    value = value.replace("\\left", "").replace("\\right", "").replace("\\,", "").replace("\\!", "").replace("\\%", "%")
    value = value.replace(" ", "").replace(",", "").rstrip(".").rstrip("%")
    if not value or len(value) > 100 or re.search(r"[eE][+-]?\d{4,}", value):
        return None
    match = re.fullmatch(r"\(?(-?[0-9.]+(?:[eE][+-]?\d+)?)\)?(?:/\(?(-?[0-9.]+)\)?)?", value)
    if not match:
        return None
    try:
        number = Fraction(match.group(1))
        if match.group(2) is not None:
            number /= Fraction(match.group(2))
    except (ValueError, ZeroDivisionError):
        return None
    return number


def job_verify_math(payload: dict) -> tuple[bool, dict]:
    found = final_answer(payload["answer"])
    expected = str(payload["expected"])
    tol = float(payload.get("tol", 1e-6))
    got_number, want_number = parse_number(found), parse_number(expected)
    if got_number is not None and want_number is not None:
        passed = abs(got_number - want_number) <= Fraction(tol) * max(Fraction(1), abs(want_number))
    else:
        squash = lambda value: re.sub(r"[\s$]|\\text\{|\}", "", str(value)).lower()  # noqa: E731
        passed = bool(squash(found)) and squash(found) == squash(expected)
    return True, {"passed": bool(passed), "found": clip(found, 200)}


def normalize_text(text: str) -> str:
    return " ".join(unicodedata.normalize("NFKC", text).lower().split())


def _minhash_params() -> list[tuple[int, int]]:
    # Fixed for every computer, so signatures from different workers compare.
    prime = (1 << 61) - 1
    params = []
    for index in range(MINHASH_PERM):
        digest = hashlib.sha256(f"neuro-minhash-{index}".encode()).digest()
        params.append((int.from_bytes(digest[:8], "big") % (prime - 1) + 1, int.from_bytes(digest[8:16], "big") % prime))
    return params


_MINHASH = _minhash_params()


def minhash(text: str) -> list[int]:
    prime = (1 << 61) - 1
    words = normalize_text(text).split()
    grams = {" ".join(words[i:i + 3]) for i in range(max(1, len(words) - 2))} if words else {""}
    hashed = [int.from_bytes(hashlib.blake2b(gram.encode("utf-8"), digest_size=8).digest(), "big") for gram in grams]
    return [min((a * value + b) % prime for value in hashed) for a, b in _MINHASH]


def job_dedup_hash(payload: dict) -> tuple[bool, dict]:
    rows = [
        {"sha256": hashlib.sha256(normalize_text(text).encode("utf-8")).hexdigest(), "minhash": minhash(text)}
        for text in payload["texts"]
    ]
    return True, {"hashes": rows, "num_perm": MINHASH_PERM}


# -- verify_tests / cross_tests: running model-written tests in a sandbox -----
# The only job kinds that RUN generated code. Each run: a fresh temp folder
# (never inside this repository) as cwd, an argument list (never a shell),
# `python -I` or node with its permission model, stdin closed, a scrubbed
# environment (no tokens, proxies, PATH extras or interpreter options), a
# 10 s wall timeout, capped output, the folder removed afterwards.
#   Windows: a Job Object with 1 GB memory, ONE active process (no child
#            processes), 10 s CPU time and kill-on-close.
#   Linux/macOS: only with NEURO_RUN_TESTS=1, never as root; rlimits set in
#            preexec_fn: CPU 10 s, address space 1 GB, file size 16 MB,
#            no core files, NPROC 0 (node only without NPROC when it cannot
#            start its threads under it; then node's permission model
#            refuses child processes and worker threads).
# Network access is NOT blocked. A worker that cannot sandbox never
# advertises these kinds. The code was adapted from
# ml/evaluation/coding_eval.py (copied, not imported).


class SandboxError(OSError):
    pass


class _WinJob:
    """A Windows Job Object: memory cap, one active process, CPU time limit,
    kill on close, no UI access."""

    JOB_OBJECT_LIMIT_JOB_TIME = 0x0004
    JOB_OBJECT_LIMIT_ACTIVE_PROCESS = 0x0008
    JOB_OBJECT_LIMIT_JOB_MEMORY = 0x0200
    JOB_OBJECT_LIMIT_DIE_ON_UNHANDLED_EXCEPTION = 0x0400
    JOB_OBJECT_LIMIT_KILL_ON_JOB_CLOSE = 0x2000
    PROCESS_ACCESS = 0x0001 | 0x0100 | 0x0800 | 0x1000  # terminate, set quota, suspend/resume, query limited

    _api = None

    @classmethod
    def api(cls):
        if cls._api is None:
            from ctypes import wintypes

            class IO_COUNTERS(ctypes.Structure):
                _fields_ = [(name, ctypes.c_ulonglong) for name in (
                    "ReadOperationCount", "WriteOperationCount", "OtherOperationCount",
                    "ReadTransferCount", "WriteTransferCount", "OtherTransferCount")]

            class BASIC(ctypes.Structure):
                _fields_ = [("PerProcessUserTimeLimit", ctypes.c_longlong), ("PerJobUserTimeLimit", ctypes.c_longlong),
                            ("LimitFlags", wintypes.DWORD), ("MinimumWorkingSetSize", ctypes.c_size_t),
                            ("MaximumWorkingSetSize", ctypes.c_size_t), ("ActiveProcessLimit", wintypes.DWORD),
                            ("Affinity", ctypes.c_size_t), ("PriorityClass", wintypes.DWORD),
                            ("SchedulingClass", wintypes.DWORD)]

            class EXTENDED(ctypes.Structure):
                _fields_ = [("BasicLimitInformation", BASIC), ("IoInfo", IO_COUNTERS),
                            ("ProcessMemoryLimit", ctypes.c_size_t), ("JobMemoryLimit", ctypes.c_size_t),
                            ("PeakProcessMemoryUsed", ctypes.c_size_t), ("PeakJobMemoryUsed", ctypes.c_size_t)]

            class UI(ctypes.Structure):
                _fields_ = [("UIRestrictionsClass", wintypes.DWORD)]

            k32 = ctypes.WinDLL("kernel32", use_last_error=True)
            ntdll = ctypes.WinDLL("ntdll")
            k32.CreateJobObjectW.argtypes = [ctypes.c_void_p, wintypes.LPCWSTR]
            k32.CreateJobObjectW.restype = wintypes.HANDLE
            k32.SetInformationJobObject.argtypes = [wintypes.HANDLE, ctypes.c_int, ctypes.c_void_p, wintypes.DWORD]
            k32.SetInformationJobObject.restype = wintypes.BOOL
            k32.OpenProcess.argtypes = [wintypes.DWORD, wintypes.BOOL, wintypes.DWORD]
            k32.OpenProcess.restype = wintypes.HANDLE
            k32.AssignProcessToJobObject.argtypes = [wintypes.HANDLE, wintypes.HANDLE]
            k32.AssignProcessToJobObject.restype = wintypes.BOOL
            k32.TerminateJobObject.argtypes = [wintypes.HANDLE, wintypes.UINT]
            k32.TerminateJobObject.restype = wintypes.BOOL
            k32.TerminateProcess.argtypes = [wintypes.HANDLE, wintypes.UINT]
            k32.TerminateProcess.restype = wintypes.BOOL
            k32.CloseHandle.argtypes = [wintypes.HANDLE]
            k32.CloseHandle.restype = wintypes.BOOL
            ntdll.NtResumeProcess.argtypes = [wintypes.HANDLE]
            ntdll.NtResumeProcess.restype = ctypes.c_long
            cls._api = (k32, ntdll, EXTENDED, UI)
        return cls._api

    def __init__(self, memory_bytes: int, cpu_seconds: float):
        k32, _, EXTENDED, UI = self.api()
        self.handle = k32.CreateJobObjectW(None, None)
        if not self.handle:
            raise OSError(ctypes.get_last_error(), "CreateJobObjectW failed")
        info = EXTENDED()
        info.BasicLimitInformation.LimitFlags = (
            self.JOB_OBJECT_LIMIT_JOB_TIME | self.JOB_OBJECT_LIMIT_ACTIVE_PROCESS | self.JOB_OBJECT_LIMIT_JOB_MEMORY
            | self.JOB_OBJECT_LIMIT_DIE_ON_UNHANDLED_EXCEPTION | self.JOB_OBJECT_LIMIT_KILL_ON_JOB_CLOSE)
        info.BasicLimitInformation.PerJobUserTimeLimit = int(cpu_seconds * 10_000_000)  # 100 ns units
        info.BasicLimitInformation.ActiveProcessLimit = 1
        info.JobMemoryLimit = memory_bytes
        if not k32.SetInformationJobObject(self.handle, 9, ctypes.byref(info), ctypes.sizeof(info)):
            error = ctypes.get_last_error()
            self.close()
            raise OSError(error, "SetInformationJobObject failed")
        ui = UI(0xFF)  # no clipboard, desktop switching, global atoms, system parameters or exit-windows
        k32.SetInformationJobObject(self.handle, 4, ctypes.byref(ui), ctypes.sizeof(ui))

    def adopt(self, pid: int) -> None:
        """Put a process created with CREATE_SUSPENDED into the job, then let it run."""
        k32, ntdll, _, _ = self.api()
        process = k32.OpenProcess(self.PROCESS_ACCESS, False, pid)
        if not process:
            raise OSError(ctypes.get_last_error(), "OpenProcess failed")
        try:
            if not k32.AssignProcessToJobObject(self.handle, process):
                error = ctypes.get_last_error()
                k32.TerminateProcess(process, 1)
                raise OSError(error, "AssignProcessToJobObject failed")
            if ntdll.NtResumeProcess(process) != 0:
                k32.TerminateProcess(process, 1)
                raise OSError(0, "NtResumeProcess failed")
        finally:
            k32.CloseHandle(process)

    def terminate(self) -> None:
        if self.handle:
            self.api()[0].TerminateJobObject(self.handle, 1)

    def close(self) -> None:
        if self.handle:
            self.api()[0].CloseHandle(self.handle)
            self.handle = None


def sandbox_env(workdir: str) -> dict:
    """A minimal environment: no tokens, proxies, PATH extras, home secrets
    or interpreter options."""
    if os.name == "nt":
        root = os.environ.get("SystemRoot", r"C:\Windows")
        return {"SystemRoot": root, "WINDIR": root, "SystemDrive": os.environ.get("SystemDrive", "C:"),
                "PATH": rf"{root}\System32;{root}", "TEMP": workdir, "TMP": workdir,
                "USERPROFILE": workdir, "HOME": workdir}
    return {"PATH": "/usr/bin:/bin", "HOME": workdir, "TMPDIR": workdir, "LANG": "C.UTF-8"}


def _posix_limits(nproc: bool, address_space: bool):
    """preexec_fn for a test child. The values are worked out here, in the
    parent; the child only calls setrlimit."""
    import resource

    def capped(which: int, value: int) -> tuple[int, int]:
        _, hard = resource.getrlimit(which)
        value = value if hard == resource.RLIM_INFINITY else min(value, hard)
        return value, value

    cpu_soft, _ = capped(resource.RLIMIT_CPU, TEST_CPU_SECONDS)
    rules = [
        (resource.RLIMIT_CPU, (cpu_soft, cpu_soft), True),
        (resource.RLIMIT_FSIZE, capped(resource.RLIMIT_FSIZE, TEST_FILE_LIMIT), True),
        (resource.RLIMIT_CORE, (0, 0), True),
    ]
    if nproc:
        rules.append((resource.RLIMIT_NPROC, (0, 0), True))
    if address_space:
        # macOS does not enforce RLIMIT_AS and may refuse it; elsewhere it must hold.
        rules.append((resource.RLIMIT_AS, capped(resource.RLIMIT_AS, TEST_MEMORY), sys.platform != "darwin"))
    setrlimit = resource.setrlimit

    def apply() -> None:
        for which, value, required in rules:
            try:
                setrlimit(which, value)
            except (ValueError, OSError):
                if required:
                    raise
    return apply


def _drain(stream, cap: int, sink: dict, flood: threading.Event | None = None) -> None:
    kept = bytearray()
    total = 0
    try:
        while True:
            chunk = stream.read1(65536)
            if not chunk:
                break
            total += len(chunk)
            if len(kept) < cap:
                kept += chunk[:cap - len(kept)]
            if flood is not None and total > TEST_OUTPUT_STOP:
                flood.set()
    except (OSError, ValueError):
        pass
    sink["data"], sink["total"] = bytes(kept), total


def run_sandboxed(argv: list[str], workdir: str, limits: dict, timeout: float = TEST_TIMEOUT) -> dict:
    """Run argv in workdir under `limits` ({"job": True} on Windows, or
    {"nproc": bool, "address_space": bool} on POSIX). Returns a result dict."""
    kwargs: dict = {"cwd": workdir, "env": sandbox_env(workdir), "stdin": subprocess.DEVNULL,
                    "stdout": subprocess.PIPE, "stderr": subprocess.PIPE, "shell": False, "close_fds": True}
    job = None
    if os.name == "nt":
        kwargs["creationflags"] = 0x08000000  # CREATE_NO_WINDOW
        if not limits.get("job"):
            raise SandboxError("no Job Object: refusing to run tests")
        job = _WinJob(TEST_MEMORY, TEST_CPU_SECONDS)
        kwargs["creationflags"] |= 0x00000004  # CREATE_SUSPENDED until it is inside the job
    else:
        kwargs["start_new_session"] = True
        kwargs["preexec_fn"] = _posix_limits(bool(limits.get("nproc")), bool(limits.get("address_space")))
    start = time.perf_counter()
    try:
        proc = subprocess.Popen(argv, **kwargs)
    except (OSError, subprocess.SubprocessError) as error:
        if job:
            job.close()
        raise SandboxError(f"could not start {Path(argv[0]).name}: {error}") from error
    sinks: tuple[dict, dict] = ({}, {})
    timed_out = flooded = False
    flood = threading.Event()
    try:
        if job:
            try:
                job.adopt(proc.pid)
            except OSError as error:
                proc.kill()
                proc.wait()
                raise SandboxError(f"could not apply the Job Object limit: {error}") from error
        readers = [threading.Thread(target=_drain, args=(pipe, TEST_OUTPUT_CAP, sink, flood), daemon=True)
                   for pipe, sink in zip((proc.stdout, proc.stderr), sinks)]
        for reader in readers:
            reader.start()
        deadline = time.monotonic() + timeout
        while proc.poll() is None and not flood.is_set() and time.monotonic() < deadline:
            flood.wait(0.02)
        if proc.poll() is None:
            flooded = flood.is_set()
            timed_out = not flooded
            if job:
                job.terminate()
            elif os.name != "nt":
                try:
                    os.killpg(proc.pid, 9)
                except OSError:
                    proc.kill()
            else:
                proc.kill()
            proc.wait(timeout=10)
        for reader in readers:
            reader.join(timeout=5)
    finally:
        if job:
            job.terminate()
            job.close()
        for pipe in (proc.stdout, proc.stderr):
            try:
                pipe.close()
            except OSError:
                pass
    out, err = sinks
    return {
        "returncode": proc.returncode, "timed_out": timed_out, "flooded": flooded,
        "seconds": round(time.perf_counter() - start, 3),
        "stdout": out.get("data", b"").decode("utf-8", "replace"),
        "stderr": err.get("data", b"").decode("utf-8", "replace"),
        "stdout_bytes": out.get("total", 0), "stderr_bytes": err.get("total", 0),
    }


def _remove_tree(path: str) -> bool:
    for attempt in range(10):
        shutil.rmtree(path, ignore_errors=True)
        if not os.path.exists(path):
            return True
        time.sleep(0.1 * (attempt + 1))
    return False


def sandbox_workdir() -> str:
    """A fresh temp folder for one run, never inside this repository."""
    folder = tempfile.mkdtemp(prefix="neuro-tests-")
    resolved = os.path.normcase(os.path.realpath(folder))
    repo = os.path.normcase(os.path.realpath(str(ROOT)))
    if resolved == repo or resolved.startswith(repo.rstrip("\\/") + os.sep):
        _remove_tree(folder)
        raise SandboxError("the temp folder is inside the repository; set TEMP/TMPDIR elsewhere")
    return folder


# Python: runs main.py as __main__, then (cross_tests) each extra assert on
# its own, then prints the run token, so exiting early with code 0 is no pass.
PY_TEST_HARNESS = '''\
import json, sys
token = sys.argv.pop(1)
checks = None
try:
    with open("checks.json", encoding="utf-8") as handle:
        checks = json.load(handle)
except FileNotFoundError:
    pass
with open("main.py", encoding="utf-8") as handle:
    source = handle.read()
namespace = {"__name__": "__main__", "__file__": "main.py", "__builtins__": __builtins__}
exec(compile(source, "main.py", "exec"), namespace)
sys.stdout.flush()
print()
if checks is not None:
    for index, check in enumerate(checks):
        try:
            exec(compile(check, "check%d" % index, "exec"), namespace)
            print(token, "P", index, flush=True)
        except BaseException:
            print(token, "F", index, flush=True)
    print(token, "END", flush=True)
else:
    print(token, flush=True)
'''

JS_TEST_TAIL = '''
;process.stdout.write("\\n@@TOKEN@@\\n");
'''

# Direct eval inside the module, so the checks see the solution's functions.
JS_CROSS_TAIL = '''
;(async () => {
  const __neuroAssert = (typeof require === "function" ? require : process.getBuiltinModule)("node:assert");
  const __neuroChecks = @@CHECKS@@;
  process.stdout.write("\\n");
  for (let __neuroIndex = 0; __neuroIndex < __neuroChecks.length; __neuroIndex++) {
    let __neuroOk = true;
    try {
      const __neuroResult = eval("{ const assert = __neuroAssert;\\n" + __neuroChecks[__neuroIndex] + "\\n}");
      if (__neuroResult && typeof __neuroResult.then === "function") await __neuroResult;
    } catch (__neuroError) {
      __neuroOk = false;
    }
    process.stdout.write("@@TOKEN@@ " + (__neuroOk ? "P " : "F ") + __neuroIndex + "\\n");
  }
  process.stdout.write("@@TOKEN@@ END\\n");
})();
'''

_sandbox_cache: dict = {}
_sandbox_lock = threading.Lock()
_PY_PROBE = "import json\nassert json.loads('[1, 2]') == [1, 2]\nprint('probe-ok')\n"
_JS_PROBE = "require('node:assert').strictEqual(1 + 1, 2); console.log('probe-ok');"


def _probe(argv_for, limits: dict, source_name: str, source: str) -> bool:
    """Run a fixed probe program (never job code) under the given limits."""
    folder = sandbox_workdir()
    try:
        script = Path(folder) / source_name
        script.write_text(source, encoding="utf-8")
        (Path(folder) / "package.json").write_text("{}\n", encoding="utf-8")
        ran = run_sandboxed(argv_for(folder) + [str(script)], folder, limits, timeout=20)
        return not ran["timed_out"] and ran["returncode"] == 0 and ran["stdout"].strip() == "probe-ok"
    except (OSError, ValueError, subprocess.SubprocessError):
        return False
    finally:
        _remove_tree(folder)


def _node_flag_sets(node: str) -> list[list[str]]:
    """Node flags to try, permission model first (no child processes, no
    worker threads, file reads only inside the run folder)."""
    base = ["--max-old-space-size=512"]
    return [["--permission", "--allow-fs-read={dir}", *base],
            ["--experimental-permission", "--allow-fs-read={dir}", *base],
            base]


def sandbox_status(refresh: bool = False) -> dict:
    """Whether this computer may run model-written tests, and how.
    {"ok", "note", "langs": {"python"|"javascript": {"argv", "limits", "how"}}}.
    Probed once per process with fixed programs."""
    with _sandbox_lock:
        if _sandbox_cache and not refresh:
            return _sandbox_cache["status"]
        status = _sandbox_probe()
        _sandbox_cache["status"] = status
        return status


def _sandbox_probe() -> dict:
    def off(why: str) -> dict:
        return {"ok": False, "note": "tests off: " + why, "langs": {}}

    if os.name == "nt":
        try:
            _WinJob(TEST_MEMORY, TEST_CPU_SECONDS).close()
        except (OSError, AttributeError, ValueError) as exc:
            return off(f"no Windows Job Object ({exc})")
        py_variants = [({"job": True}, "Job Object")]
        js_variants = [({"job": True}, "Job Object")]
        how = (f"Windows Job Object: {TEST_MEMORY >> 20} MB, 1 process (no child processes), "
               f"{TEST_CPU_SECONDS} s CPU, kill on close")
    else:
        if os.environ.get("NEURO_RUN_TESTS", "").strip() != "1":
            return off("set NEURO_RUN_TESTS=1 to run model-written tests on this computer")
        if hasattr(os, "geteuid") and os.geteuid() == 0:
            return off("refused as root (process limits do not bind root)")
        try:
            import resource  # noqa: F401
        except ImportError:
            return off("no resource module")
        py_variants = [({"nproc": True, "address_space": True}, "NPROC 0")]
        js_variants = [({"nproc": True, "address_space": True}, "NPROC 0"),
                       ({"nproc": False, "address_space": True}, "node permission model instead of NPROC"),
                       ({"nproc": False, "address_space": False}, "node permission model, heap cap instead of RLIMIT_AS")]
        how = (f"rlimits: CPU {TEST_CPU_SECONDS} s, address space {TEST_MEMORY >> 20} MB, "
               f"file size {TEST_FILE_LIMIT >> 20} MB, no core files")
    langs: dict = {}
    python = getattr(sys, "_base_executable", "") or sys.executable
    if python and Path(python).is_file():
        prefix = [str(Path(python).resolve()), "-I", "-B", "-X", "utf8"]
        for limits, label in py_variants:
            if _probe(lambda folder: list(prefix), limits, "probe.py", _PY_PROBE):
                langs["python"] = {"argv": prefix, "limits": limits, "how": label}
                break
    node = node_path()
    if node:
        node = str(Path(node).resolve())
        for limits, label in js_variants:
            for flags in _node_flag_sets(node):
                permission = any("permission" in flag for flag in flags)
                if label.startswith("node permission") and not permission:
                    continue
                argv = [node, *flags]
                if _probe(lambda folder: [arg.replace("{dir}", folder) for arg in argv], limits, "probe.js", _JS_PROBE):
                    langs["javascript"] = {"argv": argv, "limits": limits,
                                           "how": label + (", node permission model" if permission and "permission" not in label else "")}
                    break
            if "javascript" in langs:
                break
    if not langs:
        return off(how + "; but no Python or node ran under these limits")
    detail = "; ".join(f"{lang}: {spec['how']}" for lang, spec in sorted(langs.items()))
    note = (f"runs model-written tests ({', '.join(sorted(langs))}): {how}; {detail}; fresh temp folder, "
            f"scrubbed env, {TEST_TIMEOUT} s wall timeout. Network access is NOT blocked.")
    return {"ok": True, "note": note, "langs": langs}


def run_test_program(lang: str, code: str, box: dict, checks: list[str] | None = None,
                     timeout: float = TEST_TIMEOUT) -> dict:
    """Run one answer (and, for cross_tests, extra asserts after it) in the
    sandbox. Adds "finished" (the run reached its end) and "results"
    (index -> passed, cross_tests only)."""
    spec = box["langs"][lang]
    token = "NEURO-TESTS-" + secrets.token_hex(16)
    folder = sandbox_workdir()
    try:
        work = Path(folder)
        if lang == "python":
            (work / "main.py").write_text(code, encoding="utf-8")
            (work / "harness.py").write_text(PY_TEST_HARNESS, encoding="utf-8")
            if checks is not None:
                (work / "checks.json").write_text(json.dumps(checks, ensure_ascii=False), encoding="utf-8")
            argv = list(spec["argv"]) + [str(work / "harness.py"), token]
        else:
            # An empty package.json stops node from reading one in a parent folder.
            (work / "package.json").write_text("{}\n", encoding="utf-8")
            tail = (JS_CROSS_TAIL if checks is not None else JS_TEST_TAIL).replace("@@TOKEN@@", token)
            tail = tail.replace("@@CHECKS@@", json.dumps(checks or [], ensure_ascii=True))
            (work / "main.js").write_text(code + "\n" + tail, encoding="utf-8")
            argv = [arg.replace("{dir}", folder) for arg in spec["argv"]] + [str(work / "main.js")]
        result = run_sandboxed(argv, folder, spec["limits"], timeout)
    finally:
        result_cleaned = _remove_tree(folder)
    result["cleaned"] = result_cleaned
    for spelled in {folder, folder.replace("\\", "/")}:
        result["stderr"] = result["stderr"].replace(spelled, "<job>")
    lines = [line.strip() for line in result["stdout"].splitlines() if line.strip()]
    end = f"{token} END" if checks is not None else token
    result["finished"] = bool(result["returncode"] == 0 and not result["timed_out"] and not result["flooded"]
                              and lines and lines[-1] == end)
    results: dict[int, bool] = {}
    if checks is not None:
        for line in lines:
            parts = line.split()
            if len(parts) == 3 and parts[0] == token and parts[1] in ("P", "F") and parts[2].isdigit():
                results.setdefault(int(parts[2]), parts[1] == "P")
    result["results"] = results
    return result


def _stderr_head(text: str, limit: int = 300) -> str:
    """The start of stderr without the harness's own traceback frames."""
    kept, skipping = [], False
    for line in text.replace("\r\n", "\n").split("\n"):
        if 'harness.py"' in line:
            skipping = True
            continue
        if skipping and line.startswith("    "):
            continue
        skipping = False
        kept.append(line)
    return "\n".join(kept).strip()[:limit]


def _error_line(text: str) -> str:
    """The most telling stderr line: the last 'SomethingError: ...' line, else the last line."""
    fatal = [line.strip() for line in text.splitlines() if "FATAL ERROR" in line or "out of memory" in line]
    if fatal:
        return fatal[0][:300]
    errors = [line.strip() for line in text.splitlines() if re.match(r"\s*\w*(Error|Exception)\b", line)]
    lines = [line.strip() for line in text.splitlines() if line.strip()]
    line = errors[-1] if errors else (lines[-1] if lines else "")
    return line[:300]


# -- static checks on the tests: enough real asserts, nothing swallowed -------

_PY_BROAD = {"Exception", "BaseException"}


def _handler_names(node: ast.AST | None) -> set[str]:
    if node is None:
        return set()
    if isinstance(node, ast.Tuple):
        return set().union(*(_handler_names(item) for item in node.elts)) if node.elts else set()
    if isinstance(node, ast.Name):
        return {node.id}
    if isinstance(node, ast.Attribute):
        return {node.attr}
    return set()


def _trivial_py_assert(node: ast.Assert) -> bool:
    """assert True / assert 1 / assert "x" / assert 2 == 2 / assert x == x:
    no name in the test at all, or both sides of one comparison the same."""
    test = node.test
    if not any(isinstance(item, ast.Name) for item in ast.walk(test)):
        return True
    if isinstance(test, ast.Compare) and len(test.ops) == 1 and isinstance(test.ops[0], (ast.Eq, ast.Is, ast.GtE, ast.LtE)):
        return ast.dump(test.left) == ast.dump(test.comparators[0])
    return False


def _py_test_problems(code: str) -> tuple[int, str]:
    try:
        tree = ast.parse(code)
    except SyntaxError as exc:
        return 0, clip(f"SyntaxError: {exc.msg} (line {exc.lineno})", 300)
    except (ValueError, RecursionError, MemoryError, OverflowError) as exc:
        return 0, clip(f"{type(exc).__name__}: {exc}", 300)
    try_types = tuple(t for t in (ast.Try, getattr(ast, "TryStar", None)) if t is not None)
    for node in ast.walk(tree):
        if isinstance(node, ast.Name) and node.id == "AssertionError" and not isinstance(node.ctx, ast.Load):
            return 0, "the code rebinds AssertionError"
        if isinstance(node, ast.Call) and _handler_names(node.func) & {"suppress"}:
            if any(_handler_names(arg) & (_PY_BROAD | {"AssertionError"}) for arg in node.args):
                return 0, "contextlib.suppress hides assert failures"
        if isinstance(node, try_types):
            guarded = any(isinstance(item, ast.Assert) for stmt in node.body for item in ast.walk(stmt))
            for handler in node.handlers:
                names = _handler_names(handler.type)
                if "AssertionError" in names:
                    return 0, "the code catches AssertionError"
                reraises = any(isinstance(item, ast.Raise) for stmt in handler.body for item in ast.walk(stmt))
                if guarded and (handler.type is None or names & _PY_BROAD) and not reraises:
                    return 0, "a try/except around the asserts swallows failures"
    # Asserts that run: top level (also inside top-level if/for/with blocks)
    # or inside a top-level function that the top level calls.
    top = [stmt for stmt in tree.body if not isinstance(stmt, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef))]
    called = {item.func.id for stmt in top for item in ast.walk(stmt)
              if isinstance(item, ast.Call) and isinstance(item.func, ast.Name)}
    running: list[ast.Assert] = []
    for stmt in tree.body:
        if isinstance(stmt, ast.ClassDef):
            continue
        if isinstance(stmt, (ast.FunctionDef, ast.AsyncFunctionDef)) and stmt.name not in called:
            continue
        running.extend(item for item in ast.walk(stmt) if isinstance(item, ast.Assert))
    real = [item for item in running if not _trivial_py_assert(item)]
    if len(real) < MIN_ASSERTS:
        extra = f" ({len(running) - len(real)} trivial ones like `assert True` do not count)" if len(running) > len(real) else ""
        return len(real), f"only {len(real)} assert statement(s) that run; at least {MIN_ASSERTS} are needed{extra}"
    return len(real), ""


def js_mask(code: str) -> str:
    """The code with comments and string/template/regex contents blanked
    (same length, newlines kept), so structure can be scanned with regexes."""
    out = list(code)
    index, size = 0, len(code)
    last = ""  # last significant character, to tell a regex from a division

    def blank(start: int, stop: int) -> None:
        for at in range(start, min(stop, size)):
            if out[at] != "\n":
                out[at] = " "

    while index < size:
        char = code[index]
        if code.startswith("//", index):
            end = code.find("\n", index)
            end = size if end < 0 else end
            blank(index, end)
            index = end
            continue
        if code.startswith("/*", index):
            end = code.find("*/", index + 2)
            end = size if end < 0 else end + 2
            blank(index, end)
            index = end
            continue
        if char in "'\"`" or (char == "/" and (not last or last in "(,=:[!&|?{};+-*%<>~^" )):
            close = char
            at = index + 1
            in_class = False
            while at < size:
                current = code[at]
                if current == "\\":
                    at += 2
                    continue
                if current == "\n" and close != "`":
                    break
                if close == "/":
                    if current == "[":
                        in_class = True
                    elif current == "]":
                        in_class = False
                    elif current == "/" and not in_class:
                        break
                elif current == close:
                    break
                at += 1
            blank(index + 1, at)
            index = at + 1
            last = "a"
            continue
        if not char.isspace():
            last = char
        index += 1
    return "".join(out)


_JS_ASSERT_METHODS = frozenset(
    "strictEqual deepStrictEqual equal deepEqual notStrictEqual notDeepStrictEqual notEqual notDeepEqual "
    "ok throws doesNotThrow rejects doesNotReject match doesNotMatch".split()
)
_JS_ASSERT_CALL = re.compile(r"(?<![\w$.])assert\s*(?:\.\s*(?:strict\s*\.\s*)?([A-Za-z]+)\s*)?\(")
_JS_EQUALS = frozenset("strictEqual deepStrictEqual equal deepEqual".split())
_JS_TRUTHY = re.compile(r"""^(?:true|!0|!false|!!1|[1-9][0-9]*(?:\.[0-9]+)?|0\.[0-9]*[1-9][0-9]*|"[^"]+"|'[^']+'|`[^`]+`|\[\s*\]|\{\s*\})$""")


def _close_paren(masked: str, start: int) -> int:
    """Index of the bracket that closes the one at `start`, or -1."""
    depth = 0
    for at in range(start, len(masked)):
        char = masked[at]
        if char in "([{":
            depth += 1
        elif char in ")]}":
            depth -= 1
            if depth == 0:
                return at
    return -1


def _split_args(masked: str, code: str, start: int, stop: int) -> list[str]:
    parts, depth, last = [], 0, start
    for at in range(start, stop):
        char = masked[at]
        if char in "([{":
            depth += 1
        elif char in ")]}":
            depth -= 1
        elif char == "," and depth == 0:
            parts.append(code[last:at].strip())
            last = at + 1
    tail = code[last:stop].strip()
    if tail or parts:
        parts.append(tail)
    return [part for part in parts if part]


def js_assert_calls(code: str, masked: str | None = None) -> list[dict]:
    """node:assert calls: [{"start", "end" (after the close paren), "method", "args", "real"}]."""
    masked = js_mask(code) if masked is None else masked
    calls = []
    for match in _JS_ASSERT_CALL.finditer(masked):
        method = match.group(1) or ""
        if method and method not in _JS_ASSERT_METHODS:
            continue
        open_at = match.end() - 1
        close_at = _close_paren(masked, open_at)
        if close_at < 0:
            continue
        args = _split_args(masked, code, open_at + 1, close_at)
        real = bool(args)
        if method in ("", "ok") and args and _JS_TRUTHY.match(args[0]):
            real = False
        if method in _JS_EQUALS and len(args) >= 2 and args[0] == args[1]:
            real = False
        calls.append({"start": match.start(), "end": close_at + 1, "method": method or "assert", "args": args, "real": real})
    return calls


_JS_ASSERT_IMPORT = re.compile(r"""\brequire\s*\(\s*['"](?:node:)?assert(?:/strict)?['"]\s*\)|\bfrom\s+['"](?:node:)?assert(?:/strict)?['"]|\bgetBuiltinModule\s*\(\s*['"](?:node:)?assert(?:/strict)?['"]""")


def _js_test_problems(code: str) -> tuple[int, str]:
    masked = js_mask(code)
    if not _JS_ASSERT_IMPORT.search(code):
        return 0, "no node:assert import (use const assert = require('node:assert'))"
    if re.search(r"\bconsole\s*\.\s*assert\b", masked):
        return 0, "console.assert never fails a run; use node:assert"
    if re.search(r"\b(?:uncaughtException|unhandledRejection)\b", code):
        return 0, "a global error handler could hide failing asserts"
    calls = js_assert_calls(code, masked)
    for match in re.finditer(r"\btry\s*\{", masked):
        body_end = _close_paren(masked, match.end() - 1)
        if body_end < 0:
            continue
        caught = re.match(r"\s*catch\s*(?:\([^)]*\))?\s*\{", masked[body_end + 1:])
        if not caught:
            continue
        catch_open = body_end + 1 + caught.end() - 1
        catch_end = _close_paren(masked, catch_open)
        if catch_end < 0:
            continue
        guarded = any(match.end() <= call["start"] < body_end for call in calls)
        if guarded and not re.search(r"\bthrow\b", masked[catch_open:catch_end]):
            return 0, "a try/catch around the asserts swallows failures"
    real = sum(1 for call in calls if call["real"])
    if real < MIN_ASSERTS:
        trivial = len(calls) - real
        extra = f" ({trivial} trivial ones like assert(true) do not count)" if trivial else ""
        return real, f"only {real} node:assert call(s); at least {MIN_ASSERTS} are needed{extra}"
    return real, ""


def tests_problems(code: str, lang: str) -> tuple[int, str]:
    """(real asserts counted, "" or why the tests do not qualify)."""
    return _py_test_problems(code) if lang == "python" else _js_test_problems(code)


def extract_asserts(code: str, lang: str) -> list[str]:
    """The plain top-level asserts of one answer, one statement each, for
    cross_tests. Python: module-level `assert` statements (also inside
    `if __name__ == "__main__":`). JavaScript: node:assert calls that start
    a line at the top level of the file."""
    if lang == "python":
        try:
            tree = ast.parse(code)
        except (SyntaxError, ValueError, RecursionError, MemoryError, OverflowError):
            return []
        found: list[ast.Assert] = []
        for stmt in tree.body:
            if isinstance(stmt, ast.Assert):
                found.append(stmt)
            elif (isinstance(stmt, ast.If) and isinstance(stmt.test, ast.Compare)
                  and isinstance(stmt.test.left, ast.Name) and stmt.test.left.id == "__name__"):
                found.extend(item for item in stmt.body if isinstance(item, ast.Assert))
        out = []
        for item in found:
            if _trivial_py_assert(item):
                continue
            try:
                out.append(ast.unparse(item))
            except (ValueError, RecursionError):
                continue
        return out
    masked = js_mask(code)
    depth_at = []
    depth = 0
    for char in masked:
        depth_at.append(depth)
        if char in "([{":
            depth += 1
        elif char in ")]}":
            depth -= 1
    out = []
    for call in js_assert_calls(code, masked):
        line_start = masked.rfind("\n", 0, call["start"]) + 1
        if not call["real"] or depth_at[call["start"]] != 0 or masked[line_start:call["start"]].strip():
            continue
        # A whole statement only: assert.x(...).catch(...) is not a plain assert.
        if not re.match(r"[ \t]*(?:;|\r?\n|$)", masked[call["end"]:]):
            continue
        out.append(code[call["start"]:call["end"]] + ";")
    return out


def job_verify_tests(payload: dict) -> tuple[bool, dict]:
    """Run the answer's own asserts in the sandbox. Passed: exactly one code
    block, at least MIN_ASSERTS real asserts that run, nothing that hides a
    failure, exit code 0 within the time limit, and the run reached its end."""
    lang = payload["lang"]
    blocks = [block for block in payload.get("blocks") or [] if block.strip()]
    if not blocks:
        return True, {"passed": False, "error": f"no {lang} code block found"}
    if len(blocks) > 1:
        return True, {"passed": False, "error": f"expected one {lang} code block, found {len(blocks)}"}
    code = blocks[0]
    count, problem = tests_problems(code, lang)
    if problem:
        return True, {"passed": False, "error": problem, "asserts": count}
    box = sandbox_status()
    if lang not in box["langs"]:
        return False, {"error": "unavailable: " + box["note"]}
    try:
        ran = run_test_program(lang, code, box)
    except SandboxError as exc:
        return False, {"error": clip(f"unavailable: {exc}", 300)}
    result = {"passed": ran["finished"], "asserts": count, "seconds": ran["seconds"], "returncode": int(ran["returncode"] if ran["returncode"] is not None else -1)}
    if not ran["finished"]:
        if ran["timed_out"]:
            result["error"] = f"timed out after {TEST_TIMEOUT} s"
        elif ran["flooded"]:
            result["error"] = f"printed more than {TEST_OUTPUT_STOP >> 20} MB"
        elif ran["returncode"] == 0:
            result["error"] = "the program stopped before the end of its tests"
        else:
            result["error"] = _error_line(ran["stderr"]) or f"exit code {ran['returncode']}"
        result["stderr"] = _stderr_head(ran["stderr"])
    return True, result


def job_cross_tests(payload: dict) -> tuple[bool, dict]:
    """CodeT-lite: each sample's code runs with the other samples' plain
    asserts appended, one sandbox run per sample. Agreement = the share of
    the others' asserts it passes (None when the others have none)."""
    lang = payload["lang"]
    box = sandbox_status()
    if lang not in box["langs"]:
        return False, {"error": "unavailable: " + box["note"]}
    samples = payload["samples"]
    extracted = [extract_asserts(sample["code"], lang) for sample in samples]
    scores: dict[str, dict] = {}
    timeouts = 0
    for index, sample in enumerate(samples):
        checks = [check for other, found in enumerate(extracted) if other != index for check in found][:MAX_CROSS_CHECKS]
        if not checks:
            scores[sample["id"]] = {"agreement": None, "passed": 0, "total": 0}
            continue
        try:
            ran = run_test_program(lang, sample["code"], box, checks=checks)
        except SandboxError as exc:
            return False, {"error": clip(f"unavailable: {exc}", 300)}
        timeouts += bool(ran["timed_out"] or ran["flooded"])
        good = sum(1 for at in range(len(checks)) if ran["results"].get(at))
        scores[sample["id"]] = {"agreement": round(good / len(checks), 4), "passed": good, "total": len(checks)}
    detail = f"{len(samples)} samples, {sum(len(found) for found in extracted)} asserts" + (f", {timeouts} run(s) timed out" if timeouts else "")
    return True, {"scores": scores, "detail": detail}


JOB_HANDLERS = {
    "generate": job_generate,
    "verify_luau": job_verify_luau,
    "verify_python_syntax": job_verify_python_syntax,
    "verify_js": job_verify_js,
    "verify_ts": job_verify_ts,
    "verify_html_css": job_verify_html_css,
    "verify_sql": job_verify_sql,
    "verify_math": job_verify_math,
    "dedup_hash": job_dedup_hash,
    "verify_tests": job_verify_tests,
    "cross_tests": job_cross_tests,
}


def run_job(job: dict) -> tuple[bool, dict]:
    """Run one leased job with its built-in handler. The payload is checked
    again here, so a bad queue cannot push past the size caps."""
    kind = str(job.get("kind") or "")
    try:
        payload = validate_job(kind, job.get("payload"))
        return JOB_HANDLERS[kind](payload)
    except JobError as exc:
        return False, {"error": clip(str(exc), 300)}
    except (OSError, urllib.error.URLError, ValueError, KeyError, TypeError) as exc:
        return False, {"error": clip(f"{type(exc).__name__}: {exc}", 300)}


class JobAuthError(Exception):
    def __init__(self, missing: bool = False) -> None:
        super().__init__("no cluster.token" if missing else "token mismatch")
        self.missing = missing


# --- finding the main PC (serve) without NEURO_MESH_URL ---------------------
# Order: NEURO_MESH_URL, then the address remembered in mesh_url.txt, then a
# sweep of this computer's own private /24 for a serve that says so on GET
# /hello, then this computer itself. Finding a serve grants nothing: pairing still
# needs the code approved on that main PC's page, and a worker that already
# holds a cluster.token only takes a found serve that proves it holds the
# same token (HMAC of a fresh nonce), so the token is never sent to a device
# that merely answers /hello.
# Last fallback only: this computer itself (a main PC may also run a worker).
# Set NEURO_MESH_URL when discovery cannot see the main PC.
DEFAULT_MESH_URL = "http://127.0.0.1:8780"
MESH_URL_PATH = ROOT / "mesh_url.txt"
_mesh_url_lock = threading.Lock()
_mesh_url_cache: dict = {"url": None}


def lan_url(raw: str) -> str:
    """A clean http(s)://host:port on the private LAN (or 127.0.0.1), else ""."""
    try:
        url = clean_url(str(raw or ""))
    except ValueError:
        return ""
    host = urlparse(url).hostname or ""
    return url if private_lan(host) or host == "127.0.0.1" else ""


def remembered_mesh_url() -> str:
    try:
        raw = MESH_URL_PATH.read_text(encoding="utf-8", errors="replace")[:200]
    except OSError:
        return ""
    return lan_url(raw.strip())


def remember_mesh_url(url: str) -> None:
    url = lan_url(url)
    if not url:
        return
    temp = MESH_URL_PATH.parent / f"{MESH_URL_PATH.name}.tmp-{secrets.token_hex(4)}"
    try:
        temp.write_text(url + "\n", encoding="utf-8")
        os.replace(temp, MESH_URL_PATH)
    except OSError:
        pass
    finally:
        temp.unlink(missing_ok=True)


def serve_proof_ok(base: str, secret: str) -> bool:
    """True when the serve at `base` proves it holds `secret` (the same
    HMAC check serve makes of a worker). Never sends the token."""
    parsed = urlparse(base)
    if not parsed.hostname or not parsed.port:
        return False
    address = f"{parsed.hostname}:{parsed.port}"
    nonce = secrets.token_hex(16)
    try:
        with _DIRECT.open(base + "/hello/proof?nonce=" + nonce, timeout=4) as response:
            body = read_reply(response)
    except (OSError, urllib.error.URLError, ValueError):
        return False
    return isinstance(body, dict) and proof_ok(secret, nonce, address, body.get("proof"))


def probe_serve(ip: str) -> dict | None:
    """GET /hello on port 8780 of one LAN address. Returns {"url", "name"}
    only for a reply that says it is a Neuro serve. Carries no token."""
    if not private_lan(ip):
        return None
    try:
        with socket.create_connection((ip, DASHBOARD_PORT), 0.6):
            pass
    except OSError:
        return None
    url = f"http://{ip}:{DASHBOARD_PORT}"
    try:
        with _DIRECT.open(url + "/hello", timeout=3) as response:
            body = read_reply(response)
    except (OSError, urllib.error.URLError, ValueError):
        return None
    if not isinstance(body, dict) or body.get("app") != "neuroai-mesh" or body.get("role") != "serve":
        return None
    return {"url": url, "name": clip(body.get("name"), 64)}


def discover_serves(probe=probe_serve) -> list[dict]:
    """Every serve on this computer's own private /24, by the LAN sweep's
    address rules (never 26.x, 192.168.56.x or 169.254.x). This computer's
    own LAN address is asked too: the main PC may also run a worker."""
    prefixes, own = ipv4_prefixes()
    lan = lan_host()
    if lan:
        prefixes = [lan.rsplit(".", 1)[0]]
    hosts = sweep_hosts(prefixes, set())
    if lan and lan in hosts:
        hosts.remove(lan)
        hosts.insert(0, lan)
    if not hosts:
        return []
    with ThreadPoolExecutor(max_workers=32) as pool:
        return [item for item in pool.map(probe, hosts) if item]


def find_mesh_url(discover=discover_serves, prove=serve_proof_ok, say=print) -> str:
    raw = os.environ.get("NEURO_MESH_URL", "").strip()
    if raw:
        return lan_url(raw)
    remembered = remembered_mesh_url()
    if remembered:
        say(f"Main PC: {remembered} (remembered in {MESH_URL_PATH.name}; delete that file to look again).")
        return remembered
    say(f"NEURO_MESH_URL is not set. Looking for the main PC on this network (port {DASHBOARD_PORT})...")
    try:
        found = discover()
    except Exception as exc:  # a failed sweep must not stop the worker
        say(f"The search failed ({type(exc).__name__}).")
        found = []
    secret = saved_token()
    if secret and found:
        # This worker already holds a token: only a serve that holds the same
        # one may ever receive it in an Authorization header.
        proven = [item for item in found if prove(item["url"], secret)]
        if not proven:
            for item in found:
                say(f"  {item['url']} ({item.get('name') or 'no name'}) does not hold this computer's cluster.token.")
            say("Not using it. If that is your main PC, set NEURO_MESH_URL to its address to pair with it again.")
            return ""
        found = proven
    if len(found) == 1:
        url = found[0]["url"]
        remember_mesh_url(url)
        say(f"Found the main PC: {url} ({found[0].get('name') or 'no name'}). Saved in {MESH_URL_PATH.name}.")
        if not secret:
            say("Pairing still needs its code approved on that main PC's page.")
        return url
    if len(found) > 1:
        say("Found more than one main PC on this network:")
        for item in found:
            say(f"  {item['url']} ({item.get('name') or 'no name'})")
        say("Set NEURO_MESH_URL to the one this computer should work for, then start the worker again.")
        return ""
    say(f"No main PC answered. Trying {DEFAULT_MESH_URL} (this computer); set NEURO_MESH_URL=http://<main PC>:{DASHBOARD_PORT} if yours is elsewhere.")
    return lan_url(DEFAULT_MESH_URL)


def mesh_url() -> str:
    """The main PC's address for this worker, found once per process."""
    with _mesh_url_lock:
        if _mesh_url_cache["url"] is None:
            _mesh_url_cache["url"] = find_mesh_url(say=lambda line: print(line, flush=True))
        return _mesh_url_cache["url"]


def machine_busy(gpu: bool = True) -> str:
    """Why this computer should not take another job now, or "". Asked
    before every lease. gpu=False (a check slot): the GPU lock does not
    matter, only generate pauses while training holds the GPU."""
    pct = cpu_busy_pct()
    if pct is not None and pct >= 75:
        return "CPU busy"
    # A generate slot only sends a request to the local model server, whose memory (weights and KV
    # cache) was allocated when it started: it needs little free RAM. A check slot starts a sandboxed
    # process that may use up to ~1 GB. Workers running a model sat paused at ~800 MB free otherwise.
    # A Steam Deck keeps its model in GPU-mapped (GTT) memory that Linux counts as used, so it sits
    # near 250 MB available while generating fine: 128 MB still leaves room for the worker itself.
    floor = 128 if gpu else 1024
    if int(ram_mib().get("free_mib") or 0) < floor:
        return "low memory"
    # NEURO_MODEL_ON_CPU=1: the local model server runs on the CPU only (llama-server -ngl 0), so a
    # training run on this computer's GPU does not stop it. This lets the training PC's idle CPU and
    # RAM generate data too.
    on_cpu = os.environ.get("NEURO_MODEL_ON_CPU", "").strip() == "1"
    if gpu and not on_cpu and os.name == "nt" and lock_held(GPU_LOCK):
        return "training holds the GPU"
    return ""


def _mesh_post(base: str, path: str, body: dict, timeout: float) -> tuple[int, object]:
    # saved_token, not token: a worker without the main PC's token must not
    # make up its own one.
    secret = saved_token()
    if not secret:
        raise JobAuthError(missing=True)
    request = urllib.request.Request(
        base + path,
        data=json.dumps(body, ensure_ascii=False).encode("utf-8"),
        method="POST",
        headers={"Authorization": "Bearer " + secret, "Content-Type": "application/json"},
    )
    try:
        with _DIRECT.open(request, timeout=timeout) as response:
            if response.status == 204:
                return 204, None
            return response.status, read_reply(response)
    except urllib.error.HTTPError as exc:
        if exc.code == 401:
            raise JobAuthError() from None
        return exc.code, None


def slot_kinds(slot: str) -> list[str]:
    """The kinds one slot leases: generate slots only generate, check slots
    every other kind this worker runs."""
    gen = slot.startswith("g")
    return [kind for kind in worker_kinds() if (kind == "generate") == gen]


# The most generate tokens this worker asks for. Starts at GENERATE_MAX_TOKENS and drops by 1024
# each time the local llama-server says its context is full (a laptop whose parallel slots share
# 16k of context cannot finish four 4096-token replies at once).
_gen_cap = {"tokens": GENERATE_MAX_TOKENS}
CONTEXT_FULL = re.compile(r"context size has been exceeded|exceeds the available context|context is full|n_ctx", re.IGNORECASE)


def _slot_pause(slot: str, why: str) -> None:
    with _job_lock:
        if why:
            _job_paused[slot] = why
        else:
            _job_paused.pop(slot, None)


def slot_once(slot: str, base: str, node: str, post=_mesh_post, run=run_job, sleep=time.sleep) -> float:
    """One lease-run-report round of one job slot. Returns the seconds to
    wait before the next round. Leases with the same protocol as before,
    plus the slot id, so serve gives each slot its own job."""
    gen = slot.startswith("g")
    plan = slot_plan()
    if int(slot[1:]) >= (plan["gen"] if gen else plan["cpu"]):
        _slot_pause(slot, "")  # this slot is not in use now
        return 60.0
    kinds = slot_kinds(slot)
    busy = machine_busy(gpu=gen)
    _slot_pause(slot, busy)
    if not kinds or busy:
        return 15.0
    body = {"node": node, "slot": slot, "kinds": kinds, "gen_max": _gen_cap["tokens"],
            "test_langs": sorted(sandbox_status()["langs"])}
    code, job = post(base, "/jobs/next", body, 10)
    with _job_lock:
        _job_state["told"] = ""
    if not (code == 200 and isinstance(job, dict) and job.get("kind") in kinds and job.get("id") and job.get("lease")):
        return 15.0
    jid, kind = str(job["id"]), str(job["kind"])
    with _job_lock:
        if jid in _job_held:
            # An older serve gives every slot of a computer the same job.
            # One slot runs it; the others wait.
            return 15.0
        _job_held.add(jid)
        _job_running[slot] = kind
    try:
        try:
            ok, output = run(job)
        except Exception as exc:  # a handler bug fails this job, not the slot
            ok, output = False, {"error": clip(f"{type(exc).__name__} in the worker", 300)}
        if len(json.dumps(output, ensure_ascii=False)) > JOB_OUTPUT_LIMIT:
            ok, output = False, {"error": "output is too large"}
        result = {"id": job["id"], "lease": job["lease"], "node": node, "ok": ok, "output": output}
        for attempt in range(3):
            try:
                post(base, "/jobs/result", result, 30)
                break
            except (OSError, urllib.error.URLError, ValueError):
                sleep(5 * (attempt + 1))
        with _job_lock:
            _job_state["done" if ok else "failed"] = int(_job_state.get("done" if ok else "failed") or 0) + 1
    finally:
        with _job_lock:
            _job_held.discard(jid)
            _job_running.pop(slot, None)
    error = str(output.get("error") or "") if not ok and isinstance(output, dict) else ""
    if gen and CONTEXT_FULL.search(error):
        # The local model's context (shared by its parallel slots) cannot hold this long a reply:
        # ask the main PC for shorter jobs from now on. The long ones go to computers that fit them.
        with _job_lock:
            before = _gen_cap["tokens"]
            _gen_cap["tokens"] = max(LEGACY_GEN_MAX // 2, before - 1024)
        if _gen_cap["tokens"] != before:
            print(f"The local model ran out of context; asking for jobs up to {_gen_cap['tokens']} tokens now.", flush=True)
    if gen and error and TRANSIENT_MODEL_ERROR.search(error):
        # The model is overloaded or restarting: this slot rests, so the other slots finish and free
        # the model's memory (one fewer parallel reply), instead of failing job after job.
        _slot_pause(slot, "model busy, resting")
        return 90.0
    return 1.0


def slot_loop(slot: str, base: str, node: str, rounds: int | None = None) -> None:
    """One job slot, forever (or `rounds` rounds in tests). Every error stays
    inside this slot: the other slots keep working."""
    done = 0
    while rounds is None or done < rounds:
        done += 1
        try:
            wait = slot_once(slot, base, node)
        except JobAuthError as exc:
            # Never print the token. Only say that it is missing or does not match.
            state = "no token" if exc.missing else "token"
            with _job_lock:
                first = _job_state.get("told") != state
                _job_state["told"] = state
                _job_paused[slot] = state
            if first:
                print(
                    "Jobs paused: this computer has no cluster.token. Pairing with the main PC starts."
                    if exc.missing
                    else "Jobs paused: this computer's cluster.token is not the main PC's token. Pairing with the main PC starts.",
                    flush=True,
                )
            want_pairing()
            wait = 600.0
        except (OSError, urllib.error.URLError, ValueError):
            wait = 30.0
        except Exception as exc:
            print(f"job slot {slot}: {type(exc).__name__}", flush=True)
            wait = 60.0
        if rounds is not None and done >= rounds:
            return
        # A finished pairing wakes every waiting slot at once, without a restart.
        if _token_changed.wait(wait):
            time.sleep(0.5)  # let every waiting slot see it before it is cleared
            _token_changed.clear()


def job_loop() -> None:
    with _job_lock:
        _job_state["on"] = True
    base = mesh_url()
    if not base:
        with _job_lock:
            _job_state.update({"on": False, "paused": "no main PC address (set NEURO_MESH_URL to a LAN address)"})
        print("Jobs are off: no main PC address. Set NEURO_MESH_URL to the main PC's LAN address.", flush=True)
        return
    node = socket.gethostname()
    print("Tests: " + sandbox_status()["note"], flush=True)
    time.sleep(20)
    plan = slot_plan(refresh=True)
    # Generate threads: NEURO_GEN_SLOTS, else up to 4; those above the model
    # server's slot count stay idle until slot_plan() says otherwise.
    gen_threads = _slot_env("NEURO_GEN_SLOTS")
    gen_threads = GEN_SLOT_DEFAULT_CAP if gen_threads is None else gen_threads
    cpu_threads = cpu_slot_count()
    if not gen_threads and not cpu_threads:
        with _job_lock:
            _job_state.update({"on": False, "paused": "no job slots (NEURO_GEN_SLOTS and NEURO_CPU_SLOTS are 0)"})
        print("Jobs are off: NEURO_GEN_SLOTS and NEURO_CPU_SLOTS are both 0.", flush=True)
        return
    print(f"Job slots: generate {plan['gen']} (up to {gen_threads}), checks {cpu_threads}.", flush=True)
    slots = [f"g{index}" for index in range(gen_threads)] + [f"c{index}" for index in range(cpu_threads)]
    for slot in slots:
        threading.Thread(target=slot_loop, args=(slot, base, node), name=f"neuro-job-{slot}", daemon=True).start()


def read_job_file(kind_arg: str, path: str) -> tuple[list[tuple[str, dict]], list[str]]:
    """Parse and check a `jobs add` file without queueing anything or
    contacting serve. Returns ([(kind, raw row)], ["line N: why", ...])."""
    kind = kind_arg
    need_verify = kind == "generate-verify"
    if need_verify:
        kind = "generate"
    if kind not in JOB_KINDS:
        raise SystemExit("Unknown kind. Use generate-verify or one of " + ", ".join(JOB_KINDS))
    rows: list[tuple[str, dict]] = []
    errors: list[str] = []
    planned = 0
    try:
        handle = open(path, "r", encoding="utf-8-sig")
    except OSError as exc:
        raise SystemExit(f"Cannot read {path}: {exc.strerror}")
    with handle:
        for number, line in enumerate(handle, 1):
            if not line.strip():
                continue
            if planned >= 100000:
                errors.append(f"line {number}: stopped at 100000 jobs")
                break
            try:
                raw = json.loads(line)
                if need_verify and not (isinstance(raw, dict) and isinstance(raw.get("verify"), dict)):
                    raise JobError("generate-verify rows need a verify object")
                validate_job(kind, raw)
                rows.append((kind, raw))
                planned += len(expand_samples(kind, raw))
            except ValueError as exc:
                errors.append(f"line {number}: {exc}")
    return rows, errors


def jobs_main(args: list[str]) -> None:
    usage = (
        "use: mesh.py jobs add <kind|generate-verify> <file.jsonl> | mesh.py jobs status"
        " | mesh.py jobs export [--per-prompt N] [--require-agreement] <name.jsonl> | mesh.py jobs export --pairs <path.jsonl>"
    )
    if not args:
        raise SystemExit(usage)
    queue = JobQueue()
    command = args[0]
    if command == "add" and len(args) == 3:
        rows, errors = read_job_file(args[1], args[2])
        needed ={kind for kind, _ in rows} | {raw["verify"].get("kind") for _, raw in rows if isinstance(raw.get("verify"), dict)}
        known = serve_job_kinds()
        if known is not None and not needed <= known:
            missing = ", ".join(sorted(str(item) for item in needed - known))
            raise SystemExit(
                f"mesh.py serve is running an older mesh.py that does not know {missing}. "
                "Restart serve first, then add these jobs. Nothing was queued."
            )
        added = 0
        for start in range(0, len(rows), 500):
            added += len(queue.add_many(rows[start:start + 500]))
        print(f"Queued {added} {args[1]} jobs. Skipped {len(errors)} rows.", flush=True)
        for error in errors[:10]:
            print("  " + error, flush=True)
        return
    if command == "status" and len(args) == 1:
        summary = None
        try:
            with _DIRECT.open(f"http://127.0.0.1:{DASHBOARD_PORT}/api/jobs", timeout=3) as response:
                summary = read_reply(response)
            source = "live from mesh.py serve"
        except (OSError, urllib.error.URLError, ValueError):
            summary = None
        if not isinstance(summary, dict):
            summary = queue.summary()
            source = "from jobs/queue.jsonl (serve is not answering, so leases are not shown)"
        print(f"Jobs {source}")
        print(f"{'kind':<22}{'queued':>8}{'leased':>8}{'done':>8}{'failed':>8}{'passed':>8}")
        for kind, row in (summary.get("kinds") or {}).items():
            print(f"{kind:<22}{row.get('queued', 0):>8}{row.get('leased', 0):>8}{row.get('done', 0):>8}{row.get('failed', 0):>8}{row.get('passed', 0):>8}")
        for name, row in (summary.get("nodes") or {}).items():
            running = ", ".join(str(kind) for kind in row.get("running") or []) or str(row.get("now") or "")
            print(f"  {name}: done {row.get('done', 0)}, failed {row.get('failed', 0)}" + (f", running {running}" if running else ""))
        print(f"Verified rows ready to export: {summary.get('verified_rows', 0)}")
        return
    if command == "export" and len(args) == 3 and args[1] == "--pairs":
        try:
            out, count = queue.export_pairs(args[2])
        except JobError as exc:
            raise SystemExit(str(exc))
        print(f"Wrote {count} chosen/rejected pairs to {out}", flush=True)
        return
    if command == "export" and len(args) >= 2:
        per_prompt = 1
        require_agreement = False
        rest = args[1:-1]
        while rest:
            flag = rest.pop(0)
            if flag == "--per-prompt" and rest and rest[0].isdigit():
                per_prompt = int(rest.pop(0))
            elif flag == "--require-agreement":
                require_agreement = True
            else:
                raise SystemExit(usage)
        if args[-1].startswith("--"):
            raise SystemExit(usage)
        try:
            out, count = queue.export(args[-1], per_prompt, require_agreement)
        except JobError as exc:
            raise SystemExit(str(exc))
        extra = f" (cross-check agreement >= {AGREEMENT_MIN})" if require_agreement else ""
        print(f"Wrote {count} verified rows{extra} to {out}", flush=True)
        return
    raise SystemExit(usage)


def serve_job_kinds() -> set[str] | None:
    """Job kinds the running `mesh.py serve` on this PC knows, or None when
    it does not answer. An older serve silently drops jobs of kinds it does
    not know, so `jobs add` checks first."""
    try:
        with _DIRECT.open(f"http://127.0.0.1:{DASHBOARD_PORT}/api/jobs", timeout=3) as response:
            summary = read_reply(response)
    except (OSError, urllib.error.URLError, ValueError):
        return None
    kinds = summary.get("kinds") if isinstance(summary, dict) else None
    return {str(kind) for kind in kinds} if isinstance(kinds, dict) else None


class Handler(BaseHTTPRequestHandler):
    mode = "serve"

    def log_message(self, fmt: str, *args) -> None:
        return

    def parse_request(self) -> bool:
        self._body_done = False
        if not super().parse_request():
            return False
        # Both ports listen on 0.0.0.0, so they are also reachable on the VPN
        # (26.x) and host-only (192.168.56.x) adapters. Only loopback and the
        # private home LAN get an answer.
        if not lan_client(self.client_address[0]):
            self._body_done = True  # no reading from outside the LAN
            self.send_json(403, {"error": "LAN only"})
            return False
        if self.mode == "serve" and not browser_host_ok(self.headers.get("Host", "")):
            self.send_json(403, {"error": "Open this page by its LAN address."})
            return False
        if self.command in {"POST", "DELETE", "PUT", "PATCH"} and not origin_ok(self.headers.get("Origin"), self.headers.get("Host")):
            # Another web site open in a browser on the LAN cannot press Stop.
            self.send_json(403, {"error": "Requests must come from this page."})
            return False
        return True

    def send_json(self, code: int, payload: dict) -> None:
        if code >= 400 and self.command in {"POST", "PUT", "PATCH", "DELETE"} and not getattr(self, "_body_done", True):
            self.drain_body()
        raw = json.dumps(payload, ensure_ascii=False).encode("utf-8")
        self.send_response(code)
        self.send_header("Content-Type", "application/json; charset=utf-8")
        self.send_header("Content-Length", str(len(raw)))
        self.send_header("Cache-Control", "no-store")
        self.send_header("X-Content-Type-Options", "nosniff")
        self.end_headers()
        self.wfile.write(raw)

    def send_vendor(self, name: str) -> None:
        """A file from vendor/, only when its exact name is in VENDOR_FILES.
        The name is compared as given, so "../x", "a/b" or "%2e%2e" never
        match and nothing outside vendor/ can be read."""
        kind = VENDOR_FILES.get(name)
        if not kind:
            self.send_json(404, {"error": "not found"})
            return
        try:
            raw = (VENDOR_DIR / name).read_bytes()
        except OSError:
            self.send_json(404, {"error": "not found"})
            return
        self.send_response(200)
        self.send_header("Content-Type", kind)
        self.send_header("Content-Length", str(len(raw)))
        # Pinned versions: a week in the browser cache is fine.
        self.send_header("Cache-Control", "public, max-age=604800")
        self.send_header("X-Content-Type-Options", "nosniff")
        self.end_headers()
        self.wfile.write(raw)

    def send_empty(self, code: int = 204) -> None:
        self.send_response(code)
        self.send_header("Content-Length", "0")
        self.end_headers()

    def main_pc_page(self) -> bool:
        """The request comes from a browser on the main PC itself (loopback
        or this PC's own LAN address), never from another computer."""
        raw = str(self.client_address[0]).split("%", 1)[0]
        try:
            addr = ipaddress.ip_address(raw)
        except ValueError:
            return False
        if addr.version == 6 and addr.ipv4_mapped:
            addr = addr.ipv4_mapped
        if addr.is_loopback:
            return True
        own = lan_host()
        return bool(own) and str(addr) == own

    def worker_token_ok(self) -> bool:
        """Worker side: the request must carry `Authorization: Bearer
        <cluster.token>`. Sends the refusal and returns False otherwise. A
        worker with no cluster.token refuses with a clear message and does
        not make one up."""
        secret = saved_token()
        if not secret:
            self.send_json(403, {"error": NO_TOKEN_NOTE})
            return False
        if not authorized(self.headers.get("Authorization"), secret):
            self.send_json(401, {"error": "token"})
            return False
        return True

    def drain_body(self, limit: int = 65536) -> None:
        """Read (and drop) a small unread request body before a refusal.
        Closing with it unread makes Windows reset the connection, so the
        caller saw a dropped socket instead of the 401/403."""
        self._body_done = True
        try:
            length = int(self.headers.get("Content-Length", "0") or 0)
        except (TypeError, ValueError):
            return
        if 0 < length <= limit:
            try:
                self.connection.settimeout(2)
                self.rfile.read(length)
            except OSError:
                pass

    def read_json(self, limit: int = 10000) -> dict:
        try:
            length = int(self.headers.get("Content-Length", "0") or 0)
        except (TypeError, ValueError):
            return {}
        if length <= 0 or length > limit:
            return {}
        self._body_done = True
        try:
            data = json.loads(self.rfile.read(length).decode("utf-8"))
        except (UnicodeDecodeError, json.JSONDecodeError, ValueError):
            return {}
        return data if isinstance(data, dict) else {}

    def do_GET(self) -> None:
        path = urlparse(self.path).path
        if self.mode == "work":
            if path == "/hello":
                payload = local_status()
                payload["app"] = "neuroai-mesh"
                payload["role"] = "work"
                payload["port"] = WORKER_PORT
                self.send_json(200, payload)
                return
            if path == "/hello/proof":
                # Read-only. Answers an HMAC of serve's nonce; never the token
                # or a plain hash of it.
                if not proof_rate_ok(str(self.client_address[0])):
                    self.send_json(429, {"error": "Too many token checks. Try again in a minute."})
                    return
                nonce = (parse_qs(urlparse(self.path).query).get("nonce") or [""])[0]
                if not _NONCE.match(nonce):
                    self.send_json(400, {"error": "Send nonce as 32 to 128 lowercase hex characters."})
                    return
                secret = saved_token()
                if not secret:
                    self.send_json(200, {"app": "neuroai-mesh", "has_token": False, "note": NO_TOKEN_NOTE})
                    return
                local = self.connection.getsockname()
                address = f"{local[0]}:{local[1]}"
                self.send_json(200, {"app": "neuroai-mesh", "has_token": True, "proof": proof_digest(secret, nonce, address)})
                return
            if path != "/status":
                self.send_json(404, {"error": "not found"})
                return
            if not self.worker_token_ok():
                return
            self.send_json(200, local_status())
            return
        if path == "/hello":
            # Lets a worker without NEURO_MESH_URL find this main PC. Says
            # only what it is; holds nothing secret.
            self.send_json(200, {"app": "neuroai-mesh", "role": "serve", "port": DASHBOARD_PORT, "name": clip(socket.gethostname(), 64)})
            return
        if path == "/hello/proof":
            # The mirror of the worker's proof: a worker that already holds a
            # token checks a found main PC with it before sending it. Answers
            # an HMAC of the worker's nonce; never the token or a hash of it.
            if not proof_rate_ok(str(self.client_address[0])):
                self.send_json(429, {"error": "Too many token checks. Try again in a minute."})
                return
            nonce = (parse_qs(urlparse(self.path).query).get("nonce") or [""])[0]
            if not _NONCE.match(nonce):
                self.send_json(400, {"error": "Send nonce as 32 to 128 lowercase hex characters."})
                return
            secret = saved_token()
            if not secret:
                self.send_json(200, {"app": "neuroai-mesh", "role": "serve", "has_token": False})
                return
            local = self.connection.getsockname()
            address = f"{local[0]}:{local[1]}"
            self.send_json(200, {"app": "neuroai-mesh", "role": "serve", "has_token": True, "proof": proof_digest(secret, nonce, address)})
            return
        if path in {"/", "/dashboard"}:
            raw = DASHBOARD.read_bytes()
            self.send_response(200)
            self.send_header("Content-Type", "text/html; charset=utf-8")
            self.send_header("Content-Length", str(len(raw)))
            # The page has Stop buttons; it must not be framed by another site.
            self.send_header("X-Frame-Options", "DENY")
            self.send_header("Content-Security-Policy", "frame-ancestors 'none'")
            self.send_header("Cache-Control", "no-store")
            self.end_headers()
            self.wfile.write(raw)
            return
        if path == "/api/mesh":
            try:
                snapshot_body = mesh_snapshot()
            except Exception as exc:  # the page must get an answer, not a dropped socket
                print(f"mesh snapshot failed: {type(exc).__name__}: {exc}", flush=True)
                self.send_json(500, {"error": "The status could not be read. The page keeps the last one."})
                return
            self.send_json(200, snapshot_body)
            return
        if path == "/api/jobs":
            self.send_json(200, job_queue().summary())
            return
        if path == "/api/agents":
            try:
                from agents import snapshot

                body = snapshot()
            except Exception as exc:
                print(f"agents snapshot failed: {type(exc).__name__}: {exc}", flush=True)
                self.send_json(500, {"error": "agents unavailable", "workers": []})
                return
            self.send_json(200, body)
            return
        if path == "/api/token":
            self.send_json(404, {"error": "not found"})
            return
        if path == "/api/pair":
            if not self.main_pc_page():
                self.send_json(403, {"error": "Only the main PC's own page lists new computers.", "pending": []})
                return
            self.send_json(200, {"pending": pair_list()})
            return
        if path == "/api/telemetry":
            raw_window = (parse_qs(urlparse(self.path).query).get("window") or [""])[0]
            try:
                window = int(raw_window) if raw_window else TELEMETRY_WINDOW_S
            except ValueError:
                window = TELEMETRY_WINDOW_S
            self.send_json(200, TELEMETRY.view(window))
            return
        if path.startswith("/vendor/"):
            self.send_vendor(path[len("/vendor/"):])
            return
        if path.startswith("/platforms/"):
            name = Path(path).name
            if name not in {"start-mac.sh", "start-linux.sh"}:
                self.send_json(404, {"error": "not found"})
                return
            raw = (ROOT / "platforms" / name).read_bytes()
            self.send_response(200)
            self.send_header("Content-Type", "text/plain; charset=utf-8")
            self.send_header("Content-Length", str(len(raw)))
            self.end_headers()
            self.wfile.write(raw)
            return
        self.send_json(404, {"error": "not found"})

    def do_POST(self) -> None:
        path = urlparse(self.path).path
        if self.mode == "work":
            # Every POST on the worker changes or starts something (agent
            # check, process stop, RPC start/stop), so all of them need the
            # main PC's token on top of the LAN-only and Origin checks. Only
            # `mesh.py serve` sends it; the page never has it.
            if not self.worker_token_ok():
                return
            if path == "/check":
                body = self.read_json()
                try:
                    rounds = int(body.get("rounds") or 3)
                except (TypeError, ValueError):
                    rounds = 3
                rounds = max(1, min(rounds, 5))
                passed = 0
                try:
                    from agents import drill_round

                    for index in range(1, rounds + 1):
                        row = drill_round(index)
                        passed += 1 if isinstance(row, dict) and row.get("ok") else 0
                except Exception as exc:
                    print(f"agent check failed: {type(exc).__name__}: {exc}", flush=True)
                    self.send_json(500, {"ok": False, "error": "The agent check could not run on this computer.", "host": socket.gethostname()})
                    return
                self.send_json(200, {"ok": passed == rounds, "passed": passed, "rounds": rounds, "host": socket.gethostname()})
                return
            if path == "/process/stop":
                body = self.read_json()
                try:
                    pid = int(body.get("pid") or 0)
                except (TypeError, ValueError):
                    pid = 0
                self.send_json(200, stop_owned_process(pid, str(body.get("name") or "")))
                return
            if path in {"/rpc/start", "/rpc/stop"}:
                try:
                    if path == "/rpc/start":
                        self.send_json(200, start_rpc())
                        return
                    self.send_json(200, stop_rpc())
                    return
                except RuntimeError as exc:
                    self.send_json(409, {"error": str(exc)})
                    return
                except OSError:
                    self.send_json(409, {"error": "ggml-rpc-server could not be started on this machine."})
                    return
            self.send_json(404, {"error": "not found"})
            return
        if path in {"/pair/request", "/pair/status"}:
            # Worker-facing, no token yet: LAN-only, Host and Origin checks
            # already ran in parse_request. Rate-limited per client address.
            client = str(self.client_address[0])
            body = self.read_json(4000)
            handler = pair_request if path == "/pair/request" else pair_status
            code, payload = handler(client, body)
            self.send_json(code, payload)
            return
        if path in {"/api/pair/approve", "/api/pair/deny"}:
            # Only the page open on the main PC itself may approve.
            if not self.main_pc_page():
                self.send_json(403, {"error": "Approve new computers on the main PC's own page."})
                return
            code, payload = pair_decide(str(self.read_json().get("id") or ""), path.endswith("/approve"))
            self.send_json(code, payload)
            return
        if path in {"/jobs/next", "/jobs/result"}:
            # Workers pull data-factory jobs. Bearer token required; a job is
            # data for a built-in handler, never a command.
            if not authorized(self.headers.get("Authorization")):
                self.send_json(401, {"error": "token"})
                return
            body = self.read_json(JOB_RESULT_LIMIT)
            if not body:
                self.send_json(400, {"error": "Send a JSON object."})
                return
            queue = job_queue()
            if path == "/jobs/next":
                job = queue.lease(str(body.get("node") or ""), body.get("kinds"), gen_max=body.get("gen_max"),
                                  test_langs=body.get("test_langs"), slot=body.get("slot"))
                if job is None:
                    self.send_empty(204)
                    return
                self.send_json(200, job)
                return
            accepted, note = queue.finish(body)
            self.send_json(200 if accepted else 409, {"accepted": accepted, "note": note})
            return
        try:
            if path == "/api/agents/jobs":
                from agents import enqueue

                body = self.read_json()
                title = str(body.get("title", "")).strip()
                prompt = str(body.get("prompt", "")).strip()
                if not prompt:
                    self.send_json(400, {"error": "Missing task"})
                    return
                self.send_json(200, {"id": enqueue(title or prompt[:40], prompt)})
                return
            if path == "/api/agents/swarm":
                from agents import ensure_workers, seed

                body = self.read_json()
                requested = int(body.get("count", 4) or 4)
                seeded = seed()
                self.send_json(200, {"seeded": seeded, **ensure_workers(requested)})
                return
            if path == "/api/models":
                body = self.read_json()
                try:
                    self.send_json(200, save_model(str(body.get("path") or "")))
                except ValueError as exc:
                    self.send_json(400, {"error": str(exc)})
                return
            if path == "/api/training/start":
                _training_cache["at"] = 0.0
                view = training_view()
                self.send_json(
                    409,
                    {
                        "error": "A training run is already going. No dataset file was changed." if view.get("running") else "The training state is unknown, so no second run is started. No dataset file was changed." if view.get("running") is None else "This page does not start training. Start your trainer yourself; its log appears on the Training page once training.json points at it. No dataset file was changed.",
                        "datasets": view.get("datasets") or [],
                    },
                )
                return
            if path == "/api/agents/stop":
                from agents import stop_workers

                self.send_json(200, stop_workers())
                return
            if path == "/api/model/stop":
                self.send_json(200, stop_model())
                return
            if path == "/api/process/stop":
                body = self.read_json()
                try:
                    pid = int(body.get("pid") or 0)
                except (TypeError, ValueError):
                    pid = 0
                self.send_json(200, stop_at(str(body.get("url") or ""), pid, str(body.get("name") or "")))
                return
            if path == "/api/share/offer":
                self.send_json(200, offer_share())
                return
            if path == "/api/chat":
                message = str(self.read_json(64000).get("message", "")).strip()
                if not message:
                    self.send_json(400, {"error": "Missing message"})
                    return
                self.send_json(200, ask_local(message))
                return
            if path == "/api/specs":
                self.send_json(200, save_spec(self.read_json()))
                return
            if path == "/api/helpers/check":
                self.send_json(200, check_helpers())
                return
            if path == "/api/nodes":
                url = clean_url(str(self.read_json().get("url", "")))
                if not private_lan(urlparse(url).hostname or ""):
                    raise ValueError("Only a computer on the home LAN can be added.")
                add_node(url)
                self.send_json(200, mesh_snapshot())
                return
            if path == "/api/local/rpc/start":
                self.send_json(200, start_rpc())
                return
            if path == "/api/local/rpc/stop":
                self.send_json(200, stop_rpc())
                return
            if path.startswith("/api/remote/"):
                nodes = {item["url"]: item for item in read_nodes()}
                target = self.headers.get("X-Node-Url", "")
                if target not in nodes:
                    self.send_json(404, {"error": "Machine is not in the cluster"})
                    return
                action = path.rsplit("/", 1)[-1]
                if action not in {"start", "stop"}:
                    self.send_json(404, {"error": "not found"})
                    return
                try:
                    payload = worker_call(target, f"/rpc/{action}", {}, 5)
                except WorkerProofError as exc:
                    self.send_json(409, {"error": str(exc)})
                    return
                except urllib.error.HTTPError as exc:
                    self.send_json(502, {"error": worker_refusal(exc)})
                    return
                self.send_json(200, payload if isinstance(payload, dict) else {"ok": True})
                return
        except RuntimeError as exc:
            self.send_json(409, {"error": str(exc)})
            return
        except ValueError as exc:
            self.send_json(400, {"error": str(exc)})
            return
        except (OSError, urllib.error.URLError, json.JSONDecodeError) as exc:
            self.send_json(502, {"error": str(exc)})
            return
        self.send_json(404, {"error": "not found"})

    def do_DELETE(self) -> None:
        if self.mode != "serve":
            self.send_json(404, {"error": "not found"})
            return
        query = parse_qs(urlparse(self.path).query)
        name = (query.get("name") or [""])[0].strip()
        try:
            with _files_lock:
                if name:
                    write_specs([spec for spec in read_specs(strict=True) if spec.get("name") != name])
                else:
                    system = (query.get("os") or [""])[0]
                    if system in {"mac", "linux"}:
                        specs = read_specs(strict=True)
                        gone = {spec.get("url") for spec in specs if spec.get("os") == system and spec.get("url")}
                        write_specs([spec for spec in specs if spec.get("os") != system])
                        if gone:
                            write_nodes([item for item in read_nodes(strict=True) if item.get("url") not in gone])
                    else:
                        url = clean_url((query.get("url") or [""])[0])
                        write_nodes([item for item in read_nodes(strict=True) if item["url"] != url])
        except ValueError as exc:
            self.send_json(400, {"error": str(exc)})
            return
        except (RuntimeError, OSError) as exc:
            self.send_json(409, {"error": str(exc)})
            return
        self.send_json(200, mesh_snapshot())


def serve(mode: str) -> None:
    Handler.mode = mode
    _ROLE["name"] = mode
    if mode == "serve":
        server = ThreadingHTTPServer(("0.0.0.0", DASHBOARD_PORT), Handler)
        print(f"Neuro dashboard http://127.0.0.1:{DASHBOARD_PORT}", flush=True)
        if lan_host():
            print(f"On the Mac open http://{lan_host()}:{DASHBOARD_PORT}", flush=True)
        print(f"token id {token_id()}", flush=True)
        start_discovery()
        threading.Thread(target=helper_loop, name="neuro-helpers", daemon=True).start()
        if os.environ.get("NEURO_TELEMETRY", "1").strip() != "0":
            threading.Thread(target=telemetry_loop, args=(TELEMETRY,), name="neuro-telemetry", daemon=True).start()
    else:
        server = ThreadingHTTPServer(("0.0.0.0", WORKER_PORT), Handler)
        print(f"Neuro worker on port {WORKER_PORT}", flush=True)
        # A worker never makes its own token: it pairs with the main PC.
        if saved_token():
            print(f"token id {token_id()}", flush=True)
        base = mesh_url()
        if base:
            threading.Thread(target=pair_loop, args=(base,), name="neuro-pair", daemon=True).start()
            if not saved_token():
                print("This computer has no cluster.token yet. Pairing with the main PC starts.", flush=True)
                want_pairing()
        elif not saved_token():
            print(NO_TOKEN_NOTE, flush=True)
        if os.environ.get("NEURO_JOBS", "1").strip() != "0":
            threading.Thread(target=job_loop, name="neuro-jobs", daemon=True).start()
    if mode == "serve":
        token()
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        print("\nStopped.", flush=True)
    finally:
        server.server_close()


def main() -> None:
    command = sys.argv[1] if len(sys.argv) > 1 else "serve"
    if command == "agent":
        from agents import worker_main

        worker_main()
        return
    if command == "agents":
        from agents import terminal

        requested = int(sys.argv[2]) if len(sys.argv) > 2 else 4
        terminal(requested)
        return
    if command == "code":
        from agents import code_main

        code_main(" ".join(sys.argv[2:]).strip())
        return
    if command == "drill":
        from agents import drill_main

        rounds = int(sys.argv[2]) if len(sys.argv) > 2 else 10
        drill_main(rounds)
        return
    if command == "swarm":
        from agents import ensure_workers, seed

        requested = int(sys.argv[2]) if len(sys.argv) > 2 else 4
        seeded = seed()
        result = ensure_workers(requested)
        print(f"seeded {seeded} jobs; started {result['started']} of {result['requested']}", flush=True)
        print("pids " + " ".join(str(pid) for pid in result["pids"]), flush=True)
        return
    if command == "jobs":
        jobs_main(sys.argv[2:])
        return
    if command not in {"serve", "work"}:
        raise SystemExit("use: mesh.py serve | mesh.py work | mesh.py jobs ... | mesh.py code [TASK] | mesh.py agents [COUNT] | mesh.py drill [ROUNDS]")
    serve(command)


if __name__ == "__main__":
    main()
