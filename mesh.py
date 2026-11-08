"""Neuro Mesh: one dashboard for Windows, Linux, and Mac workers.

Workers report memory and can start the local llama.cpp RPC server.
They do not accept shell commands.
"""

from __future__ import annotations

import ctypes
import hashlib
import hmac
import json
import os
import platform
import shutil
import socket
import subprocess
import sys
import threading
import time
import urllib.error
import urllib.request
from concurrent.futures import ThreadPoolExecutor
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from urllib.parse import parse_qs, urlparse

ROOT = Path(__file__).resolve().parent
TOKEN_PATH = ROOT / "cluster.token"
NODES_PATH = ROOT / "nodes.json"
SPECS_PATH = ROOT / "specs.json"
PID_PATH = ROOT / "rpc.pid"
DASHBOARD = ROOT / "dashboard.html"
RPC_PORT = 50052
DASHBOARD_PORT = 8780
WORKER_PORT = 8781


def token() -> str:
    if not TOKEN_PATH.exists():
        TOKEN_PATH.write_text(os.urandom(24).hex(), encoding="utf-8")
    return TOKEN_PATH.read_text(encoding="utf-8").strip()


def token_id() -> str:
    return hashlib.sha256(token().encode("utf-8")).hexdigest()[:8]


_found_lock = threading.Lock()
_found: list[dict] = []


def authorized(header: str | None) -> bool:
    expected = "Bearer " + token()
    got = header or ""
    return hmac.compare_digest(got, expected)


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
    return []


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
    cards = []
    for line in out.splitlines():
        parts = [part.strip() for part in line.split(",")]
        if len(parts) < 4:
            continue
        cards.append(
            {
                "name": parts[0],
                "memory_used_mib": int(float(parts[1])),
                "memory_total_mib": int(float(parts[2])),
                "temperature_c": int(float(parts[3])),
            }
        )
    return cards


def process_running(image: str) -> bool:
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
    if override and Path(override).exists():
        return override
    names = ["ggml-rpc-server.exe", "ggml-rpc-server"]
    candidates = [
        Path(r"D:\liliai\tools\llama-cpp-cuda\ggml-rpc-server.exe"),
        ROOT / "ggml-rpc-server.exe",
        ROOT / "ggml-rpc-server",
    ]
    for candidate in candidates:
        if candidate.exists():
            return str(candidate)
    for name in names:
        found = shutil.which(name)
        if found:
            return found
    return None


def heavy_processes() -> list[dict]:
    rows: list[dict] = []
    try:
        if os.name == "nt":
            out = subprocess.check_output(["tasklist", "/FO", "CSV", "/NH"], text=True, timeout=8, stderr=subprocess.DEVNULL)
            for line in out.splitlines():
                parts = [part.strip().strip('"') for part in line.split('","')]
                if len(parts) < 5:
                    continue
                name = parts[0].strip('"')
                mem = parts[-1].replace(",", "").replace(" K", "").replace(" ", "")
                if not mem.isdigit():
                    continue
                mb = int(mem) // 1024
                if mb < 80:
                    continue
                role = "other"
                lowered = name.lower()
                if lowered.startswith("llama-server"):
                    role = "model"
                elif lowered.startswith("python"):
                    role = "python"
                rows.append({"name": name, "mb": mb, "role": role})
        else:
            out = subprocess.check_output(["ps", "-axo", "rss=,comm="], text=True, timeout=4)
            for line in out.splitlines():
                bits = line.strip().split(None, 1)
                if len(bits) < 2 or not bits[0].isdigit():
                    continue
                mb = int(bits[0]) // 1024
                if mb < 80:
                    continue
                name = bits[1].replace("\\", "/").split("/")[-1][:40]
                rows.append({"name": name, "mb": mb, "role": "other"})
    except (OSError, subprocess.SubprocessError, ValueError):
        return []
    rows.sort(key=lambda item: item["mb"], reverse=True)
    return rows[:6]


def dataset_index() -> list[dict]:
    root = Path(os.environ.get("NEURO_AI_DATA", r"D:\liliai\neuro-ai-data")) / "training"
    rows = []
    if not root.is_dir():
        return rows
    for path in sorted(root.rglob("*.jsonl")):
        try:
            size = path.stat().st_size
        except OSError:
            continue
        rows.append({"name": path.relative_to(root).as_posix(), "mb": round(size / (1024 * 1024), 1)})
        if len(rows) >= 16:
            break
    return rows


_training_cache: dict = {"at": 0.0, "view": None}


def training_view() -> dict:
    now = time.time()
    cached = _training_cache.get("view")
    if cached and now - float(_training_cache.get("at") or 0) < 60:
        return cached
    data_root = Path(os.environ.get("NEURO_AI_DATA", r"D:\liliai\neuro-ai-data"))
    versions_path = data_root / "versions.json"
    active = ""
    served = "NeuroAI 9B"
    if versions_path.is_file():
        try:
            versions = json.loads(versions_path.read_text(encoding="utf-8"))
            active = str(versions.get("activePeft") or "")
        except (OSError, json.JSONDecodeError):
            active = ""
    state_path = data_root / "training" / "runs" / "neuro-ai-long-v1" / "checkpoint-6933" / "trainer_state.json"
    step = 0
    max_steps = 0
    loss = None
    history: list[float] = []
    if state_path.is_file():
        try:
            state = json.loads(state_path.read_text(encoding="utf-8"))
            step = int(state.get("global_step") or 0)
            max_steps = int(state.get("max_steps") or 0)
            for row in reversed(state.get("log_history") or []):
                if isinstance(row, dict) and row.get("loss") is not None:
                    value = round(float(row["loss"]), 4)
                    if loss is None:
                        loss = value
                    history.append(value)
                    if len(history) >= 8:
                        break
        except (OSError, json.JSONDecodeError, TypeError, ValueError):
            step = 0
    view = {
        "running": False,
        "run": "neuro-ai-long-v1",
        "step": step,
        "max_steps": max_steps,
        "loss": loss,
        "history": list(reversed(history)),
        "served": served,
        "active_adapter": active,
        "reason": "Training stays on this NVIDIA PC. A joined computer can hold part of the served model only after chat is stopped and that computer is listening on its own LAN address. One training run is not split across the machines. The dataset files are unchanged.",
        "datasets": dataset_index(),
    }
    _training_cache["at"] = now
    _training_cache["view"] = view
    return view


def local_status() -> dict:
    cards = machine_cards()
    binary = rpc_binary()
    chat = process_running("llama-server.exe") or process_running("llama-server")
    return {
        "name": socket.gethostname(),
        "os": platform.system().lower(),
        "machine": platform.machine(),
        "ram": ram_mib(),
        "gpus": cards,
        "heavy": heavy_processes(),
        "chat_server": chat,
        "rpc_binary": binary,
        "rpc_listening": port_open(RPC_PORT),
        "temperature_c": cards[0]["temperature_c"] if cards else None,
    }


def read_nodes() -> list[dict]:
    if not NODES_PATH.exists():
        return []
    try:
        data = json.loads(NODES_PATH.read_text(encoding="utf-8"))
    except json.JSONDecodeError:
        return []
    if not isinstance(data, list):
        return []
    return [item for item in data if isinstance(item, dict) and item.get("url")]


def write_nodes(nodes: list[dict]) -> None:
    NODES_PATH.write_text(json.dumps(nodes, indent=2), encoding="utf-8")


def read_specs() -> list[dict]:
    if not SPECS_PATH.exists():
        return []
    try:
        data = json.loads(SPECS_PATH.read_text(encoding="utf-8"))
    except json.JSONDecodeError:
        return []
    return data if isinstance(data, list) else []


def write_specs(specs: list[dict]) -> None:
    SPECS_PATH.write_text(json.dumps(specs, ensure_ascii=False, indent=2), encoding="utf-8")


def clip(value: object, limit: int = 80) -> str:
    return str(value or "").strip()[:limit]


def save_spec(raw: dict) -> dict:
    system = clip(raw.get("os"), 16)
    if system not in {"mac", "linux"}:
        raise ValueError("Choose mac or linux")
    url = clip(raw.get("url"), 200)
    item = {
        "os": system,
        "name": clip(raw.get("name"), 40) or system,
        "url": clean_url(url) if url else "",
        "cpu": clip(raw.get("cpu")),
        "ram_gb": clip(raw.get("ram_gb"), 16),
        "gpu": clip(raw.get("gpu")),
        "vram_gb": clip(raw.get("vram_gb"), 16),
    }
    specs = [spec for spec in read_specs() if spec.get("os") != system]
    specs.append(item)
    write_specs(specs)
    if item["url"]:
        nodes = read_nodes()
        if not any(node["url"] == item["url"] for node in nodes):
            nodes.append({"url": item["url"]})
            write_nodes(nodes)
    return item


def clean_url(value: str) -> str:
    parsed = urlparse(value.strip())
    if parsed.scheme not in {"http", "https"} or not parsed.hostname or not parsed.port:
        raise ValueError("Address must look like http://192.168.7.20:8781")
    return f"{parsed.scheme}://{parsed.hostname}:{parsed.port}"


def fetch_worker(url: str) -> dict:
    request = urllib.request.Request(
        url.rstrip("/") + "/status",
        headers={"Authorization": "Bearer " + token()},
    )
    with urllib.request.urlopen(request, timeout=0.8) as response:
        body = json.loads(response.read().decode("utf-8"))
    if not isinstance(body, dict):
        raise ValueError("Bad worker response")
    return body


def fetch_public(url: str, path: str) -> dict:
    request = urllib.request.Request(url.rstrip("/") + path)
    with urllib.request.urlopen(request, timeout=0.8) as response:
        body = json.loads(response.read().decode("utf-8"))
    if not isinstance(body, dict):
        raise ValueError("Bad worker response")
    return body


def pull_legacy(url: str) -> dict:
    try:
        return {"url": url, "ok": True, "status": fetch_worker(url)}
    except urllib.error.HTTPError as exc:
        if exc.code == 401:
            return {
                "url": url,
                "ok": True,
                "limited": True,
                "status": {"name": "Mac", "os": "", "ram": {}},
            }
        return {"url": url, "ok": False, "error": f"http {exc.code}"}
    except (OSError, urllib.error.URLError, ValueError, json.JSONDecodeError):
        return {"url": url, "ok": False, "error": "no reply"}


def pull_worker(url: str) -> dict:
    try:
        status = fetch_public(url, "/hello")
    except urllib.error.HTTPError as exc:
        if exc.code == 404:
            return pull_legacy(url)
        return {"url": url, "ok": False, "error": f"http {exc.code}"}
    except (OSError, urllib.error.URLError, ValueError, json.JSONDecodeError):
        return {"url": url, "ok": False, "error": "no reply"}
    if status.get("app") != "neuroai-mesh":
        return {"url": url, "ok": False, "error": "not a mesh worker"}
    return {"url": url, "ok": True, "status": status}


def ipv4_prefixes() -> tuple[list[str], set[str]]:
    prefixes: list[str] = []
    own: set[str] = set()
    if os.name != "nt":
        try:
            ip = socket.gethostbyname(socket.gethostname())
        except OSError:
            return [], set()
        if ip.count(".") == 3 and not ip.startswith("127."):
            return [ip.rsplit(".", 1)[0]], {ip}
        return [], set()
    out = subprocess.check_output(["ipconfig"], text=True, errors="replace")
    current = ""
    for line in out.splitlines():
        if "IPv4" in line and ":" in line:
            current = line.split(":")[-1].strip()
        elif current and "Subnet Mask" in line and ":" in line:
            mask = line.split(":")[-1].strip()
            if (
                mask == "255.255.255.0"
                and current.count(".") == 3
                and not current.startswith("127.")
            ):
                own.add(current)
                prefix = current.rsplit(".", 1)[0]
                if prefix not in prefixes:
                    prefixes.append(prefix)
            current = ""
    return prefixes, own


def probe_host(ip: str) -> dict | None:
    sock = socket.socket()
    sock.settimeout(0.25)
    try:
        sock.connect((ip, WORKER_PORT))
    except OSError:
        return None
    finally:
        sock.close()
    return pull_worker(f"http://{ip}:{WORKER_PORT}")


def remember_peer(item: dict) -> None:
    if not item.get("ok") or item.get("limited"):
        return
    status = item.get("status") or {}
    url = str(item.get("url") or "")
    if not url:
        return
    nodes = read_nodes()
    if not any(node["url"] == url for node in nodes):
        nodes.append({"url": url})
        write_nodes(nodes)
    bay = {"darwin": "mac", "linux": "linux"}.get(str(status.get("os") or ""))
    if not bay:
        return
    current = next((spec for spec in read_specs() if spec.get("os") == bay), None)
    if current and current.get("url"):
        return
    total = int((status.get("ram") or {}).get("total_mib") or 0)
    save_spec(
        {
            "os": bay,
            "name": status.get("name") or bay,
            "url": url,
            "ram_gb": str(max(1, round(total / 1024))) if total else "",
            "cpu": status.get("machine") or "",
            "gpu": "",
        }
    )


def scan_lan() -> list[dict]:
    prefixes, own = ipv4_prefixes()
    hosts = [
        f"{prefix}.{n}"
        for prefix in prefixes
        for n in range(1, 255)
        if f"{prefix}.{n}" not in own
    ]
    if not hosts:
        return []
    with ThreadPoolExecutor(max_workers=48) as pool:
        found = [item for item in pool.map(probe_host, hosts) if item and item.get("ok")]
    kept: list[dict] = []
    seen: set[tuple[str, str]] = set()
    for item in found:
        status = item.get("status") or {}
        if status.get("os") and not item.get("limited"):
            key = (str(status.get("name") or ""), str(status.get("os")))
        else:
            key = ("url", str(item.get("url") or ""))
        if key in seen:
            continue
        seen.add(key)
        kept.append(item)
    return kept


def discovery_loop() -> None:
    global _found
    while True:
        try:
            found = scan_lan()
            for item in found:
                remember_peer(item)
            with _found_lock:
                _found = found
        except Exception as exc:
            print(f"discovery failed: {exc}", flush=True)
        time.sleep(12)


def start_discovery() -> None:
    threading.Thread(target=discovery_loop, name="neuro-discovery", daemon=True).start()


def known_models() -> list[dict]:
    models = []
    data_root = Path(os.environ.get("NEURO_AI_DATA", r"D:\liliai\neuro-ai-data"))
    folder = data_root / "models" / "gguf"
    paths = []
    if folder.is_dir():
        paths.extend(folder.rglob("*.gguf"))
    extra = ROOT / "models.json"
    if extra.is_file():
        try:
            saved = json.loads(extra.read_text(encoding="utf-8"))
            if isinstance(saved, list):
                paths.extend(Path(item.get("path", "")) for item in saved if isinstance(item, dict))
        except (OSError, json.JSONDecodeError):
            pass
    seen = set()
    for path in paths:
        if not path.is_file() or path.suffix.lower() != ".gguf" or "mmproj" in path.name.lower():
            continue
        key = str(path.resolve()).lower()
        if key in seen:
            continue
        seen.add(key)
        models.append(
            {
                "name": path.stem,
                "path": str(path),
                "mb": path.stat().st_size // (1024 * 1024),
                "loaded": "qwen3.5-9b" in path.name.lower() and "q5" in path.name.lower(),
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
    saved = []
    if extra.is_file():
        try:
            current = json.loads(extra.read_text(encoding="utf-8"))
            if isinstance(current, list):
                saved = current
        except (OSError, json.JSONDecodeError):
            saved = []
    item = {"name": path.stem, "path": str(path)}
    if not any(entry.get("path") == item["path"] for entry in saved if isinstance(entry, dict)):
        saved.append(item)
    extra.write_text(json.dumps(saved, indent=2), encoding="utf-8")
    return item


def mesh_snapshot() -> dict:
    local = local_status()
    nodes = [{"url": "local", "ok": True, "status": local}]
    for item in read_nodes():
        nodes.append(pull_worker(item["url"]))
    with _found_lock:
        found = list(_found)
    return {
        "nodes": nodes,
        "found": found,
        "token_id": token_id(),
        "rpc_port": RPC_PORT,
        "chat": "http://127.0.0.1:8080/v1",
        "limits": {
            "remote_shell": False,
            "joint_training": False,
            "reason": "Extra machines add memory for one inference run through llama.cpp. Training stays on the CUDA machine. This console does not run remote commands.",
        },
        "specs": read_specs(),
        "training": training_view(),
        "models": known_models(),
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
    try:
        with socket.socket(socket.AF_INET, socket.SOCK_DGRAM) as sock:
            sock.connect(("192.168.7.1", 9))
            host = sock.getsockname()[0]
    except OSError:
        return ""
    if host.startswith(("10.", "192.168.", "172.")):
        return host
    return ""


def start_rpc() -> dict:
    status = local_status()
    if status["chat_server"]:
        raise RuntimeError("The GPU is serving NeuroAI. Stop the chat server before splitting weights.")
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
    payload = json.dumps(
        {
            "messages": [
                {
                    "role": "system",
                    "content": "You are NeuroAI, running on this computer. Answer the user.",
                },
                {"role": "user", "content": message},
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
    with urllib.request.urlopen(request, timeout=180) as response:
        body = json.loads(response.read().decode("utf-8"))
    choice = body["choices"][0]["message"]
    return {
        "reply": choice.get("content") or "",
        "reasoning": choice.get("reasoning_content") or "",
        "tokens": body.get("usage") or {},
    }


def stop_rpc() -> dict:
    if not PID_PATH.exists():
        return {"stopped": False}
    pid_text = PID_PATH.read_text(encoding="utf-8").strip()
    PID_PATH.unlink(missing_ok=True)
    if not pid_text.isdigit():
        return {"stopped": False}
    pid = int(pid_text)
    if os.name == "nt":
        subprocess.run(["taskkill", "/PID", str(pid), "/F"], check=False, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
    else:
        try:
            os.kill(pid, 15)
        except OSError:
            return {"stopped": False}
    return {"stopped": True, "pid": pid}


class Handler(BaseHTTPRequestHandler):
    mode = "serve"

    def log_message(self, fmt: str, *args) -> None:
        return

    def handle_error(self, request, client_address) -> None:
        err = sys.exc_info()[1]
        print(f"request failed: {err}", flush=True)

    def send_json(self, code: int, payload: dict) -> None:
        raw = json.dumps(payload, ensure_ascii=False).encode("utf-8")
        self.send_response(code)
        self.send_header("Content-Type", "application/json; charset=utf-8")
        self.send_header("Content-Length", str(len(raw)))
        self.end_headers()
        self.wfile.write(raw)

    def read_json(self, limit: int = 10000) -> dict:
        length = int(self.headers.get("Content-Length", "0") or 0)
        if length <= 0 or length > limit:
            return {}
        body = self.rfile.read(length)
        data = json.loads(body.decode("utf-8"))
        return data if isinstance(data, dict) else {}

    def do_GET(self) -> None:
        path = urlparse(self.path).path
        if self.mode == "work":
            if path == "/hello":
                payload = local_status()
                payload["app"] = "neuroai-mesh"
                payload["port"] = WORKER_PORT
                self.send_json(200, payload)
                return
            if path != "/status":
                self.send_json(404, {"error": "not found"})
                return
            if not authorized(self.headers.get("Authorization")):
                self.send_json(401, {"error": "token"})
                return
            self.send_json(200, local_status())
            return
        if path in {"/", "/dashboard"}:
            raw = DASHBOARD.read_bytes()
            self.send_response(200)
            self.send_header("Content-Type", "text/html; charset=utf-8")
            self.send_header("Content-Length", str(len(raw)))
            self.end_headers()
            self.wfile.write(raw)
            return
        if path == "/api/mesh":
            self.send_json(200, mesh_snapshot())
            return
        if path == "/api/agents":
            from agents import snapshot

            self.send_json(200, snapshot())
            return
        if path == "/api/token":
            self.send_json(200, {"token": token()})
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
            if not authorized(self.headers.get("Authorization")):
                self.send_json(401, {"error": "token"})
                return
            try:
                if path == "/rpc/start":
                    self.send_json(200, start_rpc())
                    return
                if path == "/rpc/stop":
                    self.send_json(200, stop_rpc())
                    return
            except RuntimeError as exc:
                self.send_json(409, {"error": str(exc)})
                return
            self.send_json(404, {"error": "not found"})
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
                        "error": "Training did not start because the loaded model is using the GPU. No dataset file was changed. " + str(len(view.get("datasets") or [])) + " dataset files are listed on this page.",
                        "datasets": view.get("datasets") or [],
                    },
                )
                return
            if path == "/api/agents/stop":
                from agents import stop_workers

                self.send_json(200, stop_workers())
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
            if path == "/api/nodes":
                url = clean_url(str(self.read_json().get("url", "")))
                nodes = read_nodes()
                if not any(item["url"] == url for item in nodes):
                    nodes.append({"url": url})
                    write_nodes(nodes)
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
                request = urllib.request.Request(
                    target.rstrip("/") + f"/rpc/{action}",
                    data=b"{}",
                    method="POST",
                    headers={
                        "Authorization": "Bearer " + token(),
                        "Content-Type": "application/json",
                    },
                )
                with urllib.request.urlopen(request, timeout=5) as response:
                    payload = json.loads(response.read().decode("utf-8"))
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
        system = (query.get("os") or [""])[0]
        if system in {"mac", "linux"}:
            specs = read_specs()
            gone = {spec.get("url") for spec in specs if spec.get("os") == system and spec.get("url")}
            write_specs([spec for spec in specs if spec.get("os") != system])
            if gone:
                write_nodes([item for item in read_nodes() if item.get("url") not in gone])
            self.send_json(200, mesh_snapshot())
            return
        url = (query.get("url") or [""])[0]
        try:
            url = clean_url(url)
        except ValueError as exc:
            self.send_json(400, {"error": str(exc)})
            return
        write_nodes([item for item in read_nodes() if item["url"] != url])
        self.send_json(200, mesh_snapshot())


def serve(mode: str) -> None:
    Handler.mode = mode
    if mode == "serve":
        server = ThreadingHTTPServer(("127.0.0.1", DASHBOARD_PORT), Handler)
        print(f"Neuro Mesh dashboard http://127.0.0.1:{DASHBOARD_PORT}", flush=True)
        print(f"token id {token_id()}", flush=True)
        start_discovery()
    else:
        server = ThreadingHTTPServer(("0.0.0.0", WORKER_PORT), Handler)
        print(f"Neuro Mesh worker on port {WORKER_PORT}", flush=True)
        print(f"token id {token_id()}", flush=True)
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
    if command == "swarm":
        from agents import ensure_workers, seed

        requested = int(sys.argv[2]) if len(sys.argv) > 2 else 4
        seeded = seed()
        result = ensure_workers(requested)
        print(f"seeded {seeded} jobs; started {result['started']} of {result['requested']}", flush=True)
        print("pids " + " ".join(str(pid) for pid in result["pids"]), flush=True)
        return
    if command not in {"serve", "work"}:
        raise SystemExit("use: mesh.py serve | mesh.py work | mesh.py code [TASK] | mesh.py agents [COUNT]")
    serve(command)


if __name__ == "__main__":
    main()
