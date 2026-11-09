"""CMD workers for Neuro.

One llama-server holds the model. Workers are separate Python processes.
Only a few of them may generate at once, because the GPU has four slots.
"""

from __future__ import annotations

import ctypes
import http.client
import json
import locale
import os
import stat
import random
import re
import shutil
import socket
import sqlite3
import subprocess
import sys
import threading
import time
import urllib.error
import urllib.request
from pathlib import Path

ROOT = Path(__file__).resolve().parent
HOME = ROOT
DB_PATH = ROOT / "agents.db"
STOP_PATH = ROOT / "agents.stop"
LOG_DIR = ROOT / "agent-logs"
WORKER_SCRIPT = ROOT / "mesh.py"


def tree_for(raw: str) -> Path:
    return ROOT


SLOTS = 4
CHAT_URL = os.environ.get("NEURO_CHAT_URL") or "http://127.0.0.1:8080/v1/chat/completions"
HEALTH_URL = CHAT_URL.split("/v1/", 1)[0].rstrip("/") + "/health"
# FileNotFoundError on this file means the GPU is free. Anything else means training owns it.
GPU_LOCK = Path(os.environ.get("NEURO_GPU_LOCK") or (HOME.parent / "train_local" / "gpu.lock"))
REQUEST_TIMEOUT = 180
CONTEXT_CHARS = int(os.environ.get("NEURO_CONTEXT_CHARS") or 9000)
TOOL_CHARS = 2400
MAX_TURNS = 6
REAP_AFTER = 180
SECRET_NAMES = frozenset({"cluster.token", ".env", ".env.local", "token.txt", "nodes.json", "specs.json"})
DB_SUFFIXES = (".db", ".db-wal", ".db-shm", ".db-journal", ".sqlite", ".sqlite3")
TRAINING_WAIT = "The model is training on this PC, so this task has to wait."
OFFLINE_WAIT = "Neuro's model is not running yet, so this task has to wait."
LOADING_WAIT = "Neuro is still loading the model, so this task has to wait."
SYSTEM = "You are Neuro. Do the task. One tool call, then stop. Do not explain the tool."
LAST_STEP = "That was the last tool. Answer the task now in plain text. Do not emit a tool."
# Stop right after one call, and never let the model invent its own tool result.
STOPS = ["<tool_response>", "</tool_call>"]
# The model server is on this machine. A system proxy must never sit in between.
_OPENER = urllib.request.build_opener(urllib.request.ProxyHandler({}))


class ModelWait(RuntimeError):
    """The model cannot answer right now. The job goes back to the queue."""


class ContextFull(RuntimeError):
    """The prompt did not fit in the model's context window."""


def gpu_busy() -> bool:
    try:
        os.stat(GPU_LOCK)
    except FileNotFoundError:
        return False
    except OSError:
        return True
    return True


def append_capped(path: Path, line: str, limit: int = 200000, keep: int = 100) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    lock_path = path.with_suffix(path.suffix + ".lock")
    try:
        with lock_path.open("xb") as seed:
            seed.write(b"0")
    except FileExistsError:
        pass
    with lock_path.open("r+b") as lock:
        if os.name == "nt":
            import msvcrt

            while True:
                try:
                    lock.seek(0)
                    msvcrt.locking(lock.fileno(), msvcrt.LK_NBLCK, 1)
                    break
                except OSError:
                    time.sleep(0.05)
        else:
            import fcntl

            fcntl.flock(lock.fileno(), fcntl.LOCK_EX)
        try:
            with path.open("a", encoding="utf-8") as handle:
                handle.write(line if line.endswith("\n") else line + "\n")
            if path.stat().st_size > limit:
                lines = path.read_text(encoding="utf-8").splitlines()[-keep:]
                path.write_text("\n".join(lines) + "\n", encoding="utf-8")
        finally:
            if os.name == "nt":
                import msvcrt

                lock.seek(0)
                msvcrt.locking(lock.fileno(), msvcrt.LK_UNLCK, 1)


class _Db(sqlite3.Connection):
    """`with connect() as db:` commits or rolls back, then closes the file handle."""

    def __exit__(self, exc_type, exc, tb):
        try:
            return super().__exit__(exc_type, exc, tb)
        finally:
            self.close()


_schema_ready = False


def connect() -> sqlite3.Connection:
    global _schema_ready
    # timeout is SQLite's busy_timeout: ten processes wait for the write lock instead of failing.
    db = sqlite3.connect(DB_PATH, timeout=30, factory=_Db)
    db.row_factory = sqlite3.Row
    db.execute("PRAGMA busy_timeout=30000")
    db.execute("PRAGMA synchronous=NORMAL")
    if not _schema_ready:
        db.execute("PRAGMA journal_mode=WAL")
        db.execute(
            """
            CREATE TABLE IF NOT EXISTS jobs (
                id INTEGER PRIMARY KEY,
                title TEXT NOT NULL,
                prompt TEXT NOT NULL,
                status TEXT NOT NULL,
                worker_pid INTEGER,
                result TEXT,
                error TEXT,
                created REAL NOT NULL,
                updated REAL NOT NULL
            )
            """
        )
        db.execute(
            """
            CREATE TABLE IF NOT EXISTS workers (
                pid INTEGER PRIMARY KEY,
                started REAL NOT NULL,
                heartbeat REAL NOT NULL,
                state TEXT NOT NULL,
                job_id INTEGER
            )
            """
        )
        db.commit()
        _schema_ready = True
    return db


def db_retry(action, *args, tries: int = 5):
    for attempt in range(tries):
        try:
            return action(*args)
        except sqlite3.OperationalError:
            if attempt == tries - 1:
                raise
            time.sleep(0.4 * (attempt + 1) + random.random() * 0.3)
    return None


def free_mib() -> int:
    if os.name != "nt":
        if sys.platform == "darwin":
            try:
                out = subprocess.check_output(["vm_stat"], text=True, timeout=3)
            except (OSError, subprocess.SubprocessError):
                return 0
            page = 4096
            pages = 0
            for line in out.splitlines():
                if "page size of" in line:
                    for word in line.replace("(", " ").replace(")", " ").split():
                        if word.isdigit():
                            page = int(word)
                if line.startswith("Pages free"):
                    number = line.split(":")[-1].strip().strip(".")
                    if number.isdigit():
                        pages += int(number)
            return pages * page // (1024 * 1024)
        try:
            page = os.sysconf("SC_PAGE_SIZE")
            return int(os.sysconf("SC_AVPHYS_PAGES") * page // (1024 * 1024))
        except (ValueError, OSError, AttributeError):
            return 0

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
    return int(status.ullAvailPhys // (1024 * 1024))


def worker_cap(requested: int) -> int:
    # Leave 4 GB free. A waiting worker is about 45 MB.
    # Zero is allowed. Callers start the first worker themselves.
    room = max(0, free_mib() - 4096) // 45
    return max(0, min(int(requested), int(room), 24))


def enqueue(title: str, prompt: str) -> int:
    now = time.time()
    with connect() as db:
        cursor = db.execute(
            "INSERT INTO jobs (title, prompt, status, created, updated) VALUES (?, ?, 'queued', ?, ?)",
            (title.strip()[:80] or "task", prompt.strip()[:4000], now, now),
        )
        return int(cursor.lastrowid)


def seed() -> int:
    with connect() as db:
        existing = db.execute("SELECT COUNT(*) AS n FROM jobs").fetchone()["n"]
    if existing:
        return 0
    tasks = [
        ("Red bay", "A Mac worker is running but the bay stays red. Give the one Windows firewall check and the address format."),
        ("Copy step", "Write the exact files a person must copy to a Mac before the worker script can start."),
        ("Chat idle", "Explain in two sentences why the GPU can sit near zero while Neuro is still loaded."),
        ("Linux command", "Give the one command that starts a Linux worker from the neuro-cluster folder."),
        ("Token", "Say where cluster.token must sit, and why it must stay off the public internet."),
        ("Slots", "Explain why many CMD workers share one model and only four generate at once."),
    ]
    for title, prompt in tasks:
        enqueue(title, prompt)
    return len(tasks)


def reap(db: sqlite3.Connection) -> None:
    cutoff = time.time() - REAP_AFTER
    stale = db.execute("SELECT pid FROM workers WHERE heartbeat < ?", (cutoff,)).fetchall()
    for row in stale:
        db.execute(
            "UPDATE jobs SET status='queued', worker_pid=NULL, updated=? WHERE status='running' AND worker_pid=?",
            (time.time(), row["pid"]),
        )
        db.execute("DELETE FROM workers WHERE pid=?", (row["pid"],))


def has_queued() -> bool:
    with connect() as db:
        row = db.execute("SELECT EXISTS(SELECT 1 FROM jobs WHERE status='queued') AS q").fetchall()[0]
    return bool(row["q"])


def beat(pid: int, state: str) -> None:
    now = time.time()
    with connect() as db:
        db.execute(
            "INSERT INTO workers (pid, started, heartbeat, state, job_id) VALUES (?, ?, ?, ?, NULL) "
            "ON CONFLICT(pid) DO UPDATE SET heartbeat=excluded.heartbeat, state=excluded.state, job_id=NULL",
            (pid, now, now, state),
        )


def claim(pid: int) -> dict | None:
    now = time.time()
    with connect() as db:
        db.execute("BEGIN IMMEDIATE")
        reap(db)
        db.execute(
            "INSERT INTO workers (pid, started, heartbeat, state, job_id) VALUES (?, ?, ?, 'idle', NULL) "
            "ON CONFLICT(pid) DO UPDATE SET heartbeat=excluded.heartbeat",
            (pid, now, now),
        )
        running = db.execute("SELECT COUNT(*) AS n FROM jobs WHERE status='running'").fetchone()["n"]
        if running >= SLOTS:
            db.execute("UPDATE workers SET state='waiting', job_id=NULL, heartbeat=? WHERE pid=?", (now, pid))
            db.commit()
            return None
        job = db.execute(
            "SELECT id, title, prompt FROM jobs WHERE status='queued' ORDER BY id LIMIT 1"
        ).fetchone()
        if job is None:
            db.execute("UPDATE workers SET state='idle', job_id=NULL, heartbeat=? WHERE pid=?", (now, pid))
            db.commit()
            return None
        db.execute(
            "UPDATE jobs SET status='running', worker_pid=?, updated=? WHERE id=?",
            (pid, now, job["id"]),
        )
        db.execute(
            "UPDATE workers SET state='generating', job_id=?, heartbeat=? WHERE pid=?",
            (job["id"], now, pid),
        )
        db.commit()
        return {"id": int(job["id"]), "title": str(job["title"]), "prompt": str(job["prompt"])}


def finish(job_id: int, pid: int, result: str = "", error: str = "") -> None:
    now = time.time()
    status = "error" if error else "done"
    title = ""
    with connect() as db:
        # Skip the write if another worker took this job over after a missed heartbeat.
        db.execute(
            "UPDATE jobs SET status=?, result=?, error=?, worker_pid=?, updated=? "
            "WHERE id=? AND (worker_pid=? OR worker_pid IS NULL)",
            (status, result[:4000], error[:400], pid, now, job_id, pid),
        )
        db.execute(
            "UPDATE workers SET state='idle', job_id=NULL, heartbeat=? WHERE pid=?",
            (now, pid),
        )
        row = db.execute("SELECT title FROM jobs WHERE id=?", (job_id,)).fetchone()
        title = str(row["title"]) if row else ""
    try:
        remember(job_id, status, title, error or result)
    except OSError:
        pass


def requeue(job_id: int, pid: int) -> None:
    now = time.time()
    with connect() as db:
        db.execute(
            "UPDATE jobs SET status='queued', worker_pid=NULL, updated=? WHERE id=? AND status='running' AND worker_pid=?",
            (now, job_id, pid),
        )
        db.execute(
            "UPDATE workers SET state='waiting', job_id=NULL, heartbeat=? WHERE pid=?",
            (now, pid),
        )


def keep_alive(pid: int, done: threading.Event) -> None:
    # A long job must not look dead, or another worker would run it a second time.
    while not done.wait(20):
        try:
            with connect() as db:
                db.execute("UPDATE workers SET heartbeat=? WHERE pid=?", (time.time(), pid))
        except sqlite3.Error:
            pass


def model_state(timeout: float = 2.0) -> tuple[bool, str]:
    try:
        with _OPENER.open(HEALTH_URL, timeout=timeout) as response:
            response.read()
        return True, ""
    except urllib.error.HTTPError as exc:
        if exc.code == 503:
            return False, LOADING_WAIT
        # Some servers have no /health. Let the chat call decide.
        return True, ""
    except (urllib.error.URLError, OSError, ValueError):
        pass
    if gpu_busy():
        return False, TRAINING_WAIT
    return False, OFFLINE_WAIT


def _http_detail(exc: urllib.error.HTTPError) -> str:
    try:
        raw = exc.read().decode("utf-8", errors="replace")
    except (OSError, ValueError):
        return ""
    finally:
        exc.close()
    try:
        data = json.loads(raw)
        error = data.get("error") if isinstance(data, dict) else None
        if isinstance(error, dict):
            return str(error.get("message") or error.get("type") or raw)[:300]
        if error:
            return str(error)[:300]
    except ValueError:
        pass
    return raw[:300]


def chat(messages: list[dict], max_tokens: int = 1000, *, think: bool = False, stop: list[str] | None = None) -> dict:
    payload = {
        "messages": messages,
        "temperature": 0.6 if think else 0.4,
        "max_tokens": max(64, min(int(max_tokens), 4096)),
        # Thinking stays off unless the user asked for Deep mode.
        "chat_template_kwargs": {"enable_thinking": bool(think)},
        "cache_prompt": True,
    }
    if stop:
        payload["stop"] = list(stop)
    data = json.dumps(payload).encode("utf-8")
    delays = (0, 2, 5, 10)
    problem = ""
    body = None
    for attempt, delay in enumerate(delays):
        if delay:
            time.sleep(delay + random.random())
        request = urllib.request.Request(CHAT_URL, data=data, headers={"Content-Type": "application/json"})
        try:
            with _OPENER.open(request, timeout=REQUEST_TIMEOUT) as response:
                raw = response.read().decode("utf-8", errors="replace")
        except urllib.error.HTTPError as exc:
            detail = _http_detail(exc)
            if exc.code == 400 and "context" in detail.lower():
                raise ContextFull(detail) from exc
            if exc.code == 503:
                problem = "loading"
                continue
            if exc.code in {408, 429, 500, 502, 504}:
                problem = f"error {exc.code}"
                continue
            raise RuntimeError(f"Neuro's model refused the request ({exc.code}). {detail}".strip()) from exc
        except (TimeoutError, socket.timeout):
            problem = "timeout"
            if attempt >= 1:
                break
            continue
        except (urllib.error.URLError, ConnectionError, http.client.HTTPException) as exc:
            reason = getattr(exc, "reason", None)
            if isinstance(reason, (TimeoutError, socket.timeout)):
                problem = "timeout"
                if attempt >= 1:
                    break
                continue
            if gpu_busy():
                raise ModelWait(TRAINING_WAIT) from exc
            problem = "offline"
            if attempt >= 2:
                break
            continue
        try:
            body = json.loads(raw)
        except ValueError as exc:
            raise RuntimeError("Neuro's model sent a reply that is not JSON.") from exc
        break
    if body is None:
        if problem == "loading":
            raise ModelWait(LOADING_WAIT)
        if problem == "offline":
            raise ModelWait(TRAINING_WAIT if gpu_busy() else OFFLINE_WAIT)
        if problem == "timeout":
            raise RuntimeError("Neuro's model took too long to answer.")
        raise RuntimeError(f"Neuro's model kept failing ({problem}).")
    choices = body.get("choices") if isinstance(body, dict) else None
    if not isinstance(choices, list) or not choices or not isinstance(choices[0], dict):
        raise RuntimeError("Neuro's model sent an empty reply.")
    choice = choices[0]
    message = choice.get("message") if isinstance(choice.get("message"), dict) else {}
    content = message.get("content") or ""
    if isinstance(content, list):
        content = "".join(str(part.get("text", "")) for part in content if isinstance(part, dict))
    # reasoning_content is the model's private thinking. It is never shown and never sent back.
    return {
        "content": str(content),
        "tool_calls": message.get("tool_calls") or [],
        "finish": str(choice.get("finish_reason") or ""),
    }


def ask_model(prompt: str, max_tokens: int = 220) -> str:
    reply = chat(
        [{"role": "system", "content": SYSTEM}, {"role": "user", "content": prompt}],
        max_tokens=max(64, min(int(max_tokens), 1200)),
    )
    return visible_answer(reply["content"])


def secret_name(name: str) -> bool:
    plain = name.strip().rstrip(". ").lower()
    return plain in SECRET_NAMES or plain.endswith(DB_SUFFIXES)


def inside(relative: str) -> tuple[str, Path] | None:
    raw = (relative or "").strip().strip("`'\"").replace("\\", "/")
    while raw.startswith("./"):
        raw = raw[2:]
    raw = raw.lstrip("/")
    tail = raw[2:] if len(raw) > 1 and raw[1] == ":" and raw[0].isalpha() else raw
    # A colon past the drive letter is an NTFS stream such as name::$DATA.
    if not raw or "\0" in raw or ":" in tail or ".." in raw.split("/"):
        return None
    if any(secret_name(part) for part in raw.split("/")):
        return None
    base = tree_for(raw).resolve()
    try:
        path = (base / raw).resolve()
    except (OSError, RuntimeError, ValueError):
        return None
    if path != base and base not in path.parents:
        return None
    if secret_name(path.name):
        return None
    return raw, path


def safe_read(relative: str, lines: tuple[int, int] | None = None) -> str:
    found = inside(relative)
    if found is None:
        return "unreadable"
    raw, path = found
    try:
        info = path.stat()
    except FileNotFoundError:
        return "missing"
    except OSError:
        return "unreadable"
    if not stat.S_ISREG(info.st_mode):
        return "missing"
    try:
        with path.open("r", encoding="utf-8", errors="replace") as handle:
            if lines:
                start, end = lines
                picked = []
                for number, line in enumerate(handle, 1):
                    if number > end:
                        break
                    if number >= start:
                        picked.append(f"{number}: {line.rstrip()[:500]}")
                return "\n".join(picked) or "no lines there"
            text = handle.read(6000)
            more = bool(handle.read(1))
        if more:
            size = path.stat().st_size
            text += f"\n[file continues, {size} bytes in all. READ {raw} START-END shows other lines]"
    except OSError:
        return "unreadable"
    return text


def read_tool(argument: str) -> str:
    text = argument.strip()
    parts = text.rsplit(None, 1)
    if len(parts) == 2:
        match = re.fullmatch(r"(\d+)\s*-\s*(\d+)", parts[1])
        if match:
            start = max(1, int(match.group(1)))
            end = max(start, min(int(match.group(2)), start + 399))
            return safe_read(parts[0], (start, end))
    result = safe_read(text)
    if result == "missing" and " " in text:
        short = tool_path(text)
        if short:
            return safe_read(short)
    return result


def workspace_note() -> str:
    names = []
    try:
        children = sorted(ROOT.iterdir())
    except OSError:
        children = []
    for path in children:
        if path.name.startswith(".") or secret_name(path.name):
            continue
        names.append(path.name + ("/" if path.is_dir() else ""))
    return "Repo files: " + ", ".join(names[:24])


def safe_write(relative: str, content: str) -> str:
    shown = (relative or "").strip().replace("\\", "/").lstrip("/")
    found = inside(relative)
    if found is None:
        return "blocked " + shown
    raw, path = found
    text = content.lstrip("\r\n").rstrip() + "\n"
    if len(text.encode("utf-8")) > 150000:
        return "too large"
    if path.is_dir():
        return "blocked " + raw + " is a folder"
    try:
        path.parent.mkdir(parents=True, exist_ok=True)
    except OSError as exc:
        return f"could not write {raw}: {exc.strerror or exc}"
    # Write a side file and swap it in, so a parallel reader never sees half a file.
    temp = path.with_name(f".{path.name}.{os.getpid()}.tmp")
    try:
        temp.write_text(text, encoding="utf-8")
        os.replace(temp, path)
    except OSError:
        try:
            temp.unlink(missing_ok=True)
        except OSError:
            pass
        try:
            path.write_text(text, encoding="utf-8")
        except OSError as exc:
            return f"could not write {raw}: {exc.strerror or exc}"
    return "wrote " + raw


def apply_actions(answer: str, require_end: bool = False) -> list[str]:
    # Only the first MKDIR or WRITE runs. One tool call, then stop.
    logs = []
    lines = answer.splitlines()
    index = 0
    while index < len(lines):
        line = clean_tool_line(lines[index])
        if line.startswith("MKDIR "):
            raw = tool_path(line[6:])
            found = inside(raw) if raw else None
            if found is None:
                logs.append("blocked path")
                return logs
            try:
                found[1].mkdir(parents=True, exist_ok=True)
                logs.append("mkdir " + raw)
            except OSError as exc:
                logs.append(f"could not make {raw}: {exc.strerror or exc}")
            return logs
        if line.startswith("WRITE "):
            raw = tool_path(line[6:])
            index += 1
            body = []
            ended = False
            while index < len(lines):
                if clean_tool_line(lines[index]) == "ENDWRITE":
                    ended = True
                    break
                body.append(lines[index])
                index += 1
            if (
                len(body) >= 2
                and body[0].strip().startswith("```")
                and body[-1].strip() == "```"
                and not raw.lower().endswith(".md")
            ):
                body = body[1:-1]
            if not raw:
                logs.append("blocked path")
            elif require_end and not ended:
                logs.append(f"not written {raw}: the file was cut off before ENDWRITE. Write a shorter file.")
            else:
                logs.append(safe_write(raw, "\n".join(body)))
            return logs
        index += 1
    return logs


def visible_answer(answer: str) -> str:
    text = answer or ""
    while True:
        start = text.find("<think>")
        if start < 0:
            break
        end = text.find("</think>", start)
        if end < 0:
            # Cut off while thinking. None of it is the answer.
            text = text[:start]
            break
        text = text[:start] + text[end + len("</think>") :]
    if "</think>" in text:
        # The opening tag sat in the prompt. Everything before the close is thinking.
        text = text.rsplit("</think>", 1)[1]
    return text.strip()


TOOL_HELP = (
    "You are Neuro Code in this repo. Emit one tool and stop. No sentence before it or after it.\n"
    "LIST relative/dir\n"
    "GREP text\n"
    "READ relative/path\n"
    "READ relative/path 120-180\n"
    "SESSIONS\n"
    "MKDIR relative/path\n"
    "WRITE relative/path\n"
    "the file contents\n"
    "ENDWRITE\n"
    "RUN command\n"
    "WRITE under workspace/ is allowed. Do not say you cannot create a file there.\n"
    "Stop when the tool line is done. Do not explain the tool.\n"
    "A tool result comes back inside <tool_response>. Read it, then one tool or a plain answer.\n"
    "When the task is done, answer in plain text and do not emit a tool."
)


def remember(job_id: int, status: str, title: str, text: str) -> None:
    row = {
        "t": time.strftime("%Y-%m-%d %H:%M", time.localtime()),
        "id": job_id,
        "status": status,
        "title": title[:80],
        "text": (text or "")[:800],
    }
    # Ten workers append here. The lock keeps lines whole and the cap keeps the file small.
    append_capped(LOG_DIR / "session.jsonl", json.dumps(row, ensure_ascii=False), limit=400000, keep=400)


def recent_sessions(limit: int = 6) -> list[str]:
    path = LOG_DIR / "session.jsonl"
    try:
        info = path.stat()
    except FileNotFoundError:
        return []
    except OSError:
        return ["session log unreadable"]
    if not stat.S_ISREG(info.st_mode):
        return []
    rows = []
    try:
        text = path.read_text(encoding="utf-8", errors="replace")
    except OSError:
        return []
    for line in text.splitlines():
        if not line.strip():
            continue
        try:
            item = json.loads(line)
        except json.JSONDecodeError:
            continue
        if not isinstance(item, dict):
            continue
        rows.append(f"{item.get('t', '')}  {item.get('id', '')}  {item.get('status', '')}  {item.get('title', '')}")
    return rows[-limit:]


def tool_path(text: str) -> str:
    token = text.strip().split(" ", 1)[0] if text.strip() else ""
    token = token.strip("`'\",").replace("\\", "/").lstrip("/")
    if not token or any(mark in token for mark in (",", " ", "..")):
        return ""
    return token


def repo_dir(relative: str) -> Path | None:
    raw = (relative or "").strip().strip("`'\"").replace("\\", "/").strip("/")
    if raw in {"", "."}:
        return ROOT.resolve()
    found = inside(raw)
    if found is None:
        return None
    return found[1] if found[1].is_dir() else None


def tool_list(relative: str) -> str:
    path = repo_dir(relative)
    if path is None:
        return "missing"
    try:
        children = sorted(path.iterdir())
    except OSError:
        return "unreadable"
    names = []
    extra = 0
    for child in children:
        if secret_name(child.name):
            continue
        if len(names) >= 40:
            extra += 1
            continue
        names.append(child.name + ("/" if child.is_dir() else ""))
    if extra:
        names.append(f"... {extra} more")
    return "\n".join(names) or "empty"


GREP_SKIP_DIRS = {"agent-logs", "__pycache__", ".git", "node_modules", ".next", ".venv", "venv"}
GREP_SKIP_SUFFIXES = (
    ".png", ".jpg", ".jpeg", ".gif", ".ico", ".webp", ".zip", ".exe", ".dll", ".bin",
    ".gguf", ".safetensors", ".pt", ".pyc", ".lock", ".tmp",
)


def tool_grep(needle: str) -> str:
    needle = needle.strip().strip("`")
    if len(needle) >= 2 and needle[0] == needle[-1] and needle[0] in "'\"":
        needle = needle[1:-1]
    if not needle or len(needle) > 80:
        return "need a short search"
    lowered = needle.lower()
    hits: list[str] = []
    loose: list[str] = []
    # os.walk prunes skipped folders, so .git is never walked at all.
    for dirpath, dirnames, filenames in os.walk(ROOT):
        dirnames[:] = sorted(name for name in dirnames if name not in GREP_SKIP_DIRS)
        for name in sorted(filenames):
            if secret_name(name) or name.lower().endswith(GREP_SKIP_SUFFIXES):
                continue
            path = Path(dirpath) / name
            try:
                if path.stat().st_size > 300000:
                    continue
                text = path.read_text(encoding="utf-8", errors="ignore")
            except OSError:
                continue
            if lowered not in text.lower():
                continue
            shown = path.relative_to(ROOT).as_posix()
            for number, line in enumerate(text.splitlines(), 1):
                if needle in line:
                    hits.append(f"{shown}:{number}: {line.strip()[:160]}")
                    if len(hits) >= 20:
                        return "\n".join(hits)
                elif len(loose) < 20 and lowered in line.lower():
                    loose.append(f"{shown}:{number}: {line.strip()[:160]}")
    if hits:
        return "\n".join(hits)
    if loose:
        return "no exact matches. Other case:\n" + "\n".join(loose)
    return "no matches"


RUN_BLOCKED = (
    "cluster.token", "token.txt", ".env", "nodes.json", "specs.json", "agents.db", "gpu.lock",
    "train_local", "neuro-ai-data", ".gguf", ".safetensors", "train.py", "llama-server", "vllm",
    "comfyui", "shutdown", "format ", "format-volume", "rm -rf", "del /s", "rd /s", "rmdir /s",
    "reg delete", "reg add", "cipher ", "diskpart", "bcdedit", "--gpu-reset", "nvidia-smi -r",
)
# Ending processes from a tool could stop the training run or the model server.
RUN_WORDS = re.compile(r"\b(taskkill|tskill|kill|pkill|killall|stop-process|wmic)\b")


def _decode(data: bytes) -> str:
    try:
        return data.decode("utf-8")
    except UnicodeDecodeError:
        return data.decode(locale.getpreferredencoding(False) or "utf-8", errors="replace")


def _stop_tree(process: subprocess.Popen) -> None:
    # Only this tool's own shell and whatever it started.
    if os.name == "nt":
        subprocess.run(
            ["taskkill", "/PID", str(process.pid), "/T", "/F"],
            check=False,
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
            timeout=15,
        )
    else:
        import signal

        try:
            os.killpg(process.pid, signal.SIGKILL)
        except OSError:
            pass


def tool_run(command: str, timeout: int = 20) -> str:
    text = command.strip()
    lowered = text.lower()
    if not text or ".." in text or any(item in lowered for item in RUN_BLOCKED) or RUN_WORDS.search(lowered):
        return "blocked"
    options: dict = {}
    if os.name == "nt":
        options["creationflags"] = subprocess.CREATE_NEW_PROCESS_GROUP | subprocess.CREATE_NO_WINDOW
    else:
        options["start_new_session"] = True
    try:
        process = subprocess.Popen(
            text,
            cwd=str(ROOT),
            shell=True,
            stdin=subprocess.DEVNULL,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            **options,
        )
    except OSError as exc:
        return f"could not run: {exc.strerror or exc}"
    try:
        out, _ = process.communicate(timeout=timeout)
    except subprocess.TimeoutExpired:
        try:
            _stop_tree(process)
        except (OSError, subprocess.SubprocessError):
            pass
        try:
            out, _ = process.communicate(timeout=5)
        except subprocess.TimeoutExpired:
            out = b""
        partial = _decode(out or b"").strip()
        return clip(f"timed out after {timeout}s" + ("\n" + partial if partial else ""), 4000)
    shown = _decode(out or b"").strip()
    if process.returncode:
        shown = (shown + "\n" if shown else "") + f"exit {process.returncode}"
    return clip(shown or "exit 0", 4000)


def clean_tool_line(line: str) -> str:
    text = line.strip().strip("`").strip()
    text = text.lstrip("-*•").strip()
    while text[:1].isdigit():
        text = text[1:]
    return text.lstrip(".) ").strip()


TOOL_NAMES = {
    "list": "LIST", "list_dir": "LIST", "list_files": "LIST", "list_directory": "LIST", "ls": "LIST",
    "read": "READ", "read_file": "READ", "open_file": "READ", "cat": "READ", "view": "READ",
    "grep": "GREP", "search": "GREP", "search_files": "GREP", "find": "GREP", "grep_search": "GREP",
    "write": "WRITE", "write_file": "WRITE", "create_file": "WRITE", "save_file": "WRITE",
    "mkdir": "MKDIR", "make_dir": "MKDIR", "create_dir": "MKDIR", "create_directory": "MKDIR",
    "run": "RUN", "shell": "RUN", "run_command": "RUN", "bash": "RUN", "exec": "RUN", "terminal": "RUN",
    "sessions": "SESSIONS", "list_sessions": "SESSIONS", "recent_sessions": "SESSIONS",
}
PATH_KEYS = ("path", "file", "file_path", "filepath", "filename", "dir", "directory", "dir_path", "folder", "target")
QUERY_KEYS = ("query", "pattern", "text", "search", "needle", "regex", "q", "keyword")
COMMAND_KEYS = ("command", "cmd", "script", "shell", "code")
BODY_KEYS = ("content", "contents", "body", "text", "data", "file_text")
_JSON = json.JSONDecoder(strict=False)


def _first_line(value: object) -> str:
    if value is None:
        return ""
    text = value if isinstance(value, str) else json.dumps(value) if isinstance(value, (dict, list)) else str(value)
    lines = text.strip().splitlines()
    return lines[0].strip() if lines else ""


def _pick(args: dict, keys: tuple[str, ...]) -> object:
    for key in keys:
        if key in args and args[key] not in (None, ""):
            return args[key]
    return None


def command_from(name: object, args: object) -> str:
    command = TOOL_NAMES.get(str(name or "").strip().strip("\"'").lower().replace("-", "_"), "")
    if not command:
        return ""
    if command == "SESSIONS":
        return "SESSIONS"
    if isinstance(args, str):
        args = {"value": args}
    if not isinstance(args, dict):
        args = {}
    args = {str(key).strip().lower(): value for key, value in args.items()}
    only = next(iter(args.values())) if len(args) == 1 else None
    if command == "WRITE":
        safe = tool_path(_first_line(_pick(args, PATH_KEYS)))
        body = next((args[key] for key in BODY_KEYS if key in args and args[key] is not None), None)
        if not safe or body is None:
            return ""
        if not isinstance(body, str):
            body = json.dumps(body, indent=2, ensure_ascii=False)
        return "WRITE " + safe + "\n" + body.strip("\n") + "\nENDWRITE"
    if command == "LIST":
        return "LIST " + (_first_line(_pick(args, PATH_KEYS) or only) or ".")
    if command == "GREP":
        value = _first_line(_pick(args, QUERY_KEYS) or only)
    elif command == "RUN":
        value = _first_line(_pick(args, COMMAND_KEYS) or only)
    else:
        value = _first_line(_pick(args, PATH_KEYS) or only)
        if command == "READ" and value:
            start = _pick(args, ("start_line", "start", "line_start", "from"))
            end = _pick(args, ("end_line", "end", "line_end", "to"))
            if str(start or "").isdigit() and str(end or "").isdigit():
                value += f" {int(start)}-{int(end)}"
    return command + " " + value if value else ""


def xml_tool(answer: str) -> str:
    # <tool_call><function=name><parameter=key>value</parameter></function></tool_call>
    if "<function=" not in answer:
        return ""
    rest = answer.split("<function=", 1)[1]
    name, _, rest = rest.partition(">")
    rest = rest.split("</function>", 1)[0]
    args: dict = {}
    while "<parameter=" in rest:
        rest = rest.split("<parameter=", 1)[1]
        key, _, rest = rest.partition(">")
        close = rest.find("</parameter>")
        following = rest.find("<parameter=")
        if close < 0 and following < 0:
            break  # cut off inside the value, so the value is not whole
        if close < 0 or 0 <= following < close:
            value, rest = rest[:following], rest[following:]
        else:
            value, rest = rest[:close], rest[close + len("</parameter>") :]
        args[key.strip().strip("\"'").lower()] = value.strip("\n")
    return command_from(name, args)


def tool_from_object(item: object) -> str:
    if not isinstance(item, dict):
        return ""
    function = item.get("function") if isinstance(item.get("function"), dict) else {}
    name = item.get("name") or function.get("name") or item.get("tool")
    args = None
    for source in (item, function):
        for key in ("arguments", "parameters", "args", "input"):
            if key in source:
                args = source[key]
                break
        if args is not None:
            break
    if isinstance(args, str):
        try:
            args = _JSON.decode(args) if args.strip() else {}
        except ValueError:
            args = {"value": args}
    return command_from(name, args if args is not None else {})


def json_tool(answer: str) -> str:
    # <tool_call>{"name": "...", "arguments": {...}}</tool_call>, or the JSON alone.
    at = answer.find("<tool_call>")
    text = answer[at + len("<tool_call>") :] if at >= 0 else answer.lstrip()
    brace = text.find("{")
    if brace < 0 or text[:brace].strip():
        return ""
    try:
        item, _ = _JSON.raw_decode(text, brace)
    except ValueError:
        return ""
    return tool_from_object(item)


def native_tool(answer: str) -> str:
    # The model's own tool format. It emits one call and nothing after it.
    return xml_tool(answer) or json_tool(answer)


def openai_tool(tool_calls: object) -> str:
    if not isinstance(tool_calls, list) or not tool_calls:
        return ""
    return tool_from_object(tool_calls[0])


def looks_like_call(answer: str) -> bool:
    return "<tool_call>" in answer or "<function=" in answer or bool(re.match(r"\s*\{\s*\"(name|function)\"", answer))


def write_block(answer: str) -> str:
    lines = answer.splitlines()
    for index, line in enumerate(lines):
        if clean_tool_line(line).startswith("WRITE "):
            block = [clean_tool_line(line)]
            for other in lines[index + 1 :]:
                block.append(other)
                if clean_tool_line(other) == "ENDWRITE":
                    break
            return "\n".join(block)
    return ""


def pick_tool(answer: str, tool_calls: object = None) -> tuple[str, str]:
    """Return the first tool call and the text of only that call, for the chat history."""
    command = openai_tool(tool_calls)
    if command:
        item = tool_calls[0] if isinstance(tool_calls, list) else {}
        function = item.get("function") if isinstance(item, dict) and isinstance(item.get("function"), dict) else {}
        args = function.get("arguments", {})
        if isinstance(args, str):
            try:
                args = _JSON.decode(args) if args.strip() else {}
            except ValueError:
                pass
        call = {"name": function.get("name") or item.get("name", ""), "arguments": args}
        return command, "<tool_call>\n" + json.dumps(call, ensure_ascii=False) + "\n</tool_call>"
    command = native_tool(answer)
    if command:
        at = answer.find("<tool_call>")
        if at < 0:
            at = answer.find("<function=") if "<function=" in answer else answer.find("{")
        segment = answer[max(0, at) :]
        end = segment.find("</tool_call>")
        if end >= 0:
            segment = segment[: end + len("</tool_call>")]
        elif segment.startswith("<tool_call>"):
            segment = segment.rstrip() + "\n</tool_call>"
        return command, segment.strip()
    command = first_tool(answer)
    if command:
        return command, (write_block(answer) or command) if command.startswith("WRITE ") else command
    return "", ""


def tool_response(text: str) -> str:
    body = clip((text or "").strip(), TOOL_CHARS).replace("</tool_response>", "<\\/tool_response>")
    return "<tool_response>\n" + body + "\n</tool_response>"


def clip(text: str, limit: int) -> str:
    """Keep the start and the end of long output. Errors are often at the end."""
    text = text or ""
    if len(text) <= limit:
        return text
    head = max(0, limit * 2 // 3)
    tail = max(0, limit - head - 40)
    cut = len(text) - head - tail
    return text[:head] + f"\n[... {cut} characters cut ...]\n" + (text[-tail:] if tail else "")


def first_tool(answer: str) -> str:
    for line in answer.splitlines():
        stripped = clean_tool_line(line)
        if stripped == "SESSIONS" or stripped.startswith(("LIST ", "GREP ", "READ ", "MKDIR ", "RUN ", "WRITE ", "SESSIONS ")):
            return stripped
        if stripped == "LIST":
            return "LIST ."
    return ""


def fit_context(messages: list[dict], budget: int = CONTEXT_CHARS) -> None:
    """Keep the system prompt, the task and the newest call. Shrink or drop older turns."""

    def size() -> int:
        return sum(len(str(item.get("content") or "")) for item in messages)

    index = 2
    while size() > budget and index < len(messages) - 2:
        item = messages[index]
        content = str(item.get("content") or "")
        if item.get("role") == "user" and content.startswith("<tool_response>"):
            item["content"] = "<tool_response>\n[older result removed to save room]\n</tool_response>"
        elif item.get("role") == "assistant" and len(content) > 300:
            item["content"] = clip(content, 300)
        index += 1
    while size() > budget and len(messages) > 4:
        del messages[2:4]


def deep_request(prompt: str) -> tuple[bool, str]:
    """Deep mode turns thinking on, only when the task asks for it by name."""
    text = (prompt or "").lstrip()
    lowered = text.lower()
    for mark in ("/deep", "deep mode:"):
        if lowered.startswith(mark) and (len(lowered) == len(mark) or not lowered[len(mark)].isalnum()):
            return True, text[len(mark) :].lstrip(" :\t\r\n")
    return False, prompt


def plain_answer(answer: str) -> str:
    text = answer
    for mark in ("<tool_call>", "</tool_call>", "<|im_end|>", "<|endoftext|>"):
        text = text.replace(mark, "")
    return text.strip()


def run_tool(command: str, source: str, cut: bool) -> tuple[str, list[str]]:
    if command.startswith("WRITE "):
        wrote = apply_actions(source, require_end=cut)
        return "\n".join(wrote) or "nothing written", wrote
    if command.startswith("MKDIR "):
        made = apply_actions(command)
        return "\n".join(made) or "mkdir", made
    if command.startswith("LIST "):
        return tool_list(command[5:]), []
    if command.startswith("GREP "):
        return tool_grep(command[5:]), []
    if command.startswith("READ "):
        return read_tool(command[5:]), []
    if command.startswith("RUN "):
        return tool_run(command[4:]), []
    if command.startswith("SESSIONS"):
        return "\n".join(recent_sessions(8)) or "no saved session yet", []
    return "unknown tool", []


def ask_fitted(messages: list[dict], deep: bool) -> dict:
    budget = CONTEXT_CHARS
    for attempt in range(2):
        fit_context(messages, budget)
        try:
            return chat(messages, max_tokens=2048 if deep else 1200, think=deep, stop=STOPS)
        except ContextFull:
            if attempt:
                raise RuntimeError("The task is too long for Neuro's memory. Make it shorter.") from None
            budget //= 2
            if len(messages) > 2:
                messages[-1]["content"] = clip(str(messages[-1]["content"]), 1200)
    raise RuntimeError("The task is too long for Neuro's memory. Make it shorter.")


def run_agent(prompt: str) -> str:
    deep, prompt = deep_request(prompt)
    first = TOOL_HELP + "\n\nTask:\n" + prompt + "\n\n" + workspace_note()
    # One growing chat: the server reuses the cached prefix, and the model sees its own calls.
    messages: list[dict] = [{"role": "system", "content": SYSTEM}, {"role": "user", "content": first}]
    logs: list[str] = []
    fixes = 0
    for turn in range(MAX_TURNS):
        final_turn = turn == MAX_TURNS - 1
        if final_turn and len(messages) > 2:
            messages[-1]["content"] = str(messages[-1]["content"]) + "\n" + LAST_STEP
        reply = ask_fitted(messages, deep)
        # Only the visible answer is used. Thinking is never stored or sent back.
        answer = visible_answer(reply["content"])
        cut = reply["finish"] == "length"
        command, call = pick_tool(answer, reply["tool_calls"])
        if not command:
            if looks_like_call(answer) and not final_turn and fixes < 2:
                fixes += 1
                if cut:
                    note = "The tool call was cut off at the length limit, so nothing ran. Send a shorter call."
                else:
                    note = "The tool call could not be read, so nothing ran. Send one complete tool call or a plain answer."
                messages.append({"role": "assistant", "content": clip(answer, 600)})
                messages.append({"role": "user", "content": tool_response(note)})
                continue
            return "\n".join([*logs, plain_answer(answer)]).strip()[:4000]
        if final_turn:
            break
        print("tool " + command.splitlines()[0][:120], flush=True)
        source = command if "\n" in command else answer
        try:
            result, made = run_tool(command, source, cut)
        except Exception as exc:  # a tool failure is a result for the model, not a dead job
            result, made = f"error: {exc}", []
        if made:
            logs.extend(made)
        else:
            logs.append(command.splitlines()[0] + " -> " + (result.splitlines()[0] if result else "empty")[:140])
        messages.append({"role": "assistant", "content": clip(call, 1500)})
        messages.append({"role": "user", "content": tool_response(result)})
    return "\n".join(logs)[:4000]


def worker_main() -> None:
    pid = os.getpid()
    print(f"Neuro Code {pid} on one shared model, {SLOTS} generate at once", flush=True)
    idle = 0.8
    wait = 5.0
    said = ""
    state = ""
    beat_at = 0.0
    while not STOP_PATH.exists():
        try:
            if not has_queued():
                if state != "idle" or time.time() - beat_at > 30:
                    beat(pid, "idle")
                    state, beat_at = "idle", time.time()
                time.sleep(idle)
                idle = min(idle * 1.25, 2.0)
                continue
            idle = 0.8
            ready, why = model_state()
            if not ready:
                # Jobs stay queued. Nothing is claimed while the model is away.
                if state != "waiting" or time.time() - beat_at > 30:
                    beat(pid, "waiting")
                    state, beat_at = "waiting", time.time()
                if why != said:
                    print(f"agent {pid} waits: {why}", flush=True)
                    said = why
                time.sleep(wait + random.random())
                wait = min(wait * 2, 60.0)
                continue
            wait = 5.0
            said = ""
            job = claim(pid)
            state, beat_at = "", time.time()
        except sqlite3.Error as exc:
            print(f"agent {pid} database busy: {exc}", flush=True)
            time.sleep(1 + random.random())
            continue
        if job is None:
            time.sleep(0.8)
            continue
        print(f"agent {pid} job {job['id']} {job['title']}", flush=True)
        done = threading.Event()
        pulse = threading.Thread(target=keep_alive, args=(pid, done), daemon=True)
        pulse.start()
        try:
            try:
                result = run_agent(str(job["prompt"]))
            except Exception as exc:
                text = str(exc) or type(exc).__name__
                if isinstance(exc, ModelWait) or "training" in text.lower():
                    db_retry(requeue, int(job["id"]), pid)
                    print(f"agent {pid} waits: {text}", flush=True)
                    said = text
                    time.sleep(wait + random.random())
                    wait = min(wait * 2, 60.0)
                    continue
                db_retry(finish, int(job["id"]), pid, "", text)
                print(f"agent {pid} error job {job['id']}: {text}", flush=True)
            else:
                db_retry(finish, int(job["id"]), pid, result)
                print(f"agent {pid} done job {job['id']}", flush=True)
        except sqlite3.Error as exc:
            print(f"agent {pid} could not save job {job['id']}: {exc}", flush=True)
            time.sleep(1 + random.random())
        finally:
            done.set()
            pulse.join(timeout=2)
    try:
        with connect() as db:
            db.execute("DELETE FROM workers WHERE pid=?", (pid,))
    except sqlite3.Error:
        pass
    print(f"Neuro agent {pid} stopped", flush=True)


def spawn(requested: int, *, exact: bool = False) -> dict:
    STOP_PATH.unlink(missing_ok=True)
    count = int(requested) if exact else worker_cap(requested)
    count = max(0, min(count, 24))
    LOG_DIR.mkdir(exist_ok=True)
    flags = 0
    if os.name == "nt":
        flags = subprocess.CREATE_NEW_PROCESS_GROUP | subprocess.DETACHED_PROCESS
    pids = []
    for _ in range(count):
        log_path = LOG_DIR / f"worker-{time.time_ns()}.log"
        handle = log_path.open("a", encoding="utf-8")
        process = subprocess.Popen(
            [sys.executable, "-u", str(WORKER_SCRIPT), "agent"],
            cwd=str(HOME),
            stdout=handle,
            stderr=handle,
            creationflags=flags,
        )
        handle.close()
        pids.append(process.pid)
    return {
        "requested": int(requested),
        "started": len(pids),
        "pids": pids,
        "slots": SLOTS,
        "free_mib": free_mib(),
    }


def stop_workers() -> dict:
    STOP_PATH.write_text("stop", encoding="utf-8")
    with connect() as db:
        rows = db.execute("SELECT pid FROM workers").fetchall()
    stopped = []
    for row in rows:
        pid = int(row["pid"])
        if os.name == "nt":
            subprocess.run(
                ["taskkill", "/PID", str(pid), "/F"],
                check=False,
                stdout=subprocess.DEVNULL,
                stderr=subprocess.DEVNULL,
            )
        else:
            try:
                os.kill(pid, 15)
            except OSError:
                continue
        stopped.append(pid)
    time.sleep(0.3)
    STOP_PATH.unlink(missing_ok=True)
    with connect() as db:
        db.execute("DELETE FROM workers")
        db.execute(
            "UPDATE jobs SET status='queued', worker_pid=NULL, updated=? WHERE status='running'",
            (time.time(),),
        )
    return {"stopped": stopped}


def snapshot() -> dict:
    with connect() as db:
        reap(db)
        db.commit()
        workers = [dict(row) for row in db.execute("SELECT pid, state, job_id, heartbeat FROM workers ORDER BY pid")]
        counts = {
            row["status"]: row["n"]
            for row in db.execute("SELECT status, COUNT(*) AS n FROM jobs GROUP BY status")
        }
        jobs = [
            dict(row)
            for row in db.execute(
                "SELECT id, title, status, worker_pid, result, error FROM jobs ORDER BY id DESC LIMIT 12"
            )
        ]
    return {
        "slots": SLOTS,
        "workers": workers,
        "counts": counts,
        "jobs": jobs,
        "free_mib": free_mib(),
    }


def fit(value: object, width: int) -> str:
    text = str(value).replace("\n", " ")
    if len(text) <= width:
        return text.ljust(width)
    return text[: width - 1] + "…"


def draw_terminal(snap: dict, started: dict) -> None:
    counts = snap["counts"]
    workers = snap["workers"]
    generating = sum(1 for item in workers if item["state"] == "generating")
    width = 62
    inner = width - 2

    def row(text: str) -> str:
        return "│" + fit(text, inner) + "│"

    lines = [
        "┌" + "─" * inner + "┐",
        row("  NEURO  1.0"),
        row("  Agent console  ·  one local model"),
        "├" + "─" * inner + "┤",
        row("  Model   local"),
        row("  API     http://127.0.0.1:8080/v1"),
        row(f"  Slots   {generating} generating  ·  {SLOTS} at once  ·  {len(workers)} workers"),
        row(
            f"  Jobs    queued {counts.get('queued', 0)}   "
            f"running {counts.get('running', 0)}   done {counts.get('done', 0)}"
        ),
        row(f"  Started {started['started']} of {started['requested']}"),
        "├" + "─" * inner + "┤",
        row("  agents"),
    ]
    jobs = list(reversed(snap["jobs"][:8]))
    if not jobs:
        lines.append(row("  waiting for work"))
    for job in jobs:
        lines.append(row(f"  {job['id']:>3}  {str(job['status']):<10} {job['title']}"))
    lines.append("├" + "─" * inner + "┤")
    lines.append(row("  The website does not start these. Ctrl+C stops them."))
    lines.append(row("  > shared local model"))
    lines.append("└" + "─" * inner + "┘")
    sys.stdout.write("\x1b[H\x1b[J" + "\n".join(lines) + "\n")
    sys.stdout.flush()


def job_row(job_id: int) -> dict | None:
    with connect() as db:
        row = db.execute(
            "SELECT id, status, result, error FROM jobs WHERE id=?",
            (job_id,),
        ).fetchone()
    return dict(row) if row else None


def ensure_workers(wanted: int = 4) -> dict:
    alive = len(snapshot()["workers"])
    wanted = max(1, min(int(wanted), 24))
    if alive >= wanted:
        return {"started": 0, "already": alive, "slots": SLOTS, "free_mib": free_mib()}
    extra = worker_cap(wanted - alive)
    if alive == 0 and extra == 0:
        extra = 1
    if extra <= 0:
        return {"started": 0, "already": alive, "slots": SLOTS, "free_mib": free_mib()}
    started = spawn(extra, exact=True)
    started["already"] = alive
    return started


def wait_job(job_id: int, timeout: int = 180) -> str:
    deadline = time.time() + timeout
    while time.time() < deadline:
        row = job_row(job_id)
        if row and row["status"] in {"done", "error"}:
            return str(row["result"] or row["error"] or "")
        time.sleep(0.8)
    if gpu_busy():
        return TRAINING_WAIT
    return "The model did not answer in time."


BRAIN = (
    ("   .-~^~-.  ", "  (  .  .  ) ", "   `-._.-'  "),
    ("   .-~^~-.  ", "  ( .    . ) ", "   `-._.-'  "),
    ("   .-~^~-.  ", "  (  .  .  ) ", "    `-.-'   "),
    ("   .-^~^-.  ", "  ( .    . ) ", "   `-._.-'  "),
)


def code_screen(snap: dict, typed: str, note: str, beat: int = 0) -> tuple[str, int, str]:
    pink = "\x1b[38;2;255;92;146m"
    soft = "\x1b[38;2;255;176;198m"
    reset = "\x1b[0m"
    workers = snap["workers"]
    counts = snap["counts"]
    generating = sum(1 for item in workers if item["state"] == "generating")
    width = max(78, min(shutil.get_terminal_size(fallback=(100, 30)).columns, 100))
    inner = width - 2

    def row(text: str, color: str = "") -> str:
        body = fit(text, inner)
        if color:
            body = color + body + reset
        return pink + "│" + reset + body + pink + "│" + reset

    brain = BRAIN[beat % len(BRAIN)]
    saved = recent_sessions(4)
    left = ["", *brain, "", "Neuro Code"]
    right = ["saved on this PC"] + (saved or ["nothing saved yet"])
    while len(right) < len(left):
        right.append("")
    lines = [
        pink + "┌" + "─" * inner + "┐" + reset,
        row("  Neuro Code", soft),
        pink + "├" + "─" * inner + "┤" + reset,
    ]
    half = 34
    for index, part in enumerate(left):
        other = right[index] if index < len(right) else ""
        lines.append(row(fit(part.center(half), half) + "  " + other, soft if index and index < 4 else ""))
    lines.append(pink + "├" + "─" * inner + "┤" + reset)
    lines.append(
        row(
            f"  agents {len(workers)}   generating {generating}   "
            f"queued {counts.get('queued', 0)}   done {counts.get('done', 0)}   "
            f"free {snap.get('free_mib', 0)} MB"
        )
    )
    lines.append(row("  LIST  GREP  READ  WRITE  RUN  SESSIONS"))
    jobs = list(reversed(snap["jobs"][:6]))
    if not jobs:
        lines.append(row("  waiting for a task"))
    for job in jobs:
        lines.append(row(f"  {job['id']:>3}  {str(job['status']):<8} {job['title']}"))
    if note:
        lines.append(pink + "├" + "─" * inner + "┤" + reset)
        text = " ".join(note.split())
        while text and len(lines) < 22:
            lines.append(row("  " + text[: inner - 4], soft))
            text = text[inner - 4 :]
    lines.append(pink + "├" + "─" * inner + "┤" + reset)
    prompt_row = len(lines) + 1
    lines.append(row("  > " + typed + "▌", soft))
    lines.append(row("  exit closes this window. tasks stay in the local session."))
    lines.append(pink + "└" + "─" * inner + "┘" + reset)
    body = "\n".join(lines)
    return "\x1b[H\x1b[J" + body + "\n", prompt_row, lines[prompt_row - 1]


def code_main(initial: str = "") -> None:
    ensure_workers(4)
    typed = ""
    note = ""
    pending = enqueue(initial[:70], initial) if initial else None
    if os.name == "nt":
        import msvcrt
    else:
        msvcrt = None
    snap = snapshot()
    beat = 0
    painted = ""
    prompt_at = 1
    prompt_line = ""
    next_snap = 0.0
    next_beat = time.time() + 0.8

    def show(full: bool) -> None:
        nonlocal painted, prompt_at, prompt_line
        frame, prompt_at, prompt_line = code_screen(snap, typed, note, beat)
        if not full:
            sys.stdout.write(f"\x1b[{prompt_at};1H" + prompt_line)
            sys.stdout.flush()
            return
        if frame != painted:
            sys.stdout.write(frame)
            sys.stdout.flush()
            painted = frame

    try:
        show(True)
        while True:
            if msvcrt is not None:
                typed_changed = False
                while msvcrt.kbhit():
                    char = msvcrt.getwch()
                    if char in {"\x00", "\xe0"}:
                        if msvcrt.kbhit():
                            msvcrt.getwch()
                        continue
                    if char in {"\x03"}:
                        return
                    if char in {"\r", "\n"}:
                        text = typed.strip()
                        typed = ""
                        typed_changed = True
                        if text in {"exit", "quit"}:
                            return
                        if text:
                            pending = enqueue(text[:70], text)
                            note = ""
                            next_snap = 0.0
                        break
                    if char in {"\x08", "\x7f"}:
                        typed = typed[:-1]
                        typed_changed = True
                        continue
                    if char >= " ":
                        typed += char
                        typed_changed = True
                if typed_changed:
                    show(not typed and not note)
                    continue
            now = time.time()
            if now >= next_snap:
                snap = snapshot()
                if pending:
                    row = job_row(pending)
                    if row and row["status"] in {"done", "error"}:
                        note = str(row["result"] or row["error"] or "")[:360]
                        pending = None
                    elif row and row["status"] == "queued" and not note and gpu_busy():
                        note = TRAINING_WAIT
                next_snap = now + 0.45
                show(True)
            elif not typed and now >= next_beat:
                beat = (beat + 1) % len(BRAIN)
                next_beat = now + 0.8
                show(True)
            if msvcrt is None:
                text = input("> ").strip()
                if text in {"exit", "quit"}:
                    break
                if text:
                    pending = enqueue(text[:70], text)
                    note = ""
                continue
            time.sleep(0.02)
    except KeyboardInterrupt:
        pass
    finally:
        sys.stdout.write("\x1b[0m\nwindow closed. the workers keep running.\n")
        sys.stdout.flush()


def drill_round(index: int) -> dict:
    """Check the tool layer with no model and no server. Safe to run while training."""
    folder = LOG_DIR / "drill"
    folder.mkdir(parents=True, exist_ok=True)
    cutoff = time.time() - 600
    for old in [*folder.glob("round-*.txt"), *folder.glob(".round-*.tmp")]:
        try:
            if old.stat().st_mtime < cutoff:
                old.unlink()
        except OSError:
            pass
    name = f"round-{os.getpid()}.txt"
    relative = f"agent-logs/drill/{name}"
    wrote = safe_write(relative, f"round {index}")
    read = safe_read(relative)
    listed = tool_list("agent-logs/drill")
    xml_write = "<tool_call><function=write_file><parameter=path></parameter><parameter=content>nope</parameter></function></tool_call>"
    picked, call = pick_tool(
        visible_answer(
            "<think>secret plan</think>Sure.\n<tool_call>\n<function=list>\n"
            "<parameter=path>\nworkspace\n</parameter>\n</function>"
        )
    )
    openai_calls = [{"id": "c1", "type": "function", "function": {"name": "list_dir", "arguments": "{\"path\": \"workspace\"}"}}]
    messages = [{"role": "system", "content": "s"}, {"role": "user", "content": "task"}]
    for _ in range(6):
        messages += [{"role": "assistant", "content": "READ a"}, {"role": "user", "content": tool_response("y" * 3000)}]
    fit_context(messages, 9000)
    long_reply = tool_response("x" * 5000 + "TAIL")
    acted = apply_actions(
        "WRITE workspace/agent-check/drill-one.txt\nonly-one\nENDWRITE\nWRITE workspace/agent-check/drill-two.txt\nnope\nENDWRITE"
    )
    cut = apply_actions("WRITE workspace/agent-check/drill-cut.txt\npartial", require_end=True)
    checks = {
        "write": wrote.startswith("wrote "),
        "read": f"round {index}" in read,
        "list": name in listed or (folder / name).is_file(),
        "secret": safe_write("cluster.token", "nope").startswith("blocked"),
        "secret_case": safe_read("CLUSTER.TOKEN") == "unreadable",
        "secret_stream": safe_read("cluster.token::$DATA") == "unreadable",
        "secret_dot": safe_read("cluster.token.") == "unreadable",
        "numbered": first_tool("1. LIST workspace") == "LIST workspace",
        "native": native_tool("<tool_call><function=grep><parameter=query>def drill_round</parameter></function></tool_call>") == "GREP def drill_round",
        "json": native_tool('<tool_call>\n{"name": "read_file", "arguments": {"path": "workspace/notes.txt"}}\n</tool_call>') == "READ workspace/notes.txt",
        "json_write": native_tool('<tool_call>{"name": "write_file", "arguments": {"path": "workspace/a.txt", "content": "one\ntwo"}}</tool_call>') == "WRITE workspace/a.txt\none\ntwo\nENDWRITE",
        "openai": openai_tool(openai_calls) == "LIST workspace",
        "one_call": picked == "LIST workspace" and "secret" not in call and "Sure" not in call and call.endswith("</tool_call>"),
        "think_close": visible_answer("plan here</think>LIST workspace") == "LIST workspace",
        "think_open": visible_answer("<think>a</think>ok<think>never closed") == "ok",
        "grep": "agents.py" in tool_grep("def drill_round"),
        "one_write": acted == ["wrote workspace/agent-check/drill-one.txt"],
        "second": safe_read("workspace/agent-check/drill-two.txt") == "missing",
        "cut_write": len(cut) == 1 and cut[0].startswith("not written") and safe_read("workspace/agent-check/drill-cut.txt") == "missing",
        "escape": safe_write("../outside.txt", "nope").startswith("blocked"),
        "mkdir": apply_actions("MKDIR workspace/agent-check/drill-dir") == ["mkdir workspace/agent-check/drill-dir"],
        "run": "drill-ok" in tool_run("echo drill-ok"),
        "run_blocked": tool_run("rm -rf /") == "blocked",
        "run_kill": tool_run("taskkill /IM python.exe /F") == "blocked",
        "envelope": tool_response("only-one").startswith("<tool_response>") and "only-one" in tool_response("only-one"),
        "clip": "TAIL" in long_reply and len(long_reply) < TOOL_CHARS + 100,
        "fit": sum(len(item["content"]) for item in messages) <= 9000 and messages[-1]["content"].startswith("<tool_response>"),
        "empty_path": native_tool(xml_write) == "",
        "deep": deep_request("/deep why?") == (True, "why?") and deep_request("deep learning basics")[0] is False,
    }
    try:
        (folder / name).unlink(missing_ok=True)
    except OSError:
        pass
    ok = all(checks.values())
    row = {"t": time.strftime("%Y-%m-%d %H:%M:%S", time.localtime()), "round": index, "ok": ok}
    if not ok:
        row["miss"] = [key for key, value in checks.items() if not value]
    append_capped(LOG_DIR / "drill.jsonl", json.dumps(row))
    return row


def drill_main(rounds: int = 10) -> None:
    count = max(1, min(int(rounds), 50))
    passed = 0
    for index in range(1, count + 1):
        row = drill_round(index)
        passed += 1 if row["ok"] else 0
        miss = "" if row["ok"] else " " + ",".join(row.get("miss", []))
        print(f"drill {index} {'ok' if row['ok'] else 'fail'}{miss}", flush=True)
    print(f"drill passed {passed} of {count}", flush=True)


def terminal(workers: int = 4) -> None:
    seeded = seed()
    started = ensure_workers(workers)
    print(f"seeded {seeded}", flush=True)
    try:
        while True:
            draw_terminal(snapshot(), started)
            time.sleep(1.5)
    except KeyboardInterrupt:
        sys.stdout.write("\nstopping workers\n")
        sys.stdout.flush()
        stop_workers()


def main() -> None:
    command = sys.argv[1] if len(sys.argv) > 1 else "code"
    if command == "agent":
        worker_main()
        return
    if command == "drill":
        rounds = int(sys.argv[2]) if len(sys.argv) > 2 else 10
        drill_main(rounds)
        return
    if command == "code":
        code_main(" ".join(sys.argv[2:]).strip())
        return
    if command == "agents":
        terminal(int(sys.argv[2]) if len(sys.argv) > 2 else 4)
        return
    raise SystemExit("use: agents.py agent | agents.py drill [ROUNDS] | agents.py code [TASK]")


if __name__ == "__main__":
    main()
