"""CMD workers for Neuro.

One llama-server holds the model. Workers are separate Python processes.
Only a few of them may generate at once, because the GPU has four slots.
"""

from __future__ import annotations

import contextvars
import ctypes
import difflib
import http.client
import json
import locale
import os
import random
import re
import shutil
import socket
import sqlite3
import stat
import subprocess
import sys
import threading
import time
import urllib.error
import urllib.request
from pathlib import Path
from typing import NamedTuple

# Where this copy lives. The same file runs in two places:
# - inside neuro-cluster (mesh.py next to it, the other computers): the repo is this folder.
# - in neuro-agents on the main PC: the repo is the sibling neuro-cluster folder when it has
#   mesh.py, while the queue, logs and workspace stay in this folder.
# HOME holds agents.db, agent-logs, workspace and the stop files. ROOT is the repo the tools see.
HOME = Path(__file__).resolve().parent


def _mesh_state(folder: Path) -> str:
    """'file' when folder/mesh.py is a regular file, 'missing' when absent, 'unknown' otherwise."""
    try:
        info = (folder / "mesh.py").stat()
    except FileNotFoundError:
        return "missing"
    except OSError:
        return "unknown"
    return "file" if stat.S_ISREG(info.st_mode) else "missing"


# An unreadable mesh.py here still counts as inside the cluster: the narrower tree is the safe pick.
IN_CLUSTER = _mesh_state(HOME) != "missing"
if IN_CLUSTER:
    ROOT = HOME
else:
    ROOT = HOME.parent / "neuro-cluster"
    if _mesh_state(ROOT) == "missing":
        ROOT = HOME
DB_PATH = HOME / "agents.db"
STOP_PATH = HOME / "agents.stop"
LOG_DIR = HOME / "agent-logs"
# Workers start this same file ("agents.py agent"), so a worker always runs the code and the
# queue of the copy that started it. mesh.py on the main PC imports ../neuro-agents first, so
# "mesh.py agent" there would run the other copy against the other agents.db.
WORKER_SCRIPT = Path(__file__).resolve()


def stop_requested(path: Path) -> bool:
    try:
        path.stat()
    except FileNotFoundError:
        return False
    except OSError:
        return True
    return True


def regular_file(path: Path) -> bool:
    try:
        info = path.stat()
    except FileNotFoundError:
        return False
    except OSError:
        return True
    return stat.S_ISREG(info.st_mode)


def tree_for(raw: str) -> Path:
    if raw in {"agent-logs", "workspace"} or raw.startswith("agent-logs/") or raw.startswith("workspace/"):
        return HOME
    return ROOT


SLOTS = 4
CHAT_URL = os.environ.get("NEURO_CHAT_URL") or "http://127.0.0.1:8080/v1/chat/completions"
HEALTH_URL = CHAT_URL.split("/v1/", 1)[0].rstrip("/") + "/health"
# FileNotFoundError on this file means the GPU is free. Anything else means training owns it.
GPU_LOCK = Path(os.environ.get("NEURO_GPU_LOCK") or (HOME.parent / "train_local" / "gpu.lock"))
REQUEST_TIMEOUT = 180
# The server has 16384 tokens of context shared by 4 slots (--kv-unified). About 40000 characters
# is roughly 10-12k tokens. ContextFull from the server still halves this for one retry.
DEFAULT_CONTEXT_CHARS = 40000


def _env_chars(name: str, default: int) -> int:
    try:
        value = int(os.environ.get(name) or default)
    except ValueError:
        return default
    return max(4000, min(value, 200000))


CONTEXT_CHARS = _env_chars("NEURO_CONTEXT_CHARS", DEFAULT_CONTEXT_CHARS)
TOOL_CHARS = 2400
READ_CHARS = TOOL_CHARS - 200  # a READ fits whole inside one tool_response
MAX_TURNS = 6
REAP_AFTER = 180
SECRET_NAMES = frozenset({"cluster.token", ".env", ".env.local", "token.txt", "nodes.json", "specs.json"})
DB_SUFFIXES = (".db", ".db-wal", ".db-shm", ".db-journal", ".sqlite", ".sqlite3")
TRAINING_WAIT = "The model is training on this PC, so this task has to wait."
OFFLINE_WAIT = "Neuro's model is not running yet, so this task has to wait."
LOADING_WAIT = "Neuro is still loading the model, so this task has to wait."
SYSTEM = "You are Neuro. Do the task. One tool call, then stop. Do not explain the tool."
LAST_STEP = "That was the last tool. Answer the task now in plain text. Do not emit a tool."
# Never let the model invent its own tool result. "</tool_call>" is no longer a stop: a turn may
# carry up to MAX_BATCH read-only calls, and the parser runs only what batch_calls allows.
STOPS = ["<tool_response>"]
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


_schema_ready: set[str] = set()
# Drills point one thread at a temporary queue and log folder. Nothing else ever sets these, so a
# drill running inside mesh.py never touches the live agents.db or the live logs.
_OVERRIDE = threading.local()


def db_path() -> Path:
    return getattr(_OVERRIDE, "db", None) or DB_PATH


def logs_home() -> Path:
    return getattr(_OVERRIDE, "logs", None) or LOG_DIR


# Team mode columns. ALTER TABLE ADD COLUMN with a constant default keeps every old row valid, and
# workers still running older code keep working: they never name these columns.
TEAM_COLUMNS = (
    ("parent_id", "INTEGER DEFAULT NULL"),
    ("role", "TEXT DEFAULT NULL"),
    ("depth", "INTEGER NOT NULL DEFAULT 0"),
    ("prompt_tokens", "INTEGER NOT NULL DEFAULT 0"),
    ("completion_tokens", "INTEGER NOT NULL DEFAULT 0"),
    ("cached_tokens", "INTEGER NOT NULL DEFAULT 0"),
)


def migrate_jobs(db: sqlite3.Connection) -> list[str]:
    """Add the team columns that are missing. Safe to run again and from several processes."""
    added: list[str] = []
    db.execute("BEGIN IMMEDIATE")
    have = {str(row[1]) for row in db.execute("PRAGMA table_info(jobs)")}
    for name, declaration in TEAM_COLUMNS:
        if name in have:
            continue
        try:
            db.execute(f"ALTER TABLE jobs ADD COLUMN {name} {declaration}")
            added.append(name)
        except sqlite3.OperationalError as exc:
            if "duplicate column" not in str(exc).lower():
                raise
    db.execute("CREATE INDEX IF NOT EXISTS jobs_parent ON jobs(parent_id)")
    db.commit()
    return added


def connect() -> sqlite3.Connection:
    # timeout is SQLite's busy_timeout: ten processes wait for the write lock instead of failing.
    path = db_path()
    db = sqlite3.connect(path, timeout=30, factory=_Db)
    db.row_factory = sqlite3.Row
    db.execute("PRAGMA busy_timeout=30000")
    db.execute("PRAGMA synchronous=NORMAL")
    if path_key(Path(path)) not in _schema_ready:
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
        migrate_jobs(db)
        _schema_ready.add(path_key(Path(path)))
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


def enqueue(title: str, prompt: str, *, role: str | None = None) -> int:
    """A top-level job. role='lead' marks a team task (agents.py team). Sub-tasks come from SPAWN."""
    now = time.time()
    # The task kind rides on the first line of the prompt. run_agent strips it.
    stamped = stamp_kind(prompt)
    with connect() as db:
        cursor = db.execute(
            "INSERT INTO jobs (title, prompt, status, created, updated, role) VALUES (?, ?, 'queued', ?, ?, ?)",
            (title.strip()[:80] or "task", stamped, now, now, role),
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
        # A lead sitting in WAIT generates nothing, so it does not hold one of the model's slots.
        running = db.execute(
            "SELECT COUNT(*) AS n FROM jobs j WHERE j.status='running' AND NOT EXISTS "
            "(SELECT 1 FROM workers w WHERE w.pid=j.worker_pid AND w.state='team_wait')"
        ).fetchone()["n"]
        if running >= SLOTS:
            db.execute("UPDATE workers SET state='waiting', job_id=NULL, heartbeat=? WHERE pid=?", (now, pid))
            db.commit()
            return None
        # No deadlock: sub-tasks go before new top-level jobs, and at most max(1, workers - 1)
        # top-level jobs that may SPAWN run at once, so one worker stays free for sub-tasks.
        workers = db.execute("SELECT COUNT(*) AS n FROM workers").fetchone()["n"]
        leads = db.execute(
            "SELECT COUNT(*) AS n FROM jobs WHERE status='running' AND parent_id IS NULL "
            "AND prompt NOT LIKE '[kind:quick]%'"
        ).fetchone()["n"]
        lead_room = 1 if leads < max(1, int(workers) - 1) else 0
        job = db.execute(
            "SELECT id, title, prompt, parent_id, role, depth FROM jobs WHERE status='queued' "
            "AND (parent_id IS NOT NULL OR prompt LIKE '[kind:quick]%' OR ?) "
            "ORDER BY CASE WHEN parent_id IS NULL THEN 1 ELSE 0 END, id LIMIT 1",
            (lead_room,),
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
        return {
            "id": int(job["id"]),
            "title": str(job["title"]),
            "prompt": str(job["prompt"]),
            "parent_id": int(job["parent_id"]) if job["parent_id"] is not None else None,
            "role": str(job["role"] or ""),
            "depth": int(job["depth"] or 0),
        }


CANCELLED = "cancelled: the lead finished before this sub-task started"


def finish(job_id: int, pid: int, result: str = "", error: str = "") -> None:
    now = time.time()
    status = "error" if error else "done"
    title = ""
    team_row = None
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
        row = db.execute("SELECT title, parent_id FROM jobs WHERE id=?", (job_id,)).fetchone()
        title = str(row["title"]) if row else ""
        root = job_id
        if row is not None and row["parent_id"] is None:
            # Nobody will WAIT for sub-tasks that never started, so they do not run.
            db.execute(
                "UPDATE jobs SET status='error', error=?, updated=? WHERE parent_id=? AND status='queued'",
                (CANCELLED, now, job_id),
            )
        elif row is not None:
            root = int(row["parent_id"])
        # The write above holds the lock, so only the last job of a tree to finish sees it closed.
        tree = db.execute(
            "SELECT COUNT(*) AS jobs, SUM(status IN ('queued','running')) AS open, "
            "SUM(prompt_tokens) AS p, SUM(completion_tokens) AS c, SUM(cached_tokens) AS k, "
            "MIN(created) AS first, MAX(updated) AS last FROM jobs WHERE id=? OR parent_id=?",
            (root, root),
        ).fetchone()
        if tree is not None and int(tree["jobs"] or 0) > 1 and not int(tree["open"] or 0):
            team_row = {
                "t": time.strftime("%Y-%m-%d %H:%M:%S", time.localtime()),
                "kind": "team",
                "tree": root,
                "jobs": int(tree["jobs"]),
                "prompt_tokens": int(tree["p"] or 0),
                "completion_tokens": int(tree["c"] or 0),
                "cached_tokens": int(tree["k"] or 0),
                "seconds": round(float(tree["last"] or now) - float(tree["first"] or now), 1),
            }
    try:
        remember(job_id, status, title, error or result)
    except OSError:
        pass
    if team_row is not None:
        try:
            append_capped(logs_home() / "usage.jsonl", json.dumps(team_row), limit=2_000_000, keep=5000)
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
            response.read(4_000)
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
        raw = exc.read(4_000).decode("utf-8", errors="replace")
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


def chat(
    messages: list[dict],
    max_tokens: int = 1000,
    *,
    think: bool = False,
    stop: list[str] | None = None,
    _open=None,
    _sleep=None,
) -> dict:
    payload = {
        "messages": messages,
        "temperature": 0.6 if think else 0.4,
        "max_tokens": max(64, min(int(max_tokens), 4096)),
        # Thinking follows the task kind (KIND_BUDGET) or Deep mode.
        "chat_template_kwargs": {"enable_thinking": bool(think)},
        "cache_prompt": True,
    }
    if stop:
        payload["stop"] = list(stop)
    data = json.dumps(payload).encode("utf-8")
    opener = _open or _OPENER.open
    pause = _sleep or time.sleep
    delays = (0, 2, 5, 10)
    problem = ""
    body = None
    hard = 0
    for attempt, delay in enumerate(delays):
        if delay:
            pause(delay + random.random())
        request = urllib.request.Request(CHAT_URL, data=data, headers={"Content-Type": "application/json"})
        try:
            with opener(request, timeout=REQUEST_TIMEOUT) as response:
                raw = response.read().decode("utf-8", errors="replace")
        except urllib.error.HTTPError as exc:
            detail = _http_detail(exc)
            if exc.code == 400 and "context" in detail.lower():
                raise ContextFull(detail) from exc
            if exc.code == 503:
                problem = "loading"
                continue
            if exc.code in {500, 502, 504}:
                # A hard server error may have spent a whole generation. Retry it once only.
                problem = f"error {exc.code}"
                hard += 1
                if hard >= 2:
                    break
                continue
            if exc.code in {408, 429}:
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
    usage = body.get("usage") if isinstance(body.get("usage"), dict) else {}
    timings = body.get("timings") if isinstance(body.get("timings"), dict) else {}
    details = usage.get("prompt_tokens_details") if isinstance(usage.get("prompt_tokens_details"), dict) else {}
    # reasoning_content is the model's private thinking. It is never shown and never sent back.
    return {
        "content": str(content),
        "tool_calls": message.get("tool_calls") or [],
        "finish": str(choice.get("finish_reason") or ""),
        "usage": {
            "prompt": _count(usage.get("prompt_tokens")),
            "completion": _count(usage.get("completion_tokens")),
            "cached": _count(details.get("cached_tokens") or timings.get("cache_n")),
        },
    }


def _count(value: object) -> int:
    try:
        return max(0, int(value or 0))
    except (TypeError, ValueError):
        return 0


def ask_model(prompt: str, max_tokens: int = 220) -> str:
    reply = chat(
        [{"role": "system", "content": SYSTEM}, {"role": "user", "content": prompt}],
        max_tokens=max(64, min(int(max_tokens), 1200)),
    )
    return visible_answer(reply["content"])


def secret_name(name: str) -> bool:
    plain = name.strip().rstrip(". ").lower()
    # Backups and temp copies of a secret are secret too (cluster.token.old, cluster.token.tmp-ab12,
    # .env.backup, mesh_url.txt is not secret).
    if any(plain.startswith(base + ".") for base in SECRET_NAMES):
        return True
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
    # Every line comes back numbered ("12: text") so EDIT can quote it exactly. The output stops
    # before the tool_response clip would cut it, and says which range to READ next.
    start, end = lines or (1, 10**9)
    picked: list[str] = []
    used = 0
    stopped_at = 0
    total = 0
    try:
        with path.open("r", encoding="utf-8", errors="replace") as handle:
            for number, line in enumerate(handle, 1):
                total = number
                if stopped_at and number - stopped_at > 100000:
                    break  # a huge file: no need to count every line
                if stopped_at or number < start:
                    continue
                if number > end:
                    if not lines:
                        continue
                    break
                row = f"{number}: {line.rstrip(chr(13) + chr(10))[:500]}"
                if picked and used + len(row) + 1 > READ_CHARS:
                    stopped_at = number
                    if lines:
                        break
                    continue
                picked.append(row)
                used += len(row) + 1
    except OSError:
        return "unreadable"
    if not picked:
        return "no lines there" if lines else ""
    text = "\n".join(picked)
    if stopped_at:
        last = end if lines else total
        size = "more lines" if lines else f"file continues, {total} lines in all"
        text += f"\n[{size}. READ {raw} {stopped_at}-{min(last, stopped_at + 79)} shows the next ones]"
    return text


READ_FAILED = frozenset({"missing", "unreadable", "no lines there"})


def _read_call(argument: str) -> tuple[str, str]:
    """READ path [START-END] -> (result, the path text that was read)."""
    text = argument.strip()
    parts = text.rsplit(None, 1)
    if len(parts) == 2:
        match = re.fullmatch(r"(\d+)\s*-\s*(\d+)", parts[1])
        if match:
            start = max(1, int(match.group(1)))
            end = max(start, min(int(match.group(2)), start + 399))
            return safe_read(parts[0], (start, end)), parts[0]
    result = safe_read(text)
    if result == "missing" and " " in text:
        short = tool_path(text)
        if short:
            return safe_read(short), short
    return result, text


def read_tool(argument: str) -> str:
    return _read_call(argument)[0]


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


# A code sub-agent writes only in its own folder, workspace/team-<lead id>/<own id>/, so parallel
# sub-agents never overwrite each other or the repo. Set by AgentRun for that job only; it adds a
# block on top of inside() and never loosens anything.
_JAIL: contextvars.ContextVar[tuple[str, Path] | None] = contextvars.ContextVar("neuro_write_jail", default=None)


def jail_block(raw: str, path: Path, *, folder_ok: bool = False) -> str:
    jail = _JAIL.get()
    if jail is None:
        return ""
    shown, root = jail
    if root in path.parents or (folder_ok and path == root):
        return ""
    return f"blocked {raw}: a sub-agent writes only under {shown}/"


def safe_write(relative: str, content: str) -> str:
    shown = (relative or "").strip().replace("\\", "/").lstrip("/")
    found = inside(relative)
    if found is None:
        return "blocked " + shown
    raw, path = found
    jailed = jail_block(raw, path)
    if jailed:
        return jailed
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


def path_key(path: Path) -> str:
    return os.path.normcase(str(path))


class TaskState:
    """What one task has done so far. It lives for one task only."""

    def __init__(self, task: str = "") -> None:
        self.task = task
        self.reads: set[str] = set()  # path_key of every file READ (or written) in this task
        self.read_names: list[str] = []
        self.changed: list[str] = []
        self.failures: list[str] = []
        self.todo: list[tuple[str, str]] = []
        self.secret = False  # a call touched a secret path or a result held a secret

    def saw(self, raw: str, path: Path) -> None:
        self.reads.add(path_key(path))
        if raw not in self.read_names:
            self.read_names.append(raw)

    def change(self, raw: str, path: Path) -> None:
        self.saw(raw, path)
        if raw not in self.changed:
            self.changed.append(raw)


EDIT_OPEN = "<<<<<<< SEARCH"
EDIT_MID = "======="
EDIT_CLOSE = ">>>>>>> REPLACE"
_NUMBERED = re.compile(r"^\s*\d+:(?: |$)")


def edit_call(path: str, old: str, new: str) -> str:
    return f"EDIT {path}\n{EDIT_OPEN}\n{old}\n{EDIT_MID}\n{new}\n{EDIT_CLOSE}"


def parse_edit(block: str) -> tuple[str, str, str, bool] | None:
    """EDIT path / <<<<<<< SEARCH / old / ======= / new / >>>>>>> REPLACE -> (path, old, new, closed)."""
    lines = block.splitlines()
    if not lines:
        return None
    head = clean_tool_line(lines[0])
    if not head.startswith("EDIT "):
        return None
    path = tool_path(head[5:])
    opened = next((i for i in range(1, len(lines)) if lines[i].strip().startswith("<<<<<<<")), None)
    if opened is None:
        return None
    mid = next((i for i in range(opened + 1, len(lines)) if re.fullmatch(r"={7,}", lines[i].strip())), None)
    if mid is None:
        return None
    close = next((i for i in range(mid + 1, len(lines)) if lines[i].strip().startswith(">>>>>>>")), None)
    new_lines = lines[mid + 1 : close if close is not None else len(lines)]
    if close is None and new_lines and new_lines[-1].strip().startswith("```"):
        new_lines = new_lines[:-1]
    return path, "\n".join(lines[opened + 1 : mid]), "\n".join(new_lines), close is not None


def _unnumber(text: str) -> str | None:
    """Old text copied from READ with its "12: " prefixes still on. None when it is not that."""
    rows = text.split("\n")
    filled = [row for row in rows if row.strip()]
    if not filled or not all(_NUMBERED.match(row) for row in filled):
        return None
    return "\n".join(re.sub(r"^\s*\d+: ?", "", row) for row in rows)


def _edit_diff(raw: str, line: int, old: str, new: str) -> str:
    body = [row for row in difflib.unified_diff(old.split("\n"), new.split("\n"), lineterm="", n=1)][2:]
    body = [row for row in body if not row.startswith("@@")]
    if len(body) > 16:
        body = body[:16] + [f"... {len(body) - 16} more diff lines"]
    return clip("\n".join([f"edited {raw} at line {line}", *body]), 1200)


def tool_edit(relative: str, old: str, new: str, state: TaskState | None) -> tuple[str, list[str]]:
    """Replace old with new in an existing file. The same path checks as WRITE, plus: the file was
    READ in this task, and old matches exactly once."""
    shown = (relative or "").strip().replace("\\", "/").lstrip("/")
    found = inside(relative)
    if found is None:
        return "blocked " + shown, []
    raw, path = found
    jailed = jail_block(raw, path)
    if jailed:
        return jailed, []
    if path.is_dir():
        return "blocked " + raw + " is a folder", []
    try:
        info = path.stat()
    except FileNotFoundError:
        return f"missing {raw}: EDIT changes an existing file. Use WRITE for a new file.", []
    except OSError:
        return "unreadable", []
    if not stat.S_ISREG(info.st_mode):
        return f"missing {raw}: EDIT changes an existing file.", []
    if state is None or path_key(path) not in state.reads:
        return f"refused: READ {raw} first, then EDIT it.", []
    if not old.strip():
        return "refused: the old text is empty. Copy the exact lines to change from READ.", []
    try:
        data = path.read_bytes()
    except OSError:
        return "unreadable", []
    bom = data.startswith(b"\xef\xbb\xbf")
    try:
        text = data.decode("utf-8-sig" if bom else "utf-8")
    except UnicodeDecodeError:
        return f"refused: {raw} is not UTF-8 text.", []
    crlf = "\r\n" in text and text.count("\r\n") == text.count("\n")
    body = text.replace("\r\n", "\n") if crlf else text
    old, new = old.replace("\r\n", "\n"), new.replace("\r\n", "\n")
    count = body.count(old)
    if count == 0:
        plain = _unnumber(old)
        if plain is not None and body.count(plain) > 0:
            old, count = plain, body.count(plain)
            new = _unnumber(new) if _unnumber(new) is not None else new
    if old == new:
        return "refused: the old and new text are the same.", []
    if count == 0:
        return f"refused: the old text matches 0 times in {raw}. Copy it exactly from READ, without line numbers.", []
    if count > 1:
        return f"refused: the old text matches {count} times in {raw}. Add nearby lines so it matches once.", []
    at = body.index(old)
    result = body[:at] + new + body[at + len(old) :]
    if crlf:
        result = result.replace("\n", "\r\n")
    encoded = (b"\xef\xbb\xbf" if bom else b"") + result.encode("utf-8")
    if len(encoded) > 150000:
        return "too large", []
    # Same swap as safe_write, so a parallel reader never sees half a file.
    temp = path.with_name(f".{path.name}.{os.getpid()}.tmp")
    try:
        temp.write_bytes(encoded)
        os.replace(temp, path)
    except OSError:
        try:
            temp.unlink(missing_ok=True)
        except OSError:
            pass
        try:
            path.write_bytes(encoded)
        except OSError as exc:
            return f"could not write {raw}: {exc.strerror or exc}", []
    state.change(raw, path)
    return _edit_diff(raw, body[:at].count("\n") + 1, old, new), ["edited " + raw]


GLOB_LIMIT = 50
GLOB_WALK = 20000


def glob_regex(pattern: str) -> str:
    out = []
    index = 0
    while index < len(pattern):
        char = pattern[index]
        if pattern.startswith("**/", index):
            out.append("(?:[^/]+/)*")
            index += 3
            continue
        if pattern.startswith("**", index):
            out.append(".*")
            index += 2
            continue
        if char == "*":
            out.append("[^/]*")
        elif char == "?":
            out.append("[^/]")
        elif char == "[" and "]" in pattern[index + 2 :]:
            close = pattern.index("]", index + 2)
            inner = pattern[index + 1 : close]
            if inner.startswith("!"):
                inner = "^" + inner[1:]
            out.append("[" + inner.replace("\\", "\\\\") + "]")
            index = close
        elif char == "{" and "}" in pattern[index:]:
            close = pattern.index("}", index)
            out.append("(?:" + "|".join(glob_regex(part) for part in pattern[index + 1 : close].split(",")) + ")")
            index = close
        else:
            out.append(re.escape(char))
        index += 1
    return "".join(out)


def tool_glob(pattern: str) -> str:
    """File paths that match a glob under the repo (or workspace/), sorted by path, capped."""
    text = (pattern or "").strip().strip("`'\"").replace("\\", "/")
    while text.startswith("./"):
        text = text[2:]
    if (
        not text or len(text) > 120 or text.startswith("/") or "\0" in text or ":" in text
        or ".." in text.split("/")
    ):
        return "need a relative pattern such as **/*.py"
    parts = text.split("/")
    static: list[str] = []
    for part in parts[:-1]:
        if any(mark in part for mark in "*?[{"):
            break
        static.append(part)
    start_rel = "/".join(static)
    if start_rel:
        found = inside(start_rel)
        if found is None or not found[1].is_dir():
            return "no matches"
        start = found[1]
        tree = tree_for(found[0]).resolve()
    else:
        start = tree = ROOT.resolve()
    rest = "/".join(parts[len(static) :])
    anywhere = "/" not in rest
    try:
        matcher = re.compile("(?s:" + glob_regex(rest) + r")\Z", re.IGNORECASE if os.name == "nt" else 0)
    except re.error:
        return "need a relative pattern such as **/*.py"
    hits: list[str] = []
    extra = 0
    walked = 0
    for dirpath, dirnames, filenames in os.walk(start):
        dirnames[:] = sorted(name for name in dirnames if name not in GREP_SKIP_DIRS and not secret_name(name))
        for name in sorted(filenames):
            walked += 1
            if secret_name(name):
                continue
            full = Path(dirpath) / name
            try:
                target = name if anywhere else full.relative_to(start).as_posix()
                shown = full.relative_to(tree).as_posix()
            except ValueError:
                continue
            if not matcher.match(target):
                continue
            if len(hits) >= GLOB_LIMIT:
                extra += 1
                continue
            hits.append(shown)
        if walked >= GLOB_WALK:
            break
    if extra:
        hits.append(f"... {extra} more. Narrow the pattern.")
    return "\n".join(hits) or "no matches"


TODO_LINE = re.compile(r"^\s*(?:[-*]\s*)?\[([ xX>~-]?)\]\s*(.+?)\s*$")
TODO_STATUS = {"": "pending", " ": "pending", "x": "done", "X": "done", ">": "doing", "~": "doing", "-": "doing"}
TODO_MARK = {"pending": " ", "doing": ">", "done": "x"}
TODO_LIMIT = 8


def parse_todo(block: str) -> list[tuple[str, str]]:
    items: list[tuple[str, str]] = []
    for line in block.splitlines()[1:]:
        match = TODO_LINE.match(line)
        if not match:
            if line.strip():
                break
            continue
        items.append((TODO_STATUS[match.group(1)], " ".join(match.group(2).split())[:80]))
        if len(items) >= TODO_LIMIT:
            break
    return items


def todo_line(items: list[tuple[str, str]]) -> str:
    """The checklist on one line. It rides on the newest message only, never in history."""
    if not items:
        return ""
    done = sum(1 for status, _ in items if status == "done")
    return f"TODO {done}/{len(items)}: " + " | ".join(f"[{TODO_MARK[status]}] {text[:60]}" for status, text in items)


def tool_todo(block: str, state: TaskState | None) -> str:
    items = parse_todo(block)
    if not items:
        return "TODO needs item lines: [ ] step, [>] current step, [x] done step"
    if state is not None:
        state.todo = items
    done = sum(1 for status, _ in items if status == "done")
    return f"todo saved, {done} of {len(items)} done"


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
            jailed = jail_block(found[0], found[1], folder_ok=True)
            if jailed:
                logs.append(jailed)
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


class Budget(NamedTuple):
    think: bool  # thinking on for the first (planning) turn
    max_tokens: int  # cap for turns that may WRITE/EDIT (code) and for the final answer
    max_turns: int  # generations per job
    write: bool  # WRITE, EDIT and MKDIR allowed
    run: bool  # RUN allowed
    verify: bool  # after a WRITE or EDIT, ask once for a checking RUN before "done"
    fixes: int  # repair turns for a tool call that could not be read


# The router picks one of these once per job. Safety checks below still apply to every kind.
KIND_BUDGET: dict[str, Budget] = {
    "quick": Budget(False, 600, 3, False, False, False, 1),
    "research": Budget(True, 1200, 6, False, True, False, 2),
    "review": Budget(True, 2560, 6, False, True, False, 2),
    "code": Budget(True, 3072, 12, True, True, True, 2),
}
TOOL_TOKENS = 700  # a routine tool turn: thinking off, a few calls or a short answer
THINK_TOKENS = 2048  # a planning or diagnosis turn: room to think, then call
DEEP_TOKENS = 4096  # /deep forces thinking on every turn with this budget
DEFAULT_KIND = "research"
READ_ONLY = frozenset({"LIST", "GREP", "GLOB", "READ", "SESSIONS", "TODO"})
MAX_BATCH = 3  # read-only calls that may run together in one turn
# The model's own syntax. The parsers (text_tools, parse_edit, apply_actions) expect exactly these.
TOOL_SYNTAX = (
    ("LIST", "LIST relative/dir"),
    ("GLOB", "GLOB **/*.py"),
    ("GREP", "GREP text"),
    ("READ", "READ relative/path\nREAD relative/path 120-180"),
    ("SESSIONS", "SESSIONS"),
    ("TODO", "TODO\n[x] done step\n[>] current step\n[ ] next step"),
    ("EDIT", f"EDIT relative/path\n{EDIT_OPEN}\nexact old lines\n{EDIT_MID}\nnew lines\n{EDIT_CLOSE}"),
    ("WRITE", "WRITE relative/path\nthe file contents\nENDWRITE"),
    ("MKDIR", "MKDIR relative/path"),
    ("RUN", "RUN command"),
)
KIND_NOTES = {
    "quick": "Answer briefly. Use a tool only when the answer needs the repo.",
    "research": "Look before you answer. Name the files and lines you used.",
    "review": "Find bugs and risks, with file and line for each. Do not fix them.",
    "code": "Read first, then EDIT (WRITE only for a new file), then RUN a compile or a test.",
}


def kind_tools(kind: str) -> frozenset[str]:
    budget = KIND_BUDGET.get(kind) or KIND_BUDGET["quick"]
    names = {"LIST", "GLOB", "GREP", "READ", "SESSIONS"}
    if kind in KIND_BUDGET and kind != "quick":
        names.add("TODO")  # quick tasks do not need a checklist
    if budget.write:
        names |= {"MKDIR", "WRITE", "EDIT"}
    if budget.run:
        names.add("RUN")
    return frozenset(names)


def _env_int(name: str, default: int, low: int, high: int) -> int:
    try:
        value = int(os.environ.get(name) or default)
    except ValueError:
        return default
    return max(low, min(value, high))


# Team mode: a lead (a top-level code, research or review job) splits a large task into sub-tasks
# and runs them on other workers in parallel. Sub-agents cannot SPAWN (depth limit 1).
TEAM_ROLES = {"research": "research", "code": "code", "review": "review", "test": "code"}
TEAM_KINDS = frozenset({"research", "review", "code"})
TEAM_HEADS = frozenset({"SPAWN", "WAIT", "CHILDREN"})
TEAM_MAX = _env_int("NEURO_TEAM_MAX", 6, 1, 12)  # sub-agents per lead, ever
TEAM_WAIT = _env_int("NEURO_TEAM_WAIT", 480, 5, 1800)  # seconds one WAIT may block
TEAM_WAITS = _env_int("NEURO_TEAM_WAITS", 4, 1, 12)  # WAIT calls per lead run
TEAM_WALL = _env_int("NEURO_TEAM_WALL", 3600, 60, 14400)  # seconds from the first SPAWN
TEAM_REPORT = _env_int("NEURO_TEAM_REPORT", 1500, 200, 4000)  # characters of each report in WAIT
TEAM_POLL = 2.0
TEAM_BONUS_TURNS = 4  # a lead that spawned needs turns to WAIT, verify and integrate
TEAM_RESULT_CHARS = min(12000, (TEAM_REPORT + 200) * TEAM_MAX)


def team_roles_for(kind: str) -> list[str]:
    """A lead may only start sub-agents with no more power than itself: a task that cannot change
    files cannot get files changed through a code or test sub-agent either."""
    write = (KIND_BUDGET.get(kind) or KIND_BUDGET["quick"]).write
    return [role for role, child in TEAM_ROLES.items() if write or not KIND_BUDGET[child].write]


def tool_help(kind: str, mode: str = "solo") -> str:
    """The whole system prompt for one kind. Stable text only: it is the cached prefix of every turn.
    mode: solo (no queue job), lead (a top-level job that may SPAWN), child (a sub-agent)."""
    allowed = kind_tools(kind)
    budget = KIND_BUDGET.get(kind) or KIND_BUDGET["quick"]
    lead = mode == "lead" and kind in TEAM_KINDS
    child = mode == "child"
    batch = " ".join(name for name in ("LIST", "GLOB", "GREP", "READ", "SESSIONS", "TODO") if name in allowed)
    alone = [name for name in ("EDIT", "WRITE", "MKDIR", "RUN") if name in allowed]
    lines = ["You are Neuro, a careful agent working in a local repo.", "- Understand the task first."]
    if child:
        lines[0] = "You are Neuro, a sub-agent in a local repo. Do only your sub-task; you cannot SPAWN."
    if "TODO" in allowed:
        lines.append("- For work with 3 or more steps, keep a short TODO and update it as you go.")
    lines.append("- Never guess what a file says. GLOB, GREP or READ it first.")
    if budget.write:
        lines.append("- Make the smallest correct change with EDIT. WRITE is for new files.")
        lines.append("- After a change, RUN a test or a compile and read the result.")
    else:
        lines.append("- This task does not change files.")
    lines += [
        "- If a call fails, find the cause before trying again. Never repeat the same call.",
        "- Do not narrate or explain tools. A turn is tool calls only, or the final answer.",
        f"Tools, one per line. Up to {MAX_BATCH} of {batch} may go in one turn"
        + (f"; {', '.join(alone)} {'goes' if len(alone) == 1 else 'go'} alone." if alone else "."),
    ]
    lines += [syntax for name, syntax in TOOL_SYNTAX if name in allowed]
    if budget.write and child:
        lines.append("WRITE, EDIT and MKDIR only under your folder named in the task. READ shows numbered lines; leave the numbers out of EDIT.")
    elif budget.write:
        lines.append("WRITE under workspace/ is allowed. READ shows numbered lines; leave the numbers out of EDIT.")
    if lead:
        roles = "|".join(team_roles_for(kind))
        lines += [
            "Team: a small task you do yourself. A large task with independent parts: plan, keep a TODO,"
            f" SPAWN up to {TEAM_MAX} sub-agents in parallel, each with one clear deliverable, then WAIT,"
            " review and verify their results" + (" (RUN the tests), integrate them yourself," if budget.write else ",")
            + " and answer.",
            f"SPAWN {roles.split('|')[0]} short title\nthe sub-task: what to do and the deliverable\nENDSPAWN",
            "WAIT\nWAIT 12 13\nCHILDREN",
            f"SPAWN roles: {roles}. Several SPAWNs may go in one turn; each returns an id at once."
            + (" Code and test sub-agents write only under workspace/team-<your id>/<their id>/." if budget.write else ""),
            "WAIT returns each report (all sub-agents if no ids); still running means WAIT later or keep working.",
        ]
    lines.append("Results come back inside <tool_response>.")
    lines.append(KIND_NOTES.get(kind, ""))
    if child:
        lines.append(
            "Final report, plain text, at most 8 lines: what was done, the files (paths), how it was verified"
            " and the result, and anything not done. No tool syntax. Stop when done."
        )
    elif budget.write:
        lines.append(
            "Final answer, plain text, at most 8 lines: what changed (files), how it was checked and the"
            " result, and anything not done. No tool syntax. Stop when done."
        )
    else:
        lines.append("Final answer: plain text, short and direct, no tool syntax. Stop when done.")
    return "\n".join(line for line in lines if line)


# Built once: the system message is the same for every job of a kind, so the server cache reuses it.
KIND_TOOLS = {kind: tool_help(kind) for kind in KIND_BUDGET}
LEAD_TOOLS = {kind: tool_help(kind, "lead") for kind in TEAM_KINDS}
CHILD_TOOLS = {kind: tool_help(kind, "child") for kind in KIND_BUDGET}
TOOL_HELP = KIND_TOOLS["code"]


def agent_system(kind: str, mode: str = "solo") -> str:
    if mode == "child":
        return CHILD_TOOLS.get(kind, CHILD_TOOLS["quick"])
    if mode == "lead" and kind in LEAD_TOOLS:
        return LEAD_TOOLS[kind]
    return KIND_TOOLS.get(kind, KIND_TOOLS["quick"])


def turn_budget(
    kind: str, *, turn: int, deep: bool = False, after_failure: bool = False, writing: bool = False,
    final_turn: bool = False,
) -> tuple[bool, int]:
    """(think, max_tokens) for one turn. Thinking plans the first turn of a real task and
    diagnoses the turn after a failure; routine tool turns stay short and do not think."""
    budget = KIND_BUDGET.get(kind) or KIND_BUDGET["quick"]
    if deep:
        return True, max(budget.max_tokens, DEEP_TOKENS)
    think = (turn == 0 and budget.think) or after_failure
    tokens = min(TOOL_TOKENS, budget.max_tokens)
    if final_turn or (writing and budget.write):
        tokens = budget.max_tokens
    if think:
        tokens = max(tokens, THINK_TOKENS)
    return think, tokens


_HE_PREFIX = r"(?<!\w)[ובהלשמכ]{0,2}"


def _en(*words: str) -> re.Pattern:
    return re.compile(r"\b(?:" + "|".join(words) + r")\b")


def _he(*forms: str) -> re.Pattern:
    # Hebrew glues ו/ב/ה/ל/ש/מ/כ to the front of a word. The end must still be a word end,
    # so קוד matches הקוד but not קודם.
    return re.compile(_HE_PREFIX + "(?:" + "|".join(forms) + r")(?!\w)")


# Asking for a change: a coding task even with no file named.
CODE_STRONG = (
    _en(r"fix(?:es|ed|ing)?", r"implement\w*", r"refactor\w*", r"patch(?:es|ed|ing)?", r"debug\w*",
        r"rewrite|rewrites|rewriting|rewrote", r"rename[sd]?", r"scaffold\w*"),
    _he("תקן", "תקני", "תקנו", "תתקן", "תתקני", "תיקון", "תקון", "תוסיף", "תוסיפי", "תוסיפו",
        "הוסף", "הוסיפי", "הוסיפו", "הוסיף", "תממש", "תממשי", "לממש", "שכתב", "תשכתב", "דבג", "תדבג",
        "ערוך", "תערוך", "ערכי"),
)
# Words that make a change only when code or a file is in the task.
CODE_WEAK = (
    _en(r"write|writes|writing", r"add|adds|adding", r"create|creates|creating", r"make", r"change|changes",
        r"update|updates", r"edit|edits", r"modify", r"remove", r"delete", r"insert", r"replace", r"generate",
        r"build"),
    _he("כתוב", "כתבי", "כתבו", "תכתוב", "תכתבי", "צור", "תיצור", "ליצור", "צרו", "תשנה", "תשני", "למחוק",
        "תמחק", "מחק"),
)
CODE_NOUN = (
    _en(r"code|codebase", r"bugs?|buggy", r"files?", r"functions?", r"methods?", r"class(?:es)?", r"modules?",
        r"scripts?", r"repo|repository", r"tests?|unittest|pytest", r"compile[sr]?|compiling", r"errors?",
        r"traceback", r"exceptions?", r"crash\w*", r"endpoints?", r"api", r"regex", r"variables?",
        r"folders?", r"director(?:y|ies)", r"python|javascript|typescript|html|css|sql|json|yaml",
        r"workspace", r"readme", r"database|db", r"server", r"config"),
    _he("קוד", "קודים", "באג", "באגים", "קובץ", "קבצים", "פונקציה", "פונקציות", "סקריפט", "סקריפטים",
        "שגיאה", "שגיאות", "טסט", "טסטים", "תיקייה", "תיקיה", "ריפו", "שרת", "מסד"),
)
REVIEW = (
    _en(r"review\w*", r"audit\w*", r"security", r"secure", r"vulnerab\w*", r"critique", r"inspect\w*",
        r"(?:find|look for|spot|hunt for|any)\s+(?:the\s+)?bugs?", r"code smells?"),
    _he("סקירה", "סקירת", "סקור", "תסקור", "תסקרי", "ביקורת", "אבטחה", "פרצה", "פרצות", "חולשה", "חולשות"),
)
# Checking words are a review when they point at code, a lookup otherwise.
CHECK = (
    _en(r"check|checks", r"verify", r"validate"),
    _he("בדוק", "בדקי", "בדקו", "תבדוק", "תבדקי"),
)
RESEARCH = (
    _en(r"where", r"explain\w*", r"find", r"search\w*", r"grep", r"locate", r"trace", r"investigate",
        r"analy[sz]e", r"compare", r"how (?:does|do|is|are)", r"which file", r"look (?:up|into|at)",
        r"show me", r"list", r"read", r"summari[sz]e", r"describe"),
    _he("איפה", "היכן", "הסבר", "הסבירי", "הסבירו", "תסביר", "תסבירי", "הסביר", "חפש", "חפשי", "תחפש",
        "מצא", "מצאי", "תמצא", "נתח", "תנתח", "השווה", "תשווה", "תאר", "תסכם", "סכם"),
)
# "do not change files", "בלי לשנות" — a refused action is not a request.
NEGATED = re.compile(
    r"\b(?:do not|don't|dont|never|without|no need to|not)\s+(?:\w+\s+){0,2}?\w+"
    r"|(?<!\w)(?:אל|בלי|לא|אין צורך)\s+\S+"
)
CODE_SIGN = re.compile(
    r"[\w-]\.(?:py|pyw|js|mjs|ts|tsx|jsx|json|md|txt|ps1|cmd|bat|sh|html?|css|toml|ya?ml|ini|cfg|log|go|rs"
    r"|java|c|cpp|h|hpp|cs|sql|db)\b"
    r"|\w[/\\]\w|\w\(\)|`|\b[a-z0-9]+_[a-z0-9_]+\b|\b[a-z]+[A-Z]\w*|\b[A-Z]{3,}\b|\bdef |\bimport |=>|\{|\}"
)


def _hit(groups: tuple[re.Pattern, ...], text: str) -> bool:
    return any(pattern.search(text) for pattern in groups)


def classify_task(prompt: str) -> str:
    """quick | research | review | code. When unsure: research. Files or code never mean quick."""
    text = (prompt or "").strip()
    if not text:
        return DEFAULT_KIND
    lowered = text.lower()
    asked = NEGATED.sub(" ", lowered)
    coded = bool(CODE_SIGN.search(text)) or _hit(CODE_NOUN, lowered)
    if _hit(CODE_STRONG, asked) or (coded and _hit(CODE_WEAK, asked)):
        return "code"
    if _hit(REVIEW, lowered) or (coded and _hit(CHECK, lowered)):
        return "review"
    if coded or _hit(RESEARCH, lowered) or _hit(CHECK, lowered):
        return "research"
    if len(text) <= 160 and text.count("\n") <= 1:
        return "quick"
    return DEFAULT_KIND


KIND_MARK = re.compile(r"\s*\[kind:([A-Za-z_-]{1,16})\][ \t]*(?:\r?\n|$)")


def split_kind(prompt: str) -> tuple[str | None, str]:
    """Take a leading [kind:x] line off a stored prompt. The tag is not trusted yet."""
    text = prompt or ""
    match = KIND_MARK.match(text)
    if not match:
        return None, text
    return match.group(1).lower(), text[match.end() :]


def job_kind(prompt: str) -> tuple[str, str]:
    """The kind for a job, and the prompt with the tag removed. A bad tag is classified again."""
    tag, rest = split_kind(prompt)
    if tag in KIND_BUDGET:
        return tag, rest
    return classify_task(deep_request(rest)[1]), rest


def stamp_kind(prompt: str) -> str:
    kind, rest = job_kind(prompt)
    head = f"[kind:{kind}]\n"
    return head + rest.strip()[: 4000 - len(head)]


def remember(job_id: int, status: str, title: str, text: str) -> None:
    row = {
        "t": time.strftime("%Y-%m-%d %H:%M", time.localtime()),
        "id": job_id,
        "status": status,
        "title": title[:80],
        "text": (text or "")[:800],
    }
    # Ten workers append here. The lock keeps lines whole and the cap keeps the file small.
    append_capped(logs_home() / "session.jsonl", json.dumps(row, ensure_ascii=False), limit=400000, keep=400)


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
            creationflags=getattr(subprocess, "CREATE_NO_WINDOW", 0),
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
    "edit": "EDIT", "edit_file": "EDIT", "str_replace": "EDIT", "replace_in_file": "EDIT", "apply_edit": "EDIT",
    "glob": "GLOB", "glob_files": "GLOB", "find_files": "GLOB", "file_search": "GLOB",
    "todo": "TODO", "todos": "TODO", "todo_write": "TODO", "todowrite": "TODO", "update_todo": "TODO",
    "update_todos": "TODO", "set_todo": "TODO", "update_plan": "TODO",
    "spawn": "SPAWN", "spawn_agent": "SPAWN", "spawn_subagent": "SPAWN", "subagent": "SPAWN",
    "wait": "WAIT", "wait_children": "WAIT", "wait_agents": "WAIT",
    "children": "CHILDREN", "list_children": "CHILDREN", "list_agents": "CHILDREN",
}
SPAWN_BODY_KEYS = ("task", "prompt", "description", "body", "content", "text", "subtask", "sub_task")
PATH_KEYS = ("path", "file", "file_path", "filepath", "filename", "dir", "directory", "dir_path", "folder", "target")
QUERY_KEYS = ("query", "pattern", "text", "search", "needle", "regex", "q", "keyword")
COMMAND_KEYS = ("command", "cmd", "script", "shell", "code")
BODY_KEYS = ("content", "contents", "body", "text", "data", "file_text")
OLD_KEYS = ("old_string", "old_str", "old_text", "old", "search", "find", "before", "original")
NEW_KEYS = ("new_string", "new_str", "new_text", "new", "replace", "replacement", "after", "updated")
GLOB_KEYS = ("pattern", "glob", "query", "name", "value")
TODO_KEYS = ("todos", "items", "tasks", "steps", "todo", "list", "plan", "value")
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


TODO_DONE = {"done", "completed", "complete", "finished", "x"}
TODO_DOING = {"in_progress", "in-progress", "doing", "active", "current", "started", "working"}


def _todo_rows(value: object) -> list[str]:
    """Native todo arguments (a list of {content, status} or strings) as TODO item lines."""
    if isinstance(value, str):
        text = value.strip()
        if text.startswith("["):
            try:
                value = _JSON.decode(text)
            except ValueError:
                value = text.splitlines()
        else:
            value = text.splitlines()
    if not isinstance(value, list):
        return []
    rows = []
    for item in value[:TODO_LIMIT]:
        status = ""
        if isinstance(item, dict):
            text = _first_line(
                item.get("content") or item.get("text") or item.get("task") or item.get("step")
                or item.get("title") or item.get("description")
            )
            status = str(item.get("status") or item.get("state") or "").strip().lower()
        else:
            text = _first_line(item)
            match = TODO_LINE.match(text)
            if match:
                rows.append(f"[{match.group(1) or ' '}] {match.group(2)}")
                continue
        if not text:
            continue
        mark = "x" if status in TODO_DONE else ">" if status in TODO_DOING else " "
        rows.append(f"[{mark}] {text}")
    return rows


def command_from(name: object, args: object) -> str:
    command = TOOL_NAMES.get(str(name or "").strip().strip("\"'").lower().replace("-", "_"), "")
    if not command:
        return ""
    if command in {"SESSIONS", "CHILDREN"}:
        return command
    if isinstance(args, str):
        args = {"value": args}
    if not isinstance(args, dict):
        args = {}
    args = {str(key).strip().lower(): value for key, value in args.items()}
    only = next(iter(args.values())) if len(args) == 1 else None
    if command == "WAIT":
        ids = _pick(args, ("ids", "id", "children", "agents", "value"))
        found = re.findall(r"\d+", json.dumps(ids) if isinstance(ids, (list, dict)) else str(ids or ""))
        return ("WAIT " + " ".join(found[:TEAM_MAX])) if found else "WAIT"
    if command == "SPAWN":
        role = _first_line(_pick(args, ("role", "type", "kind", "agent")))
        title = _first_line(_pick(args, ("title", "name", "summary")))
        body = _pick(args, SPAWN_BODY_KEYS)
        if not role or not isinstance(body, str) or not body.strip():
            return ""
        return f"SPAWN {role.split()[0]} {title}".rstrip() + "\n" + body.strip("\n") + "\nENDSPAWN"
    if command == "EDIT":
        safe = tool_path(_first_line(_pick(args, PATH_KEYS)))
        old = next((args[key] for key in OLD_KEYS if key in args and args[key] is not None), None)
        new = next((args[key] for key in NEW_KEYS if key in args and args[key] is not None), None)
        if not safe or not isinstance(old, str) or not old or not isinstance(new, str):
            return ""
        return edit_call(safe, old, new)
    if command == "GLOB":
        pattern = _first_line(_pick(args, GLOB_KEYS))
        folder = _first_line(_pick(args, PATH_KEYS))
        if folder and pattern and folder != pattern:
            pattern = folder.rstrip("/\\") + "/" + pattern
        return "GLOB " + pattern if pattern else ""
    if command == "TODO":
        rows = _todo_rows(_pick(args, TODO_KEYS))
        return "TODO\n" + "\n".join(rows) if rows else ""
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


def openai_tools(tool_calls: object) -> list[tuple[str, str]]:
    calls: list[tuple[str, str]] = []
    if not isinstance(tool_calls, list):
        return calls
    for item in tool_calls[: MAX_BATCH + 1]:
        command = tool_from_object(item)
        if not command:
            continue
        function = item.get("function") if isinstance(item, dict) and isinstance(item.get("function"), dict) else {}
        args = function.get("arguments", {})
        if isinstance(args, str):
            try:
                args = _JSON.decode(args) if args.strip() else {}
            except ValueError:
                pass
        call = {"name": function.get("name") or item.get("name", ""), "arguments": args}
        calls.append((command, "<tool_call>\n" + json.dumps(call, ensure_ascii=False) + "\n</tool_call>"))
    return calls


def native_tools(answer: str) -> list[tuple[str, str]]:
    """Every call in the model's own format, each with the text of only that call."""
    calls: list[tuple[str, str]] = []
    if "<tool_call>" in answer:
        for piece in answer.split("<tool_call>")[1:]:
            segment = "<tool_call>" + piece
            end = segment.find("</tool_call>")
            segment = segment[: end + len("</tool_call>")] if end >= 0 else segment.rstrip() + "\n</tool_call>"
            command = native_tool(segment)
            if command:
                calls.append((command, segment.strip()))
        return calls
    if "<function=" in answer:
        for piece in answer.split("<function=")[1:]:
            segment = "<function=" + piece
            end = segment.find("</function>")
            if end >= 0:
                segment = segment[: end + len("</function>")]
            command = xml_tool(segment)
            if command:
                calls.append((command, segment.strip()))
        return calls
    command = json_tool(answer)
    if command:
        calls.append((command, answer[max(0, answer.find("{")) :].strip()))
    return calls


SINGLE_HEADS = ("LIST ", "GREP ", "GLOB ", "READ ", "MKDIR ", "RUN ", "SESSIONS ")


def team_line(stripped: str) -> bool:
    """A line that starts a team call. It also ends the body of the SPAWN before it."""
    return stripped.startswith(("SPAWN ", "WAIT ")) or stripped in {"WAIT", "CHILDREN"}


def parse_spawn(command: str) -> tuple[str, str, str, bool] | None:
    """SPAWN role title / sub-task lines / ENDSPAWN -> (role, title, body, closed)."""
    lines = command.splitlines()
    if not lines:
        return None
    match = re.match(r"SPAWN\s+(\S+)[ \t]*(.*)$", clean_tool_line(lines[0]))
    if not match:
        return None
    role = match.group(1).strip("`'\"<>[](),:;").lower()
    title = one_line(match.group(2).strip("`'\" -:"), 80)
    body = lines[1:]
    closed = bool(body) and clean_tool_line(body[-1]) == "ENDSPAWN"
    if closed:
        body = body[:-1]
    return role, title, "\n".join(body).strip(), closed


def text_tools(answer: str) -> list[tuple[str, str]]:
    """Every call in the plain line syntax. WRITE, EDIT, TODO and SPAWN carry their body lines."""
    lines = answer.splitlines()
    calls: list[tuple[str, str]] = []
    index = 0
    while index < len(lines):
        stripped = clean_tool_line(lines[index])
        if stripped.startswith("SPAWN "):
            block = [stripped]
            index += 1
            closed = False
            while index < len(lines):
                following = clean_tool_line(lines[index])
                if following == "ENDSPAWN":
                    closed = True
                    index += 1
                    break
                if team_line(following):
                    closed = True  # the next team call starts, so this one was whole
                    break
                block.append(lines[index])
                index += 1
            text = "\n".join(block + (["ENDSPAWN"] if closed else []))
            calls.append((text, text))
            continue
        if team_line(stripped):
            calls.append((stripped, stripped))
            index += 1
            continue
        if stripped.startswith(("WRITE ", "EDIT ")):
            is_write = stripped.startswith("WRITE ")
            block = [stripped]
            index += 1
            while index < len(lines):
                block.append(lines[index])
                index += 1
                end = clean_tool_line(block[-1])
                if (end == "ENDWRITE") if is_write else block[-1].strip().startswith(">>>>>>>"):
                    break
            text = "\n".join(block)
            calls.append((text, text))
            continue
        if stripped in {"TODO", "TODO:"}:
            block = ["TODO"]
            index += 1
            while index < len(lines) and (not lines[index].strip() or TODO_LINE.match(lines[index])):
                if lines[index].strip():
                    block.append(lines[index].strip())
                index += 1
            text = "\n".join(block)
            calls.append((text, text))
            continue
        if stripped == "SESSIONS" or stripped.startswith(SINGLE_HEADS):
            calls.append((stripped, stripped))
        elif stripped == "LIST":
            calls.append(("LIST .", "LIST ."))
        index += 1
    return calls


def pick_tools(answer: str, tool_calls: object = None) -> list[tuple[str, str]]:
    """All tool calls in one reply, in order, as (command, text of only that call)."""
    return openai_tools(tool_calls) or native_tools(answer) or text_tools(answer)


def tool_head(command: str) -> str:
    return command.strip().split(None, 1)[0].upper() if command.strip() else ""


def batch_calls(calls: list[tuple[str, str]]) -> list[tuple[str, str]]:
    """Up to MAX_BATCH read-only calls run together. WRITE, EDIT, MKDIR and RUN run alone.
    A lead's SPAWN calls run together too, so its sub-agents start in parallel."""
    if calls and tool_head(calls[0][0]) == "SPAWN":
        spawns = []
        for call in calls:
            if tool_head(call[0]) != "SPAWN" or len(spawns) >= TEAM_MAX:
                break
            spawns.append(call)
        return spawns
    if not calls or tool_head(calls[0][0]) not in READ_ONLY:
        return calls[:1]
    picked = []
    for call in calls:
        if tool_head(call[0]) not in READ_ONLY or len(picked) >= MAX_BATCH:
            break
        picked.append(call)
    return picked


def pick_tool(answer: str, tool_calls: object = None) -> tuple[str, str]:
    """The first tool call and the text of only that call."""
    calls = pick_tools(answer, tool_calls)
    return calls[0] if calls else ("", "")


def tool_response(text: str, limit: int = TOOL_CHARS) -> str:
    body = clip((text or "").strip(), limit).replace("</tool_response>", "<\\/tool_response>")
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
        if stripped in {"SESSIONS", "TODO"} or stripped.startswith((*SINGLE_HEADS, "WRITE ", "EDIT ")):
            return stripped
        if stripped == "LIST":
            return "LIST ."
    return ""


def fit_context(messages: list[dict], budget: int = CONTEXT_CHARS) -> None:
    """Fit the chat in budget characters.

    messages[0] (system), messages[1] (the task) and a running summary at messages[2] are never
    touched, and the newest call and its result stay whole. First the oldest tool results are
    emptied, then old calls are clipped, and only then the oldest assistant+tool pairs are dropped.
    run_agent folds old turns into the summary (compact) before this; this is the last resort.
    """

    def size() -> int:
        return sum(len(str(item.get("content") or "")) for item in messages)

    keep = 3 if len(messages) > 2 and is_summary(messages[2]) else 2
    removed = "<tool_response>\n[older result removed to save room]\n</tool_response>"
    for index in range(keep, len(messages) - 2):
        if size() <= budget:
            return
        item = messages[index]
        content = str(item.get("content") or "")
        if item.get("role") == "user" and content.startswith("<tool_response>") and content != removed:
            item["content"] = removed
    for index in range(keep, len(messages) - 2):
        if size() <= budget:
            return
        item = messages[index]
        content = str(item.get("content") or "")
        if item.get("role") == "assistant" and len(content) > 300:
            item["content"] = clip(content, 300)
    while size() > budget and len(messages) > keep + 2:
        del messages[keep : keep + 2]


SUMMARY_HEAD = "[summary of earlier turns]"
KEEP_NEWEST = 4  # the newest two turns (call + result each) always stay whole


def is_summary(item: dict) -> bool:
    return item.get("role") == "user" and str(item.get("content") or "").startswith(SUMMARY_HEAD)


def one_line(text: str, limit: int) -> str:
    return " ".join(str(text or "").split())[:limit]


def summary_text(state: TaskState, steps: list[str]) -> str:
    lines = [SUMMARY_HEAD, "Task: " + one_line(state.task, 160)]
    if state.read_names:
        lines.append("Files read: " + ", ".join(state.read_names[-12:]))
    if state.changed:
        lines.append("Files changed: " + ", ".join(state.changed[-12:]))
    if steps:
        lines.append("Steps so far:")
        lines += ["- " + step for step in steps[-10:]]
    if state.failures:
        lines.append("Failed attempts: " + " | ".join(state.failures[-4:]))
    if state.todo:
        lines.append(todo_line(state.todo))
    return clip("\n".join(lines), 2000)


def compact(history: list[dict], budget: int, state: TaskState, reserve: int = 0) -> bool:
    """Like a long session: when the chat is over budget, fold the oldest turns into one running
    summary message (messages[2]). System, task, summary and the newest turns are never dropped.
    It shrinks to about 60% of the budget, so the cached prefix holds for several turns after."""

    def size(items: list[dict]) -> int:
        return sum(len(str(item.get("content") or "")) for item in items) + reserve

    if size(history) <= budget:
        return False
    start = 3 if len(history) > 2 and is_summary(history[2]) else 2
    steps = list(history[2].get("steps") or []) if start == 3 else []
    target = budget * 3 // 5
    fixed = size(history[:2]) + 2000
    cut = start
    while len(history) - cut > KEEP_NEWEST and fixed + size(history[cut:]) - reserve > target:
        cut += 2
    if cut == start:
        return False
    for item in history[start:cut]:
        if item.get("role") == "assistant":
            steps.append(one_line(item.get("note") or item.get("content"), 160))
    history[2:cut] = [{"role": "user", "content": summary_text(state, steps), "steps": steps[-10:]}]
    return True


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


TEAM_STATUS = {"done": "done", "error": "failed", "running": "still running", "queued": "still running (waiting for a worker)"}
TEAM_LEAD_NOTE = (
    "Team mode is on: if this task has independent parts, plan them, SPAWN sub-agents for them in"
    " parallel, WAIT for their reports, verify and integrate. Do a small task yourself."
)


def team_jail(parent_id: int, job_id: int) -> str:
    return f"workspace/team-{int(parent_id)}/{int(job_id)}"


def team_children(parent_id: int, ids: list[int] | None = None) -> list[dict]:
    with connect() as db:
        rows = db.execute(
            "SELECT id, title, status, role, result, error, created, prompt_tokens, completion_tokens "
            "FROM jobs WHERE parent_id=? ORDER BY id",
            (parent_id,),
        ).fetchall()
    found = [dict(row) for row in rows]
    if ids is not None:
        wanted = set(ids)
        found = [row for row in found if row["id"] in wanted]
    return found


def live_workers(db: sqlite3.Connection) -> int:
    cutoff = time.time() - REAP_AFTER
    return int(db.execute("SELECT COUNT(*) AS n FROM workers WHERE heartbeat >= ?", (cutoff,)).fetchone()["n"])


def child_prompt(kind: str, role: str, title: str, body: str, parent_id: int, parent_title: str, job_id: int) -> str:
    """The compact task a sub-agent gets: role, sub-task, workspace rule, report contract."""
    lines = [
        f"Role: {role} sub-agent #{job_id} of lead task #{parent_id} ({one_line(parent_title, 60)}).",
        f"Sub-task: {title}",
        clip(body, 2600),
    ]
    if KIND_BUDGET[kind].write:
        lines.append(
            f"Workspace rule: write files only under {team_jail(parent_id, job_id)}/ (yours alone; other"
            " sub-agents work in parallel). READ anything in the repo. The lead integrates your work."
        )
    else:
        lines.append("This sub-task does not change files.")
    lines.append("Final report, at most 8 lines: what was done, the files, how it was verified.")
    head = f"[kind:{kind}]\n"
    return head + "\n".join(lines)[: 4000 - len(head)]


class Team:
    """Team mode for one job. A lead (depth 0, a code/research/review job in the queue) may SPAWN,
    WAIT and list CHILDREN; a sub-agent (depth 1) may not, and writes only in its own folder."""

    def __init__(
        self, job: dict | None, kind: str, *, sleep=None, clock=None, wait_seconds: float | None = None,
        wall_seconds: float | None = None,
    ) -> None:
        job = job or {}
        self.job_id = int(job["id"]) if job.get("id") is not None else None
        self.parent_id = int(job["parent_id"]) if job.get("parent_id") is not None else None
        self.role = str(job.get("role") or "")
        self.depth = int(job.get("depth") or 0)
        self.pid = int(job.get("pid") or os.getpid())
        self.kind = kind
        self.sleep = sleep or time.sleep
        self.clock = clock or time.time
        self.wait_seconds = float(TEAM_WAIT if wait_seconds is None else wait_seconds)
        self.wall_seconds = float(TEAM_WALL if wall_seconds is None else wall_seconds)
        self.waits = 0
        self.spawned = 0

    @property
    def child(self) -> bool:
        return self.depth >= 1 or self.parent_id is not None

    @property
    def lead(self) -> bool:
        return self.job_id is not None and not self.child and self.kind in TEAM_KINDS

    @property
    def mode(self) -> str:
        return "child" if self.child else "lead" if self.lead else "solo"

    def jail(self) -> str | None:
        if self.child and self.job_id is not None and self.parent_id is not None:
            return team_jail(self.parent_id, self.job_id)
        return None

    def children(self, ids: list[int] | None = None) -> list[dict]:
        return team_children(self.job_id, ids) if self.job_id is not None else []

    def time_left(self, kids: list[dict] | None = None) -> float:
        kids = self.children() if kids is None else kids
        if not kids:
            return self.wall_seconds
        first = min(float(row["created"]) for row in kids)
        return self.wall_seconds - (self.clock() - first)

    def save_tokens(self, usage: dict) -> None:
        if self.job_id is None:
            return
        try:
            with connect() as db:
                db.execute(
                    "UPDATE jobs SET prompt_tokens=?, completion_tokens=?, cached_tokens=? WHERE id=?",
                    (usage["prompt"], usage["completion"], usage["cached"], self.job_id),
                )
        except sqlite3.Error:
            pass

    def set_state(self, state: str) -> None:
        try:
            with connect() as db:
                db.execute("UPDATE workers SET state=?, heartbeat=? WHERE pid=?", (state, time.time(), self.pid))
        except sqlite3.Error:
            pass

    def note(self) -> str:
        """For the lead's task message: team mode, and sub-agents left from an earlier attempt."""
        parts = [TEAM_LEAD_NOTE] if self.role == "lead" else []
        kids = self.children() if self.lead else []
        if kids:
            listed = "; ".join(f"#{row['id']} {row['role']} {row['status']}: {one_line(row['title'], 40)}" for row in kids)
            parts.append(f"Sub-agents already started for this task: {listed}. WAIT or CHILDREN shows them.")
        return "\n".join(parts)

    def tool(self, command: str, head: str, cut: bool) -> str:
        if self.job_id is None:
            return "blocked: SPAWN, WAIT and CHILDREN work only for a task in the queue. Do the work yourself."
        if self.child:
            return f"blocked: a sub-agent cannot use {head} (depth limit 1). Do your sub-task yourself."
        if self.kind not in TEAM_KINDS:
            return "blocked: this task cannot start sub-agents. Answer it directly."
        if head == "SPAWN":
            return self.spawn(command, cut)
        if head == "WAIT":
            return self.wait(command[4:])
        return self.children_text()

    def spawn(self, command: str, cut: bool) -> str:
        parsed = parse_spawn(command)
        if parsed is None:
            return "SPAWN needs: SPAWN role short title, the sub-task on the next lines, then ENDSPAWN"
        role, title, body, closed = parsed
        if cut and not closed:
            return "not spawned: the call was cut off before ENDSPAWN. Send a shorter sub-task."
        if role not in TEAM_ROLES:
            return "refused: the role must be research, code, review or test."
        kind = TEAM_ROLES[role]
        if role not in team_roles_for(self.kind):
            return f"refused: a {self.kind} task cannot change files, so it cannot SPAWN a {role} sub-agent. Use research or review."
        if not body:
            return "refused: put the sub-task text on the lines after SPAWN, then ENDSPAWN."
        title = title or one_line(body, 60)
        now = self.clock()
        with connect() as db:
            db.execute("BEGIN IMMEDIATE")
            parent = db.execute("SELECT title, parent_id FROM jobs WHERE id=?", (self.job_id,)).fetchone()
            if parent is None:
                return "refused: this task is not in the queue."
            if parent["parent_id"] is not None:  # the queue itself also holds the depth limit
                return "blocked: a sub-agent cannot use SPAWN (depth limit 1). Do your sub-task yourself."
            kids = db.execute(
                "SELECT COUNT(*) AS n, MIN(created) AS first FROM jobs WHERE parent_id=?", (self.job_id,)
            ).fetchone()
            if int(kids["n"]) >= TEAM_MAX:
                return f"refused: this task already has {TEAM_MAX} sub-agents (the limit). WAIT for them or do the rest yourself."
            if kids["first"] is not None and now - float(kids["first"]) >= self.wall_seconds:
                return f"refused: the team time limit ({int(self.wall_seconds)}s) is used up. Integrate what you have."
            if live_workers(db) < 2:
                return "refused: fewer than 2 agent workers are running, so a sub-agent could not run in parallel. Do the work yourself."
            cursor = db.execute(
                "INSERT INTO jobs (title, prompt, status, created, updated, parent_id, role, depth) "
                "VALUES (?, '', 'queued', ?, ?, ?, ?, ?)",
                (title[:80], now, now, self.job_id, role, self.depth + 1),
            )
            job_id = int(cursor.lastrowid)
            prompt = child_prompt(kind, role, title, body, self.job_id, str(parent["title"]), job_id)
            db.execute("UPDATE jobs SET prompt=? WHERE id=?", (prompt, job_id))
        self.spawned += 1
        where = f", writes under {team_jail(self.job_id, job_id)}/" if KIND_BUDGET[kind].write else ""
        return f"spawned #{job_id} {role}: {title}{where}. WAIT {job_id} returns its report."

    def wait(self, argument: str) -> str:
        kids = self.children()
        if not kids:
            return "refused: there are no sub-agents yet. SPAWN first, or do the work yourself."
        if self.waits >= TEAM_WAITS:
            return f"refused: WAIT was used {self.waits} times (the limit). Integrate what you have; CHILDREN shows the status."
        asked = [int(item) for item in re.findall(r"\d+", argument or "")]
        known = {row["id"] for row in kids}
        wanted = [item for item in dict.fromkeys(asked) if item in known] if asked else sorted(known)
        if not wanted:
            return "refused: those ids are not sub-agents of this task. CHILDREN lists them."
        self.waits += 1
        left = self.time_left(kids)
        limit = max(0.0, min(self.wait_seconds, left))
        deadline = self.clock() + limit
        self.set_state("team_wait")
        try:
            while True:
                rows = self.children(wanted)
                if all(row["status"] in {"done", "error"} for row in rows) or self.clock() >= deadline:
                    break
                self.sleep(TEAM_POLL)
        finally:
            self.set_state("generating")
        finished = sum(1 for row in rows if row["status"] in {"done", "error"})
        head = f"{finished} of {len(rows)} sub-agents finished."
        if finished < len(rows):
            head += (
                " The team time limit is used up; integrate what you have." if left <= 0
                else f" The rest are still running after {int(limit)}s: WAIT again later or keep working."
            )
        unknown = [item for item in asked if item not in known]
        if unknown:
            head += " Not sub-agents of this task: " + ", ".join(f"#{item}" for item in unknown) + "."
        out = [head]
        for row in rows:
            tokens = int(row["prompt_tokens"] or 0) + int(row["completion_tokens"] or 0)
            out.append(f"#{row['id']} {row['role']} {TEAM_STATUS.get(row['status'], row['status'])} ({tokens} tokens): {one_line(row['title'], 60)}")
            report = str(row["result"] or row["error"] or "").strip()
            if row["status"] in {"done", "error"} and report:
                if len(report) > TEAM_REPORT:
                    report = "[...] " + report[-(TEAM_REPORT - 6) :]
                out.append(report)
        return "\n".join(out)

    def children_text(self) -> str:
        kids = self.children()
        if not kids:
            return f"no sub-agents yet (limit {TEAM_MAX})"
        lines = [
            f"#{row['id']} {row['role']} {TEAM_STATUS.get(row['status'], row['status'])}: {one_line(row['title'], 60)}"
            for row in kids
        ]
        lines.append(f"{len(kids)} of {TEAM_MAX} sub-agents used, WAIT used {self.waits} of {TEAM_WAITS}")
        return "\n".join(lines)


def run_tool(
    command: str, source: str, cut: bool, *, kind: str = "code", state: TaskState | None = None,
    team: Team | None = None,
) -> tuple[str, list[str]]:
    # Gates by task kind come first. They only add blocks; every check below still runs.
    budget = KIND_BUDGET.get(kind) or KIND_BUDGET["quick"]
    head = tool_head(command)
    if head in {"WRITE", "MKDIR", "EDIT"} and not budget.write:
        return "blocked: this task cannot change files. Use LIST, GREP or READ, or answer in plain text.", []
    if head == "RUN" and not budget.run:
        return "blocked: this task cannot RUN commands. Use LIST, GREP or READ, or answer in plain text.", []
    if head in TEAM_HEADS:
        if kind not in TEAM_KINDS:
            return "blocked: this task cannot start sub-agents. Answer it directly.", []
        if team is None:
            return "blocked: SPAWN, WAIT and CHILDREN work only for a task in the queue. Do the work yourself.", []
        return team.tool(command, head, cut), []
    if command.startswith("WRITE "):
        wrote = apply_actions(source, require_end=cut)
        if state is not None:
            for item in wrote:
                found = inside(item[6:]) if item.startswith("wrote ") else None
                if found:
                    state.change(found[0], found[1])
        return "\n".join(wrote) or "nothing written", wrote
    if command.startswith("EDIT "):
        parsed = parse_edit(command if "\n" in command else source)
        if cut and (parsed is None or not parsed[3]):
            name = parsed[0] if parsed else tool_path(command[5:].split("\n", 1)[0])
            return f"not edited {name}: the call was cut off before {EDIT_CLOSE}. Send a smaller EDIT.", []
        if parsed is None:
            return f"EDIT needs: EDIT path, {EDIT_OPEN}, the exact old lines, {EDIT_MID}, the new lines, {EDIT_CLOSE}", []
        if not parsed[0]:
            return "blocked path", []
        return tool_edit(parsed[0], parsed[1], parsed[2], state)
    if command.startswith("MKDIR "):
        made = apply_actions(command)
        return "\n".join(made) or "mkdir", made
    if command.startswith("LIST "):
        return tool_list(command[5:]), []
    if command.startswith("GREP "):
        return tool_grep(command[5:]), []
    if command.startswith("GLOB "):
        return tool_glob(command[5:]), []
    if command.startswith("READ "):
        result, raw = _read_call(command[5:])
        found = inside(raw) if state is not None and result not in READ_FAILED else None
        if found:
            state.saw(found[0], found[1])
        return result, []
    if head == "TODO":
        return tool_todo(command, state), []
    if command.startswith("RUN "):
        return tool_run(command[4:]), []
    if command.startswith("SESSIONS"):
        return "\n".join(recent_sessions(8)) or "no saved session yet", []
    return "unknown tool", []


def ask_fitted(messages: list[dict], think: bool = False, max_tokens: int = 1200) -> dict:
    budget = CONTEXT_CHARS
    for attempt in range(2):
        fit_context(messages, budget)
        try:
            return chat(messages, max_tokens=max_tokens, think=think, stop=STOPS)
        except ContextFull:
            if attempt:
                raise RuntimeError("The task is too long for Neuro's memory. Make it shorter.") from None
            budget //= 2
            if len(messages) > 2:
                messages[-1]["content"] = clip(str(messages[-1]["content"]), 1200)
    raise RuntimeError("The task is too long for Neuro's memory. Make it shorter.")


# On the last turn these still run once. Their result goes into the answer instead of being lost.
FINAL_TOOLS = frozenset({"RUN", "READ", "GREP", "GLOB", "LIST", "SESSIONS", "WAIT", "CHILDREN"})
# A RUN that checks written work: a compiler, a test runner, or running the program.
VERIFY_RUN = re.compile(
    r"\b(?:py_compile|compileall|pytest|unittest|tests?|tsc|mypy|ruff|flake8|pylint|lint|build|check|cargo|go"
    r"|npm|npx|node|deno|python3?|py|make|dotnet|javac|java|gcc|clang|rustc|bash|sh|powershell|pwsh)\b",
    re.IGNORECASE,
)
VERIFY_NUDGE = (
    "Not done yet: you changed a file but did not RUN a compile or a test. "
    "Send one RUN that checks it, for example RUN python -m py_compile path, "
    "or answer in plain text with the reason no check applies."
)
LOOP_NUDGE = (
    "You sent the same call as last turn, so it did not run again; its result is above. "
    "Change approach: a different call, a smaller step, or the final answer."
)
BATCH_NOTE = (
    "Only the first {n} call(s) ran. Up to 3 of LIST GLOB GREP READ SESSIONS TODO run together; "
    "EDIT, WRITE, MKDIR and RUN run alone, one per turn."
)
FAIL_STARTS = (
    "blocked", "refused", "error:", "could not", "not written", "not edited", "unknown tool", "timed out",
    "missing", "unreadable", "too large", "EDIT needs", "TODO needs", "need a ",
)
EXIT_FAIL = re.compile(r"(?:^|\n)exit [1-9]\d*\s*$")
# Patterns of real credentials. A trajectory that holds one is never saved.
SECRET_VALUE = re.compile(
    r"sk-[A-Za-z0-9_-]{20,}|ghp_[A-Za-z0-9]{30,}|github_pat_\w{30,}|hf_[A-Za-z0-9]{30,}|AKIA[0-9A-Z]{16}"
    r"|xox[abp]-[A-Za-z0-9-]{20,}|-----BEGIN [A-Z ]*PRIVATE KEY-----"
)


def tool_failed(result: str) -> bool:
    text = (result or "").strip()
    return text.startswith(FAIL_STARTS) or bool(EXIT_FAIL.search(text))


def touches_secret(command: str) -> bool:
    """A call whose path, pattern or command names a secret file (cluster.token, .env, *.db ...)."""
    head = (command or "").split("\n", 1)[0].lower()
    return any(part and secret_name(part) for part in re.split(r"[\s/\\'\"`<>|;,&()=:*?\[\]{}]+", head))


def secret_values() -> list[str]:
    """The live secrets on this computer, only to make sure a saved trajectory never holds one.
    They are compared in memory and never written or shown."""
    values: list[str] = []
    for base in {ROOT.resolve(), HOME.resolve()}:
        for name in ("cluster.token", "token.txt"):
            try:
                text = (base / name).read_text(encoding="utf-8", errors="replace")
            except OSError:
                continue
            values += [line.strip() for line in text.splitlines() if len(line.strip()) >= 8]
        for name in (".env", ".env.local"):
            try:
                text = (base / name).read_text(encoding="utf-8", errors="replace")
            except OSError:
                continue
            for line in text.splitlines():
                if "=" in line and not line.lstrip().startswith("#"):
                    value = line.split("=", 1)[1].strip().strip("'\"")
                    if len(value) >= 8:
                        values.append(value)
    return values


def has_secret(text: str, values: list[str]) -> bool:
    return bool(SECRET_VALUE.search(text or "")) or any(value in (text or "") for value in values)


def compose(logs: list[str], tail: str, limit: int = 4000) -> str:
    """The job result: the step log, then the answer. The answer is kept whole; old steps go first."""
    tail = (tail or "").strip()[: limit - 300]
    room = limit - len(tail) - 60
    kept: list[str] = []
    for line in reversed(logs):
        if len(line) + 1 > room:
            break
        kept.append(line)
        room -= len(line) + 1
    kept.reverse()
    if len(kept) < len(logs):
        kept.insert(0, f"[{len(logs) - len(kept)} earlier steps not shown]")
    return "\n".join([*kept, tail]).strip()


def _say_line(line: str) -> None:
    print(line, flush=True)


class AgentRun:
    """One job, run the way a long coding session runs: plan, look, change, check, report."""

    def __init__(
        self, prompt: str, ask, say, log_dir: Path | None, context_chars: int | None,
        job: dict | None = None, team_options: dict | None = None,
    ) -> None:
        self.ask = ask or ask_fitted
        self.say = say or _say_line
        self.log_dir = log_dir or LOG_DIR
        self.context = int(context_chars or CONTEXT_CHARS)
        self.started = time.time()
        kind, prompt = job_kind(prompt)
        self.deep, prompt = deep_request(prompt)
        self.kind = kind
        self.budget = KIND_BUDGET[kind]
        self.max_turns = self.budget.max_turns
        self.team = Team(job, kind, **(team_options or {}))
        self.state = TaskState(prompt)
        self.secrets = secret_values() if kind == "code" else []
        # Stable first (cached by the server for every job of this kind), the task and live files after.
        note = self.team.note() if self.team.lead else ""
        first = "Task:\n" + prompt + ("\n\n" + note if note else "") + "\n\n" + workspace_note()
        self.history: list[dict] = [
            {"role": "system", "content": agent_system(kind, self.team.mode)},
            {"role": "user", "content": first},
        ]
        # The full conversation, never compacted. Saved only for verified code tasks.
        self.transcript: list[dict] = [dict(item) for item in self.history]
        self.logs: list[str] = []
        self.usage = {"prompt": 0, "completion": 0, "cached": 0}
        self.turns = 0
        self.think_turns = 0
        self.wrote = self.verified = self.passed = False

    def view(self, suffix: list[str]) -> list[dict]:
        """What the model sees this turn. The volatile suffix (TODO, last-step note) rides on the
        newest message only and is never stored in history."""
        messages = [{"role": item["role"], "content": str(item["content"])} for item in self.history]
        extra = "\n\n".join(part for part in suffix if part)
        if extra and len(messages) > 1:
            messages[-1]["content"] += "\n\n" + extra
        self.transcript[-1]["content"] = messages[-1]["content"]
        return messages

    def add_turn(self, call: str, response: str, note: str) -> None:
        self.history.append({"role": "assistant", "content": clip(call, 1500), "note": note})
        self.history.append({"role": "user", "content": response})
        self.transcript.append({"role": "assistant", "content": call})
        self.transcript.append({"role": "user", "content": response})

    def call_model(self, think: bool, tokens: int, suffix: list[str]) -> dict:
        compact(self.history, self.context, self.state, reserve=sum(len(part) + 2 for part in suffix))
        reply = self.ask(self.view(suffix), think, tokens)
        self.turns += 1
        self.think_turns += 1 if think else 0
        used = reply.get("usage") if isinstance(reply.get("usage"), dict) else {}
        for key in self.usage:
            self.usage[key] += _count(used.get(key))
        self.team.save_tokens(self.usage)  # the live tree shows tokens per agent
        return reply

    def track(self, command: str, result: str) -> None:
        if touches_secret(command) or has_secret(result, self.secrets):
            self.state.secret = True

    def finish(self, text: str, outcome: str, answer: str = "") -> str:
        seconds = round(time.time() - self.started, 1)
        row = {
            "t": time.strftime("%Y-%m-%d %H:%M:%S", time.localtime()),
            "kind": self.kind,
            "outcome": outcome,
            "turns": self.turns,
            "think_turns": self.think_turns,
            "prompt_tokens": self.usage["prompt"],
            "completion_tokens": self.usage["completion"],
            "cached_tokens": self.usage["cached"],
            "seconds": seconds,
            "changed": len(self.state.changed),
            "verified": bool(self.verified and self.passed),
        }
        if self.team.job_id is not None:
            # "tree" is the lead's id for the lead and every sub-agent, so tokens add up per team.
            row["job_id"] = self.team.job_id
            row["tree"] = self.team.parent_id if self.team.parent_id is not None else self.team.job_id
            row["role"] = self.team.role or ("lead" if self.team.spawned else "")
            row["depth"] = self.team.depth
            row["spawned"] = self.team.spawned
        try:
            append_capped(self.log_dir / "usage.jsonl", json.dumps(row), limit=2_000_000, keep=5000)
        except OSError:
            pass
        self.say(
            f"usage {self.kind}: {self.turns} turns ({self.think_turns} thinking), prompt {self.usage['prompt']}"
            f" tokens ({self.usage['cached']} cached), completion {self.usage['completion']}, {seconds}s, {outcome}"
        )
        if (
            self.kind == "code" and outcome == "answered" and answer and self.wrote and self.verified
            and self.passed and not self.state.secret
        ):
            self.save_trajectory(answer)
        return text

    def save_trajectory(self, answer: str) -> bool:
        messages = [{"role": item["role"], "content": item["content"]} for item in self.transcript]
        messages.append({"role": "assistant", "content": answer})
        row = {
            "t": time.strftime("%Y-%m-%d %H:%M:%S", time.localtime()),
            "kind": self.kind,
            "turns": self.turns,
            "changed": self.state.changed,
            "messages": messages,
        }
        line = json.dumps(row, ensure_ascii=False)
        if self.state.secret or has_secret(line, self.secrets):
            return False
        try:
            append_capped(self.log_dir / "trajectories.jsonl", line, limit=50_000_000, keep=2000)
        except OSError:
            return False
        return True

    def run(self) -> str:
        jail = self.team.jail()
        if jail is None:
            return self._run()
        found = inside(jail)
        if found is None:
            raise RuntimeError("the sub-agent folder is not allowed")
        if self.budget.write:
            found[1].mkdir(parents=True, exist_ok=True)
        token = _JAIL.set((found[0], found[1].resolve()))
        try:
            return self._run()
        finally:
            _JAIL.reset(token)

    def _run(self) -> str:
        budget, state, logs = self.budget, self.state, self.logs
        fixes = 0
        nudged = False
        after_failure = False
        writing = True  # the first turn of a code task may WRITE at once
        no_think = False
        retry_tokens = 0
        last_sig = ""
        repeats = 0
        turn = -1
        while True:
            turn += 1
            if self.team.spawned and self.max_turns == budget.max_turns:
                self.max_turns += TEAM_BONUS_TURNS  # once: room to WAIT, verify and integrate
            if turn >= self.max_turns:
                break
            final_turn = turn == self.max_turns - 1
            think, tokens = turn_budget(
                self.kind, turn=turn, deep=self.deep, after_failure=after_failure, writing=writing,
                final_turn=final_turn,
            )
            if no_think:
                think, no_think = False, False
            if retry_tokens:
                tokens, retry_tokens = max(tokens, retry_tokens), 0
            suffix = [todo_line(state.todo)]
            if final_turn and len(self.history) > 2:
                suffix.append(LAST_STEP)
            reply = self.call_model(think, tokens, suffix)
            # Only the visible answer is used. Thinking is never stored or sent back.
            answer = visible_answer(reply.get("content") or "")
            cut = reply.get("finish") == "length"
            calls = pick_tools(answer, reply.get("tool_calls"))
            after_failure = writing = False
            if not calls:
                if not answer and cut and think and not final_turn and fixes < budget.fixes:
                    # Every token went to thinking. Ask again with thinking off.
                    fixes += 1
                    no_think = True
                    continue
                if looks_like_call(answer) and not final_turn and fixes < budget.fixes:
                    fixes += 1
                    if cut:
                        note = "The tool call was cut off at the length limit, so nothing ran. Send a shorter call."
                    else:
                        note = "The tool call could not be read, so nothing ran. Send one complete tool call or a plain answer."
                    self.add_turn(clip(answer, 600), tool_response(note), "unreadable tool call")
                    after_failure, writing = True, cut
                    continue
                if cut and answer and tokens < budget.max_tokens and not final_turn and fixes < budget.fixes:
                    # The answer hit the small tool-turn cap. Ask again with room for a whole answer.
                    fixes += 1
                    retry_tokens = budget.max_tokens
                    continue
                if budget.verify and self.wrote and not self.verified and not nudged and not final_turn:
                    # One nudge only. The next plain answer is accepted as it is.
                    nudged = True
                    self.add_turn(clip(plain_answer(answer), 600), tool_response(VERIFY_NUDGE), "answered before a check")
                    continue
                final = plain_answer(answer)
                return self.finish(compose(logs, final), "answered", final)
            batch = batch_calls(calls)
            heads = [tool_head(command) for command, _ in batch]
            if final_turn and any(head not in FINAL_TOOLS for head in heads):
                break
            signature = "\n".join(command for command, _ in batch)
            if heads == ["WAIT"]:
                # WAIT again is how a lead polls; TEAM_WAITS bounds it instead of the loop breaker.
                repeats, last_sig = 0, ""
            else:
                repeats = repeats + 1 if signature == last_sig else 1
                last_sig = signature
            call_text = "\n".join(text for _, text in batch)
            first_line = batch[0][0].splitlines()[0][:120]
            if repeats >= 3:
                note = (
                    f"Stopped: the same call ({first_line}) came three times in a row with no progress. "
                    "The task is not finished."
                )
                return self.finish(compose(logs, note), "loop")
            if repeats == 2:
                self.add_turn(call_text, tool_response(LOOP_NUDGE), "repeated " + first_line)
                after_failure = True
                continue
            results: list[str] = []
            limits: list[int] = []
            notes: list[str] = []
            for command, _ in batch:
                head = tool_head(command)
                self.say("tool " + command.splitlines()[0][:120])
                source = command if "\n" in command else answer
                try:
                    result, made = run_tool(command, source, cut, kind=self.kind, state=state, team=self.team)
                except Exception as exc:  # a tool failure is a result for the model, not a dead job
                    result, made = f"error: {exc}", []
                self.track(command, result)
                line = command.splitlines()[0] + " -> " + (result.splitlines()[0] if result else "empty")[:140]
                if made:
                    logs.extend(made)
                else:
                    logs.append(line)
                notes.append(one_line(line, 160))
                if any(item.startswith(("wrote ", "edited ")) for item in made):
                    self.wrote, self.verified, self.passed = True, False, False
                elif head == "RUN" and self.wrote and not result.startswith("blocked") and VERIFY_RUN.search(command[4:]):
                    self.verified, self.passed = True, not tool_failed(result)
                if tool_failed(result):
                    after_failure = True
                    tail = result.strip().splitlines()[-1] if result.strip() else ""
                    state.failures.append(one_line(command.splitlines()[0] + " -> " + tail, 160))
                if head in {"READ", "EDIT"} or (head == "RUN" and tool_failed(result)) or result.startswith(("not written", "not edited")):
                    writing = True  # the next turn may well be an EDIT or a WRITE
                results.append(result)
                # Sub-agent reports are each capped already; one WAIT may carry all of them.
                limits.append(TEAM_RESULT_CHARS if head == "WAIT" else TOOL_CHARS)
            if final_turn:
                return self.finish(compose(logs, "last tool result:\n" + clip("\n".join(results), 1500)), "last_turn")
            response = "\n".join(tool_response(result, limit) for result, limit in zip(results, limits))
            if len(calls) > len(batch):
                response += "\n" + tool_response(BATCH_NOTE.format(n=len(batch)))
            self.add_turn(call_text, response, "; ".join(notes))
        return self.finish(compose(logs, "[stopped: out of turns before a final answer]"), "out_of_turns")


def run_agent(
    prompt: str, *, ask=None, say=None, log_dir: Path | None = None, context_chars: int | None = None,
    job: dict | None = None, team_options: dict | None = None,
) -> str:
    """One job. ask(messages, think, max_tokens) is the model call; drills pass a fake one.
    job is the queue row (id, parent_id, role, depth, pid); without it there is no team mode."""
    job = AgentRun(prompt, ask, say, log_dir, context_chars, job, team_options)
    try:
        return job.run()
    except ModelWait:
        raise  # the job goes back to the queue; it did not finish
    except Exception as exc:
        job.finish("", "error " + type(exc).__name__)
        raise


def worker_main() -> None:
    pid = os.getpid()
    print(f"Neuro Code {pid} on one shared model, {SLOTS} generate at once", flush=True)
    idle = 0.8
    wait = 5.0
    said = ""
    state = ""
    beat_at = 0.0
    while not stop_requested(STOP_PATH):
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
                result = run_agent(str(job["prompt"]), job={**job, "pid": pid})
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
                creationflags=getattr(subprocess, "CREATE_NO_WINDOW", 0),
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


def tree_rows(job_id: int) -> list[dict]:
    """The lead and its sub-agents. A sub-agent's id shows its whole tree."""
    with connect() as db:
        row = db.execute("SELECT parent_id FROM jobs WHERE id=?", (job_id,)).fetchone()
        if row is None:
            return []
        root = int(row["parent_id"]) if row["parent_id"] is not None else int(job_id)
        rows = db.execute(
            "SELECT id, parent_id, title, status, role, prompt_tokens, completion_tokens, cached_tokens, "
            "created, updated, result, error FROM jobs WHERE id=? OR parent_id=? "
            "ORDER BY CASE WHEN parent_id IS NULL THEN 0 ELSE 1 END, id",
            (root, root),
        ).fetchall()
    return [dict(item) for item in rows]


def tree_finished(rows: list[dict]) -> bool:
    return bool(rows) and all(row["status"] in {"done", "error"} for row in rows)


def tree_text(job_id: int, full: bool = False) -> str:
    rows = tree_rows(job_id)
    if not rows:
        return f"no task #{job_id}"
    lines = []
    total = cached = 0
    for row in rows:
        tokens = int(row["prompt_tokens"] or 0) + int(row["completion_tokens"] or 0)
        total += tokens
        cached += int(row["cached_tokens"] or 0)
        child = row["parent_id"] is not None
        role = str(row["role"] or ("" if child else "lead"))
        status = "failed" if row["status"] == "error" else str(row["status"])
        seconds = max(0.0, float(row["updated"]) - float(row["created"]))
        lines.append(
            f"{'  +- ' if child else ''}#{row['id']} {role:<8} {status:<8} {tokens:>7} tok {seconds:>6.0f}s  "
            + one_line(row["title"], 60)
        )
        if full:
            text = str(row["result"] or row["error"] or "").strip()
            for part in clip(text, 600).splitlines()[-8:]:
                lines.append(("      " if child else "    ") + part[:160])
    lead = rows[0]
    lines.append(
        f"team #{lead['id']}: {len(rows) - 1} sub-agents, {total} tokens ({cached} cached), "
        + ("finished" if tree_finished(rows) else "working")
    )
    return "\n".join(lines)


def team_main(task: str) -> None:
    """agents.py team "<task>": queue a lead task and show its tree until the whole team is done."""
    task = (task or "").strip()
    if not task:
        raise SystemExit('use: agents.py team "<task>"')
    job_id = enqueue(task[:70], task, role="lead")
    with connect() as db:
        row = db.execute("SELECT prompt FROM jobs WHERE id=?", (job_id,)).fetchone()
        workers = live_workers(db)
    kind = job_kind(str(row["prompt"]))[0] if row else "?"
    print(f"team task #{job_id} queued ({kind})", flush=True)
    if kind not in TEAM_KINDS:
        print("this is a quick task, so it runs alone without sub-agents", flush=True)
    if workers < 2:
        print(
            f"note: {workers} agent worker(s) running. Sub-agents need 2 or more; start them with"
            " 'agents.py agents' or from the mesh page. This window only watches.",
            flush=True,
        )
    shown = ""
    live = sys.stdout.isatty()
    try:
        while True:
            rows = tree_rows(job_id)
            text = tree_text(job_id)
            if text != shown:
                sys.stdout.write(("\x1b[H\x1b[J" if live else "") + text + "\n" + ("" if live else "\n"))
                sys.stdout.flush()
                shown = text
            if tree_finished(rows):
                break
            time.sleep(1.5)
    except KeyboardInterrupt:
        print(f"\nstopped watching. The team keeps running; 'agents.py tree {job_id}' shows it later.", flush=True)
        return
    print("\n" + tree_text(job_id, full=True), flush=True)


def tree_main(raw: str) -> None:
    if not raw.strip().lstrip("#").isdigit():
        raise SystemExit("use: agents.py tree <id>")
    print(tree_text(int(raw.strip().lstrip("#")), full=True), flush=True)


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
    lines.append(row("  LIST  GLOB  GREP  READ  EDIT  WRITE  RUN  TODO"))
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


ROUTER_CASES = (
    ("fix agents.py enqueue", "code"),
    ("add a retry to chat()", "code"),
    ("write a script that prints hello", "code"),
    ("refactor the queue", "code"),
    ("תקן את הבאג בקובץ agents.py", "code"),
    ("תוסיף בדיקה לפונקציה", "code"),
    ("כתוב סקריפט שמדפיס שלום", "code"),
    ("review agents.py for security problems", "review"),
    ("find bugs in tool_run", "review"),
    ("סקירה של הקוד ב-mesh.py", "review"),
    ("בדוק את הקוד של התור", "review"),
    ("where is cluster.token read?", "research"),
    ("איפה מוגדר SLOTS?", "research"),
    ("הסבר איך עובד התור", "research"),
    ("explain how claim works", "research"),
    ("do not change any files, just explain the queue", "research"),
    ("what is in the file list", "research"),
    ("מה יש בקובץ הזה", "research"),
    ("", "research"),
    ("say hello", "quick"),
    ("prefix the title with a star", "quick"),
    ("what is the address format", "quick"),
    ("קודם תגיד שלום", "quick"),
    ("word " * 60, "research"),
)


def _scripted(replies: list):
    """A fake model for drills: returns replies in order and records each call.
    A reply is the text, or a dict with content, tool_calls, finish and usage."""
    seen: list[dict] = []

    def ask(messages: list[dict], think: bool, max_tokens: int) -> dict:
        seen.append(
            {
                "think": think,
                "max_tokens": max_tokens,
                "system": messages[0]["content"],
                "task": messages[1]["content"],
                "last": str(messages[-1]["content"]),
                "count": len(messages),
                "messages": [dict(item) for item in messages],
            }
        )
        item = replies[min(len(seen) - 1, len(replies) - 1)]
        if isinstance(item, dict):
            return {
                "content": item.get("content", ""),
                "tool_calls": item.get("tool_calls", []),
                "finish": item.get("finish", "stop"),
                "usage": item.get("usage", {}),
            }
        return {"content": item, "tool_calls": [], "finish": "stop"}

    return ask, seen


def _fake_opener(codes: list[int]):
    import io

    calls: list[int] = []

    def opener(request, timeout=0):
        code = codes[min(len(calls), len(codes) - 1)]
        calls.append(code)
        if code == 200:
            return io.BytesIO(b'{"choices":[{"message":{"content":"ok"},"finish_reason":"stop"}]}')
        raise urllib.error.HTTPError(CHAT_URL, code, "drill", {}, io.BytesIO(b"{}"))

    return opener, calls


def router_drill(round_file: str, round_text: str) -> dict:
    """Router, kind gates, verify nudge, final-turn tool, retries and fit_context. No model, no DB."""
    pid = os.getpid()
    quiet = lambda _line: None  # noqa: E731
    checks: dict[str, bool] = {}
    drill_logs = LOG_DIR / "drill" / f"logs-{pid}"

    def drill_agent(prompt: str, **options) -> str:
        return run_agent(prompt, log_dir=drill_logs, **options)

    missed = [(text, want, classify_task(text)) for text, want in ROUTER_CASES if classify_task(text) != want]
    checks["router"] = not missed
    checks["router_kinds"] = set(KIND_BUDGET) == {"quick", "research", "review", "code"} and all(
        classify_task(text) in KIND_BUDGET for text, _ in ROUTER_CASES
    )
    stamped = stamp_kind("fix agents.py enqueue")
    checks["stamp"] = (
        stamped == "[kind:code]\nfix agents.py enqueue"
        and job_kind(stamped) == ("code", "fix agents.py enqueue")
        and stamp_kind(stamped) == stamped
        and job_kind("[kind:bogus]\nsay hello") == ("quick", "say hello")
        and job_kind("[kind:quick]\n/deep say hi") == ("quick", "/deep say hi")
        and job_kind("say hello") == ("quick", "say hello")
    )
    # Per-kind tool help lists only the tools that kind may use, in the parser's exact syntax.
    checks["kind_help"] = (
        "WRITE" not in KIND_TOOLS["quick"] and "RUN " not in KIND_TOOLS["quick"]
        and "MKDIR" not in KIND_TOOLS["research"] and "WRITE" not in KIND_TOOLS["review"]
        and "RUN command" in KIND_TOOLS["research"]
        and "WRITE relative/path\nthe file contents\nENDWRITE" in KIND_TOOLS["code"]
        and "READ relative/path 120-180" in KIND_TOOLS["quick"]
        and all(first_tool(line) for line in ("LIST relative/dir", "GREP text", "READ relative/path", "SESSIONS", "MKDIR relative/path", "RUN command"))
        and "EDIT relative/path" in KIND_TOOLS["code"] and "EDIT" not in KIND_TOOLS["research"]
        and "EDIT" not in KIND_TOOLS["quick"] and "GLOB **/*.py" in KIND_TOOLS["quick"]
        and "TODO" not in KIND_TOOLS["quick"] and "TODO\n[x]" in KIND_TOOLS["code"]
        and all(first_tool(line) for line in ("GLOB **/*.py", "EDIT a.py", "TODO"))
        and "at most 8 lines" in KIND_TOOLS["code"] and len(KIND_TOOLS["code"]) < 2400
    )
    # Gates by kind. They add to the safety checks and never replace them.
    gate_file = f"workspace/agent-check/drill-gate-{pid}.txt"
    gate_dir = f"workspace/agent-check/drill-gate-dir-{pid}"
    write_gate = run_tool(f"WRITE {gate_file}\nnope\nENDWRITE", f"WRITE {gate_file}\nnope\nENDWRITE", False, kind="quick")
    mkdir_gate = run_tool(f"MKDIR {gate_dir}", f"MKDIR {gate_dir}", False, kind="review")
    checks["gate_write"] = write_gate[0].startswith("blocked") and write_gate[1] == [] and safe_read(gate_file) == "missing"
    checks["gate_mkdir"] = mkdir_gate[0].startswith("blocked") and repo_dir(gate_dir) is None
    checks["gate_run_quick"] = run_tool("RUN echo gate-no", "RUN echo gate-no", False, kind="quick")[0].startswith("blocked")
    checks["gate_unknown"] = run_tool("RUN echo gate-no", "RUN echo gate-no", False, kind="bogus")[0].startswith("blocked")
    checks["gate_run_research"] = "gate-ok" in run_tool("RUN echo gate-ok", "RUN echo gate-ok", False, kind="research")[0]
    checks["gate_run_safety"] = (
        run_tool("RUN rm -rf /", "RUN rm -rf /", False, kind="code")[0] == "blocked"
        and run_tool("RUN taskkill /IM python.exe /F", "", False, kind="code")[0] == "blocked"
        and run_tool("RUN type cluster.token", "", False, kind="research")[0] == "blocked"
    )
    ok_file = f"workspace/agent-check/drill-gate-ok-{pid}.txt"
    code_write = run_tool(f"WRITE {ok_file}\nyes\nENDWRITE", f"WRITE {ok_file}\nyes\nENDWRITE", False, kind="code")
    checks["gate_write_code"] = code_write[1] == [f"wrote {ok_file}"] and "yes" in safe_read(ok_file)
    checks["gate_secret_code"] = run_tool("WRITE cluster.token\nx\nENDWRITE", "WRITE cluster.token\nx\nENDWRITE", False, kind="code")[0].startswith("blocked")
    # Budgets reach the model call, and the system prompt stays the same across jobs of a kind.
    ask, seen = _scripted(["hello"])
    quick_out = drill_agent("[kind:quick]\nsay hello", ask=ask, say=quiet)
    ask_deep, seen_deep = _scripted(["hi"])
    drill_agent("[kind:quick]\n/deep say hi", ask=ask_deep, say=quiet)
    ask_a, seen_a = _scripted(["fine"])
    ask_b, seen_b = _scripted(["fine"])
    drill_agent("[kind:code]\nfix the first thing", ask=ask_a, say=quiet)
    drill_agent("[kind:code]\nfix another thing", ask=ask_b, say=quiet)
    checks["budget"] = (
        quick_out == "hello"
        and seen[0]["think"] is False and seen[0]["max_tokens"] == KIND_BUDGET["quick"].max_tokens
        and seen_deep[0]["think"] is True and seen_deep[0]["max_tokens"] == DEEP_TOKENS
        and seen_a[0]["think"] is KIND_BUDGET["code"].think and seen_a[0]["max_tokens"] == KIND_BUDGET["code"].max_tokens
    )
    checks["cache_layout"] = (
        seen_a[0]["system"] == seen_b[0]["system"] == agent_system("code")
        and "fix the first thing" not in seen_a[0]["system"]
        and seen_a[0]["task"].startswith("Task:\nfix the first thing\n\nRepo files:")
        and "[kind:" not in seen_a[0]["task"]
        and seen[0]["system"] == agent_system("quick") != agent_system("code")
    )
    # Verify nudge: WRITE, then "done" with no RUN, gets exactly one nudge.
    verify_file = f"workspace/agent-check/drill-verify-{pid}.txt"
    write_call = f"WRITE {verify_file}\nok\nENDWRITE"
    ask, seen = _scripted([write_call, "Done.", "RUN echo check verify-ok", "Done and checked."])
    out = drill_agent("[kind:code]\nmake a drill file", ask=ask, say=quiet)
    checks["verify_nudge"] = (
        len(seen) == 4 and "did not RUN a compile" in seen[2]["last"]
        and out.endswith("Done and checked.") and f"wrote {verify_file}" in out
    )
    ask, seen = _scripted([write_call, "Done.", "Still done.", "never asked"])
    out = drill_agent("[kind:code]\nmake a drill file", ask=ask, say=quiet)
    checks["verify_once"] = len(seen) == 3 and out.endswith("Still done.")
    ask, seen = _scripted([write_call, "RUN echo check verify-ok", "Done."])
    out = drill_agent("[kind:code]\nmake a drill file", ask=ask, say=quiet)
    checks["verify_skip"] = len(seen) == 3 and out.endswith("Done.") and all("did not RUN" not in item["last"] for item in seen)
    ask, seen = _scripted([write_call, "Done."])
    drill_agent("[kind:research]\nlook at a drill file", ask=ask, say=quiet)
    checks["verify_code_only"] = len(seen) == 2
    # A gated WRITE inside a job is blocked and reported to the model.
    gated_file = f"workspace/agent-check/drill-gate-job-{pid}.txt"
    ask, seen = _scripted([f"WRITE {gated_file}\nno\nENDWRITE", "ok"])
    out = drill_agent("[kind:research]\nlook around", ask=ask, say=quiet)
    checks["gate_job"] = safe_read(gated_file) == "missing" and "cannot change files" in seen[1]["last"] and out.endswith("ok")
    # Final turn: a cheap read still runs once and its result is kept.
    ask, seen = _scripted(["LIST agent-logs/drill", "LIST agent-logs", f"READ {round_file}"])
    out = drill_agent("[kind:quick]\nsay hello", ask=ask, say=quiet)
    checks["final_tool"] = len(seen) == 3 and LAST_STEP in seen[2]["last"] and "last tool result:" in out and round_text in out
    turns = KIND_BUDGET["research"].max_turns
    # Alternating calls are not a loop; only the same call back to back is.
    ask, seen = _scripted([("LIST agent-logs/drill", "LIST agent-logs")[n % 2] for n in range(turns - 1)] + ["RUN rm -rf /"])
    out = drill_agent("[kind:research]\nsay hello", ask=ask, say=quiet)
    checks["final_tool_run"] = len(seen) == turns and out.endswith("last tool result:\nblocked") and "RUN rm -rf / -> blocked" in out
    # Malformed calls: one repair turn for quick, two otherwise.
    bad = '<tool_call>{"name": "read_file", "arguments": {"path": '
    ask_q, seen_q = _scripted([bad])
    ask_r, seen_r = _scripted([bad])
    drill_agent("[kind:quick]\nsay hello", ask=ask_q, say=quiet)
    drill_agent("[kind:research]\nsay hello", ask=ask_r, say=quiet)
    checks["fixes"] = len(seen_q) == 2 and len(seen_r) == 3
    # Retries: hard 500 once, 503 still ends in ModelWait.
    no_sleep = lambda _s: None  # noqa: E731
    opener, calls = _fake_opener([500, 500, 500, 500])
    try:
        chat([{"role": "user", "content": "x"}], _open=opener, _sleep=no_sleep)
        hard_ok = False
    except ModelWait:
        hard_ok = False
    except RuntimeError:
        hard_ok = len(calls) == 2
    opener, calls = _fake_opener([502, 200])
    retry_ok = chat([{"role": "user", "content": "x"}], _open=opener, _sleep=no_sleep)["content"] == "ok" and len(calls) == 2
    opener, calls = _fake_opener([503])
    try:
        chat([{"role": "user", "content": "x"}], _open=opener, _sleep=no_sleep)
        wait_ok = False
    except ModelWait:
        wait_ok = True
    checks["retry_hard"] = hard_ok
    checks["retry_once_ok"] = retry_ok
    checks["retry_503"] = wait_ok
    # fit_context: oldest tool results go first, then pairs. System and task never change.
    def history() -> list[dict]:
        items = [{"role": "system", "content": "S" * 500}, {"role": "user", "content": "T" * 5000}]
        for number in range(8):
            items.append({"role": "assistant", "content": "READ a " + "c" * 1000})
            items.append({"role": "user", "content": tool_response(f"r{number} " + "y" * 2300)})
        return items

    def total(items: list[dict]) -> int:
        return sum(len(str(item["content"])) for item in items)

    soft = history()
    fit_context(soft, 25000)
    removed = [item for item in soft[3::2] if "older result removed" in item["content"]]
    checks["fit_order"] = (
        len(soft) == 18 and total(soft) <= 25000
        and soft[0]["content"] == "S" * 500 and soft[1]["content"] == "T" * 5000
        and "older result removed" in soft[3]["content"] and 0 < len(removed) < 7
        and all(len(item["content"]) == 1007 for item in soft[2::2])
        and "r7 " in soft[-1]["content"]
    )
    hard_fit = history()
    fit_context(hard_fit, 10000)
    checks["fit_pairs"] = (
        4 <= len(hard_fit) < 18 and total(hard_fit) <= 10000
        and hard_fit[0]["content"] == "S" * 500 and hard_fit[1]["content"] == "T" * 5000
        and hard_fit[2]["role"] == "assistant" and "r7 " in hard_fit[-1]["content"]
    )
    checks["fit_budget"] = DEFAULT_CONTEXT_CHARS == 40000 and (
        CONTEXT_CHARS == 40000 or bool(os.environ.get("NEURO_CONTEXT_CHARS"))
    )
    for leftover in (ok_file, verify_file):
        found = inside(leftover)
        if found:
            try:
                found[1].unlink(missing_ok=True)
            except OSError:
                pass
    shutil.rmtree(drill_logs, ignore_errors=True)
    if missed:
        print("router missed: " + "; ".join(f"{text[:40]!r} want {want} got {got}" for text, want, got in missed), flush=True)
    return checks


def session_drill(round_file: str) -> dict:
    """EDIT, GLOB, TODO, batches, thinking, compaction, loop breaker, usage and trajectories.
    Mocked model only. Every file it makes is under workspace/agent-check or agent-logs/drill."""
    pid = os.getpid()
    quiet = lambda _line: None  # noqa: E731
    checks: dict[str, bool] = {}
    logs = LOG_DIR / "drill" / f"session-{pid}"
    shutil.rmtree(logs, ignore_errors=True)
    logs.mkdir(parents=True, exist_ok=True)

    def agent(prompt: str, replies: list, **options) -> tuple[str, list[dict]]:
        ask, seen = _scripted(replies)
        return run_agent(prompt, ask=ask, say=quiet, log_dir=logs, **options), seen

    def lines_of(name: str) -> list[str]:
        try:
            return [line for line in (logs / name).read_text(encoding="utf-8").splitlines() if line.strip()]
        except OSError:
            return []

    target = f"workspace/agent-check/drill-edit-{pid}.py"
    big = f"workspace/agent-check/drill-big-{pid}.txt"
    made_file = f"workspace/agent-check/drill-traj-{pid}.txt"
    path = inside(target)[1]
    path.parent.mkdir(parents=True, exist_ok=True)
    original = b"a = 1\r\nb = 1\r\nc = 2\r\n"
    path.write_bytes(original)
    inside(big)[1].write_text("\n".join(f"line {n} " + "x" * 40 for n in range(1, 301)) + "\n", encoding="utf-8")

    def edit(old: str, new: str, where: str = target) -> str:
        return edit_call(where, old, new)

    # EDIT: refused before READ, refused when ambiguous or absent, exact once works and keeps CRLF.
    state = TaskState("drill")
    unread = run_tool(edit("a = 1", "a = 5"), "", False, kind="code", state=state)[0]
    checks["edit_unread"] = unread.startswith("refused: READ") and path.read_bytes() == original
    numbered = run_tool(f"READ {target}", "", False, kind="code", state=state)[0]
    checks["read_numbered"] = numbered.splitlines() == ["1: a = 1", "2: b = 1", "3: c = 2"]
    long_read = read_tool(big)
    ranged = read_tool(big + " 290-400")
    checks["read_budget"] = (
        long_read.startswith("1: line 1 ") and len(long_read) <= TOOL_CHARS and "300 lines in all" in long_read
        and f"READ {big} " in long_read and ranged.splitlines()[0].startswith("290: ") and "300: line 300" in ranged
    )
    ambiguous = run_tool(edit("= 1", "= 9"), "", False, kind="code", state=state)[0]
    absent = run_tool(edit("zzz = 0", "y = 1"), "", False, kind="code", state=state)[0]
    checks["edit_ambiguous"] = ambiguous.startswith("refused") and "matches 2 times" in ambiguous and path.read_bytes() == original
    checks["edit_absent"] = absent.startswith("refused") and "matches 0 times" in absent
    done, made = run_tool(edit("a = 1", "a = 5"), "", False, kind="code", state=state)
    checks["edit_unique"] = (
        made == [f"edited {target}"] and "-a = 1" in done and "+a = 5" in done and "line 1" in done
        and path.read_bytes() == b"a = 5\r\nb = 1\r\nc = 2\r\n" and state.changed == [target]
    )
    renumbered = run_tool(edit("2: b = 1", "2: b = 7"), "", False, kind="code", state=state)[1]
    checks["edit_numbers"] = renumbered == [f"edited {target}"] and path.read_bytes() == b"a = 5\r\nb = 7\r\nc = 2\r\n"
    cut_edit = run_tool(f"EDIT {target}\n{EDIT_OPEN}\nb = 7\n{EDIT_MID}\nb = 8", "", True, kind="code", state=state)[0]
    checks["edit_safety"] = (
        run_tool(edit("x", "y", "cluster.token"), "", False, kind="code", state=state)[0].startswith("blocked")
        and run_tool(edit("x", "y", "../outside.txt"), "", False, kind="code", state=state)[0].startswith("blocked")
        and run_tool(edit("x", "y", "agents.db"), "", False, kind="code", state=state)[0].startswith("blocked")
        and run_tool(edit("b = 7", "b = 8"), "", False, kind="research", state=state)[0].startswith("blocked")
        and run_tool(edit("b = 7", "b = 8"), "", False, kind="quick", state=state)[0].startswith("blocked")
        and cut_edit.startswith("not edited") and b"b = 7" in path.read_bytes()
        and run_tool(edit("x", "y", f"workspace/agent-check/none-{pid}.py"), "", False, kind="code", state=state)[0].startswith("missing")
    )
    checks["edit_native"] = (
        native_tool('<tool_call>{"name": "edit_file", "arguments": {"path": "workspace/a.py", "old_string": "x = 1", "new_string": "x = 2"}}</tool_call>')
        == edit_call("workspace/a.py", "x = 1", "x = 2")
        and parse_edit(edit_call("w/a.py", "one\ntwo", "")) == ("w/a.py", "one\ntwo", "", True)
    )
    # GLOB: patterns under the repo or workspace, never a secret, never outside.
    globbed = tool_glob(f"workspace/agent-check/drill-edit-{pid}.*")
    checks["glob"] = (
        globbed == target and "agents.py" in tool_glob("*.py").splitlines()
        and tool_glob("../*").startswith("need") and tool_glob("C:/Windows/*").startswith("need")
        and tool_glob("*.token") == "no matches" and tool_glob("**/nodes.json") == "no matches"
        and target in tool_glob("workspace/**/*.{py,md}")
        and native_tool('<tool_call>{"name": "glob", "arguments": {"pattern": "**/*.py"}}</tool_call>') == "GLOB **/*.py"
    )
    # TODO: kept by the agent, shown on the newest message only, never stored in history.
    todo_call = "TODO\n[>] read the file\n[ ] edit it\n[ ] run a check"
    out, seen = agent("[kind:code]\nlook at the drill file", [todo_call, f"READ {target}", "GLOB workspace/agent-check/*.py", "Looked."])
    pin = "TODO 0/3: [>] read the file | [ ] edit it | [ ] run a check"
    checks["todo_pinned"] = (
        len(seen) == 4 and pin not in seen[0]["last"] and pin in seen[1]["last"] and pin in seen[3]["last"]
        and all(pin not in item["content"] for item in seen[3]["messages"][:-1])
        and native_tool('<tool_call>{"name": "todo_write", "arguments": {"todos": [{"content": "a", "status": "completed"}, {"content": "b", "status": "in_progress"}]}}</tool_call>')
        == "TODO\n[x] a\n[>] b"
    )
    # Batches: up to 3 read-only calls run together; WRITE/EDIT/RUN go alone.
    four = f"READ {target}\nLIST workspace/agent-check\nGLOB workspace/agent-check/drill-edit-{pid}.*\nGREP drill_marker_never"
    out, seen = agent("[kind:research]\nlook at the drill file", [four, "Read them."])
    last = seen[1]["last"]
    checks["batch_reads"] = (
        last.count("<tool_response>") == 4 and "1: a = 5" in last and f"drill-edit-{pid}.py" in last
        and "Only the first 3" in last and "GREP drill_marker_never" not in out and out.endswith("Read them.")
    )
    out, seen = agent("[kind:research]\nlook", [f"READ {target}\nRUN echo batch-never", "ok"])
    out2, seen2 = agent("[kind:research]\nlook", [f"RUN echo solo-ok\nREAD {target}", "ok"])
    native_two = (
        f'<tool_call>\n{{"name": "read_file", "arguments": {{"path": "{target}"}}}}\n</tool_call>\n'
        '<tool_call>\n{"name": "list_dir", "arguments": {"path": "workspace/agent-check"}}\n</tool_call>'
    )
    out3, seen3 = agent("[kind:research]\nlook", [native_two, "ok"])
    checks["batch_alone"] = (
        "batch-never" not in seen[1]["last"] and "Only the first 1" in seen[1]["last"]
        and "solo-ok" in seen2[1]["last"] and "1: a = 5" not in seen2[1]["last"]
        and seen3[1]["last"].count("<tool_response>") == 2 and "1: a = 5" in seen3[1]["last"]
    )
    # Thinking: on to plan the first turn, off for routine turns, on right after a failure.
    fail_run = "echo check & exit 3" if os.name == "nt" else "echo check; exit 3"
    out, seen = agent("[kind:code]\nlook at the drill file", [f"READ {target}", f"RUN {fail_run}", "GLOB workspace/agent-check/*.py", "Done."])
    checks["think_first"] = seen[0]["think"] is True and seen[0]["max_tokens"] == KIND_BUDGET["code"].max_tokens
    checks["think_routine"] = seen[1]["think"] is False and seen[3]["think"] is False
    checks["think_after_failure"] = seen[2]["think"] is True
    out, seen = agent("[kind:research]\nwhere is drill", ["GREP drill_marker_never", "LIST workspace", "Nowhere."])
    out_q, seen_q = agent("[kind:quick]\nsay hi", ["hi"])
    out_d, seen_d = agent("[kind:research]\n/deep where is drill", ["LIST workspace", "ok"])
    checks["think_tokens"] = (
        seen[0]["think"] is True and seen[1]["think"] is False and seen[1]["max_tokens"] == TOOL_TOKENS
        and seen[2]["max_tokens"] == TOOL_TOKENS and seen_q[0]["think"] is False
        and all(item["think"] for item in seen_d) and seen_d[1]["max_tokens"] == DEEP_TOKENS
    )
    cut_answer = {"content": "a long answer that", "finish": "length"}
    out, seen = agent("[kind:research]\nexplain the drill", ["LIST workspace", cut_answer, "Whole answer."])
    checks["final_tokens"] = len(seen) == 3 and seen[2]["max_tokens"] == KIND_BUDGET["research"].max_tokens and out.endswith("Whole answer.")
    # Compaction: the oldest turns fold into one summary; system, task, summary, newest stay.
    task_state = TaskState("compact drill task")
    task_state.read_names, task_state.changed = ["a.py"], ["b.py"]
    task_state.failures, task_state.todo = ["RUN x -> exit 1"], [("doing", "fix b")]
    history = [{"role": "system", "content": "S" * 300}, {"role": "user", "content": "Task:\n" + "T" * 400}]
    for number in range(10):
        history.append({"role": "assistant", "content": f"READ f{number} " + "c" * 200, "note": f"READ f{number} -> ok{number}"})
        history.append({"role": "user", "content": tool_response(f"r{number} " + "y" * 1500)})
    compact(history, 9000, task_state)
    first_summary = history[2]["content"] if len(history) > 2 else ""
    for number in range(10, 16):
        history.append({"role": "assistant", "content": f"READ f{number} " + "c" * 200, "note": f"READ f{number} -> ok{number}"})
        history.append({"role": "user", "content": tool_response(f"r{number} " + "y" * 1500)})
    compact(history, 9000, task_state)
    view = [{"role": item["role"], "content": item["content"]} for item in history]
    fit_context(view, 3000)
    checks["compact"] = (
        history[0]["content"] == "S" * 300 and history[1]["content"] == "Task:\n" + "T" * 400
        and first_summary.startswith(SUMMARY_HEAD) and "READ f0 -> ok0" in first_summary
        and "Files read: a.py" in first_summary and "Files changed: b.py" in first_summary
        and "Failed attempts:" in first_summary and "TODO 0/1: [>] fix b" in first_summary
        and is_summary(history[2]) and sum(1 for item in history if is_summary(item)) == 1
        and "READ f9 -> ok9" in history[2]["content"]
        and history[-2]["content"].startswith("READ f15") and "r15 " in history[-1]["content"]
        and "r14 " in history[-3]["content"] and sum(len(item["content"]) for item in history) <= 9000
        and is_summary(view[2]) and view[1]["content"] == history[1]["content"] and "r15 " in view[-1]["content"]
    )
    reads = [f"READ {big}", f"READ {big} 60-140", f"READ {big} 140-220", f"READ {big} 220-300", "LIST workspace/agent-check", "Done."]
    out, seen = agent("[kind:research]\nread the big drill file", reads, context_chars=9000)
    final_view = seen[-1]["messages"]
    checks["compact_agent"] = (
        is_summary(final_view[2]) and final_view[1]["content"].startswith("Task:\nread the big drill file")
        and final_view[0]["content"] == agent_system("research") and "Files read: " + big in final_view[2]["content"]
        and f"drill-edit-{pid}.py" in final_view[-1]["content"] and final_view[-2]["role"] == "assistant"
    )
    # Loop breaker: the same call twice gets a nudge, three times ends the job honestly.
    out, seen = agent("[kind:research]\nlook", [f"READ {target}"] * 5)
    checks["loop_breaker"] = (
        len(seen) == 3 and "same call as last turn" in seen[2]["last"] and "Stopped:" in out
        and "three times" in out and "not finished" in out
    )
    # Usage: one line per finished task, with the API's token counts.
    before = len(lines_of("usage.jsonl"))
    replies = [
        {"content": "LIST workspace", "usage": {"prompt": 120, "completion": 7, "cached": 100}},
        {"content": "hello", "usage": {"prompt": 130, "completion": 3, "cached": 120}},
    ]
    agent("[kind:quick]\nsay hello", replies)
    rows = lines_of("usage.jsonl")
    row = json.loads(rows[-1]) if rows else {}
    usage_body = (
        b'{"choices":[{"message":{"content":"ok"},"finish_reason":"stop"}],'
        b'"usage":{"prompt_tokens":11,"completion_tokens":2},"timings":{"cache_n":5}}'
    )
    import io

    parsed = chat([{"role": "user", "content": "x"}], _open=lambda request, timeout=0: io.BytesIO(usage_body), _sleep=lambda _s: None)
    checks["usage_line"] = (
        len(rows) == before + 1 and row.get("prompt_tokens") == 250 and row.get("completion_tokens") == 10
        and row.get("cached_tokens") == 220 and row.get("turns") == 2 and row.get("kind") == "quick"
        and row.get("outcome") == "answered" and parsed["usage"] == {"prompt": 11, "completion": 2, "cached": 5}
    )
    # Trajectories: only a code task whose last check passed, and never one that touched a secret.
    write = f"WRITE {made_file}\nok\nENDWRITE"
    agent("[kind:code]\nmake a drill file", [write, "RUN echo check traj-ok", "Wrote the file. Check passed."])
    saved = lines_of("trajectories.jsonl")
    good = json.loads(saved[-1]) if saved else {}
    messages = good.get("messages") or [{}]
    agent("[kind:code]\nmake a drill file", [write, f"RUN {fail_run}", "Done, but the check failed."])
    agent("[kind:code]\nmake a drill file", [write, "Done.", "Still done."])
    agent("[kind:code]\nmake a drill file", ["READ cluster.token", write, "RUN echo check traj-ok", "Done."])
    agent("[kind:code]\nmake a drill file", [write, "RUN echo check sk-" + "a" * 24, "Done."])
    agent("[kind:code]\nmake a drill file", ["GREP nodes.json", write, "RUN echo check traj-ok", "Done."])
    agent("[kind:research]\nlook", ["RUN echo check traj-ok", "Done."])
    after = lines_of("trajectories.jsonl")
    whole = "\n".join(after)
    checks["trajectory"] = (
        len(saved) == 1 and len(after) == 1 and messages[0].get("role") == "system"
        and messages[0].get("content") == agent_system("code")
        and str(messages[1].get("content", "")).startswith("Task:\nmake a drill file")
        and any(item.get("role") == "assistant" and "ENDWRITE" in item.get("content", "") for item in messages)
        and any(item.get("role") == "user" and "traj-ok" in item.get("content", "") for item in messages)
        and messages[-1] == {"role": "assistant", "content": "Wrote the file. Check passed."}
    )
    checks["trajectory_secret"] = "cluster.token" not in whole and "sk-aaaa" not in whole and "nodes.json" not in whole
    # Public repo: peers come from the env or nodes.json, optional watchers never need the scripts.
    peers_file = logs / "peers.json"
    peers_file.write_text(json.dumps([{"url": "http://10.0.0.5:8781/"}, {"url": 5}, {"name": "x"}]), encoding="utf-8")
    missing = logs / "no-such-script.py"
    checks["peers_env"] = (
        peer_urls({"NEURO_PEER_URLS": " http://a:1/, ,ftp://x, http://b:2"}) == ("http://a:1", "http://b:2")
        and peer_urls({"NEURO_PEER_URLS": ""}) == ()
        and peer_urls({}, nodes=peers_file) == ("http://10.0.0.5:8781",)
        and peer_urls({}, nodes=logs / "none.json") == ()
    )
    checks["optional_watchers"] = all(
        "optional watcher" in watch_report(role, {role: missing}) for role in ("train", "night", "brain")
    )
    for leftover in (target, big, made_file):
        found = inside(leftover)
        if found:
            try:
                found[1].unlink(missing_ok=True)
            except OSError:
                pass
    shutil.rmtree(logs, ignore_errors=True)
    return checks


OLD_JOBS_SCHEMA = (
    "CREATE TABLE jobs (id INTEGER PRIMARY KEY, title TEXT NOT NULL, prompt TEXT NOT NULL, status TEXT NOT NULL,"
    " worker_pid INTEGER, result TEXT, error TEXT, created REAL NOT NULL, updated REAL NOT NULL)"
)
OLD_WORKERS_SCHEMA = (
    "CREATE TABLE workers (pid INTEGER PRIMARY KEY, started REAL NOT NULL, heartbeat REAL NOT NULL,"
    " state TEXT NOT NULL, job_id INTEGER)"
)


def team_drill() -> dict:
    """Team mode on a temporary queue: SPAWN, WAIT, limits, the write jail, scheduling, migration.
    Mocked model only. It never opens the live agents.db and never writes the live logs."""
    pid = os.getpid()
    quiet = lambda _line: None  # noqa: E731
    checks: dict[str, bool] = {}
    home = LOG_DIR / "drill" / f"team-{pid}"
    old_db, team_db = home / "old-schema.db", home / "team.db"
    before = (getattr(_OVERRIDE, "db", None), getattr(_OVERRIDE, "logs", None))
    # High ids keep the drill's workspace/team-<id> folders apart from real team folders.
    base = 900_000_000 + (pid % 1_000_000) * 100
    w1, w2 = 990_000_001, 990_000_002  # fake worker pids: rows in the temporary queue only
    escape = f"workspace/agent-check/team-escape-{pid}.txt"
    folders: list[str] = []
    for item in (old_db, team_db):
        _schema_ready.discard(path_key(item))
    shutil.rmtree(home, ignore_errors=True)
    home.mkdir(parents=True, exist_ok=True)
    try:
        # Migration: an old-schema queue gains the team columns and keeps its rows.
        raw = sqlite3.connect(old_db)
        raw.execute(OLD_JOBS_SCHEMA)
        raw.execute(OLD_WORKERS_SCHEMA)
        raw.execute("INSERT INTO jobs (title, prompt, status, created, updated) VALUES ('old', '[kind:quick]\nhi', 'done', 1, 1)")
        raw.commit()
        raw.close()
        _OVERRIDE.db, _OVERRIDE.logs = old_db, home
        with connect() as db:
            columns = {str(row[1]) for row in db.execute("PRAGMA table_info(jobs)")}
            kept = [tuple(row) for row in db.execute("SELECT title, status, parent_id, role, depth, prompt_tokens FROM jobs")]
            again = migrate_jobs(db)
        checks["team_migrate"] = (
            all(name in columns for name, _ in TEAM_COLUMNS) and kept == [("old", "done", None, None, 0, 0)] and again == []
        )
        # A fresh temporary queue with two workers.
        _OVERRIDE.db = team_db
        now = time.time()
        with connect() as db:
            db.execute(
                "INSERT INTO jobs (id, title, prompt, status, created, updated) VALUES (?, 'drill base', '[kind:quick]\nbase', 'done', ?, ?)",
                (base, now, now),
            )
            for worker in (w1, w2):
                db.execute("INSERT INTO workers (pid, started, heartbeat, state, job_id) VALUES (?, ?, ?, 'idle', NULL)", (worker, now, now))
        lead1 = enqueue("drill lead one", "[kind:code]\nbuild three drill parts", role="lead")
        lead2 = enqueue("drill lead two", "[kind:research]\nlook at the drill parts", role="lead")
        folders += [f"workspace/team-{lead1}", f"workspace/team-{lead2}"]
        first = claim(w1)
        capped = claim(w2)  # 2 workers: one lead at a time, the other worker stays free for sub-agents
        # Lead one SPAWNs 3 sub-agents in one turn; while it WAITs, worker two runs them.
        runs: dict[str, tuple[dict, str, list]] = {}
        lead_states: list[str] = []

        def child_replies(job: dict) -> list:
            jail = team_jail(lead1, job["id"])
            if job["role"] == "code":
                return [
                    "SPAWN code nested\nmore work\nENDSPAWN",
                    f"WRITE {escape}\nno\nENDWRITE",
                    f"WRITE {jail}/part.py\nprint('part')\nENDWRITE",
                    "RUN echo check part-ok",
                    {"content": "Wrote part.py in my folder. Check passed.", "usage": {"prompt": 50, "completion": 9}},
                ]
            if job["role"] == "test":
                return [f"WRITE {jail}/test_part.py\nassert True\nENDWRITE", "RUN echo check test-ok", "Wrote test_part.py. Check passed."]
            return [{"content": "Research: the drill parts are independent.", "usage": {"prompt": 40, "completion": 6}}]

        def run_children(_seconds: float) -> None:
            with connect() as db:
                lead_states.append(str(db.execute("SELECT state FROM workers WHERE pid=?", (w1,)).fetchone()["state"]))
            while True:
                job = claim(w2)
                if job is None:
                    break
                ask, seen = _scripted(child_replies(job))
                out = run_agent(job["prompt"], ask=ask, say=quiet, log_dir=home, job={**job, "pid": w2})
                finish(job["id"], w2, out)
                runs[job["role"]] = (job, out, seen)

        spawns = (
            "SPAWN code part one\nWrite part.py that prints a word.\nENDSPAWN\n"
            "SPAWN research survey\nSay whether the parts are independent.\nENDSPAWN\n"
            "SPAWN test part tests\nWrite test_part.py."
        )
        ask, seen = _scripted(
            [{"content": spawns, "usage": {"prompt": 100, "completion": 20}}, "WAIT", "CHILDREN", "Integrated the three parts."]
        )
        lead_out = run_agent(
            str(first["prompt"]) if first else "", ask=ask, say=quiet, log_dir=home,
            job={**(first or {}), "pid": w1}, team_options={"sleep": run_children},
        )
        finish(lead1, w1, lead_out)
        kids = team_children(lead1)
        ids = {row["role"]: row["id"] for row in kids}
        checks["team_parent_cap"] = bool(first) and first["id"] == lead1 and capped is None
        checks["team_spawn3"] = (
            len(seen) == 4 and seen[1]["last"].count("spawned #") == 3 and [row["role"] for row in kids] == ["code", "research", "test"]
            and seen[0]["system"] == agent_system("code", "lead") and TEAM_LEAD_NOTE in seen[0]["task"]
            and "SPAWN" in seen[0]["system"] and "WAIT" in seen[0]["system"] and "SPAWN" not in agent_system("code")
        )
        report = seen[2]["last"] if len(seen) > 2 else ""
        checks["team_wait_reports"] = (
            "3 of 3 sub-agents finished." in report and f"#{ids.get('code')} code done" in report
            and "Wrote part.py in my folder" in report and "Research: the drill parts are independent." in report
            and "Wrote test_part.py" in report and lead_states == ["team_wait"] and lead_out.endswith("Integrated the three parts.")
            and all(row["status"] == "done" for row in kids)
        )
        code_job, _code_out, code_seen = runs.get("code", ({}, "", []))
        code_jail = team_jail(lead1, ids.get("code", 0))
        with connect() as db:
            grandchildren = db.execute(
                "SELECT COUNT(*) AS n FROM jobs WHERE parent_id IN (SELECT id FROM jobs WHERE parent_id=?)", (lead1,)
            ).fetchone()["n"]
        checks["team_child_no_spawn"] = (
            len(code_seen) == 5 and "cannot use SPAWN" in code_seen[1]["last"] and grandchildren == 0
            and code_seen[0]["system"] == agent_system("code", "child") and "SPAWN <" not in code_seen[0]["system"]
            and "SPAWN research" not in code_seen[0]["system"]
            and f"Role: code sub-agent #{ids.get('code')} of lead task #{lead1}" in code_seen[0]["task"]
            and f"{code_jail}/" in code_seen[0]["task"] and "at most 8 lines" in code_seen[0]["task"]
            and "at most 8 lines" in code_seen[0]["system"]
        )
        part = inside(f"{code_jail}/part.py")
        test_part = inside(f"{team_jail(lead1, ids.get('test', 0))}/test_part.py")
        child_team = Team({"id": ids.get("code"), "parent_id": lead1, "depth": 1, "pid": w2}, "code")
        token = _JAIL.set((code_jail, inside(code_jail)[1].resolve()))
        try:
            jail_write = safe_write(f"workspace/team-{lead1}/{ids.get('test')}/x.py", "no")
            jail_mkdir = apply_actions(f"MKDIR workspace/team-{lead1}/other-{pid}")
            jail_state = TaskState("jail")
            jail_state.saw("x", test_part[1])
            jail_edit = tool_edit(f"{team_jail(lead1, ids.get('test', 0))}/test_part.py", "assert True", "assert 1", jail_state)[0]
            jail_ok = safe_write(f"{code_jail}/more.py", "x = 1")
        finally:
            _JAIL.reset(token)
        checks["team_jail"] = (
            safe_read(escape) == "missing" and "a sub-agent writes only under" in code_seen[2]["last"]
            and part is not None and part[1].is_file() and test_part is not None and test_part[1].is_file()
            and jail_write.startswith("blocked") and jail_mkdir[0].startswith("blocked") and jail_edit.startswith("blocked")
            and jail_ok.startswith("wrote ") and "assert True" in safe_read(f"{team_jail(lead1, ids.get('test', 0))}/test_part.py")
            and child_team.tool("SPAWN code x\ny", "SPAWN", False).startswith("blocked")
            and safe_write(f"workspace/agent-check/team-free-{pid}.txt", "ok").startswith("wrote ")  # the jail ended
        )
        folders.append(f"workspace/agent-check/team-free-{pid}.txt")
        tree = tree_text(lead1)
        rows = [json.loads(line) for line in (home / "usage.jsonl").read_text(encoding="utf-8").splitlines() if line.strip()]
        team_rows = [row for row in rows if row.get("kind") == "team"]
        checks["team_tree_usage"] = (
            f"#{lead1} lead" in tree and all(f"#{item}" in tree for item in ids.values()) and "finished" in tree
            and "120 tok" in tree and "3 sub-agents" in tree
            and len(team_rows) == 1 and team_rows[0]["tree"] == lead1 and team_rows[0]["jobs"] == 4
            and team_rows[0]["prompt_tokens"] == 190 and team_rows[0]["completion_tokens"] == 35
            and sum(1 for row in rows if row.get("tree") == lead1 and row.get("kind") != "team") == 4
        )
        # Lead two: limits, refusals, the WAIT timeout path and cancel on finish.
        second = claim(w1)
        clock = [time.time()]
        sleeps: list[float] = []

        def fake_sleep(seconds: float) -> None:
            sleeps.append(seconds)
            clock[0] += seconds

        team2 = Team({**(second or {}), "pid": w1}, "research", sleep=fake_sleep, clock=lambda: clock[0], wait_seconds=30)

        def call(text: str, kind: str = "research", team: Team | None = team2, cut: bool = False) -> str:
            return run_tool(text, text, cut, kind=kind, team=team)[0]

        no_code = call("SPAWN code build\nwrite it\nENDSPAWN")
        made = [call(f"SPAWN research part {n}\nlook at part {n}\nENDSPAWN") for n in range(TEAM_MAX)]
        seventh = call("SPAWN research one more\nlook again\nENDSPAWN")
        cut_spawn = call("SPAWN research cut\npartial", cut=True)
        checks["team_limit"] = (
            (TEAM_MAX == 6 or bool(os.environ.get("NEURO_TEAM_MAX")))
            and (TEAM_WAIT == 480 or bool(os.environ.get("NEURO_TEAM_WAIT")))
            and bool(second) and second["id"] == lead2 and no_code.startswith("refused")
            and all(item.startswith("spawned #") for item in made) and seventh.startswith("refused")
            and f"already has {TEAM_MAX} sub-agents" in seventh and cut_spawn.startswith("not spawned")
            and len(team_children(lead2)) == TEAM_MAX
            and call("SPAWN research x\ny", kind="quick").startswith("blocked")
            and call("SPAWN research x\ny", team=None).startswith("blocked")
            and call("SPAWN bogus x\ny").startswith("refused")
        )
        waited = call("WAIT")
        with connect() as db:
            restored = str(db.execute("SELECT state FROM workers WHERE pid=?", (w1,)).fetchone()["state"])
        unknown = call("WAIT 12345")
        for _ in range(TEAM_WAITS - 1):
            call("WAIT " + str(team_children(lead2)[0]["id"]))
        over = call("WAIT")
        checks["team_wait_timeout"] = (
            f"0 of {TEAM_MAX} sub-agents finished." in waited and "still running after 30s" in waited
            and waited.count("waiting for a worker") == TEAM_MAX and len(sleeps) >= 15 and sleeps[:15] == [TEAM_POLL] * 15
            and restored == "generating" and unknown.startswith("refused") and over.startswith("refused")
            and "limit" in over and team2.waits == TEAM_WAITS
        )
        finish(lead2, w1, "done")
        cancelled = team_children(lead2)
        # Wall time and worker count: a third lead with its own clock.
        lead3 = enqueue("drill lead three", "[kind:review]\nreview the drill parts", role="lead")
        folders.append(f"workspace/team-{lead3}")
        team3 = Team({"id": lead3, "pid": w1}, "review", sleep=fake_sleep, clock=lambda: clock[0], wall_seconds=60)
        one = call("SPAWN review look\nreview part one\nENDSPAWN", kind="review", team=team3)
        with connect() as db:
            db.execute("DELETE FROM workers WHERE pid=?", (w2,))
        alone = call("SPAWN review look\nreview part two\nENDSPAWN", kind="review", team=team3)
        with connect() as db:
            db.execute("INSERT INTO workers (pid, started, heartbeat, state, job_id) VALUES (?, ?, ?, 'idle', NULL)", (w2, now, time.time()))
        clock[0] += 120
        late = call("SPAWN review late\nreview part three\nENDSPAWN", kind="review", team=team3)
        count = len(sleeps)
        late_wait = call("WAIT", kind="review", team=team3)
        checks["team_wall_cancel"] = (
            all(row["status"] == "error" and row["error"] == CANCELLED for row in cancelled) and len(cancelled) == TEAM_MAX
            and one.startswith("spawned #") and alone.startswith("refused") and "fewer than 2" in alone
            and late.startswith("refused") and "team time limit" in late and "time limit is used up" in late_wait
            and len(sleeps) == count
        )
        # Slots: a lead in WAIT does not hold one of the model's slots; four generating jobs do.
        stamp = time.time()
        with connect() as db:
            db.execute("UPDATE jobs SET status='done' WHERE status IN ('queued', 'running')")
            db.execute("UPDATE workers SET state='team_wait' WHERE pid=?", (w1,))
            for worker in (w1, 990_000_003, 990_000_004, 990_000_005):
                db.execute(
                    "INSERT INTO jobs (title, prompt, status, worker_pid, created, updated) VALUES ('busy', '[kind:quick]\nx', 'running', ?, ?, ?)",
                    (worker, stamp, stamp),
                )
        quick1 = enqueue("drill quick", "[kind:quick]\nsay hi")
        took = claim(w2)
        with connect() as db:
            db.execute("UPDATE workers SET state='generating' WHERE pid=?", (w1,))
        enqueue("drill quick two", "[kind:quick]\nsay hi")
        full = claim(w2)
        checks["team_slots"] = bool(took) and took["id"] == quick1 and full is None
        # Parsing: plain SPAWN blocks, native spawn and wait calls, SPAWN batches.
        parsed = text_tools("SPAWN code a\ndo a\n\nmore a\nSPAWN research b\ndo b\nENDSPAWN\nWAIT 3 4\nCHILDREN")
        checks["team_parse"] = (
            [item[0].split("\n")[0] for item in parsed] == ["SPAWN code a", "SPAWN research b", "WAIT 3 4", "CHILDREN"]
            and parse_spawn(parsed[0][0]) == ("code", "a", "do a\n\nmore a", True)
            and len(batch_calls(parsed)) == 2
            and native_tool('<tool_call>{"name": "spawn_agent", "arguments": {"role": "test", "title": "t", "task": "write tests"}}</tool_call>')
            == "SPAWN test t\nwrite tests\nENDSPAWN"
            and native_tool('<tool_call>{"name": "wait", "arguments": {"ids": [5, 6]}}</tool_call>') == "WAIT 5 6"
        )
    finally:
        _OVERRIDE.db, _OVERRIDE.logs = before
        for item in (old_db, team_db):
            _schema_ready.discard(path_key(item))
        for leftover in [*folders, escape]:
            found = inside(leftover)
            if found is None:
                continue
            if found[1].is_dir():
                shutil.rmtree(found[1], ignore_errors=True)
            else:
                try:
                    found[1].unlink(missing_ok=True)
                except OSError:
                    pass
        shutil.rmtree(home, ignore_errors=True)
    return checks


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
        "list": name in listed or regular_file(folder / name),
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
        checks.update(router_drill(relative, f"round {index}"))
    except Exception as exc:  # a broken drill is a failed check, not a dead fleet loop
        checks["router_drill_" + type(exc).__name__] = False
    try:
        checks.update(session_drill(relative))
    except Exception as exc:
        checks["session_drill_" + type(exc).__name__] = False
    try:
        checks.update(team_drill())
    except Exception as exc:
        checks["team_drill_" + type(exc).__name__] = False
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


# Detached agents and watchers have no console. Any console program they start (powershell,
# taskkill) would otherwise open a new visible window every time it runs.
NO_WINDOW = {"creationflags": subprocess.CREATE_NO_WINDOW} if os.name == "nt" else {}


WATCH_ROLES = ("train", "lock", "night", "mesh", "peers", "drill", "review", "brain")
WATCH_STOP = HOME / "watch.stop"
# Optional watchers. The training run, the night loop and the brain server exist only on some
# computers; where the script is absent the watcher says so instead of looking for a process.
TRAIN_SCRIPT = Path(os.environ.get("NEURO_TRAIN_SCRIPT") or (HOME.parent / "ml" / "training" / "train.py"))
NIGHT_SCRIPT = Path(os.environ.get("NEURO_NIGHT_SCRIPT") or (HOME / "night-loop.ps1"))
BRAIN_SCRIPT = Path(os.environ.get("NEURO_BRAIN_SCRIPT") or (HOME.parent / "brain" / "server.py"))
OPTIONAL_SCRIPTS = {"train": TRAIN_SCRIPT, "night": NIGHT_SCRIPT, "brain": BRAIN_SCRIPT}


def optional_script(path: Path) -> bool:
    try:
        return path.is_file()
    except OSError:
        return False


def peer_urls(env=None, nodes: Path | None = None) -> tuple[str, ...]:
    """Other computers to glance at. NEURO_PEER_URLS (comma-separated) wins; when it is unset,
    the saved computers in nodes.json next to mesh.py; otherwise none."""
    source = os.environ if env is None else env
    raw = source.get("NEURO_PEER_URLS")
    items: list[str] = []
    if raw is not None:
        items = raw.split(",")
    else:
        path = nodes if nodes is not None else (ROOT / "nodes.json" if _mesh_state(ROOT) == "file" else None)
        data: object = []
        if path is not None:
            try:
                data = json.loads(path.read_text(encoding="utf-8"))
            except (OSError, ValueError):
                data = []
        if isinstance(data, list):
            items = [item["url"] for item in data if isinstance(item, dict) and isinstance(item.get("url"), str)]
    urls: list[str] = []
    for item in items:
        url = item.strip().rstrip("/")
        if re.fullmatch(r"https?://[^\s/]+(?:/[^\s]*)?", url) and url not in urls:
            urls.append(url)
    return tuple(urls[:16])


def matching_pids(fragment: str) -> list[int]:
    """PIDs whose command line contains fragment. A failed listing is not 'none'."""
    completed = subprocess.run(
        [
            "powershell",
            "-NoProfile",
            "-Command",
            "Get-CimInstance Win32_Process | Where-Object { $_.CommandLine -like $env:WATCH_MATCH } | ForEach-Object { $_.ProcessId }",
        ],
        capture_output=True,
        text=True,
        timeout=40,
        env={**os.environ, "WATCH_MATCH": f"*{fragment}*"},
        check=False,
        **NO_WINDOW,
    )
    if completed.returncode != 0:
        raise OSError("process list failed")
    found = []
    for line in completed.stdout.splitlines():
        line = line.strip()
        if line.isdigit():
            found.append(int(line))
    return found


def watch_report(role: str, scripts: dict[str, Path] | None = None) -> str:
    scripts = OPTIONAL_SCRIPTS if scripts is None else scripts
    if role in scripts and not optional_script(scripts[role]):
        return f"{role}: no script at {scripts[role].name} on this computer (optional watcher)"
    if role == "train":
        try:
            found = matching_pids(str(Path(*TRAIN_SCRIPT.parts[-3:])))
        except (OSError, subprocess.TimeoutExpired):
            return "could not list processes. resume not attempted"
        if not found:
            return "train.py missing. resume not attempted"
        return "train alive pid " + " ".join(str(pid) for pid in found[:4])
    if role == "lock":
        try:
            os.stat(GPU_LOCK)
        except FileNotFoundError:
            return "gpu lock missing"
        except OSError:
            return "gpu lock unreadable, treated as busy. resume not attempted"
        return "gpu lock present, treated as busy. resume not attempted"
    if role == "night":
        try:
            found = matching_pids(NIGHT_SCRIPT.name)
        except (OSError, subprocess.TimeoutExpired):
            return "could not list the night loop"
        tail = ""
        try:
            lines = (LOG_DIR / "night-loop.txt").read_text(encoding="utf-8", errors="replace").splitlines()
            tail = lines[-1][:180] if lines else "empty"
        except OSError:
            tail = "log unreadable"
        state = "alive pid " + str(found[0]) if found else "missing"
        return f"night loop {state}. last {tail}"
    if role == "mesh":
        try:
            found = matching_pids("mesh.py serve")
        except (OSError, subprocess.TimeoutExpired):
            return "could not list mesh"
        return "mesh serve alive pid " + str(found[0]) if found else "mesh serve missing"
    if role == "peers":
        parts = []
        for url in peer_urls():
            try:
                request = urllib.request.Request(url + "/hello", method="GET")
                with _OPENER.open(request, timeout=4) as response:
                    response.read(80)
                    parts.append(f"{url} {response.status}")
            except (urllib.error.URLError, OSError, TimeoutError, http.client.HTTPException):
                parts.append(f"{url} no reply")
        return "glance " + " | ".join(parts)
    if role == "drill":
        try:
            lines = (LOG_DIR / "night-loop.txt").read_text(encoding="utf-8", errors="replace").splitlines()
        except OSError:
            return "night log unreadable"
        picked = [line for line in lines if "drill" in line or "check " in line]
        return (picked[-1] if picked else "no drill line")[:180]
    if role == "review":
        try:
            raw = (LOG_DIR / "review-cursor.txt").read_text(encoding="utf-8", errors="replace").strip()
        except OSError:
            return "review cursor unreadable"
        return "review cursor " + (raw[:20] if raw.isdigit() else "unreadable")
    if role == "brain":
        try:
            found = matching_pids(str(Path(*BRAIN_SCRIPT.parts[-2:])))
        except (OSError, subprocess.TimeoutExpired):
            return "could not list the brain server"
        return "brain server alive pid " + str(found[0]) if found else "brain server missing"
    return "unknown role"


def watch_main(role: str, index: int) -> None:
    if role not in WATCH_ROLES:
        raise SystemExit("unknown watch role")
    time.sleep(max(0, min(index, 8)) * 8)
    while not stop_requested(WATCH_STOP):
        try:
            detail = watch_report(role)
        except Exception as exc:
            detail = "watch error " + type(exc).__name__
        stamp = time.strftime("%Y-%m-%d %H:%M:%S", time.localtime())
        line = f"{stamp} watch {role}: {detail}"
        append_capped(LOG_DIR / "watch-notes.txt", line, limit=200000, keep=400)
        append_capped(LOG_DIR / "training-notes.txt", line, limit=400000, keep=800)
        print(line, flush=True)
        time.sleep(300)


def spawn_watchers() -> dict:
    already: list[int] = []
    for role in WATCH_ROLES:
        try:
            already.extend(matching_pids(f"agents.py watch {role}"))
        except (OSError, subprocess.TimeoutExpired):
            continue
    if already:
        return {"started": 0, "already": already}
    LOG_DIR.mkdir(exist_ok=True)
    flags = 0
    if os.name == "nt":
        flags = subprocess.CREATE_NEW_PROCESS_GROUP | subprocess.DETACHED_PROCESS
    pids = []
    for index, role in enumerate(WATCH_ROLES):
        handle = (LOG_DIR / f"watch-{role}.log").open("a", encoding="utf-8")
        process = subprocess.Popen(
            [sys.executable, "-u", str(WORKER_SCRIPT), "watch", role, str(index)],
            cwd=str(HOME),
            stdout=handle,
            stderr=handle,
            creationflags=flags,
        )
        handle.close()
        pids.append(process.pid)
    return {"started": len(pids), "pids": pids, "roles": list(WATCH_ROLES)}


def fleet_main() -> None:
    """CPU drills on this PC and on the other computers. No model and no weight training."""
    cores = os.cpu_count() or 4
    local_rounds = max(4, min(cores - 4, 12))
    while not stop_requested(HOME / "fleet.stop"):
        local_ok = 0
        for index in range(local_rounds):
            if drill_round(index + 1).get("ok"):
                local_ok += 1
        peer_bits = []
        for url in peer_urls():
            try:
                request = urllib.request.Request(url + "/hello", method="GET")
                with _OPENER.open(request, timeout=5) as response:
                    raw = response.read(12_000)
                data = json.loads(raw.decode("utf-8", "replace"))
                caps = data.get("capabilities") if isinstance(data, dict) and isinstance(data.get("capabilities"), dict) else {}
                peer_cores = int(caps.get("cpu_cores") or 0)
                host = str(data.get("name") or data.get("host") or url)[:80] if isinstance(data, dict) else url
            except (urllib.error.URLError, OSError, TimeoutError, http.client.HTTPException, ValueError, TypeError):
                peer_bits.append(url + " no reply")
                continue
            width = max(1, min(peer_cores or 2, 4))
            def one_check(target: str = url) -> str:
                try:
                    body = json.dumps({"rounds": 5}).encode("utf-8")
                    call = urllib.request.Request(target + "/check", data=body, method="POST")
                    call.add_header("Content-Type", "application/json")
                    with _OPENER.open(call, timeout=60) as response:
                        payload = json.loads(response.read(4_000).decode("utf-8", "replace"))
                    if not isinstance(payload, dict):
                        return "bad"
                    return str(payload.get("passed")) + "/" + str(payload.get("rounds"))
                except (urllib.error.URLError, OSError, TimeoutError, http.client.HTTPException, ValueError):
                    return "no reply"
            results = [one_check() for _n in range(width)]
            peer_bits.append(f"{host} cores {peer_cores or '?'} x{width} " + ",".join(results))
        stamp = time.strftime("%Y-%m-%d %H:%M:%S", time.localtime())
        line = (
            f"{stamp} fleet cpu: local {local_ok}/{local_rounds} serial on {cores} threads. "
            + " | ".join(peer_bits)
            + ". weight training was not started."
        )
        append_capped(LOG_DIR / "fleet-notes.txt", line, limit=200000, keep=400)
        append_capped(LOG_DIR / "training-notes.txt", line, limit=400000, keep=800)
        print(line, flush=True)
        time.sleep(60)


def main() -> None:
    command = sys.argv[1] if len(sys.argv) > 1 else "code"
    if command == "agent":
        worker_main()
        return
    if command == "drill":
        rounds = int(sys.argv[2]) if len(sys.argv) > 2 else 10
        drill_main(rounds)
        return
    if command == "watch":
        if len(sys.argv) > 2 and sys.argv[2] != "start":
            watch_main(sys.argv[2], int(sys.argv[3]) if len(sys.argv) > 3 and sys.argv[3].isdigit() else 0)
            return
        print(json.dumps(spawn_watchers()), flush=True)
        return
    if command == "fleet":
        fleet_main()
        return
    if command == "code":
        code_main(" ".join(sys.argv[2:]).strip())
        return
    if command == "agents":
        terminal(int(sys.argv[2]) if len(sys.argv) > 2 else 4)
        return
    if command == "team":
        team_main(" ".join(sys.argv[2:]))
        return
    if command == "tree":
        tree_main(sys.argv[2] if len(sys.argv) > 2 else "")
        return
    raise SystemExit(
        "use: agents.py agent | agents.py drill [ROUNDS] | agents.py watch start | agents.py fleet"
        ' | agents.py code [TASK] | agents.py team "TASK" | agents.py tree ID'
    )


if __name__ == "__main__":
    main()
