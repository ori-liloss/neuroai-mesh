"""CMD workers for Neuro.

One llama-server holds the model. Workers are separate Python processes.
Only a few of them may generate at once, because the GPU has four slots.
"""

from __future__ import annotations

import ctypes
import json
import os
import shutil
import sqlite3
import subprocess
import sys
import time
import urllib.error
import urllib.request
from pathlib import Path

ROOT = Path(__file__).resolve().parent
DB_PATH = ROOT / "agents.db"
STOP_PATH = ROOT / "agents.stop"
LOG_DIR = ROOT / "agent-logs"
SLOTS = 4
CHAT_URL = "http://127.0.0.1:8080/v1/chat/completions"


def append_capped(path: Path, line: str, limit: int = 200000, keep: int = 100) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    lock_path = path.with_suffix(path.suffix + ".lock")
    with lock_path.open("a+b") as lock:
        lock.seek(0)
        if lock.read(1) == b"":
            lock.write(b"0")
            lock.flush()
        lock.seek(0)
        if os.name == "nt":
            import msvcrt

            msvcrt.locking(lock.fileno(), msvcrt.LK_LOCK, 1)
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


def connect() -> sqlite3.Connection:
    db = sqlite3.connect(DB_PATH, timeout=30)
    db.row_factory = sqlite3.Row
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
    return db


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
    cutoff = time.time() - 180
    stale = db.execute("SELECT pid FROM workers WHERE heartbeat < ?", (cutoff,)).fetchall()
    for row in stale:
        db.execute(
            "UPDATE jobs SET status='queued', worker_pid=NULL, updated=? WHERE status='running' AND worker_pid=?",
            (time.time(), row["pid"]),
        )
        db.execute("DELETE FROM workers WHERE pid=?", (row["pid"],))


def claim(pid: int) -> sqlite3.Row | None:
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
        db.execute(
            "UPDATE jobs SET status=?, result=?, error=?, updated=? WHERE id=?",
            (status, result[:4000], error[:400], now, job_id),
        )
        db.execute(
            "UPDATE workers SET state='idle', job_id=NULL, heartbeat=? WHERE pid=?",
            (now, pid),
        )
        row = db.execute("SELECT title FROM jobs WHERE id=?", (job_id,)).fetchone()
        title = str(row["title"]) if row else ""
    remember(job_id, status, title, error or result)


def ask_model(prompt: str, max_tokens: int = 220) -> str:
    payload = json.dumps(
        {
            "messages": [
                {
                    "role": "system",
                    "content": "You are Neuro. Do the task. One tool call, then stop. Do not explain the tool.",
                },
                {"role": "user", "content": prompt},
            ],
            "temperature": 0.4,
            "max_tokens": max(64, min(int(max_tokens), 1200)),
            "chat_template_kwargs": {"enable_thinking": False},
        }
    ).encode("utf-8")
    request = urllib.request.Request(
        CHAT_URL,
        data=payload,
        headers={"Content-Type": "application/json"},
    )
    try:
        with urllib.request.urlopen(request, timeout=120) as response:
            body = json.loads(response.read().decode("utf-8"))
    except urllib.error.URLError as exc:
        raise RuntimeError("The model is training on this PC, so this task has to wait.") from exc
    message = body["choices"][0]["message"]
    # Qwen puts reasoning in a side field. The reply is only the content.
    return (message.get("content") or "").strip()


def safe_read(relative: str) -> str:
    raw = relative.strip().replace("\\", "/").lstrip("/")
    if not raw or ".." in raw.split("/"):
        return "unreadable"
    path = (ROOT / raw).resolve()
    root = ROOT.resolve()
    if path != root and root not in path.parents:
        return "unreadable"
    if path.name in {"cluster.token", ".env", ".env.local", "token.txt"}:
        return "unreadable"
    if not path.is_file():
        return "missing"
    return path.read_text(encoding="utf-8", errors="replace")[:6000]


def workspace_note() -> str:
    names = []
    for path in sorted(ROOT.iterdir()):
        if path.name.startswith(".") or path.name in {"cluster.token", "agents.db", "nodes.json", "specs.json"}:
            continue
        names.append(path.name + ("/" if path.is_dir() else ""))
    return "Repo files: " + ", ".join(names[:24])


def safe_write(relative: str, content: str) -> str:
    raw = relative.strip().replace("\\", "/").lstrip("/")
    if not raw or ".." in raw.split("/"):
        return "blocked " + raw
    path = (ROOT / raw).resolve()
    root = ROOT.resolve()
    if path != root and root not in path.parents:
        return "blocked " + raw
    if path.name in {"cluster.token", ".env", ".env.local", "token.txt"}:
        return "blocked " + raw
    if len(content.encode("utf-8")) > 150000:
        return "too large"
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(content.strip() + "\n", encoding="utf-8")
    return "wrote " + raw


def apply_actions(answer: str) -> list[str]:
    logs = []
    lines = answer.splitlines()
    index = 0
    while index < len(lines):
        line = clean_tool_line(lines[index])
        if line.startswith("MKDIR "):
            raw = tool_path(line[6:])
            if raw and ".." not in raw.split("/") and raw not in {"cluster.token", ".env", ".env.local"}:
                (ROOT / raw).mkdir(parents=True, exist_ok=True)
                logs.append("mkdir " + raw)
            else:
                logs.append("blocked path")
            return logs
        if line.startswith("WRITE "):
            raw = tool_path(line[6:])
            index += 1
            body = []
            while index < len(lines) and clean_tool_line(lines[index]) != "ENDWRITE":
                body.append(lines[index])
                index += 1
            if not raw:
                logs.append("blocked path")
            else:
                logs.append(safe_write(raw, "\n".join(body)))
            return logs
        index += 1
    return logs


def visible_answer(answer: str) -> str:
    text = answer or ""
    while "<think>" in text and "</think>" in text:
        start = text.find("<think>")
        end = text.find("</think>", start)
        if end < 0:
            break
        text = text[:start] + text[end + len("</think>") :]
    return text.strip()


TOOL_HELP = (
    "You are Neuro Code in this repo. Emit one tool and stop. No sentence before it or after it.\n"
    "LIST relative/dir\n"
    "GREP text\n"
    "READ relative/path\n"
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
    LOG_DIR.mkdir(exist_ok=True)
    row = {
        "t": time.strftime("%Y-%m-%d %H:%M", time.localtime()),
        "id": job_id,
        "status": status,
        "title": title[:80],
        "text": (text or "")[:800],
    }
    with (LOG_DIR / "session.jsonl").open("a", encoding="utf-8") as handle:
        handle.write(json.dumps(row, ensure_ascii=False) + "\n")


def recent_sessions(limit: int = 6) -> list[str]:
    path = LOG_DIR / "session.jsonl"
    if not path.is_file():
        return []
    rows = []
    for line in path.read_text(encoding="utf-8", errors="replace").splitlines():
        if not line.strip():
            continue
        try:
            item = json.loads(line)
        except json.JSONDecodeError:
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
    raw = relative.strip().replace("\\", "/").lstrip("/")
    path = ROOT if not raw else (ROOT / raw).resolve()
    root = ROOT.resolve()
    if path != root and root not in path.parents:
        return None
    if ".." in raw.split("/"):
        return None
    return path if path.is_dir() else None


def tool_list(relative: str) -> str:
    path = repo_dir(relative)
    if path is None:
        return "missing"
    names = []
    for child in sorted(path.iterdir()):
        if child.name in {"cluster.token", ".env", ".env.local", "token.txt"}:
            continue
        names.append(child.name + ("/" if child.is_dir() else ""))
        if len(names) >= 40:
            break
    return "\n".join(names) or "empty"


def tool_grep(needle: str) -> str:
    needle = needle.strip()
    if not needle or len(needle) > 80:
        return "need a short search"
    hits = []
    skip = {"agent-logs", "__pycache__", ".git"}
    secret = {"cluster.token", ".env", ".env.local", "token.txt"}
    for path in ROOT.rglob("*"):
        if not path.is_file() or path.name in secret or any(part in skip for part in path.parts):
            continue
        try:
            if path.stat().st_size > 300000:
                continue
            text = path.read_text(encoding="utf-8", errors="ignore")
        except OSError:
            continue
        for number, line in enumerate(text.splitlines(), 1):
            if needle not in line:
                continue
            hits.append(f"{path.relative_to(ROOT).as_posix()}:{number}: {line.strip()[:160]}")
            if len(hits) >= 20:
                return "\n".join(hits)
    return "\n".join(hits) or "no matches"


def tool_run(command: str) -> str:
    text = command.strip()
    lowered = text.lower()
    blocked = ("cluster.token", "token.txt", ".env", "shutdown", "format ", "rm -rf", "del /s", "reg delete", "cipher ")
    if not text or ".." in text or any(item in lowered for item in blocked):
        return "blocked"
    try:
        completed = subprocess.run(
            text,
            cwd=str(ROOT),
            shell=True,
            capture_output=True,
            text=True,
            timeout=20,
        )
    except subprocess.TimeoutExpired:
        return "timed out"
    out = ((completed.stdout or "") + (completed.stderr or "")).strip()
    return (out or f"exit {completed.returncode}")[:4000]


def clean_tool_line(line: str) -> str:
    text = line.strip().strip("`").strip()
    text = text.lstrip("-*•").strip()
    while text[:1].isdigit():
        text = text[1:]
    return text.lstrip(".) ").strip()


def native_tool(answer: str) -> str:
    # Qwen3.5 emits one <tool_call> block and nothing after it.
    if "<function=" not in answer:
        return ""
    name = answer.split("<function=", 1)[1].split(">", 1)[0].strip().lower()
    names = {
        "list": "LIST",
        "list_dir": "LIST",
        "read": "READ",
        "read_file": "READ",
        "grep": "GREP",
        "search": "GREP",
        "write": "WRITE",
        "write_file": "WRITE",
        "mkdir": "MKDIR",
        "run": "RUN",
        "shell": "RUN",
        "sessions": "SESSIONS",
    }
    command = names.get(name, "")
    if not command:
        return ""
    if command == "SESSIONS":
        return "SESSIONS"
    rest = answer.split("<function=", 1)[1]
    path = ""
    body = ""
    while "<parameter=" in rest:
        rest = rest.split("<parameter=", 1)[1]
        key, _, rest = rest.partition(">")
        value, _, rest = rest.partition("</parameter>")
        key = key.strip().lower()
        picked = value.strip().splitlines()
        picked = picked[0].strip() if picked else ""
        if key in {"path", "file", "dir", "directory", "command", "cmd", "query", "text"} and not path:
            path = picked
        elif key in {"content", "contents", "body"}:
            body = value.strip("\n")
    if command == "WRITE":
        safe = tool_path(path)
        if not safe:
            return ""
        return "WRITE " + safe + "\n" + body + "\nENDWRITE"
    if not path:
        return ""
    return command + " " + path


def tool_response(text: str) -> str:
    body = (text or "").strip()[:1800]
    return "<tool_response>\n" + body + "\n</tool_response>"


def first_tool(answer: str) -> str:
    for line in answer.splitlines():
        stripped = clean_tool_line(line)
        if stripped == "SESSIONS" or stripped.startswith(("LIST ", "GREP ", "READ ", "MKDIR ", "RUN ", "WRITE ", "SESSIONS ")):
            return stripped
    return ""


def run_agent(prompt: str) -> str:
    logs: list[str] = []
    last = ""
    for turn in range(4):
        context = TOOL_HELP + "\n\nTask:\n" + prompt
        if turn == 0:
            context += "\n\n" + workspace_note()
        if last:
            context += "\n\n" + tool_response(last)
        answer = visible_answer(ask_model(context, max_tokens=1000))
        native = native_tool(answer)
        # Qwen's <tool_call> is the action. A sentence before it is not a second command.
        tool = (native.splitlines()[0] if native else "") or first_tool(answer)
        if tool:
            print("tool " + tool.splitlines()[0][:120], flush=True)
        if not tool:
            return "\n".join([*logs, answer]).strip()[:4000]
        if tool.startswith("WRITE "):
            wrote = apply_actions(native or answer)
            logs.extend(wrote)
            last = "\n".join(wrote)
            continue
        if tool.startswith("LIST "):
            result = tool_list(tool[5:])
        elif tool.startswith("GREP "):
            result = tool_grep(tool[5:])
        elif tool.startswith("READ "):
            result = safe_read(tool[5:])
        elif tool.startswith("MKDIR "):
            made = apply_actions(tool)
            logs.extend(made)
            result = "\n".join(made) or "mkdir"
        elif tool.startswith("RUN "):
            result = tool_run(tool[4:])
        elif tool == "SESSIONS" or tool.startswith("SESSIONS"):
            result = "\n".join(recent_sessions(8)) or "no saved session yet"
        else:
            return answer[:4000]
        logs.append(tool + " -> " + (result.splitlines()[0] if result else "empty")[:140])
        last = result
    return "\n".join(logs)[:4000]


def worker_main() -> None:
    pid = os.getpid()
    print(f"Neuro Code {pid} on one shared model, {SLOTS} generate at once", flush=True)
    while not STOP_PATH.exists():
        job = claim(pid)
        if job is None:
            time.sleep(0.8)
            continue
        print(f"agent {pid} job {job['id']} {job['title']}", flush=True)
        try:
            result = run_agent(str(job["prompt"]))
            finish(int(job["id"]), pid, result=result)
            print(f"agent {pid} done job {job['id']}", flush=True)
        except (OSError, urllib.error.URLError, RuntimeError, KeyError, json.JSONDecodeError, TimeoutError) as exc:
            if "training" in str(exc).lower():
                now = time.time()
                with connect() as db:
                    db.execute(
                        "UPDATE jobs SET status='queued', worker_pid=NULL, updated=? WHERE id=? AND status='running'",
                        (now, int(job["id"])),
                    )
                    db.execute(
                        "UPDATE workers SET state='waiting', job_id=NULL, heartbeat=? WHERE pid=?",
                        (now, pid),
                    )
                print(f"agent {pid} waits, the model is training", flush=True)
                time.sleep(30)
                continue
            finish(int(job["id"]), pid, error=str(exc))
            print(f"agent {pid} error job {job['id']}: {exc}", flush=True)
    with connect() as db:
        db.execute("DELETE FROM workers WHERE pid=?", (pid,))
    print(f"Neuro agent {pid} stopped", flush=True)


def spawn(requested: int, *, exact: bool = False) -> dict:
    if STOP_PATH.exists():
        STOP_PATH.unlink()
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
            [sys.executable, "-u", str(ROOT / "mesh.py"), "agent"],
            cwd=str(ROOT),
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
    if STOP_PATH.exists():
        STOP_PATH.unlink()
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
    return "The model did not answer in time."


BRAIN = (
    ("   .-~^~-.  ", "  (  .  .  ) ", "   `-._.-'  "),
    ("   .-~^~-.  ", "  ( .    . ) ", "   `-._.-'  "),
    ("   .-~^~-.  ", "  (  .  .  ) ", "    `-.-'   "),
    ("   .-^~^-.  ", "  ( .    . ) ", "   `-._.-'  "),
)


def code_screen(snap: dict, typed: str, note: str, beat: int = 0) -> str:
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
    folder = LOG_DIR / "drill"
    folder.mkdir(parents=True, exist_ok=True)
    relative = f"agent-logs/drill/round-{os.getpid()}.txt"
    wrote = safe_write(relative, f"round {index}")
    read = safe_read(relative)
    listed = tool_list("agent-logs/drill")
    secret = safe_write("cluster.token", "nope")
    numbered = first_tool("1. LIST workspace")
    native = native_tool("<tool_call><function=grep><parameter=query>def drill_round</parameter></function></tool_call>")
    searched = tool_grep("def drill_round")
    acted = apply_actions(
        "WRITE workspace/agent-check/drill-one.txt\nonly-one\nENDWRITE\nWRITE workspace/agent-check/drill-two.txt\nnope\nENDWRITE"
    )
    second = safe_read("workspace/agent-check/drill-two.txt")
    escape = safe_write("../outside.txt", "nope")
    made = apply_actions("MKDIR workspace/agent-check/drill-dir")
    ran = tool_run("echo drill-ok")
    blocked_run = tool_run("rm -rf /")
    ok = (
        wrote.startswith("wrote ")
        and f"round {index}" in read
        and f"round-{os.getpid()}.txt" in listed
        and secret.startswith("blocked")
        and numbered == "LIST workspace"
        and native == "GREP def drill_round"
        and "agents.py" in searched
        and acted == ["wrote workspace/agent-check/drill-one.txt"]
        and second == "missing"
        and escape.startswith("blocked")
        and made == ["mkdir workspace/agent-check/drill-dir"]
        and "drill-ok" in ran
        and blocked_run == "blocked"
        and tool_response("only-one").startswith("<tool_response>")
        and "only-one" in tool_response("only-one")
        and native_tool("<tool_call><function=write_file><parameter=path></parameter><parameter=content>nope</parameter></function></tool_call>") == ""
    )
    row = {"t": time.strftime("%Y-%m-%d %H:%M:%S", time.localtime()), "round": index, "ok": ok}
    append_capped(LOG_DIR / "drill.jsonl", json.dumps(row))
    return row


def drill_main(rounds: int = 10) -> None:
    count = max(1, min(int(rounds), 50))
    passed = 0
    for index in range(1, count + 1):
        row = drill_round(index)
        passed += 1 if row["ok"] else 0
        print(f"drill {index} {'ok' if row['ok'] else 'fail'}", flush=True)
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
