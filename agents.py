"""CMD workers for NeuroAI Mesh.

One llama-server holds the model. Workers are separate Python processes.
Only a few of them may generate at once, because the GPU has four slots.
"""

from __future__ import annotations

import ctypes
import json
import os
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
    room = max(0, free_mib() - 4096) // 45
    return max(1, min(int(requested), int(room), 24))


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
        ("Chat idle", "Explain in two sentences why the GPU can sit near zero while NeuroAI is still loaded."),
        ("Linux command", "Give the one command that starts a Linux worker from the neuro-cluster folder."),
        ("Token", "Say where cluster.token must sit, and why it must stay off the public internet."),
        ("Slots", "Explain why many CMD workers share one model and only four generate at once."),
    ]
    for title, prompt in tasks:
        enqueue(title, prompt)
    return len(tasks)


def reap(db: sqlite3.Connection) -> None:
    cutoff = time.time() - 20
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
    with connect() as db:
        db.execute(
            "UPDATE jobs SET status=?, result=?, error=?, updated=? WHERE id=?",
            (status, result[:4000], error[:400], now, job_id),
        )
        db.execute(
            "UPDATE workers SET state='idle', job_id=NULL, heartbeat=? WHERE pid=?",
            (now, pid),
        )


def ask_model(prompt: str) -> str:
    payload = json.dumps(
        {
            "messages": [
                {
                    "role": "system",
                    "content": "You are NeuroAI 1.0, a local mesh worker. Be concrete. Stay under 70 words.",
                },
                {"role": "user", "content": prompt},
            ],
            "temperature": 0.4,
            "max_tokens": 220,
            "chat_template_kwargs": {"enable_thinking": False},
        }
    ).encode("utf-8")
    request = urllib.request.Request(
        CHAT_URL,
        data=payload,
        headers={"Content-Type": "application/json"},
    )
    with urllib.request.urlopen(request, timeout=120) as response:
        body = json.loads(response.read().decode("utf-8"))
    return (body["choices"][0]["message"].get("content") or "").strip()


def worker_main() -> None:
    pid = os.getpid()
    print(f"NeuroAI agent {pid} on one shared model, {SLOTS} generate at once", flush=True)
    while not STOP_PATH.exists():
        job = claim(pid)
        if job is None:
            time.sleep(0.8)
            continue
        print(f"agent {pid} job {job['id']} {job['title']}", flush=True)
        try:
            result = ask_model(str(job["prompt"]))
            finish(int(job["id"]), pid, result=result)
            print(f"agent {pid} done job {job['id']}", flush=True)
        except (OSError, urllib.error.URLError, KeyError, json.JSONDecodeError, TimeoutError) as exc:
            finish(int(job["id"]), pid, error=str(exc))
            print(f"agent {pid} error job {job['id']}: {exc}", flush=True)
    with connect() as db:
        db.execute("DELETE FROM workers WHERE pid=?", (pid,))
    print(f"NeuroAI agent {pid} stopped", flush=True)


def spawn(requested: int) -> dict:
    if STOP_PATH.exists():
        STOP_PATH.unlink()
    count = worker_cap(requested)
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
        row("  NEUROAI  1.0"),
        row("  Agent console  ·  one local model"),
        "├" + "─" * inner + "┤",
        row("  Model   NeuroAI 9B"),
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
    lines.append(row("  > shared NeuroAI 9B"))
    lines.append("└" + "─" * inner + "┘")
    sys.stdout.write("\x1b[H\x1b[J" + "\n".join(lines) + "\n")
    sys.stdout.flush()


def terminal(workers: int = 4) -> None:
    seeded = seed()
    started = spawn(workers)
    print(f"seeded {seeded}", flush=True)
    try:
        while True:
            draw_terminal(snapshot(), started)
            time.sleep(1.5)
    except KeyboardInterrupt:
        sys.stdout.write("\nstopping workers\n")
        sys.stdout.flush()
        stop_workers()
