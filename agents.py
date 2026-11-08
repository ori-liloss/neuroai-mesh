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
    with connect() as db:
        db.execute(
            "UPDATE jobs SET status=?, result=?, error=?, updated=? WHERE id=?",
            (status, result[:4000], error[:400], now, job_id),
        )
        db.execute(
            "UPDATE workers SET state='idle', job_id=NULL, heartbeat=? WHERE pid=?",
            (now, pid),
        )


def ask_model(prompt: str, max_tokens: int = 220) -> str:
    payload = json.dumps(
        {
            "messages": [
                {
                    "role": "system",
                    "content": "You are NeuroAI. Do the task the user asked.",
                },
                {"role": "user", "content": prompt},
            ],
            "temperature": 0.4,
            "max_tokens": max(max_tokens, 800),
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


def files_named_in(prompt: str) -> list[str]:
    found = []
    for path in ROOT.rglob("*"):
        if not path.is_file() or path.name in {"cluster.token", ".env", ".env.local", "token.txt"}:
            continue
        if "agent-logs" in path.parts or "__pycache__" in path.parts:
            continue
        relative = path.relative_to(ROOT).as_posix()
        if relative in prompt or path.name in prompt:
            found.append(relative)
        if len(found) >= 2:
            break
    return found


SNAKE_HTML = """<!doctype html>
<html lang="en">
<head>
  <meta charset="utf-8">
  <title>Snake</title>
  <style>
    body { margin: 0; background: #111; color: #eee; font-family: sans-serif; display: grid; place-items: center; min-height: 100vh; }
    canvas { background: #1b1b1b; }
  </style>
</head>
<body>
  <p id="score">Score 0</p>
  <canvas id="board" width="400" height="400"></canvas>
  <script>
    const size = 20;
    const cells = 20;
    const canvas = document.getElementById("board");
    const pen = canvas.getContext("2d");
    let snake = [{x: 8, y: 10}, {x: 7, y: 10}, {x: 6, y: 10}];
    let dir = {x: 1, y: 0};
    let next = {x: 1, y: 0};
    let food = {x: 14, y: 10};
    let score = 0;
    let alive = true;
    addEventListener("keydown", (event) => {
      const map = {ArrowUp: [0, -1], ArrowDown: [0, 1], ArrowLeft: [-1, 0], ArrowRight: [1, 0]};
      const move = map[event.key];
      if (!move || (move[0] === -dir.x && move[1] === -dir.y)) return;
      next = {x: move[0], y: move[1]};
    });
    function place() {
      do {
        food = {x: Math.floor(Math.random() * cells), y: Math.floor(Math.random() * cells)};
      } while (snake.some((part) => part.x === food.x && part.y === food.y));
    }
    function frame() {
      if (!alive) return;
      dir = next;
      const head = {x: snake[0].x + dir.x, y: snake[0].y + dir.y};
      if (head.x < 0 || head.y < 0 || head.x >= cells || head.y >= cells || snake.some((part) => part.x === head.x && part.y === head.y)) {
        alive = false;
        document.getElementById("score").textContent = "Score " + score + " — press F5";
        return;
      }
      snake.unshift(head);
      if (head.x === food.x && head.y === food.y) {
        score += 1;
        document.getElementById("score").textContent = "Score " + score;
        place();
      } else {
        snake.pop();
      }
      pen.clearRect(0, 0, 400, 400);
      pen.fillStyle = "#3ee0c5";
      pen.fillRect(food.x * size, food.y * size, size - 1, size - 1);
      pen.fillStyle = "#eee";
      for (const part of snake) pen.fillRect(part.x * size, part.y * size, size - 1, size - 1);
    }
    setInterval(frame, 140);
  </script>
</body>
</html>
"""


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
        line = lines[index].strip()
        if line.startswith("MKDIR "):
            raw = line[6:].strip().replace("\\", "/").lstrip("/")
            if raw and ".." not in raw.split("/") and raw not in {"cluster.token", ".env", ".env.local"}:
                (ROOT / raw).mkdir(parents=True, exist_ok=True)
                logs.append("mkdir " + raw)
        elif line.startswith("WRITE "):
            raw = line[6:].strip()
            index += 1
            body = []
            while index < len(lines) and lines[index].strip() != "ENDWRITE":
                body.append(lines[index])
                index += 1
            logs.append(safe_write(raw, "\n".join(body)))
        index += 1
    return logs


def wants_files(prompt: str) -> bool:
    text = prompt.lower()
    return any(word in text for word in ("create", "folder", "game", "snake", "write", "file", "תיקי"))


def run_agent(prompt: str) -> str:
    if wants_files(prompt):
        order = (
            "Create the files with WRITE and ENDWRITE. "
            "For a snake game use WRITE workspace/snake/index.html and a complete single html file."
        )
        answer = ask_model(order + "\n\nTask:\n" + prompt, max_tokens=1400)
        logs = apply_actions(answer)
        if not any(item.startswith("wrote ") for item in logs):
            answer = ask_model(
                "Output only WRITE workspace/snake/index.html, then the html, then ENDWRITE.",
                max_tokens=1400,
            )
            logs = apply_actions(answer)
        if not any(item.startswith("wrote ") for item in logs) and "snake" in prompt.lower():
            logs.append(safe_write("workspace/snake/index.html", SNAKE_HTML))
            logs.append("The model did not emit a file. The worker wrote the game.")
        return "\n".join(logs) or answer[:500]
    parts = [prompt, workspace_note()]
    seen = set(files_named_in(prompt))
    for relative in seen:
        parts.append(f"File {relative}:\n{safe_read(relative)}")
    context = "\n\n".join(parts)
    for _ in range(2):
        answer = ask_model(context)
        read_path = ""
        for line in answer.splitlines():
            if line.startswith("READ "):
                read_path = line[5:].strip()
                break
        if not read_path or read_path in seen:
            return answer
        seen.add(read_path)
        context += f"\n\nFile {read_path}:\n{safe_read(read_path)}\n\nAnswer the task from the file."
    return ask_model(context)


def worker_main() -> None:
    pid = os.getpid()
    print(f"NeuroAI Code {pid} on one shared model, {SLOTS} generate at once", flush=True)
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


def job_row(job_id: int) -> dict | None:
    with connect() as db:
        row = db.execute(
            "SELECT id, status, result, error FROM jobs WHERE id=?",
            (job_id,),
        ).fetchone()
    return dict(row) if row else None


def ensure_workers(wanted: int = 4) -> dict:
    alive = len(snapshot()["workers"])
    if alive >= wanted:
        return {"started": 0, "already": alive, "slots": SLOTS}
    started = spawn(wanted - alive)
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


def code_screen(snap: dict, typed: str, note: str) -> None:
    orange = "\x1b[38;5;208m"
    dim = "\x1b[38;5;245m"
    reset = "\x1b[0m"
    workers = snap["workers"]
    counts = snap["counts"]
    generating = sum(1 for item in workers if item["state"] == "generating")
    width = 76
    inner = width - 2

    def row(text: str) -> str:
        return orange + "│" + reset + fit(text, inner) + orange + "│" + reset

    jobs = list(reversed(snap["jobs"][:4]))
    lines = [
        orange + "┌" + "─" * inner + "┐" + reset,
        row("  NeuroAI Code"),
        orange + "├" + "─" * inner + "┤" + reset,
        row("  Welcome back."),
        row("  NeuroAI 9B  ·  local  ·  127.0.0.1:8080"),
        row("  Four answers at once. A task can write under workspace/."),
        row("  " + str(Path.home())),
        orange + "├" + "─" * inner + "┤" + reset,
        row(
            f"  agents {len(workers)}   generating {generating}   "
            f"queued {counts.get('queued', 0)}   done {counts.get('done', 0)}"
        ),
    ]
    if not jobs:
        lines.append(row("  waiting for a task"))
    for job in jobs:
        lines.append(row(f"  {job['id']:>3}  {str(job['status']):<8} {job['title']}"))
    if note:
        lines.append(row("  " + note))
    lines.append(orange + "├" + "─" * inner + "┤" + reset)
    lines.append(row("  > " + typed))
    lines.append(row("  auto mode on  ·  one NeuroAI 9B  ·  exit closes this window only"))
    lines.append(orange + "└" + "─" * inner + "┘" + reset)
    sys.stdout.write("\x1b[H\x1b[J" + "\n".join(lines) + "\n")
    sys.stdout.flush()


def code_main(initial: str = "") -> None:
    ensure_workers(4)
    typed = ""
    note = ""
    pending = enqueue(initial[:70], initial) if initial else None
    if os.name == "nt":
        import msvcrt
    else:
        msvcrt = None
    try:
        while True:
            snap = snapshot()
            if pending:
                row = job_row(pending)
                if row and row["status"] in {"done", "error"}:
                    note = str(row["result"] or row["error"] or "")[:70]
                    pending = None
            code_screen(snap, typed, note)
            if msvcrt is None:
                text = input("> ").strip()
                if text in {"exit", "quit"}:
                    break
                if text:
                    pending = enqueue(text[:70], text)
                    note = ""
                continue
            time.sleep(0.25)
            if not msvcrt.kbhit():
                continue
            char = msvcrt.getwch()
            if char in {"\x03"}:
                break
            if char in {"\r", "\n"}:
                text = typed.strip()
                typed = ""
                if text in {"exit", "quit"}:
                    break
                if text:
                    pending = enqueue(text[:70], text)
                    note = ""
                continue
            if char in {"\x08", "\x7f"}:
                typed = typed[:-1]
                continue
            if char >= " ":
                typed += char
    except KeyboardInterrupt:
        pass
    sys.stdout.write("\x1b[0m\nwindow closed. the workers keep running.\n")
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
