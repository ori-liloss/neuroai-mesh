"""In-process self-test for mesh.py. Binds only ephemeral ports on 127.0.0.1.

    python tests/test_mesh.py

It never touches the real cluster.token, the queue or the training folder."""
import json
import os
import sys
import tempfile
import threading
import time
import urllib.error
import urllib.request
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO))
import mesh  # noqa: E402

FAKE = "unit-test-token"
REAL_TOKEN = mesh.token  # only used on temp paths in the pairing tests
mesh.token = lambda: FAKE  # never touch the real cluster.token
passed = 0


def ok(cond, label):
    global passed
    if not cond:
        raise AssertionError(label)
    passed += 1
    print("ok  ", label)


# --- network filters -------------------------------------------------------
ok(mesh.private_lan("192.168.7.247"), "192.168.7.x is LAN")
ok(not mesh.private_lan("26.10.2.3"), "26.x never LAN")
ok(not mesh.private_lan("192.168.56.1"), "192.168.56.x ignored")
ok(not mesh.private_lan("169.254.3.4"), "169.254 ignored")
ok(not mesh.private_lan("8.8.8.8"), "public not LAN")
ok(mesh.lan_client("127.0.0.1") and not mesh.lan_client("26.1.1.1"), "client filter")
hosts = mesh.sweep_hosts(["192.168.7", "26.0.0", "192.168.56"], {"192.168.7.14"})
ok(len(hosts) == 253 and all(h.startswith("192.168.7.") for h in hosts), "sweep only 192.168.7 minus own")
ipconfig = """
Ethernet adapter Radmin VPN:
   IPv4 Address. . . . . . . . . . . : 26.44.1.2
   Subnet Mask . . . . . . . . . . . : 255.0.0.0
Ethernet adapter VirtualBox Host-Only Network:
   IPv4 Address. . . . . . . . . . . : 192.168.56.1
   Subnet Mask . . . . . . . . . . . : 255.255.255.0
Ethernet adapter Ethernet:
   IPv4 Address. . . . . . . . . . . : 192.168.7.14(Preferred)
   Subnet Mask . . . . . . . . . . . : 255.255.255.0
"""
prefixes, own = mesh.parse_ipconfig(ipconfig)
ok(prefixes == ["192.168.7"] and "192.168.7.14" in own, f"ipconfig parse {prefixes}")
ok(mesh.browser_host_ok("192.168.7.14:8780") and mesh.browser_host_ok("localhost:8780") and mesh.browser_host_ok("ori:8780"), "host ok")
ok(not mesh.browser_host_ok("evil.example.com:8780"), "rebinding host refused")
ok(mesh.origin_ok(None, "x") and mesh.origin_ok("http://192.168.7.14:8780", "192.168.7.14:8780"), "same origin ok")
ok(not mesh.origin_ok("http://evil.example.com", "192.168.7.14:8780") and not mesh.origin_ok("null", "a"), "cross origin refused")

# --- dedupe and misses -------------------------------------------------------
mac = lambda url, name: {"url": url, "ok": True, "status": {"name": name, "os": "darwin"}}
items = [mac("http://192.168.7.247:8781", "Mac-mini-sl-ai"), mac("http://192.168.7.248:8781", "Mac-mini-sl-ai.local"),
         {"url": "http://192.168.7.14:8781", "ok": True, "status": {"name": "ORI", "os": "windows"}},
         {"url": "http://192.168.7.31:8781", "ok": True, "status": {"name": "LAPTOP-M906OEFU", "os": "windows"}}]
settled = mesh.settle_peers(items, set(), {"http://192.168.7.247:8781": 2}, ("ori", "windows"))
urls = sorted(i["url"] for i in settled)
ok(urls == ["http://192.168.7.248:8781", "http://192.168.7.31:8781"], f"Mac once, fresher address wins, ORI skipped {urls}")
settled = mesh.settle_peers(items[:2], {"http://192.168.7.247:8781"}, {}, None)
ok(len(settled) == 1 and settled[0]["url"].endswith(".247:8781"), "tie keeps the shown address")
hold = {}
mesh.count_misses(hold, items[:1], 0)
for n in range(3):
    mesh.count_misses(hold, [], n + 1)
ok(hold["http://192.168.7.247:8781"]["misses"] == 3, "kept through 3 misses")
mesh.count_misses(hold, [], 5)
ok(not hold, "dropped after the 4th miss")

# --- process safety --------------------------------------------------------
ok(mesh.process_role("python.exe", 999999, set(), True) == ("python", False), "python guarded while lock busy")
ok(mesh.process_role("python.exe", 999999, set(), False) == ("python", True), "python stoppable when free")
ok(mesh.process_role("svchost.exe", 5, set(), False)[1] is False, "svchost protected")
ok(mesh.process_role("powershell.exe", 5, set(), False)[1] is False, "console host protected")
ok(mesh.process_role("x.exe", os.getpid(), set(), False) == ("page", False), "own page protected")
ok(mesh.process_role("python.exe", 4242, {4242}, False) == ("training", False), "lock pid protected")
ok(mesh.process_role("llama-server.exe", 77, set(), True) == ("model", True), "model stoppable")
cards = mesh.parse_nvidia("NVIDIA GeForce RTX 3050 Laptop GPU, [N/A], 6144, [N/A]\n")
ok(cards[0]["memory_total_mib"] == 6144 and cards[0]["temperature_c"] is None, "nvidia [N/A] tolerated")

# --- training state --------------------------------------------------------
with tempfile.TemporaryDirectory() as tmp:
    ok(mesh.newest_log(Path(tmp) / "missing")[2] is True, "missing log dir is readable-empty")
    real_lock, real_newest = mesh.lock_held, mesh.newest_log
    mesh.lock_held = lambda path: False
    mesh.newest_log = lambda d, p="neuro-ai-long-v1-": (None, 0.0, False)
    try:
        view = mesh.live_training(Path(tmp))
        ok(view["running"] is None and view["log_readable"] is False, "unreadable logs + free lock -> unknown, not dead")
    finally:
        mesh.lock_held, mesh.newest_log = real_lock, real_newest
r = mesh.newest_log(mesh.ROOT.parent / "neuro-ai-data" / "training" / "logs")
print("     the default training logs folder is readable here =", r[2])
tv = mesh.training_view()
print("     training_view on this PC:", {k: tv[k] for k in ("running", "state", "log_readable", "step")})
tail = "{'loss': '0.8123', 'epoch': 1}\n  4500/6933 [..]\n eval 12/120\n"
p = mesh.parse_log_tail(tail)
ok(p["step"] == 4500 and p["max_steps"] == 6933 and p["loss"] == 0.8123, f"eval bar ignored {p}")

# --- training log: configurable source, tail parsing, states -----------------
ok(mesh.clock_seconds("20:31:00") == 73860 and mesh.clock_seconds("10:05") == 605 and mesh.clock_seconds("?") is None, "tqdm clock parsed")
LOG_HEAD = ("2026-11-09T17:10:00 start\r\n2026-11-09T17:10:00 Preflight.\r\n{\r\n  \"max_length\": 1536\r\n}\r\n"
            "2026-11-09T17:13:34 Training. Run folder: D:\\runs\\demo-run \r\n")
LOADING = "Loading weights:   0%|          | 1/760 [00:21<4:33:29, 21.62s/it]\rLoading weights:  17%|█▋        | 132/760 [01:04<01:02, 10.09it/s]"
TRAIN = ("\r  0%|          | 0/14688 [00:00<?, ?it/s]\r  1%|          | 119/14688 [10:00<20:31:10, 5.07s/it]\n"
         "{'loss': '1.037', 'grad_norm': '0.53', 'learning_rate': '6.5e-06', 'epoch': '0.004'}\n"
         "\r  1%|          | 120/14688 [10:05<20:31:00, 5.07s/it]\n{'loss': '0.912', 'grad_norm': '0.5', 'learning_rate': '7e-06', 'epoch': '0.008'}\n")
EVAL = "\r 40%|████      | 31/77 [00:20<00:30, 1.5it/s]\n{'eval_loss': '0.8431', 'eval_runtime': '50.1', 'epoch': '0.01'}\n"
info = mesh.parse_training_log(LOG_HEAD + LOADING)
ok(info["phase"] == "loading" and info["step"] is None and info["load"]["total"] == 760 and info["load"]["step"] == 132, f"loading bar is not a train step {info['load']}")
ok(info["run_dir"] == "D:\\runs\\demo-run" and info["started"] == "2026-11-09T17:10:00", "run folder and start time from runner lines")
info = mesh.parse_training_log(LOG_HEAD + LOADING + TRAIN)
ok(info["step"] == 120 and info["max_steps"] == 14688 and info["phase"] == "training", f"train bar after \\r {info['step']}/{info['max_steps']}")
ok(info["s_per_it"] == 5.07 and info["eta_s"] == 73860 and info["elapsed_s"] == 605, "s/it, ETA and elapsed from the tqdm bar")
ok(info["loss"] == 0.912 and info["history"] == [1.037, 0.912] and info["learning_rate"] == 7e-06 and info["epoch"] == 0.008, "loss, history, lr and epoch from Trainer dicts")
info = mesh.parse_training_log(LOG_HEAD + TRAIN + EVAL)
ok(info["step"] == 120 and info["max_steps"] == 14688 and info["eval_loss"] == 0.8431 and info["eval_history"] == [0.8431], "eval bar ignored for steps, eval_loss read")
ok(mesh.parse_training_log("", EVAL, 14688)["step"] is None, "a lone eval bar is not taken for the train bar when max_steps is known")
ok(mesh.parse_training_log("", "\r  2%| | 200/9000 [1:00<2:00:00, 0.5it/s]")["s_per_it"] == 2.0, "it/s turned into s/it")
TRACE = "Traceback (most recent call last):\n  File \"train.py\", line 9, in <module>\ntorch.OutOfMemoryError: CUDA out of memory. Tried to allocate 1.20 GiB\n"
info = mesh.parse_training_log(LOG_HEAD + TRAIN + TRACE)
ok(info["error"].startswith("torch.OutOfMemoryError: CUDA out of memory"), f"traceback -> error line {info['error']!r}")
ok(not mesh.parse_training_log(LOG_HEAD + TRAIN + TRACE + "2026-11-09T18:00:00 start\n")["error"], "a new start clears an old traceback")
ok(not mesh.parse_training_log(TRAIN + "C:\\x\\ops.py:213: FutureWarning: soon\n")["error"], "a warning is not an error")
ok(mesh.parse_training_log(LOG_HEAD + TRAIN + "{'train_runtime': '9000.1', 'train_loss': '0.5', 'epoch': '2.0'}\n")["finished"], "train_runtime dict -> finished")
ok(len(mesh.parse_training_log("", "".join(f"{{'loss': '{n / 100}'}}\n" for n in range(100)))["history"]) == 40, "loss history capped at 40")

with tempfile.TemporaryDirectory() as tmp:
    big = Path(tmp) / "train.log"
    filler = "".join(f"\r  {n // 147}%|          | {n}/14688 [10:00<20:31:10, 5.07s/it]" for n in range(1, 14000))
    big.write_text(LOG_HEAD + filler + "\n{'loss': '0.4242', 'epoch': '1.9'}\n", encoding="utf-8")
    head, tail_text = mesh.read_log(big)
    ok(big.stat().st_size > 500_000 and len(head) <= mesh.LOG_HEAD_BYTES and len(tail_text) <= mesh.LOG_TAIL_BYTES, "only the head and the tail of a big log are read")
    info = mesh.parse_training_log(head, tail_text)
    ok(info["step"] == 13999 and info["loss"] == 0.4242 and info["run_dir"] == "D:\\runs\\demo-run", "step from the tail, run folder from the head")
    small = Path(tmp) / "small.log"
    small.write_text(LOG_HEAD + LOADING + TRAIN, encoding="utf-8")
    fresh = (small, time.time(), True)
    old = (small, time.time() - 3600, True)
    real = (mesh.lock_held, mesh.trainer_alive)
    try:
        mesh.lock_held = lambda path: False
        mesh.trainer_alive = lambda: False
        view = mesh.live_training(Path(tmp), fresh)
        ok(view["running"] is True and view["state"] == "running" and view["step"] == 120 and view["updated_s"] < 5, "fresh log -> running")
        view = mesh.live_training(Path(tmp), old)
        ok(view["running"] is False and view["state"] == "stopped" and view["updated_s"] >= 3600, "old log, no trainer, free lock -> stopped")
        mesh.lock_held = lambda path: True
        view = mesh.live_training(Path(tmp), old)
        ok(view["state"] == "stopped_lock_held" and view["stale_lock"] is True and view["running"] is False, "old log, lock held, no trainer -> stale lock")
        mesh.trainer_alive = lambda: None
        view = mesh.live_training(Path(tmp), old)
        ok(view["running"] is True and view["quiet"] is True, "lock held, trainer hidden (SYSTEM) -> still running, log quiet")
        loading = Path(tmp) / "loading.log"
        loading.write_text(LOG_HEAD + LOADING, encoding="utf-8")
        ok(mesh.live_training(Path(tmp), (loading, time.time(), True))["state"] == "loading", "fresh log with only loading bars -> loading")
        crashed = Path(tmp) / "crashed.log"
        crashed.write_text(LOG_HEAD + TRAIN + TRACE, encoding="utf-8")
        view = mesh.live_training(Path(tmp), (crashed, time.time(), True))
        ok(view["state"] == "error" and view["running"] is False and "OutOfMemory" in view["error"], "traceback -> error even while fresh")
        ok(mesh.live_training(Path(tmp), (None, 0.0, True))["state"] in {"running", "stopped_lock_held"}, "missing configured log falls back to the lock")
    finally:
        mesh.lock_held, mesh.trainer_alive = real

    real_cfg = mesh.TRAINING_CONFIG
    cfg = Path(tmp) / "training.json"
    try:
        mesh.TRAINING_CONFIG = cfg
        ok(mesh.training_config() == {}, "no training.json -> empty config")
        cfg.write_text("{broken", encoding="utf-8")
        ok(mesh.training_config() == {}, "broken training.json -> empty config")
        cfg.write_text(json.dumps({"log": str(small), "run_dir": str(Path(tmp) / "run"), "evil": "x", "name": 5}), encoding="utf-8")
        got = mesh.training_config()
        ok(got == {"log": str(small), "run_dir": str(Path(tmp) / "run")}, f"only known string keys are read {got}")
        os.environ.pop("NEURO_TRAINING_LOG", None)
        ok(mesh.configured_log()[0] == small, "training.json names the log")
        os.environ["NEURO_TRAINING_LOG"] = str(big)
        ok(mesh.configured_log()[0] == big, "NEURO_TRAINING_LOG wins over training.json")
        os.environ.pop("NEURO_TRAINING_LOG", None)
        ckpt = Path(tmp) / "run" / "checkpoint-200"
        ckpt.mkdir(parents=True)
        (Path(tmp) / "run" / "checkpoint-100").mkdir()
        (Path(tmp) / "run" / "checkpoint-100" / "trainer_state.json").write_text(json.dumps({"global_step": 100, "max_steps": 14688, "log_history": [{"loss": 2.0}]}), encoding="utf-8")
        (ckpt / "trainer_state.json").write_text(json.dumps({"global_step": 200, "max_steps": 14688, "log_history": [{"loss": 1.5}, {"eval_loss": 1.2}, {"loss": 1.25}]}), encoding="utf-8")
        state = mesh.run_state(Path(tmp) / "run")
        ok(state["step"] == 200 and state["history"] == [1.5, 1.25] and state["checkpoint"] == "checkpoint-200", "newest checkpoint's trainer_state.json read")
        mesh._training_cache["at"] = 0.0
        real_lock = mesh.lock_held
        mesh.lock_held = lambda path: False
        try:
            tv = mesh.training_view()
        finally:
            mesh.lock_held = real_lock
            mesh._training_cache["at"] = 0.0
        ok(tv["configured"] and tv["log"] == str(small) and tv["run"] == "run" and tv["step"] == 120 and tv["eta_s"] == 73860, f"training_view uses training.json {tv['state']}")
        ok({"state", "phase", "history", "eval_loss", "s_per_it", "updated_s", "run_dir", "error"} <= set(tv), "training_view sends the new fields")
    finally:
        mesh.TRAINING_CONFIG = real_cfg
        mesh._training_cache["at"] = 0.0
repo = Path(mesh.ROOT)
example = json.loads((repo / "training.example.json").read_text(encoding="utf-8"))
ok({"log", "run_dir"} <= set(example), "training.example.json ships log and run_dir")
ok("training.json" in (repo / ".gitignore").read_text(encoding="utf-8").split(), "training.json is gitignored")

# --- public repo: no author-only defaults -------------------------------------
mesh_src = (repo / "mesh.py").read_text(encoding="utf-8")
ok("192.168.7." not in mesh_src and "liliai" not in mesh_src.lower() and "neuro-ai-long" not in mesh_src, "mesh.py has no home LAN address, D:\\liliai path or old run name")
ok(mesh.lan_url(mesh.DEFAULT_MESH_URL) == "http://127.0.0.1:8780", "last-fallback main PC is this computer, not a fixed LAN address")
deck_src = (repo / "platforms" / "steamdeck.sh").read_text(encoding="utf-8")
ok("192.168.7." not in deck_src and "NEURO_MESH_URL=${NEURO_MESH_URL:-" not in deck_src, "steamdeck.sh lets discovery find the main PC")
ok(mesh.GPU_LOCK.name == "gpu.lock" and "liliai" not in (repo / "side_train.py").read_text(encoding="utf-8").lower(), "GPU lock and side_train paths are not hardcoded")
for name in ("plasmashell", "kwin_wayland", "Xwayland", "explorer.exe", "WindowServer", "konsole"):
    ok(mesh.process_role(name, 4321, set(), False)[1] is False, f"{name} never gets a Stop button")
ok(mesh.process_role("steamwebhelper", 4321, set(), False)[1] is True, "an ordinary app keeps its Stop button")

# --- page: grouped processes, no Stop for desktop processes, new training view ---
page = mesh.DASHBOARD.read_text(encoding="utf-8")
ok("innerHTML" not in page and "insertAdjacentHTML" not in page, "page builds every node with textContent / createElement")
ok("function groupProcs(" in page and '"plasmashell"' in page and '"kwin_wayland"' in page and "SYSTEM_APPS.has(" in page, "page groups same-name processes and hides Stop for desktop processes")
ok(all(f'id="{element}"' in page for element in ("train-chip", "train-bar", "train-eta", "train-speed", "train-age", "losses", "eval-losses", "train-error")), "training view shows state, progress, ETA, speed, age, sparklines and errors")
ok("neuro-ai-long" not in page and "start-train" not in page and "cdn" not in page.lower(), "page has no author run name, no dead Start button, no CDN")

# --- side jobs ---------------------------------------------------------------
old = {"name": "OhProggraming", "os": "windows", "ram": {"total_mib": 16000, "free_mib": 9000}, "gpus": []}
jobs = {j["job"] for j in mesh.side_jobs(old, False, True, True)}
ok({"verify", "dedup"} <= jobs and "eval" not in jobs, f"old worker without caps {jobs}")
new = {"name": "LAPTOP-M906OEFU", "os": "windows", "ram": {"total_mib": 16000, "free_mib": 8000},
       "gpus": [{"name": "RTX 3050", "memory_total_mib": 6144, "memory_used_mib": 300}],
       "caps": {"cpu_cores": 12, "free_ram_mib": 8000, "vram_total_mib": 6144, "vram_free_mib": 5844, "cpu_busy_pct": 5}}
fit = {j["job"]: j for j in mesh.side_jobs(new, False, True, True)}
ok(fit["eval"]["ready"], "3050 laptop fits eval")
macs = {"name": "Mac-mini-sl-ai", "os": "darwin", "ram": {"total_mib": 16384, "free_mib": 6000}, "gpus": [{"unified": True}], "rpc_binary": "/x/ggml-rpc-server"}
rpc = [j for j in mesh.side_jobs(macs, False, False, True) if j["job"] == "rpc"][0]
ok(not rpc["ready"], "RPC waits while chat is on")
ok([j["job"] for j in mesh.side_jobs({}, True, True, False)] == ["train"], "home row is training")

# --- saved lists -------------------------------------------------------------
with tempfile.TemporaryDirectory() as tmp:
    bad = Path(tmp) / "nodes.json"
    bad.write_text("[{", encoding="utf-8")
    ok(mesh.read_json_list(bad) == [], "damaged list reads empty")
    try:
        mesh.read_json_list(bad, strict=True)
        ok(False, "strict should raise")
    except RuntimeError:
        ok(True, "strict read refuses to overwrite a damaged file")
    good = Path(tmp) / "specs.json"
    mesh.write_json_list(good, [{"a": 1}, {"b": "ä"}])
    ok(mesh.read_json_list(good) == [{"a": 1}, {"b": "ä"}], "atomic write round trip")

# --- job handlers ------------------------------------------------------------
ok(mesh.job_verify_python_syntax({"code": "def f(x):\n    return x + 1\n"})[1]["passed"], "python syntax ok")
bad_py = mesh.job_verify_python_syntax({"code": "def f(:\n"})[1]
ok(bad_py["passed"] is False and bad_py["line"] == 1, "python syntax error caught")
ok(mesh.job_verify_python_syntax({"code": "import os; os.remove('x')"})[1]["passed"], "compile does not run code")
for answer, expected, want in [("so \\boxed{42}", "42", True), ("Final answer: 3/4", "0.75", True),
                               ("\\boxed{\\frac{1}{2}}", "0.5", True), ("The answer is 1,234.", "1234", True),
                               ("\\boxed{41}", "42", False), ("\\boxed{x+1}", "x + 1", True), ("1e99999", "1", False)]:
    got = mesh.job_verify_math({"answer": answer, "expected": expected})[1]
    ok(got["passed"] is want, f"verify_math {answer!r} vs {expected} -> {got}")
d = mesh.job_dedup_hash({"texts": ["Hello   World, this is a test of dedup.", "hello world, this is a test of dedup.",
                                   "hello world, this is a test of dedup!", "completely different words here today"]})[1]["hashes"]
ok(d[0]["sha256"] == d[1]["sha256"], "normalized sha256 equal")
sim = lambda a, b: sum(x == y for x, y in zip(a["minhash"], b["minhash"])) / mesh.MINHASH_PERM
print(f"     minhash near-dup {sim(d[0], d[2]):.2f}, unrelated {sim(d[0], d[3]):.2f}")
ok(sim(d[0], d[2]) > sim(d[0], d[3]), "minhash ranks near-duplicate higher")
if mesh.luau_path():
    good = mesh.job_verify_luau({"code": "print(1+1)", "expect_stdout": "2"})[1]
    ok(good["passed"], "luau ok")
    loop = mesh.job_verify_luau({"code": "while true do end"})[1]
    ok(not loop["passed"] and "timed out" in loop.get("error", ""), "luau timeout")
else:
    print("skip verify_luau: no luau binary on this PC")

# --- fake local model for generate ------------------------------------------
seen = {}


class Fake(BaseHTTPRequestHandler):
    def log_message(self, *a):
        pass

    def do_GET(self):
        body = json.dumps({"data": [{"id": "fake"}]}).encode() if self.path == "/v1/models" else b"{}"
        self.send_response(200 if self.path == "/v1/models" else 404)
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def do_POST(self):
        seen["body"] = json.loads(self.rfile.read(int(self.headers["Content-Length"])))
        body = json.dumps({"model": "fake", "choices": [{"message": {"content": "Work: 6*7.\n\\boxed{42}"}, "finish_reason": "stop"}],
                           "usage": {"prompt_tokens": 5, "completion_tokens": 9}}).encode()
        self.send_response(200)
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)


fake = ThreadingHTTPServer(("127.0.0.1", 0), Fake)
threading.Thread(target=fake.serve_forever, daemon=True).start()
os.environ["NEURO_LOCAL_LLM"] = str(fake.server_address[1])
os.environ["NEURO_LOCAL_LLM"] = str(fake.server_address[1])
ok(mesh.local_model_ready(), "generate advertised when /v1/models answers")
gen_ok, gen_out = mesh.job_generate(mesh.validate_job("generate", {"prompt": "6*7?", "max_tokens": 99999}))
ok(gen_ok and "\\boxed{42}" in gen_out["text"], "generate returns text")
ok(seen["body"]["max_tokens"] == 4096 and seen["body"]["chat_template_kwargs"] == {"enable_thinking": False}, "max_tokens capped, thinking off")
os.environ["NEURO_LOCAL_LLM"] = "http://evil.example.com:9999"
ok(mesh.local_llm_base() == "http://127.0.0.1:8080", "never another host")
os.environ["NEURO_LOCAL_LLM"] = str(fake.server_address[1])

# --- queue -------------------------------------------------------------------
with tempfile.TemporaryDirectory() as tmp:
    q = mesh.JobQueue(Path(tmp))
    for bad_row in [{"prompt": 5}, {"messages": [{"role": "root", "content": "x"}]}, {"prompt": "x", "verify": {"kind": "rm"}}]:
        try:
            q.add("generate", bad_row)
            ok(False, "bad row accepted")
        except mesh.JobError:
            ok(True, f"rejects {bad_row}")
    try:
        q.add("shell", {"cmd": "rm -rf"})
        ok(False, "unknown kind accepted")
    except mesh.JobError:
        ok(True, "unknown kind refused")
    g1 = q.add("generate", {"prompt": "6*7?", "verify": {"kind": "verify_math", "expected": "42"}})
    g2 = q.add("generate", {"prompt": "6*8?", "verify": {"kind": "verify_math", "expected": "48"}})
    t0 = 1000.0
    a = q.lease("LAPTOP-M906OEFU", ["generate"], t0)
    ok(a["id"] == g1, "first job leased")
    ok(q.lease("LAPTOP-M906OEFU", ["generate"], t0 + 1)["id"] == g1, "one job per worker; same lease returned")
    b = q.lease("Mac-mini-sl-ai", ["generate"], t0 + 2)
    ok(b["id"] == g2, "second worker gets the second job")
    ok(q.lease("OhProggraming", ["generate"], t0 + 3) is None, "nothing left -> 204")
    ok(q.finish({"id": g1, "lease": "wrong", "node": "LAPTOP-M906OEFU", "ok": True, "output": {"text": "x"}}, t0 + 4)[0] is False, "wrong lease refused")
    ok(q.finish({"id": g1, "lease": a["lease"], "node": "OhProggraming", "ok": True, "output": {"text": "x"}}, t0 + 4)[0] is False, "other node refused")
    run_ok, run_out = mesh.run_job(a)
    accepted, note = q.finish({"id": g1, "lease": a["lease"], "node": "LAPTOP-M906OEFU", "ok": run_ok, "output": run_out}, t0 + 5)
    ok(accepted and "verify_math queued" in note, f"generate saved and verify spawned: {note}")
    ok(q.finish({"id": g1, "lease": a["lease"], "node": "LAPTOP-M906OEFU", "ok": True, "output": {"text": "x"}}, t0 + 6)[0] is False, "double result refused")
    q.finish({"id": g2, "lease": b["lease"], "node": "Mac-mini-sl-ai", "ok": True, "output": {"text": "\\boxed{47}"}}, t0 + 6)
    v1 = q.lease("OhProggraming", ["verify_math"], t0 + 7)
    q.finish({"id": v1["id"], "lease": v1["lease"], "node": "OhProggraming", "ok": True, "output": mesh.run_job(v1)[1]}, t0 + 8)
    v2 = q.lease("OhProggraming", ["verify_math"], t0 + 9)
    q.finish({"id": v2["id"], "lease": v2["lease"], "node": "OhProggraming", "ok": True, "output": mesh.run_job(v2)[1]}, t0 + 10)
    s = q.summary()
    ok(s["verified_rows"] == 1 and s["kinds"]["verify_math"]["passed"] == 1, f"one verified row {s['verified_rows']}")
    out, count = q.export("verified.jsonl")
    rows = [json.loads(line) for line in out.read_text(encoding="utf-8").splitlines()]
    ok(count == 1 and rows[0]["messages"][-1]["role"] == "assistant" and "42" in rows[0]["messages"][-1]["content"], "export holds only the verified row")
    try:
        q.export("..\\..\\escape.jsonl")
        ok((Path(tmp) / "escape.jsonl").exists(), "export path stays in jobs folder")
    except mesh.JobError:
        ok(True, "export path refused")
    # lease expiry and requeue
    d1 = q.add("dedup_hash", {"texts": ["a b c d"]})
    first = q.lease("Mac-mini-sl-ai", ["dedup_hash"], t0 + 20)
    again = q.lease("OhProggraming", ["dedup_hash"], t0 + 20 + 301)
    ok(again and again["id"] == d1, "expired lease goes back to the queue")
    ok(q.finish({"id": d1, "lease": first["lease"], "node": "Mac-mini-sl-ai", "ok": True, "output": {}}, t0 + 330)[0] is False, "late result from old lease refused")
    bad_out = q.finish({"id": d1, "lease": again["lease"], "node": "OhProggraming", "ok": True, "output": {"hashes": [{"sha256": "zz", "minhash": []}]}}, t0 + 331)
    ok(bad_out[0] and q.jobs[d1]["state"] == "failed", "malformed output stored as failed")
    # a second process (the CLI) appending is picked up
    q2 = mesh.JobQueue(Path(tmp))
    q2.add("verify_python_syntax", {"code": "x = 1"})
    ok(q.lease("OhProggraming", ["verify_python_syntax"], t0 + 400) is not None, "server sees rows added by the CLI")
    # CLI round trip
    src = Path(tmp) / "prompts.jsonl"
    src.write_text(json.dumps({"prompt": "2+2?", "verify": {"kind": "verify_math", "expected": 4}}) + "\n{bad json\n" + json.dumps({"prompt": "no verify"}) + "\n", encoding="utf-8")
    mesh.JobQueue.__init__.__defaults__ = (Path(tmp),)
    mesh.jobs_main(["add", "generate-verify", str(src)])

# --- HTTP handler on an ephemeral loopback port -------------------------------
with tempfile.TemporaryDirectory() as tmp:
    mesh._queue = mesh.JobQueue(Path(tmp))
    mesh._queue.add("verify_math", {"answer": "\\boxed{1}", "expected": "1"})
    mesh.Handler.mode = "serve"
    srv = ThreadingHTTPServer(("127.0.0.1", 0), mesh.Handler)
    threading.Thread(target=srv.serve_forever, daemon=True).start()
    base = f"http://127.0.0.1:{srv.server_address[1]}"

    def call(path, body=None, headers=None, method=None):
        req = urllib.request.Request(base + path, data=None if body is None else json.dumps(body).encode(), headers=headers or {}, method=method)
        try:
            with mesh._DIRECT.open(req, timeout=5) as r:
                return r.status, r.read()
        except urllib.error.HTTPError as e:
            return e.code, e.read()

    ok(call("/api/token")[0] == 404, "GET /api/token is 404")
    ok(call("/jobs/next", {"node": "x", "kinds": ["verify_math"]})[0] == 401, "/jobs/next needs the token")
    auth = {"Authorization": "Bearer " + FAKE, "Content-Type": "application/json"}
    code, raw = call("/jobs/next", {"node": "OhProggraming", "kinds": ["verify_math"]}, auth)
    job = json.loads(raw)
    ok(code == 200 and job["kind"] == "verify_math", "/jobs/next leases with the token")
    ok(call("/jobs/next", {"node": "Other", "kinds": ["verify_math"]}, auth)[0] == 204, "204 when nothing fits")
    code, raw = call("/jobs/result", {"id": job["id"], "lease": job["lease"], "node": "OhProggraming", "ok": True, "output": mesh.run_job(job)[1]}, auth)
    ok(code == 200 and json.loads(raw)["accepted"], "/jobs/result accepted")
    ok(call("/api/process/stop", {"pid": 1, "name": "x"}, {"Origin": "http://evil.example.com", "Content-Type": "application/json"})[0] == 403, "cross-site Stop refused")
    ok(call("/api/jobs", headers={"Host": "evil.example.com"})[0] == 403, "rebinding Host refused")
    ok(json.loads(call("/api/jobs")[1])["kinds"]["verify_math"]["passed"] == 1, "/api/jobs summary")
    srv.shutdown()
    srv.server_close()
# --- coding verifiers (inserted into test_mesh.py before fake.shutdown) ----------
import glob
import subprocess as _sp

TMPROOT = tempfile.gettempdir()


def leftovers(prefix):
    return set(glob.glob(os.path.join(TMPROOT, prefix + "*")))


md = "Intro\n```js\nconst a = 1;\n```\ntext\n~~~typescript\nlet b: number = 2;\n~~~\n```\nplain();\n```\n```python\nx = 1\n```\n"
blocks = mesh.code_blocks(md)
ok([b["lang"] for b in blocks] == ["js", "ts", "", "python"], f"code_blocks languages {[b['lang'] for b in blocks]}")
ok(mesh.pick_blocks(md, "js") == ["const a = 1;\n"], "pick_blocks takes the js-tagged block")
ok(mesh.pick_blocks("```\nfoo();\n```", "js") == ["foo();\n"], "untagged block is the fallback")
ok(mesh.pick_blocks("let x = 1;", "js") == ["let x = 1;"], "no fence -> whole reply")
ok(mesh.pick_blocks("```python\nx=1\n```", "js") == [], "only other languages -> nothing (fails later)")
cut = mesh.code_blocks("```js\nfunction f() {\n")
ok(len(cut) == 1 and cut[0]["closed"] is False, "a cut-off fence is kept, not skipped")
ok(mesh.pick_blocks("<think>```js\nbad(\n```</think>```js\ngood();\n```", "js") == ["good();\n"], "thinking is dropped first")
h, c = mesh.split_web("```html\n<p>x</p>\n```\n```css\np { color: red; }\n```")
ok(h.strip() == "<p>x</p>" and "color" in c, "split_web separates html and css")

# validate_job for the new kinds
ok(mesh.validate_job("verify_js", {"answer": md})["blocks"] == ["const a = 1;\n"], "verify_js answer -> blocks")
ok(mesh.validate_job("verify_sql", {"answer": "```sql\nSELECT 1\n```"})["code"].startswith("SELECT 1"), "verify_sql answer -> code")
for kind, bad_row in [("verify_js", {}), ("verify_js", {"blocks": ["x"] * 17}), ("verify_html_css", {"html": "<p>", "require_tags": ["<script>"]}),
                      ("generate", {"prompt": "x", "verify": {"kind": "verify_sql", "schema": ["no"]}})]:
    try:
        mesh.validate_job(kind, bad_row)
        ok(False, f"bad {kind} row accepted")
    except mesh.JobError:
        ok(True, f"rejects bad {kind} row")
spec_row = mesh.validate_job("generate", {"prompt": "x", "samples": 99, "seed": 7, "verify": {"kind": "verify_html_css", "require_tags": ["FORM"]}})
ok(spec_row["samples"] == mesh.MAX_SAMPLES and spec_row["seed"] == 7 and spec_row["verify"]["require_tags"] == ["form"], "generate keeps samples/seed/spec")
ok(mesh.check_output("verify_js", {}, True, {"passed": False, "error": "x"})[1] is False, "check_output knows verify_js")

# --- verify_html_css
page = """<!DOCTYPE html>
<html><head><meta charset="utf-8"><title>T</title>
<style>body { margin: 0; } a[href^="http"]::after { content: "}"; }</style></head>
<body><ul><li>one<li>two</ul><p>para<p>next<br><img src="x.png" alt="">
<svg viewBox="0 0 1 1"><path d="M0 0"/></svg>
<script>if (a < b && c > d) { document.write("</div>"); }</script>
<form><input name=q></form></body></html>"""
got = mesh.job_verify_html_css({"html": page, "css": "", "require_tags": ["form", "ul"]})[1]
ok(got["passed"], f"html page passes {got}")
for html, css, why in [
    ("<div><span>x</div>", "", "misnested"), ("<div>x", "", "unclosed div"), ("x</section>", "", "stray close"),
    ("<p>x<script>let a = 1;", "", "unclosed script"), ("<div class=\"a", "", "unfinished tag"), ("<div><!-- never", "", "unfinished comment"),
    ("<p>x</p>", "p { color: red;", "css brace"), ("", "a { content: \"x; }", "css string"), ("", "Here is some prose.", "css without rule"),
    ("just words", "", "no tags"), ("<style>a { color: red; </style>", "", "inline style braces"), ("", "", "empty"),
]:
    got = mesh.job_verify_html_css({"html": html, "css": css})[1]
    ok(got["passed"] is False, f"html/css fails: {why} -> {got.get('error')}")
ok(mesh.job_verify_html_css({"html": "<p>x</p>", "css": "", "require_tags": ["form"]})[1]["passed"] is False, "require_tags enforced")
ok(mesh.job_verify_html_css(mesh.validate_job("verify_html_css", {"answer": "```css\n.a { b: c }\n```"}))[1]["passed"], "css-only answer passes")

# --- verify_sql
good_sql = """CREATE TABLE users(id INTEGER PRIMARY KEY, name TEXT NOT NULL, created TEXT DEFAULT (datetime('now')));
CREATE INDEX users_name ON users(name);
CREATE TRIGGER stamp AFTER INSERT ON users BEGIN UPDATE users SET name = trim(new.name) WHERE id = new.id; END;
INSERT INTO users(name) VALUES ('a;b');
-- comment; with a semicolon
WITH RECURSIVE c(x) AS (SELECT 1 UNION ALL SELECT x + 1 FROM c) SELECT count(*) FROM c;
SELECT u.name, COUNT(*) FROM users u GROUP BY u.name ORDER BY 2 DESC"""
started = time.time()
got = mesh.job_verify_sql({"code": good_sql})[1]
ok(got["passed"] and time.time() - started < 5, f"sql passes; infinite CTE only compiled, not run {got}")
for code, why in [("SELEC 1;", "unknown verb"), ("CREATE TABLE t(a); SELECT a FROM t WHERE ;", "syntax"), ("SELECT a FROM t;", "unknown table"), ("SELECT * FROM missing;", "unknown table"), ("ATTACH DATABASE 'x.db' AS x;", "attach"),
                  ("PRAGMA table_info(t);", "pragma"), ("VACUUM INTO 'copy.db';", "vacuum into"), ("SELECT load_extension('x');", "load_extension"),
                  ("SELECT 'unterminated", "unterminated string"), ("", "empty"), ("CREATE TABLE t(a) AUTO_INCREMENT;", "other dialect")]:
    got = mesh.job_verify_sql({"code": code})[1]
    ok(got["passed"] is False, f"sql fails: {why} -> {got.get('error')}")
ok(not (Path.cwd() / "x.db").exists() and not (Path.cwd() / "copy.db").exists(), "sql never wrote a file")
ok(mesh.job_verify_sql({"code": "SELECT name FROM users", "schema": "CREATE TABLE users(id, name); INSERT INTO users VALUES (1, 'a');"})[1]["passed"], "spec schema resolves names")
ok(mesh.job_verify_sql({"code": "CREATE TABLE b AS SELECT 1 AS x; SELECT 2;"})[1]["passed"], "CTAS is only compiled")
ok(mesh.verify_payload({"kind": "verify_sql", "schema": "CREATE TABLE t(a);"}, "```sql\nSELECT a FROM t\n```")["schema"], "verify_payload carries the schema")

# --- verify_js / verify_ts (node, parse only)
marker = Path(tempfile.gettempdir()) / f"neuro-js-ran-{os.getpid()}.txt"
marker.unlink(missing_ok=True)
side_effect = f"require('fs').writeFileSync({json.dumps(str(marker))}, 'ran');\nwhile (true) {{}}\n"
node = mesh.node_path()
if node:
    before = leftovers("neuro-js-") | leftovers("neuro-ts-")
    got = mesh.job_verify_js({"blocks": ["const f = (a, b) => a + b;\nmodule.exports = f;\n", "import x from 'nope';\nexport const y = await Promise.resolve(1);\n"]})[1]
    ok(got["passed"], f"verify_js passes CommonJS and ES module blocks {got}")
    got = mesh.job_verify_js({"blocks": [side_effect]})[1]
    ok(got["passed"] and not marker.exists(), "verify_js never runs the code (no file, no endless loop)")
    got = mesh.job_verify_js({"blocks": ["const ok = 1;\n", "function (\n"]})
    ok(got[0] and got[1]["passed"] is False and "block 2" in got[1]["error"] and "SyntaxError" in got[1]["error"], f"verify_js syntax error {got[1].get('error', '')[:80]}")
    ok("<job>" in got[1]["error"] and tempfile.gettempdir() not in got[1]["error"], "temp path is hidden in errors")
    ok(mesh.job_verify_js({"blocks": []})[1]["passed"] is False, "no JS block fails")
    ok(mesh.job_verify_js(mesh.validate_job("verify_js", {"answer": "```python\nprint(1)\n```"}))[1]["passed"] is False, "python-only answer fails verify_js")
    seen_popen = []
    real_popen = _sp.Popen

    def spy(*args, **kwargs):
        seen_popen.append((args, kwargs))
        return real_popen(*args, **kwargs)

    os.environ["NODE_OPTIONS"] = "--require ./evil.js"
    mesh.subprocess.Popen = spy
    try:
        mesh.job_verify_js({"blocks": ["let a = 1;\n"]})
    finally:
        mesh.subprocess.Popen = real_popen
        os.environ.pop("NODE_OPTIONS", None)
    argv, kwargs = seen_popen[0][0][0], seen_popen[0][1]
    ok(isinstance(argv, list) and argv[1] == "--check" and kwargs.get("shell") is False, "node gets an argument list, shell=False")
    ok("NODE_OPTIONS" not in kwargs["env"] and "PATH" not in kwargs["env"] and kwargs["stdin"] == _sp.DEVNULL, "scrubbed env, no stdin")
    if mesh.node_strips_types(node):
        got = mesh.job_verify_ts({"blocks": ["interface P { x: number }\nexport function g<T>(a: T[]): T | undefined { return a[0] as T; }\nconst p: P = { x: 1 };\n"]})[1]
        ok(got["passed"], f"verify_ts passes typed code {got}")
        ok(mesh.job_verify_ts({"blocks": ["const f = (p: number => 1;\n"]})[1]["passed"] is False, "verify_ts syntax error fails")
        ok(mesh.job_verify_ts({"blocks": ["let s: string = 1 +;\n"]})[1]["passed"] is False, "verify_ts JS-level error fails")
        enum = mesh.job_verify_ts({"blocks": ["enum E { A, B }\n"]})
        ok(enum[0] is False and enum[1]["error"].startswith("inconclusive"), "enum: no verdict, not a pass or a reject")
        got = mesh.job_verify_ts({"blocks": ["const n: number = 1;\n" + side_effect]})[1]
        ok(got["passed"] and not marker.exists(), "verify_ts never runs the code")
    else:
        print("skip verify_ts: this node has no stripTypeScriptTypes")
    ok((leftovers("neuro-js-") | leftovers("neuro-ts-")) - before == set(), "temp folders are deleted")
    mesh._kinds_cache["at"] = 0.0
    kinds_now = mesh.worker_kinds()
    ok({"verify_js", "verify_html_css", "verify_sql"} <= set(kinds_now), f"worker advertises the new kinds {kinds_now}")
else:
    print("skip verify_js/verify_ts: no node on this PC")
os.environ["NEURO_NODE"] = str(Path(tempfile.gettempdir()) / "no-such-node.exe")
try:
    missing = mesh.job_verify_js({"blocks": ["let a = 1;"]})
    ok(missing[0] is False and missing[1]["error"].startswith("unavailable"), "no node -> unavailable, not a pass")
    ok(mesh.job_verify_ts({"blocks": ["let a = 1;"]})[0] is False, "no node -> verify_ts unavailable")
    mesh._kinds_cache["at"] = 0.0
    ok("verify_js" not in mesh.worker_kinds(), "no node -> verify_js not advertised")
finally:
    os.environ.pop("NEURO_NODE", None)
    mesh._kinds_cache["at"] = 0.0

# --- generate sends the seed --------------------------------------------------
mesh.job_generate(mesh.validate_job("generate", {"prompt": "x", "seed": 12345}))
ok(seen["body"].get("seed") == 12345, "generate passes the per-sample seed to the local model")

# --- rejection sampling, export, pairs ----------------------------------------
GOOD1 = "Here:\n```sql\nSELECT id FROM t;\n```"
GOOD2 = "```sql\nSELECT t.id FROM t ORDER BY id;\n```"
BAD1 = "```sql\nSELEC id FROM t;\n```"
BAD2 = "```sql\nSELECT nope FROM missing;\n```"
with tempfile.TemporaryDirectory() as tmp:
    q = mesh.JobQueue(Path(tmp))
    ids = q.add_many([("generate", {"prompt": "List ids from t (SQLite).", "samples": 5, "meta": {"lang": "sql"},
                                     "verify": {"kind": "verify_sql", "schema": "CREATE TABLE t(id INTEGER);"}})])
    jobs5 = [q.jobs[j]["payload"] for j in ids]
    ok(len(ids) == 5 and len({p["group"] for p in jobs5}) == 1 and [p["sample"] for p in jobs5] == [0, 1, 2, 3, 4], "samples=5 -> 5 jobs, one group")
    ok(len({p["seed"] for p in jobs5}) == 5 and all(p["temperature"] == 0.8 for p in jobs5), "samples differ by seed")
    single = q.add("generate", {"prompt": "Plain.", "verify": {"kind": "verify_sql", "schema": "CREATE TABLE t(id INTEGER);"}})
    ok("group" not in q.jobs[single]["payload"], "a single job has no group")
    texts = [GOOD1, BAD1, GOOD2, BAD2, GOOD1]
    t0 = 5000.0
    for index, text in enumerate(texts):
        lease = q.lease(f"node{index}", ["generate"], t0 + index)
        q.finish({"id": lease["id"], "lease": lease["lease"], "node": f"node{index}", "ok": True, "output": {"text": text}}, t0 + index)
    lease = q.lease("nodeS", ["generate"], t0 + 9)
    q.finish({"id": lease["id"], "lease": lease["lease"], "node": "nodeS", "ok": True, "output": {"text": GOOD1}}, t0 + 9)
    # Every verify runs, except the last of the group: its worker had no verifier.
    for step in range(6):
        lease = q.lease("checker", ["verify_sql"], t0 + 20 + step)
        if q.jobs[lease["id"]]["parent"] == ids[4]:
            body = {"ok": False, "output": {"error": "unavailable: this Python has no sqlite3"}}
        else:
            run_ok, run_out = mesh.run_job(lease)
            body = {"ok": run_ok, "output": run_out}
        q.finish({"id": lease["id"], "lease": lease["lease"], "node": "checker", **body}, t0 + 20 + step)
    verdicts = [q.jobs[j]["verified"] for j in ids]
    ok(verdicts == [True, False, True, False, False], f"verify verdicts {verdicts}")
    out, count = q.export("sft.jsonl")
    rows = [json.loads(line) for line in out.read_text(encoding="utf-8").splitlines()]
    ok(count == 2 and {r["job"] for r in rows} == {ids[0], single}, "export keeps one verified candidate per prompt group")
    ok(all(r["skill"] == "code" and r["verified_by"] == "verify_sql" for r in rows) and rows[0]["group"] == jobs5[0]["group"], "export rows carry skill and group")
    out, count = q.export("all.jsonl", per_prompt=0)
    ok(count == 3, "per_prompt=0 keeps every verified distinct candidate")
    out, count = q.export_pairs("pairs.jsonl")
    pairs = [json.loads(line) for line in out.read_text(encoding="utf-8").splitlines()]
    ok(out.parent == Path(tmp).resolve() and count == 2, f"two pairs from 2 passed / 2 failed {count}")
    ok({p["chosen"][0]["content"] for p in pairs} == {mesh.strip_thinking(GOOD1), GOOD2} and {p["rejected"][0]["content"] for p in pairs} == {BAD1, BAD2}, "chosen = verified, rejected = failed")
    ok(all(p["prompt"][-1]["role"] == "user" and p["rejected_error"] and p["skill"] == "code" for p in pairs), "pair rows: prompt messages, reason, skill")
    ok(all(ids[4] not in (p["chosen_job"], p["rejected_job"]) for p in pairs), "an unavailable verifier is neither chosen nor rejected")
    # Where pairs may be written
    data_root = Path(tmp) / "data"
    (data_root / "training" / "mix").mkdir(parents=True)
    os.environ["NEURO_AI_DATA"] = str(data_root)
    try:
        default_training = mesh.ROOT.parent / "neuro-ai-data" / "training"
        cased = [str(default_training / "x" / "p.jsonl").swapcase()] if os.name == "nt" else []
        for target in [str(default_training / "pairs.jsonl"), *cased,
                       str(data_root / "training" / "mix" / "p.jsonl"), str(data_root / "training" / ".." / "training" / "p.jsonl"),
                       str(Path(tmp) / "pairs.txt"), "queue.jsonl", str(Path(tmp) / "nope" / "p.jsonl"), str(Path(tmp))]:
            try:
                q.export_pairs(target)
                ok(False, f"pairs path accepted: {target}")
            except mesh.JobError as exc:
                ok(True, f"pairs path refused: {target} ({str(exc)[:50]})")
        ok(not list((data_root / "training").rglob("*.jsonl")), "nothing written into the training folder")
        outside = Path(tmp) / "elsewhere"
        outside.mkdir()
        out, count = q.export_pairs(str(outside / "pairs.jsonl"))
        ok(out == (outside / "pairs.jsonl").resolve() and count == 2 and not list(outside.glob("*.tmp")), "pairs written atomically to a named path outside training")
        # CLI
        mesh.JobQueue.__init__.__defaults__ = (Path(tmp),)
        mesh.jobs_main(["export", "--pairs", str(outside / "cli.jsonl")])
        ok(len((outside / "cli.jsonl").read_text(encoding="utf-8").splitlines()) == 2, "jobs export --pairs writes the file")
        mesh.jobs_main(["export", "--per-prompt", "0", "cli-all.jsonl"])
        ok(len((Path(tmp) / "cli-all.jsonl").read_text(encoding="utf-8").splitlines()) == 3, "jobs export --per-prompt 0")
        try:
            mesh.jobs_main(["export", "--pairs", str(data_root / "training" / "p.jsonl")])
            ok(False, "CLI wrote pairs into training")
        except SystemExit as exc:
            ok("training folder" in str(exc), "CLI refuses pairs inside training")
        src = Path(tmp) / "coding.jsonl"
        src.write_text(json.dumps({"prompt": "Write a JS add function.", "samples": 3, "verify": {"kind": "verify_js"}}) + "\n", encoding="utf-8")
        real_kinds = mesh.serve_job_kinds
        mesh.serve_job_kinds = lambda: {"generate", "verify_luau", "verify_python_syntax", "verify_math", "dedup_hash"}
        try:
            mesh.jobs_main(["add", "generate-verify", str(src)])
            ok(False, "jobs add went to an old serve")
        except SystemExit as exc:
            ok("Restart serve" in str(exc), "jobs add refuses new kinds while an old serve runs")
        mesh.serve_job_kinds = lambda: None
        try:
            mesh.jobs_main(["add", "generate-verify", str(src)])
        finally:
            mesh.serve_job_kinds = real_kinds
        fresh = mesh.JobQueue(Path(tmp))
        fresh.sync()
        added = [j for j in fresh.jobs.values() if j["payload"].get("samples") == 3]
        ok(len(added) == 3 and len({j["payload"]["group"] for j in added}) == 1, "jobs add expands samples into one group")
    finally:
        os.environ.pop("NEURO_AI_DATA", None)

# generate -> verify_js through the queue (node only)
if node:
    with tempfile.TemporaryDirectory() as tmp:
        q = mesh.JobQueue(Path(tmp))
        g = q.add("generate", {"prompt": "js", "verify": {"kind": "verify_js"}})
        lease = q.lease("n1", ["generate"], 1.0)
        accepted, note = q.finish({"id": lease["id"], "lease": lease["lease"], "node": "n1", "ok": True,
                                   "output": {"text": "```javascript\nexport const add = (a, b) => a + b;\n```"}}, 2.0)
        v = q.lease("n2", ["verify_js"], 3.0)
        ok("verify_js queued" in note and v["payload"]["blocks"] == ["export const add = (a, b) => a + b;\n"], "generate spawns verify_js with the extracted block")
        q.finish({"id": v["id"], "lease": v["lease"], "node": "n2", "ok": True, "output": mesh.run_job(v)[1]}, 4.0)
        ok(q.jobs[g]["verified"] is True, "verify_js verdict marks the generation verified")


import re
# --- worker control routes need the cluster token -----------------------------
from urllib.parse import urlparse as _urlparse
import types as _types

ok(mesh.authorized("Bearer " + FAKE) and not mesh.authorized("Bearer wrong") and not mesh.authorized(None), "authorized compares the bearer token")
ok(mesh.authorized("Bearer é" + FAKE) is False, "non-ASCII Authorization header is refused, not an exception")
ok(not mesh.authorized("Bearer ", ""), "an empty token never authorizes")

seen_auth = []
calls = {"stop": [], "rpc_start": 0, "rpc_stop": 0}


class Worker(mesh.Handler):
    mode = "work"

    def do_GET(self):
        seen_auth.append(("GET", _urlparse(self.path).path, self.headers.get("Authorization")))
        super().do_GET()

    def do_POST(self):
        seen_auth.append(("POST", _urlparse(self.path).path, self.headers.get("Authorization")))
        super().do_POST()


real = {name: getattr(mesh, name) for name in ("local_status", "stop_owned_process", "start_rpc", "stop_rpc", "TOKEN_PATH", "saved_token")}
real_agents = sys.modules.get("agents")
mesh.local_status = lambda: {"name": "test-worker", "os": "windows", "ram": {}}


def _spy_stop(pid, name=""):
    calls["stop"].append((pid, name))
    return {"stopped": True, "note": "spy stopped " + name}


def _spy_start():
    calls["rpc_start"] += 1
    return {"started": True, "port": 50052}


def _spy_rpc_stop():
    calls["rpc_stop"] += 1
    return {"stopped": True}


mesh.stop_owned_process, mesh.start_rpc, mesh.stop_rpc = _spy_stop, _spy_start, _spy_rpc_stop
sys.modules["agents"] = _types.SimpleNamespace(drill_round=lambda index: {"ok": True})
wsrv = ThreadingHTTPServer(("127.0.0.1", 0), Worker)
threading.Thread(target=wsrv.serve_forever, daemon=True).start()
wurl = f"http://127.0.0.1:{wsrv.server_address[1]}"


def wcall(path, body=None, headers=None, method=None):
    req = urllib.request.Request(wurl + path, data=None if body is None else json.dumps(body).encode(), headers=headers or {}, method=method)
    try:
        with mesh._DIRECT.open(req, timeout=10) as r:
            return r.status, json.loads(r.read() or b"{}")
    except urllib.error.HTTPError as e:
        return e.code, json.loads(e.read() or b"{}")


good = {"Authorization": "Bearer " + FAKE, "Content-Type": "application/json"}
try:
    code, hello = wcall("/hello")
    ok(code == 200 and hello.get("app") == "neuroai-mesh", "worker GET /hello stays open without a token")
    ok(wcall("/status")[0] == 401, "worker GET /status still needs the token")
    for route, body in (("/process/stop", {"pid": 4242, "name": "x.exe"}), ("/rpc/start", {}), ("/rpc/stop", {}), ("/check", {"rounds": 1}), ("/anything", {})):
        ok(wcall(route, body, {"Content-Type": "application/json"})[0] == 401, f"worker POST {route} without a token -> 401")
        ok(wcall(route, body, {"Authorization": "Bearer wrong", "Content-Type": "application/json"})[0] == 401, f"worker POST {route} with a wrong token -> 401")
    ok(wcall("/process/stop", {"pid": 1}, {"Authorization": "Bearer é", "Content-Type": "application/json"})[0] == 401, "non-ASCII token header -> 401")
    ok(calls == {"stop": [], "rpc_start": 0, "rpc_stop": 0}, "nothing was stopped or started without the token")
    code, body = wcall("/process/stop", {"pid": 4242, "name": "x.exe"}, good)
    ok(code == 200 and calls["stop"] == [(4242, "x.exe")], "worker POST /process/stop works with the token")
    ok(wcall("/rpc/stop", {}, good)[0] == 200 and calls["rpc_stop"] == 1, "worker POST /rpc/stop works with the token")
    ok(wcall("/anything", {}, good)[0] == 404, "unknown worker POST with the token -> 404")
    bad_origin = dict(good, Origin="http://evil.example.com")
    ok(wcall("/rpc/stop", {}, bad_origin)[0] == 403 and calls["rpc_stop"] == 1, "Origin check still applies with a valid token")

    # Callers on the main PC send the header; the LAN sweep does not.
    seen_auth.clear()
    swept = mesh.pull_worker(wurl, legacy=False)
    ok(swept["ok"] and seen_auth == [("GET", "/hello", None)], "LAN sweep /hello carries no token")
    with mesh._found_lock:
        saved_found = list(mesh._found)
        mesh._found[:] = [{"url": wurl, "ok": True, "status": {"name": "test-worker"}}]
    try:
        mesh._proofs.clear()
        seen_auth.clear()
        result = mesh.stop_at(wurl, 777, "llama-server.exe")
        ok(result["stopped"] and calls["stop"][-1] == (777, "llama-server.exe"), f"stop_at reaches the worker {result}")
        ok(seen_auth[0][:2] == ("GET", "/hello/proof") and seen_auth[0][2] is None, "serve asks for a proof first, without the token")
        check = mesh.ask_check(wurl)
        ok(check["ok"] and check["passed"] == 3, f"ask_check reaches the worker {check}")
        share = mesh.offer_share()
        ok(share["peers"] and share["peers"][0]["ok"] and calls["rpc_start"] == 1, f"offer_share starts RPC on the worker {share['peers']}")
        ok(mesh.fetch_worker(wurl)["name"] == "test-worker", "fetch_worker gets /status after the proof")
        sent = [(m, p, a) for m, p, a in seen_auth if a is not None]
        ok([p for _, p, _ in sent] == ["/process/stop", "/check", "/rpc/start", "/status"] and all(a == "Bearer " + FAKE for _, _, a in sent), "the right-token worker receives the bearer token on every control call")
        ok([p for m, p, _ in seen_auth if p == "/hello/proof"] == ["/hello/proof"], "a good proof is cached per URL")

        # Proof route: an HMAC of the nonce, never the token or its hash; a replayed answer fails for another nonce.
        nonce_a, nonce_b = "ab" * 16, "cd" * 16
        code, proof_body = wcall("/hello/proof?nonce=" + nonce_a)
        raw_reply = json.dumps(proof_body)
        address = "127.0.0.1:" + str(wsrv.server_address[1])
        ok(code == 200 and mesh.proof_ok(FAKE, nonce_a, address, proof_body["proof"]), "worker proof verifies for its nonce")
        ok(not mesh.proof_ok(FAKE, nonce_b, address, proof_body["proof"]), "replayed proof fails for a different nonce")
        ok(not mesh.proof_ok(FAKE, nonce_a, "192.168.7.99:8781", proof_body["proof"]), "a proof relayed for another address fails")
        import hashlib as _hashlib
        ok(FAKE not in raw_reply and _hashlib.sha256(FAKE.encode()).hexdigest() not in raw_reply, "proof reply never holds the token or its hash")
        ok(wcall("/hello/proof?nonce=zz")[0] == 400 and wcall("/hello/proof")[0] == 400, "bad nonce refused")

        # Worker with a different token: it never receives the Authorization header.
        mesh._proofs.clear()
        mesh.saved_token = lambda: "another-token"
        before = list(calls["stop"])
        seen_auth.clear()
        result = mesh.stop_at(wurl, 778, "x.exe")
        ok(not result["stopped"] and "no matching cluster.token" in result["note"] and calls["stop"] == before, f"token mismatch explained {result['note']}")
        ok("no matching cluster.token" in mesh.ask_check(wurl)["error"], "ask_check explains a token mismatch")
        ok("no matching" in mesh.offer_share()["peers"][0]["detail"], "offer_share explains a token mismatch")
        limited = mesh.pull_legacy(wurl)
        ok(limited.get("limited") and limited["ok"], "status pull of an unproven worker is address-only")
        ok(seen_auth and all(a is None for _, _, a in seen_auth) and all(p == "/hello/proof" for _, p, _ in seen_auth), "wrong-token worker never receives the Authorization header")

        # Proof cached, then the worker's token changes: 401, cache dropped, next call does not send it.
        mesh.saved_token = real["saved_token"]
        ok(mesh.stop_at(wurl, 779, "x.exe")["stopped"], "proof again with the right token")
        mesh.saved_token = lambda: "another-token"
        result = mesh.stop_at(wurl, 780, "x.exe")
        ok(not result["stopped"] and "did not accept" in result["note"], f"token changed after the proof is explained {result['note']}")
        seen_auth.clear()
        mesh.stop_at(wurl, 781, "x.exe")
        ok(all(a is None for _, _, a in seen_auth), "after a 401 the cached proof is dropped")
        mesh.saved_token = real["saved_token"]
        before = list(calls["stop"])

        # Worker without any cluster.token: refuses with a clear message and does not make one.
        with tempfile.TemporaryDirectory() as tmp:
            missing = Path(tmp) / "cluster.token"
            mesh.TOKEN_PATH = missing
            mesh._proofs.clear()
            ok(mesh.saved_token() == "", "saved_token is empty when there is no cluster.token")
            code, body = wcall("/process/stop", {"pid": 5, "name": "x.exe"}, good)
            ok(code == 403 and "no cluster.token" in body.get("error", ""), f"tokenless worker refuses with a clear message {code}")
            ok(wcall("/hello")[0] == 200, "tokenless worker still answers /hello")
            code, body = wcall("/hello/proof?nonce=" + nonce_a)
            ok(code == 200 and body.get("has_token") is False and "proof" not in body, "tokenless worker says it has no token")
            seen_auth.clear()
            result = mesh.stop_at(wurl, 5, "x.exe")
            ok(not result["stopped"] and "no cluster.token" in result["note"], f"serve explains the missing token {result['note']}")
            ok(all(a is None for _, _, a in seen_auth), "tokenless worker never receives the Authorization header")
            try:
                mesh._mesh_post("http://127.0.0.1:9", "/jobs/next", {}, 1)
                ok(False, "job loop posted without a token")
            except mesh.JobAuthError as exc:
                ok(exc.missing, "job loop pauses (missing token) instead of making one up")
            ok(not missing.exists(), "no cluster.token was created on the worker")
            mesh.TOKEN_PATH = real["TOKEN_PATH"]
        ok(calls["stop"] == before, "nothing stopped while the token was missing")

        # Light rate limit on the proof route.
        mesh._proof_hits.clear()
        codes = [wcall("/hello/proof?nonce=" + nonce_a)[0] for _ in range(mesh.PROOF_RATE[0] + 1)]
        ok(codes[:-1] == [200] * mesh.PROOF_RATE[0] and codes[-1] == 429, "proof route is rate-limited per client")
        mesh._proof_hits.clear()
        mesh._proofs.clear()
    finally:
        with mesh._found_lock:
            mesh._found[:] = saved_found
finally:
    for name, value in real.items():
        setattr(mesh, name, value)
    if real_agents is None:
        sys.modules.pop("agents", None)
    else:
        sys.modules["agents"] = real_agents
    wsrv.shutdown()
    wsrv.server_close()

page = (Path(mesh.ROOT) / "dashboard.html").read_text(encoding="utf-8")
ok("Authorization" not in page and ":8781" not in page and "/api/token" not in page, "the page never handles the token or calls a worker directly")

# --- coding seed set ---------------------------------------------------------------
seed_path = Path(mesh.ROOT) / "seeds" / "coding-v1.jsonl"
seed_rows, seed_errors = mesh.read_job_file("generate-verify", str(seed_path))
ok(not seed_errors, f"jobs add parser accepts every seed row {seed_errors[:3]}")
ok(250 <= len(seed_rows) <= 350 and all(kind == "generate" for kind, _ in seed_rows), f"about 300 generate rows ({len(seed_rows)})")
tags = {"verify_js": "javascript", "verify_ts": "typescript", "verify_html_css": "html", "verify_sql": "sql", "verify_python_syntax": "python", "verify_luau": "luau"}
ok(all(raw["samples"] == 4 and raw["verify"]["kind"] in tags for _, raw in seed_rows), "every seed row has samples 4 and a code verifier")
ok(all("```" + tags[raw["verify"]["kind"]] in raw["prompt"] for _, raw in seed_rows), "every prompt asks for the right fence tag")
ok(all(mesh.LANG_ALIASES[tags[raw["verify"]["kind"]]] == mesh.CODE_KINDS.get(raw["verify"]["kind"], "luau") for _, raw in seed_rows), "fence tags map to the verifier's language")
ok(len({raw["prompt"] for _, raw in seed_rows}) == len(seed_rows), "seed prompts are unique")
ok(set(tags) == {raw["verify"]["kind"] for _, raw in seed_rows}, "all six code verifiers are used")
hebrew = sum(1 for _, raw in seed_rows if re.search("[֐-׿]", raw["prompt"]))
ok(0.07 <= hebrew / len(seed_rows) <= 0.13, f"about 10% Hebrew prompts ({hebrew})")
ok({raw["meta"]["difficulty"] for _, raw in seed_rows} == {"easy", "medium", "hard"}, "easy, medium and hard rows")
planned = sum(len(mesh.expand_samples(kind, raw)) for kind, raw in seed_rows)
ok(planned == 4 * len(seed_rows), f"seed file expands to {planned} generate jobs")
schemas = {raw["verify"]["schema"] for _, raw in seed_rows if raw["verify"].get("schema")}
ok(schemas and all(mesh.job_verify_sql({"code": "SELECT 1;", "schema": s})[1]["passed"] for s in schemas), "seed SQL schemas build in verify_sql")


# --- pairing: a new computer joins without copying cluster.token ------------------
import contextlib as _contextlib
import hashlib as _hl
import io as _io

LONG = "f0" * 24  # a token in the real form (48 hex), only inside this block
real_token_fn, real_token_path, real_role = mesh.token, mesh.TOKEN_PATH, dict(mesh._ROLE)
mesh.token = lambda: LONG
mesh._pairs.clear()
mesh._pair_hits.clear()
sec = "ab" * 32
dig = _hl.sha256(sec.encode()).hexdigest()
ip1, ip2 = "192.168.7.31", "192.168.7.247"
t = 10_000.0
code, reply = mesh.pair_request(ip1, {"name": "LAPTOP-M906OEFU\x1b[2J", "code": "482913", "secret_sha256": dig, "machine_key": "laptop-m906oefu/windows"}, t)
pid1 = reply.get("id", "")
ok(code == 200 and reply["state"] == "pending" and "token" not in reply, "pair request stored")
listed = mesh.pair_list(t)
ok(len(listed) == 1 and listed[0]["code"] == "482913" and listed[0]["ip"] == ip1 and "hash" not in listed[0], "pending list has name, ip and code, never the hash")
ok("\x1b" not in listed[0]["name"], "control characters dropped from the name")
ok(mesh.pair_status(ip1, {"id": pid1, "secret": sec}, t + 1) == (200, {"state": "pending"}), "pending request gives no token")
ok(mesh.pair_decide(pid1, True, t + 2)[0] == 200, "approve")
c, r = mesh.pair_status(ip1, {"id": pid1, "secret": "cd" * 32}, t + 3)
ok(c == 403 and "token" not in r, "wrong pair_secret gets no token")
c, r = mesh.pair_status(ip2, {"id": pid1, "secret": sec}, t + 3)
ok(c == 403 and "token" not in r, "another client address gets no token")
c, r = mesh.pair_status(ip1, {"id": pid1, "secret": sec}, t + 4)
ok(c == 200 and r == {"state": "approved", "token": LONG}, "approved + right secret + same address returns the token")
c, r = mesh.pair_status(ip1, {"id": pid1, "secret": sec}, t + 5)
ok(c == 404 and "token" not in r, "the token is handed out once")
# expired
_, r = mesh.pair_request(ip1, {"name": "x", "code": "111111", "secret_sha256": dig}, t + 10)
c, r2 = mesh.pair_status(ip1, {"id": r["id"], "secret": sec}, t + 10 + mesh.PAIR_TTL + 1)
ok(c == 410 and r2["state"] == "expired" and "token" not in r2, "expired request gets no token")
ok(mesh.pair_decide(r["id"], True, t + 10 + mesh.PAIR_TTL + 2)[0] == 404, "an expired request cannot be approved")
# denied
mesh._pair_hits.clear()
_, r = mesh.pair_request(ip2, {"name": "Mac-mini-sl-ai", "code": "222222", "secret_sha256": dig}, t + 20)
mesh.pair_decide(r["id"], False, t + 21)
c, r2 = mesh.pair_status(ip2, {"id": r["id"], "secret": sec}, t + 22)
ok(c == 200 and r2["state"] == "denied" and "token" not in r2, "denied request gets no token")
ok(mesh.pair_status(ip2, {"id": r["id"], "secret": sec}, t + 23)[0] == 404, "denied request is gone after it was read")
# a new code from the same computer replaces the old request
_, a1 = mesh.pair_request(ip1, {"name": "x", "code": "333333", "secret_sha256": dig}, t + 30)
_, a2 = mesh.pair_request(ip1, {"name": "x", "code": "444444", "secret_sha256": dig}, t + 31)
ok([row["code"] for row in mesh.pair_list(t + 32)] == ["444444"], "a new code replaces the older request")
ok(mesh.pair_request(ip1, {"name": "x", "code": "12345", "secret_sha256": dig}, t + 33)[0] == 400, "bad code refused")
# rate limit per client and cap on pending requests
mesh._pairs.clear()
mesh._pair_hits.clear()
codes = [mesh.pair_request(ip1, {"code": "555555", "secret_sha256": dig}, t + 40 + n)[0] for n in range(mesh.PAIR_REQUEST_RATE[0] + 1)]
ok(codes[-1] == 429 and codes[:-1] == [200] * mesh.PAIR_REQUEST_RATE[0], "pair requests are rate-limited per client")
mesh._pairs.clear()
mesh._pair_hits.clear()
codes = [mesh.pair_request(f"192.168.7.{100 + n}", {"code": "666666", "secret_sha256": dig}, t + 60)[0] for n in range(mesh.PAIR_MAX_PENDING + 1)]
ok(codes[:-1] == [200] * mesh.PAIR_MAX_PENDING and codes[-1] == 429, f"at most {mesh.PAIR_MAX_PENDING} pending requests")
mesh._pairs.clear()
mesh._pair_hits.clear()

# main PC page check: loopback or this PC's own LAN address only
real_lan_host = mesh.lan_host
mesh.lan_host = lambda: "192.168.7.14"
ok(mesh.Handler.main_pc_page(_types.SimpleNamespace(client_address=("127.0.0.1", 1))), "loopback page may approve")
ok(mesh.Handler.main_pc_page(_types.SimpleNamespace(client_address=("192.168.7.14", 1))), "this PC's own LAN address may approve")
ok(not mesh.Handler.main_pc_page(_types.SimpleNamespace(client_address=("192.168.7.31", 1))), "another LAN computer may not approve")
mesh.lan_host = real_lan_host

# over HTTP on the serve handler
class ServePage(mesh.Handler):
    mode = "serve"


psrv = ThreadingHTTPServer(("127.0.0.1", 0), ServePage)
threading.Thread(target=psrv.serve_forever, daemon=True).start()
pbase = f"http://127.0.0.1:{psrv.server_address[1]}"


def pcall(path, body=None, headers=None):
    req = urllib.request.Request(pbase + path, data=None if body is None else json.dumps(body).encode(),
                                 headers=dict({"Content-Type": "application/json"}, **(headers or {})))
    try:
        with mesh._DIRECT.open(req, timeout=5) as resp:
            return resp.status, json.loads(resp.read() or b"{}")
    except urllib.error.HTTPError as e:
        return e.code, json.loads(e.read() or b"{}")


try:
    c, r = pcall("/pair/request", {"name": "OhProggraming", "code": "123456", "secret_sha256": dig})
    ok(c == 200, "POST /pair/request over HTTP")
    c, lst = pcall("/api/pair")
    ok(c == 200 and lst["pending"][0]["code"] == "123456", "GET /api/pair lists it for the local page")
    ok(pcall("/api/pair/approve", {"id": r["id"]}, {"Origin": "http://evil.example.com"})[0] == 403, "cross-site approve refused")
    try:
        with mesh._DIRECT.open(urllib.request.Request(pbase + "/pair/request", data=b"", method="POST", headers={"Host": "evil.example.com"}), timeout=5) as resp:
            host_code = resp.status
    except urllib.error.HTTPError as e:
        host_code = e.code
    ok(host_code == 403, "pair request Host check kept")
    ok(pcall("/pair/status", {"id": r["id"], "secret": sec})[1] == {"state": "pending"}, "status pending over HTTP")
    ok(pcall("/api/pair/approve", {"id": r["id"]})[0] == 200, "approve from the local page")
    c, r2 = pcall("/pair/status", {"id": r["id"], "secret": sec})
    ok(c == 200 and r2.get("token") == LONG, "status returns the token after approval")
    ok("token" not in pcall("/pair/status", {"id": r["id"], "secret": sec})[1], "second fetch over HTTP gets no token")
    ok("pending" in pcall("/api/pair")[1] and pcall("/api/pair")[1]["pending"] == [], "nothing pending afterwards")
finally:
    psrv.shutdown()
    psrv.server_close()
mesh._pairs.clear()
mesh._pair_hits.clear()

# worker side: pair_once against the serve functions, token written atomically
with tempfile.TemporaryDirectory() as tmp:
    mesh.TOKEN_PATH = Path(tmp) / "cluster.token"
    mesh.TOKEN_PATH.write_text("0123456789abcdef-old-random", encoding="utf-8")
    mesh.token = REAL_TOKEN  # the worker reads its file; serve's answer comes from serve_token below
    mesh._ROLE["name"] = "work"
    serve_token = "9e" * 24
    polls = {"n": 0}

    def fake_post(base, path, body, timeout=10.0):
        if path == "/pair/request":
            return mesh.pair_request(ip1, body, 50_000.0)
        polls["n"] += 1
        if polls["n"] == 2:
            mesh.pair_decide(next(iter(mesh._pairs)), True, 50_001.0)
        saved = mesh.token
        mesh.token = lambda: serve_token
        try:
            return mesh.pair_status(ip1, body, 50_002.0 + polls["n"])
        finally:
            mesh.token = saved

    out = _io.StringIO()
    with _contextlib.redirect_stdout(out):
        result = mesh.pair_once("http://192.168.7.14:8780", post=fake_post, sleep=lambda s: None, clock=lambda: 0.0)
    printed = out.getvalue()
    ok(result == "paired" and mesh.TOKEN_PATH.read_text(encoding="utf-8") == serve_token, "worker pairs and writes the main PC's token")
    ok((Path(tmp) / "cluster.token.old").read_text(encoding="utf-8") == "0123456789abcdef-old-random", "old token kept as cluster.token.old")
    ok(sorted(p.name for p in Path(tmp).iterdir()) == ["cluster.token", "cluster.token.old"], "no temp file left behind")
    ok("Pairing code:" in printed and serve_token not in printed and "-old-random" not in printed, "the code is printed, the token never")
    ok(mesh.saved_token() == serve_token and mesh.authorized("Bearer " + serve_token, mesh.saved_token()), "the new token is used at once")
    try:
        mesh.write_token("../../evil token")
        ok(False, "bad token written")
    except ValueError:
        ok(mesh.TOKEN_PATH.read_text(encoding="utf-8") == serve_token, "a malformed token from the main PC is refused")

    def denying_post(base, path, body, timeout=10.0):
        if path == "/pair/request":
            return mesh.pair_request(ip2, body, 60_000.0)
        mesh.pair_decide(next(iter(mesh._pairs)), False, 60_001.0)
        return mesh.pair_status(ip2, body, 60_002.0)

    with _contextlib.redirect_stdout(_io.StringIO()):
        ok(mesh.pair_once("http://192.168.7.14:8780", post=denying_post, sleep=lambda s: None, clock=lambda: 0.0) == "denied", "a denied worker keeps its file and backs off")
        ok(mesh.pair_once("http://192.168.7.14:8780", post=lambda *a, **k: (429, {"error": "busy"}), sleep=lambda s: None) == "retry", "a refused pairing request is retried later")
    ok(mesh.TOKEN_PATH.read_text(encoding="utf-8") == serve_token, "denial does not touch cluster.token")

    # pair_loop wakes the job loop after pairing (no restart)
    real_pair_once = mesh.pair_once
    mesh.pair_once = lambda base: "paired"
    mesh._token_changed.clear()
    mesh.want_pairing()
    pair_thread = threading.Thread(target=mesh.pair_loop, args=("http://192.168.7.14:8780",), daemon=True)
    pair_thread.start()
    ok(mesh._token_changed.wait(5), "pairing wakes the job loop")
    time.sleep(0.2)
    ok(not mesh._pair_wanted.is_set(), "and stops asking once paired")

    def _end_loop(base):
        raise SystemExit()

    mesh.pair_once = _end_loop  # let the test thread end instead of waiting forever
    mesh._pair_wanted.set()
    pair_thread.join(5)
    mesh._pair_wanted.clear()
    mesh._token_changed.clear()
    mesh.pair_once = real_pair_once
    ok(not pair_thread.is_alive(), "test pair thread ended")

    # a worker without a token never makes one; serve still does on first launch
    mesh.TOKEN_PATH = Path(tmp) / "none" / "cluster.token"
    (Path(tmp) / "none").mkdir()
    try:
        REAL_TOKEN()
        ok(False, "worker made a token")
    except RuntimeError:
        ok(not mesh.TOKEN_PATH.exists() and mesh.saved_token() == "", "a worker without cluster.token never creates one")
    try:
        mesh._mesh_post("http://127.0.0.1:9", "/jobs/next", {}, 1)
    except mesh.JobAuthError as exc:
        ok(exc.missing, "the job loop sees the missing token (and starts pairing)")
    mesh._ROLE["name"] = "serve"
    made = REAL_TOKEN()
    ok(len(made) == 48 and mesh.TOKEN_PATH.exists(), "serve still creates its token on first launch")
mesh.token, mesh.TOKEN_PATH = real_token_fn, real_token_path
mesh._ROLE.update(real_role)
mesh.token = lambda: FAKE
mesh._pairs.clear()
mesh._pair_hits.clear()
mesh._pair_wanted.clear()


# --- Linux GPU detection from sysfs (AMD / Intel), no extra packages ---------
with tempfile.TemporaryDirectory() as tmp:
    drm = Path(tmp) / "drm"

    def fake_card(name, vendor, device, files=None, temp=None):
        dev = drm / name / "device"
        dev.mkdir(parents=True)
        (dev / "vendor").write_text(vendor + "\n")
        (dev / "device").write_text(device + "\n")
        for key, value in (files or {}).items():
            (dev / key).write_text(str(value) + "\n")
        if temp is not None:
            (dev / "hwmon" / "hwmon3").mkdir(parents=True)
            (dev / "hwmon" / "hwmon3" / "temp1_input").write_text(str(temp) + "\n")

    G = 1024 ** 3
    fake_card("card0", "0x1002", "0x163f", {"mem_info_vram_total": G, "mem_info_vram_used": G // 4, "mem_info_gtt_total": 8 * G, "mem_info_gtt_used": G}, temp=45000)
    (drm / "card0-eDP-1").mkdir()  # a connector, not a GPU
    fake_card("card1", "0x1002", "0x73df", {"mem_info_vram_total": 12 * G, "mem_info_vram_used": 2 * G, "mem_info_gtt_total": 16 * G})
    fake_card("card2", "0x8086", "0x46a6")
    (drm / "card3" / "device").mkdir(parents=True)  # no ids: skipped
    names = {(0x1002, 0x73DF): "Navi 22"}
    cards = mesh.linux_cards(drm, names)
    ok(len(cards) == 3, f"three GPUs from sysfs, connector and id-less entries skipped ({len(cards)})")
    deck, dgpu, igpu = cards
    ok(deck["name"] == "AMD Radeon (Steam Deck APU)" and deck["vendor"] == "amd" and deck.get("unified"), "Steam Deck APU named and marked unified")
    ok(deck["vram_mib"] == 1024 and deck["gtt_mib"] == 8192 and deck["memory_total_mib"] == 9216 and deck["memory_used_mib"] == 256 + 1024, "APU memory is VRAM carve-out plus GTT")
    ok(deck["temperature_c"] == 45 and deck["pci_id"] == "1002:163f", "APU temperature from hwmon")
    ok(dgpu["name"] == "AMD Navi 22" and not dgpu.get("unified") and dgpu["memory_total_mib"] == 12288 and dgpu["memory_used_mib"] == 2048, "AMD dGPU named from lspci, VRAM only")
    ok(igpu["name"] == "Intel GPU" and igpu["vendor"] == "intel" and igpu.get("unified") and igpu["temperature_c"] is None, "Intel iGPU without lspci name")
    ok(mesh.gpu_name(0x1002, 0x1435) == "AMD Radeon (Steam Deck APU)" and mesh.gpu_name(0x1002, 0x1234, {}) == "AMD GPU", "Deck OLED id and plain AMD fallback")
    ok(mesh.linux_cards(Path(tmp) / "missing", {}) == [], "no /sys/class/drm -> no cards, no error")
    lspci_out = (
        '04:00.0 "VGA compatible controller [0300]" "Advanced Micro Devices, Inc. [AMD/ATI] [1002]" "VanGogh [AMD Custom GPU 0405] [163f]" -rae "Valve [1e44]" "Device [1234]"\n'
        '03:00.0 "Ethernet controller [0200]" "Realtek [10ec]" "RTL8111 [8168]"\n'
        '00:02.0 "VGA compatible controller [0300]" "Intel Corporation [8086]" "Alder Lake-P GT2 [Iris Xe Graphics] [46a6]"\n'
    )
    parsed = mesh.parse_lspci(lspci_out)
    ok(parsed == {(0x1002, 0x163F): "VanGogh [AMD Custom GPU 0405]", (0x8086, 0x46A6): "Alder Lake-P GT2 [Iris Xe Graphics]"}, f"lspci -mm -nn parsed, non-display skipped {parsed}")
    ok(mesh.gpu_name(0x1002, 0x163F, parsed) == "AMD Radeon (Steam Deck APU)", "Deck name wins over the lspci code name")
    ok(mesh.gpu_name(0x8086, 0x46A6, parsed) == "Intel Alder Lake-P GT2 [Iris Xe Graphics]", "Intel named from lspci")

    # accelerator and CUDA rules
    ok(mesh.pick_accelerator([deck], False, True) == "vulkan", "AMD + local llama-server up -> vulkan")
    ok(mesh.pick_accelerator([deck], False, False) == "cpu", "AMD without a running model stays cpu")
    ok(mesh.pick_accelerator([igpu], False, True) == "vulkan", "Intel + model up -> vulkan")
    ok(mesh.pick_accelerator([{"name": "RTX 4070", "memory_total_mib": 12282}], False, True) == "cuda", "nvidia-smi card (no vendor field) still cuda")
    ok(mesh.pick_accelerator([{"name": "Apple M2", "unified": True}], True, True) == "metal", "Apple path still metal")
    ok(not mesh.is_cuda_card(dgpu) and mesh.is_cuda_card({"name": "RTX 3050"}), "an AMD dGPU is never counted as CUDA")
    amd_box = {"name": "maor", "os": "linux", "ram": {"total_mib": 16000, "free_mib": 9000}, "gpus": [dgpu],
               "caps": {"cpu_cores": 8, "free_ram_mib": 9000, "cpu_busy_pct": 5}}
    ok(not any(job["job"] == "eval" for job in mesh.side_jobs(amd_box, False, None, False)), "no 'Eval on CUDA' for an AMD GPU")
    real_cached = mesh.local_model_cached
    mesh.local_model_cached = lambda ttl=15.0: True
    caps = mesh.capabilities([deck], {"free_mib": 4000}, False, False)
    ok(caps["accelerator"] == "vulkan" and caps["local_model"] is True and caps["vram_total_mib"] == 0, "caps: Deck with a model is vulkan, reports local_model")
    mesh.local_model_cached = lambda ttl=15.0: False
    caps = mesh.capabilities([deck], {"free_mib": 4000}, True, False)
    ok(caps["accelerator"] == "vulkan" and caps["local_model"] is False, "a running llama-server process alone also means vulkan")
    mesh.local_model_cached = real_cached
    mesh._model_cache.update({"at": 0.0, "up": False})

# --- data factory rates, pass rates and ETA from finish timestamps -----------
with tempfile.TemporaryDirectory() as tmp:
    q = mesh.JobQueue(Path(tmp))
    T = 2_000_000.0
    gens = q.add_many([("generate", {"prompt": f"q{i}", "verify": {"kind": "verify_math", "expected": "1"}}) for i in range(6)])
    plain = q.add_many([("verify_python_syntax", {"code": "x = 1"}) for _ in range(4)])
    luau = q.add("verify_luau", {"code": "local x = 1"})
    # one finish long before the window: counted in totals, not in the rate
    old = q.lease("Mac-mini-sl-ai", ["generate"], T - 7200)
    q.finish({"id": old["id"], "lease": old["lease"], "node": "Mac-mini-sl-ai", "ok": True, "output": {"text": "\\boxed{1}"}}, T - 7100)
    for n, at in enumerate([T - 1800, T - 600]):
        job = q.lease("Mac-mini-sl-ai", ["generate"], at - 10)
        q.finish({"id": job["id"], "lease": job["lease"], "node": "Mac-mini-sl-ai", "ok": True, "output": {"text": "\\boxed{2}"}}, at)
    for at, kind in ((T - 900, "verify_math"), (T - 300, "verify_math"), (T - 100, "verify_math"), (T - 50, "verify_python_syntax")):
        job = q.lease("OhProggraming", [kind], at - 5)
        good, out = mesh.run_job(job)
        q.finish({"id": job["id"], "lease": job["lease"], "node": "OhProggraming", "ok": good, "output": out}, at)
    leased_now = q.lease("maor", ["generate"], T - 20)
    s = q.summary(T)
    nodes = s["nodes"]
    ok(nodes["mac-mini-sl-ai"]["done"] == 3 and nodes["mac-mini-sl-ai"]["hour_done"] == 2, f"per computer: total done 3, 2 in the last hour {nodes['mac-mini-sl-ai']}")
    ok(nodes["mac-mini-sl-ai"]["last_s"] == 600, "seconds since that computer's last finished job")
    ok(nodes["ohproggraming"]["hour_done"] == 4 and nodes["ohproggraming"]["last_s"] == 50, "verifier computer counted")
    ok(nodes["maor"]["now"] == "generate" and nodes["maor"]["done"] == 0 and nodes["maor"]["last_s"] is None, "a computer with only a lease shows what it runs now")
    rate = s["rate"]
    ok(rate["per_hour"] == 6.0 and rate["window_s"] == 3600, f"6 jobs finished in the last hour {rate['per_hour']}")
    gen_left = rate["by_kind"]["generate"]["left"]
    ok(gen_left == 3, f"generate: 2 queued + 1 leased left ({gen_left})")
    ok(rate["by_kind"]["generate"]["eta_s"] == 3 * 3600 // 2, "generate ETA = left / last-hour rate")
    vm = rate["by_kind"]["verify_math"]
    ok(vm["later"] == 3 and vm["left"] == 3 and vm["eta_s"] == rate["by_kind"]["generate"]["eta_s"], f"verify jobs still to be made wait on generate {vm}")
    ok(rate["stalled"] == ["verify_luau"] and "verify_python_syntax" not in rate["stalled"] and rate["by_kind"]["verify_luau"]["eta_s"] is None, "a kind nobody takes is stalled, without an ETA")
    ok(rate["eta_s"] == max(row["eta_s"] for row in rate["by_kind"].values() if row["eta_s"] is not None), "queue ETA is the slowest moving kind")
    ok(rate["left"] == sum(row["left"] for row in rate["by_kind"].values()), "left adds up")
    ver = s["verifiers"]
    ok(set(ver) == set(mesh.VERIFY_KINDS) and ver["verify_luau"]["pct"] is None, "pass rate per verifier kind, none when nothing checked")
    vm_row = s["kinds"]["verify_math"]
    ok(ver["verify_math"]["checked"] == vm_row["done"] + vm_row["failed"] and ver["verify_math"]["passed"] == vm_row["passed"], "verifier pass counts match the kind table")
    py = ver["verify_python_syntax"]
    ok(py["checked"] >= 1 and py["pct"] == round(100 * py["passed"] / py["checked"]), f"pass rate is passed / checked {py}")
    ok(set(["kinds", "nodes", "verified_rows", "total"]) <= set(s), "old summary keys kept")
    # a restarted serve rebuilds the same numbers from the log's timestamps
    again = mesh.JobQueue(Path(tmp)).summary(T)
    ok(again["rate"]["per_hour"] == 6.0 and again["nodes"]["mac-mini-sl-ai"]["hour_done"] == 2, "rates rebuilt from finish times in queue.jsonl after a restart")
    # finish events without a time (older logs) are still counted, just not in rates
    q._append(q.queue_path, [{"op": "finish", "id": plain[-1], "ok": True, "passed": True, "node": "oldbox"}])
    s2 = q.summary(T)
    ok(s2["nodes"]["oldbox"]["done"] == 1 and s2["nodes"]["oldbox"]["hour_done"] == 0 and s2["nodes"]["oldbox"]["last_s"] is None, "untimed finish counted in totals only")
    empty = mesh.JobQueue(Path(tmp) / "none").summary(T)
    ok(empty["rate"]["eta_s"] == 0 and empty["rate"]["left"] == 0 and empty["rate"]["per_hour"] == 0, "empty queue: nothing left")
    ok(mesh.queue_rate({"generate": {"queued": 4, "leased": 0}}, {}, {})["eta_s"] is None, "no recent rate -> no ETA")

# --- finding the main PC without NEURO_MESH_URL ------------------------------
ok(mesh.lan_url("http://192.168.7.14:8780") == "http://192.168.7.14:8780" and mesh.lan_url("http://127.0.0.1:8780"), "LAN and loopback mesh URLs ok")
ok(not mesh.lan_url("http://26.1.2.3:8780") and not mesh.lan_url("http://192.168.56.1:8780") and not mesh.lan_url("http://8.8.8.8:8780") and not mesh.lan_url("http://169.254.1.1:8780"), "VPN, host-only, link-local and public mesh URLs refused")
ok(mesh.probe_serve("26.1.1.1") is None and mesh.probe_serve("127.0.0.1") is None and mesh.probe_serve("8.8.8.8") is None, "probe_serve never touches a non-LAN address")
real_disc = {name: getattr(mesh, name) for name in ("ipv4_prefixes", "lan_host", "MESH_URL_PATH", "saved_token")}
real_env = os.environ.pop("NEURO_MESH_URL", None)
with tempfile.TemporaryDirectory() as tmp:
    try:
        probed = []
        mesh.lan_host = lambda: "192.168.7.14"
        mesh.ipv4_prefixes = lambda: (["26.0.0", "192.168.56"], {"26.0.0.5"})
        found = mesh.discover_serves(lambda ip: probed.append(ip) or ({"url": f"http://{ip}:8780", "name": "ORI"} if ip == "192.168.7.14" else None))
        ok(found == [{"url": "http://192.168.7.14:8780", "name": "ORI"}], "discovery finds the one serve")
        ok(len(probed) == 254 and probed[0] == "192.168.7.14" and all(ip.startswith("192.168.7.") for ip in probed), "only its own /24, own address first (main PC may run a worker)")
        mesh.lan_host = lambda: ""
        mesh.ipv4_prefixes = lambda: (["26.0.0", "192.168.56"], set())
        probed.clear()
        ok(mesh.discover_serves(lambda ip: probed.append(ip)) == [] and probed == [], "VPN and host-only nets never swept")

        mesh.MESH_URL_PATH = Path(tmp) / "mesh_url.txt"
        mesh.saved_token = lambda: ""
        said = []
        one = lambda: [{"url": "http://192.168.7.14:8780", "name": "ORI"}]
        os.environ["NEURO_MESH_URL"] = "http://192.168.7.20:8780"
        ok(mesh.find_mesh_url(discover=lambda: 1 / 0, say=said.append) == "http://192.168.7.20:8780" and not mesh.MESH_URL_PATH.exists(), "NEURO_MESH_URL wins, no search")
        os.environ["NEURO_MESH_URL"] = "http://26.1.1.1:8780"
        ok(mesh.find_mesh_url(discover=lambda: 1 / 0, say=said.append) == "", "a VPN NEURO_MESH_URL is still refused")
        del os.environ["NEURO_MESH_URL"]
        url = mesh.find_mesh_url(discover=one, say=said.append)
        ok(url == "http://192.168.7.14:8780" and mesh.MESH_URL_PATH.read_text().strip() == url, "one serve found -> used and remembered in mesh_url.txt")
        ok(any("approved" in line for line in said), "says pairing still needs approval on the main PC")
        said.clear()
        ok(mesh.find_mesh_url(discover=lambda: 1 / 0, say=said.append) == url and any("remembered" in line for line in said), "remembered address used later without a search")
        mesh.MESH_URL_PATH.write_text("http://26.1.1.1:8780\n")
        ok(mesh.remembered_mesh_url() == "", "a remembered non-LAN address is ignored")
        mesh.MESH_URL_PATH.unlink()
        said.clear()
        two = lambda: [{"url": "http://192.168.7.14:8780", "name": "ORI"}, {"url": "http://192.168.7.30:8780", "name": "OTHER"}]
        ok(mesh.find_mesh_url(discover=two, say=said.append) == "" and not mesh.MESH_URL_PATH.exists(), "several serves -> none picked, nothing remembered")
        ok(any("192.168.7.30" in line for line in said) and any("NEURO_MESH_URL" in line for line in said), "several serves are listed with the NEURO_MESH_URL hint")
        ok(mesh.find_mesh_url(discover=lambda: [], say=said.append) == mesh.DEFAULT_MESH_URL and not mesh.MESH_URL_PATH.exists(), "none answer -> old default as last fallback, not remembered")
        # a worker that already holds a token only takes a serve that proves it
        mesh.saved_token = lambda: "worker-token"
        asked = []
        ok(mesh.find_mesh_url(discover=one, prove=lambda u, s: asked.append((u, s)) or False, say=said.append) == "" and not mesh.MESH_URL_PATH.exists(), "found serve without this worker's token is not used")
        ok(asked == [("http://192.168.7.14:8780", "worker-token")], "the proof check gets the token locally (it is never sent)")
        ok(mesh.find_mesh_url(discover=two, prove=lambda u, s: u.endswith(".30:8780"), say=said.append) == "http://192.168.7.30:8780", "of two serves, the one that proves the token is used")
        ok(not any("worker-token" in line for line in said), "the token is never printed")
    finally:
        for name, value in real_disc.items():
            setattr(mesh, name, value)
        os.environ.pop("NEURO_MESH_URL", None)
        if real_env is not None:
            os.environ["NEURO_MESH_URL"] = real_env
        mesh._mesh_url_cache["url"] = None

# serve answers /hello with role "serve" and an HMAC proof; a worker says "work"
mesh.Handler.mode = "serve"
srv = ThreadingHTTPServer(("127.0.0.1", 0), mesh.Handler)
threading.Thread(target=srv.serve_forever, daemon=True).start()
sbase = f"http://127.0.0.1:{srv.server_address[1]}"
with mesh._DIRECT.open(sbase + "/hello", timeout=5) as r:
    hello = json.loads(r.read())
ok(hello.get("app") == "neuroai-mesh" and hello.get("role") == "serve" and hello.get("port") == mesh.DASHBOARD_PORT, "serve GET /hello says role serve")
ok(not any(FAKE in json.dumps(v) for v in hello.values()), "serve /hello holds no token")
ok(mesh.serve_proof_ok(sbase, FAKE), "serve proves it holds the token (HMAC of the worker's nonce)")
ok(not mesh.serve_proof_ok(sbase, "some-other-token"), "a serve with another token fails the proof")
try:
    with mesh._DIRECT.open(sbase + "/hello/proof?nonce=zz", timeout=5) as r:
        code = r.status
except urllib.error.HTTPError as exc:
    code = exc.code
ok(code == 400, "serve /hello/proof refuses a bad nonce")
try:
    with mesh._DIRECT.open(urllib.request.Request(sbase + "/hello", headers={"Host": "evil.example.com"}), timeout=5) as r:
        code = r.status
except urllib.error.HTTPError as exc:
    code = exc.code
ok(code == 403, "serve /hello keeps the rebinding Host check")
srv.shutdown()
srv.server_close()
real_ls = mesh.local_status
mesh.local_status = lambda: {"name": "w", "os": "linux", "ram": {}}
mesh.Handler.mode = "work"
wsrv = ThreadingHTTPServer(("127.0.0.1", 0), mesh.Handler)
threading.Thread(target=wsrv.serve_forever, daemon=True).start()
with mesh._DIRECT.open(f"http://127.0.0.1:{wsrv.server_address[1]}/hello", timeout=5) as r:
    whello = json.loads(r.read())
ok(whello.get("role") == "work" and whello.get("app") == "neuroai-mesh", "worker GET /hello says role work")
wsrv.shutdown()
wsrv.server_close()
mesh.local_status = real_ls
mesh.Handler.mode = "serve"
ok(mesh.pull_worker.__doc__ and "legacy" in mesh.pull_worker.__doc__, "LAN sweep of workers unchanged")

# --- page: data factory view renders with textContent only -------------------
page = mesh.DASHBOARD.read_text(encoding="utf-8")
for element in ("f-rows", "f-rate", "f-left", "f-eta", "jobs-verifiers", "jobs-nodes", "jobs-stalled"):
    ok(f'id="{element}"' in page, f"page has #{element}")
factory = page[page.index("function duration("):page.index("function appRows(")]
ok("innerHTML" not in factory and "insertAdjacentHTML" not in factory and "textContent" in factory, "factory view uses textContent only")
ok("modelLine(item)" in page and "No local model on this computer" in page, "page shows local model status and a no-model hint")


# refusals read a small body first, so the client gets the 401/403 instead of a reset
mesh.Handler.mode = "serve"
dsrv = ThreadingHTTPServer(("127.0.0.1", 0), mesh.Handler)
threading.Thread(target=dsrv.serve_forever, daemon=True).start()
dbase = f"http://127.0.0.1:{dsrv.server_address[1]}"
codes = []
for _ in range(40):
    req = urllib.request.Request(dbase + "/jobs/next", data=json.dumps({"node": "x", "kinds": ["verify_math"], "pad": "y" * 2000}).encode(), headers={"Content-Type": "application/json"})
    try:
        with mesh._DIRECT.open(req, timeout=5) as r:
            codes.append(r.status)
    except urllib.error.HTTPError as exc:
        codes.append(exc.code)
    except OSError as exc:
        codes.append(type(exc).__name__)
ok(codes == [401] * 40, f"40 tokenless POSTs all get a clean 401 {set(map(str, codes))}")
dsrv.shutdown()
dsrv.server_close()

# --- truncated generations: one long retry, never verified or exported when cut off ---
ok(mesh.GENERATE_MAX_TOKENS == 4096 and mesh.LONG_RETRY_TOKENS == 4096, "generate cap and long retry are 4096 tokens")
ok(mesh.validate_job("generate", {"prompt": "x", "max_tokens": 99999})["max_tokens"] == 4096, "server-side clamp is 4096")
ok(mesh.validate_job("generate", {"prompt": "x", "long_retry": True}).get("long_retry") is True
   and "long_retry" not in mesh.validate_job("generate", {"prompt": "x", "long_retry": "yes"}), "long_retry kept only as true")
ok(mesh.is_truncated({"finish_reason": "length"}) and not mesh.is_truncated({"finish_reason": "stop"}) and not mesh.is_truncated(None), "is_truncated reads finish_reason")
CUT_HTML = "```html\n<!doctype html><html><head><style>body { color: red; "
FULL_HTML = "```html\n<!doctype html>\n<html><head><title>t</title><style>body { color: red; }</style></head><body><main><p>hi</p></main></body></html>\n```"


def cut(text, tokens=2048):
    return {"text": text, "finish_reason": "length", "model": "fake", "usage": {"prompt_tokens": 50, "completion_tokens": tokens}}


def full(text):
    return {"text": text, "finish_reason": "stop", "model": "fake", "usage": {"prompt_tokens": 50, "completion_tokens": 300}}


def kind_ids(q, kind):
    return [jid for jid in q.order if q.jobs[jid]["kind"] == kind]


with tempfile.TemporaryDirectory() as tmp:
    q = mesh.JobQueue(Path(tmp))
    T = 3_000_000.0
    gid = q.add_many([("generate", {"prompt": "Make a page.", "samples": 2, "max_tokens": 2048, "meta": {"lang": "html", "id": "p1"},
                                    "verify": {"kind": "verify_html_css"}})])
    first = q.lease("oldbox", ["generate"], T)
    ok(first["id"] == gid[0], "first sample leased")
    accepted, note = q.finish({"id": gid[0], "lease": first["lease"], "node": "oldbox", "ok": True, "output": cut(CUT_HTML)}, T + 1)
    ok(accepted and "long retry queued" in note, f"cut-off result saved, long retry queued: {note}")
    ok(not kind_ids(q, "verify_html_css"), "no verify job for a cut-off answer")
    retries = [jid for jid in kind_ids(q, "generate") if q.jobs[jid]["payload"].get("long_retry")]
    ok(len(retries) == 1, "exactly one replacement generate job")
    rp, op = q.jobs[retries[0]]["payload"], q.jobs[gid[0]]["payload"]
    ok(rp["max_tokens"] == 4096 and rp["messages"] == op["messages"] and rp["group"] == op["group"] and rp["sample"] == op["sample"]
       and rp["seed"] == op["seed"] and rp["verify"] == op["verify"] and rp["meta"] == op["meta"] and "samples" not in rp,
       "retry: same prompt, group, sample, seed, verify, meta; 4096 tokens")
    ok(q.jobs[retries[0]]["parent"] == gid[0], "retry points at the cut-off job")
    # A worker that does not say gen_max ran the old 2048 cap: it never gets the 4096 retry.
    second = q.lease("oldbox", ["generate"], T + 2)
    ok(second["id"] == gid[1], "old worker gets the 2048-token sample, not the retry")
    q.finish({"id": gid[1], "lease": second["lease"], "node": "oldbox", "ok": True, "output": full(FULL_HTML)}, T + 3)
    ok(q.lease("oldbox", ["generate"], T + 4) is None, "old worker: no job above 2048 tokens")
    long_lease = q.lease("newbox", ["generate"], T + 5, gen_max=4096)
    ok(long_lease["id"] == retries[0] and long_lease["payload"]["max_tokens"] == 4096, "new worker (gen_max 4096) gets the long retry")
    accepted, note = q.finish({"id": retries[0], "lease": long_lease["lease"], "node": "newbox", "ok": True, "output": cut(CUT_HTML + "x" * 50, 4096)}, T + 6)
    ok(accepted and "truncated, not verified" in note, f"long retry still cut off: kept as truncated ({note})")
    ok(len([j for j in kind_ids(q, "generate") if q.jobs[j]["payload"].get("long_retry")]) == 1, "no second retry")
    ok(len(kind_ids(q, "verify_html_css")) == 1, "only the complete answer is checked")
    check = q.lease("checker", ["verify_html_css"], T + 7)
    good, out = mesh.run_job(check)
    q.finish({"id": check["id"], "lease": check["lease"], "node": "checker", "ok": good, "output": out}, T + 8)
    ok(good and out["passed"] and q.jobs[gid[1]]["verified"] is True, f"complete answer passes {out}")
    # A retry that finishes gets its verify job like any answer.
    single = q.add("generate", {"prompt": "Another page.", "max_tokens": 1024, "verify": {"kind": "verify_html_css"}})
    lease = q.lease("newbox", ["generate"], T + 10, gen_max=4096)
    ok(lease["id"] == single, "1024-token job leased")
    q.finish({"id": single, "lease": lease["lease"], "node": "newbox", "ok": True, "output": cut(CUT_HTML, 1024)}, T + 11)
    lease = q.lease("newbox", ["generate"], T + 12, gen_max=4096)
    ok(q.jobs[lease["id"]]["parent"] == single and lease["payload"]["max_tokens"] == 4096, "a 1024-token cut-off also gets the long retry")
    accepted, note = q.finish({"id": lease["id"], "lease": lease["lease"], "node": "newbox", "ok": True, "output": full(FULL_HTML)}, T + 13)
    ok("verify_html_css queued" in note, f"a finished retry queues its check: {note}")
    check = q.lease("checker", ["verify_html_css"], T + 14)
    ok(q.jobs[check["id"]]["parent"] == lease["id"], "the check belongs to the retry")
    good, out = mesh.run_job(check)
    q.finish({"id": check["id"], "lease": check["lease"], "node": "checker", "ok": good, "output": out}, T + 15)
    # Jobs already 4096 tokens long are not retried.
    big = q.add("generate", {"prompt": "Big page.", "max_tokens": 4096, "verify": {"kind": "verify_html_css"}})
    lease = q.lease("newbox", ["generate"], T + 16, gen_max=4096)
    ok(lease["id"] == big, "4096-token job leased to the new worker")
    note = q.finish({"id": big, "lease": lease["lease"], "node": "newbox", "ok": True, "output": cut(CUT_HTML, 4096)}, T + 17)[1]
    ok("truncated, not verified" in note and q.lease("newbox", ["generate"], T + 18, gen_max=4096) is None, "a 4096-token cut-off is not retried")
    out, count = q.export("v.jsonl", per_prompt=0)
    rows = [json.loads(line) for line in out.read_text(encoding="utf-8").splitlines()]
    ok(count == 2 and {r["job"] for r in rows} == {gid[1], q.jobs[check["id"]]["parent"]}, "export: only complete, verified answers")
    out, count = q.export_pairs("pairs.jsonl")
    ok(count == 0, "a cut-off answer is never a rejected pair")
    s = q.summary(T + 20)
    vh = s["verifiers"]["verify_html_css"]
    ok(s["kinds"]["generate"]["truncated"] == 4 and vh["truncated"] == 4 and vh["retried"] == 2 and vh["truncated_final"] == 2,
       f"/api/jobs counts truncated per kind {vh} {s['kinds']['generate']}")
    ok(vh["checked"] == 2 and vh["passed"] == 2 and vh["pct"] == 100, "pass rate counts only checked answers")
    again = mesh.JobQueue(Path(tmp)).summary(T + 20)["verifiers"]["verify_html_css"]
    ok(again == vh, "truncated counts rebuilt from the logs after a restart")
    ok(s["verifiers"]["verify_sql"]["truncated"] == 0 and "truncated" in s["verifiers"]["verify_luau"], "every verifier row carries truncated counts")

# Older log lines: cut-off answers that already got a verify job are skipped too.
with tempfile.TemporaryDirectory() as tmp:
    q = mesh.JobQueue(Path(tmp))
    T = 3_100_000.0
    gid = q.add_many([("generate", {"prompt": "Write Python.", "samples": 3, "max_tokens": 2048, "verify": {"kind": "verify_python_syntax"}})])
    texts = {gid[0]: ("```python\nx = 1\n```", "stop"), gid[1]: ("```python\ny = 2\n```", "length"), gid[2]: ("```python\ndef f(:\n", "length")}
    for n, jid in enumerate(gid):
        text, reason = texts[jid]
        # what the old server wrote: the result line, the finish line and a verify job
        q._append(q.results_path, [{"id": jid, "kind": "generate", "parent": "", "node": "oldbox", "ok": True, "passed": None,
                                    "output": {"text": text, "finish_reason": reason, "model": "m", "usage": {"completion_tokens": 2048}}, "at": int(T + n)}])
        q._append(q.queue_path, [{"op": "finish", "id": jid, "ok": True, "passed": None, "node": "oldbox", "at": int(T + n)}])
        q.add("verify_python_syntax", mesh.verify_payload({"kind": "verify_python_syntax"}, text), parent=jid)
    for step in range(3):
        lease = q.lease("checker", ["verify_python_syntax"], T + 10 + step)
        good, out = mesh.run_job(lease)
        q.finish({"id": lease["id"], "lease": lease["lease"], "node": "checker", "ok": good, "output": out}, T + 10 + step)
    ok([q.jobs[j]["verified"] for j in gid] == [True, True, False], "old log: verdicts as the verifier gave them")
    out, count = q.export("v.jsonl", per_prompt=0)
    rows = [json.loads(line) for line in out.read_text(encoding="utf-8").splitlines()]
    ok(count == 1 and rows[0]["job"] == gid[0], "verified export skips an old cut-off answer even when it passed")
    out, count = q.export_pairs("pairs.jsonl")
    ok(count == 0, "pairs skip old cut-off answers (neither chosen nor rejected)")
    ok(q.summary(T + 20)["verified_rows"] == 1, "verified_rows leaves out cut-off answers, like the export")
    vp = q.summary(T + 20)["verifiers"]["verify_python_syntax"]
    ok(vp["truncated"] == 2 and vp["truncated_final"] == 2 and vp["retried"] == 0, f"old cut-off answers counted from results.jsonl {vp}")

# HTTP: /jobs/next passes gen_max; the worker loop sends it.
with tempfile.TemporaryDirectory() as tmp:
    mesh._queue = mesh.JobQueue(Path(tmp))
    mesh._queue.add("generate", {"prompt": "long", "max_tokens": 4096})
    mesh.Handler.mode = "serve"
    gsrv = ThreadingHTTPServer(("127.0.0.1", 0), mesh.Handler)
    threading.Thread(target=gsrv.serve_forever, daemon=True).start()

    def gcall(body, headers):
        req = urllib.request.Request(f"http://127.0.0.1:{gsrv.server_address[1]}/jobs/next", data=json.dumps(body).encode(), headers=headers)
        try:
            with mesh._DIRECT.open(req, timeout=5) as r:
                return r.status, r.read()
        except urllib.error.HTTPError as e:
            return e.code, e.read()

    auth = {"Authorization": "Bearer " + FAKE, "Content-Type": "application/json"}
    ok(gcall({"node": "x", "kinds": ["generate"], "gen_max": 4096}, {"Content-Type": "application/json"})[0] == 401, "gen_max does not skip the token check")
    ok(gcall({"node": "oldbox", "kinds": ["generate"]}, auth)[0] == 204, "/jobs/next: a worker without gen_max gets no 4096 job")
    ok(gcall({"node": "oldbox", "kinds": ["generate"], "gen_max": "lots"}, auth)[0] == 204, "a junk gen_max counts as 2048")
    code, raw = gcall({"node": "newbox", "kinds": ["generate"], "gen_max": 4096}, auth)
    ok(code == 200 and json.loads(raw)["payload"]["max_tokens"] == 4096, "/jobs/next: gen_max 4096 gets it")
    gsrv.shutdown()
    gsrv.server_close()
    mesh._queue = None
mesh_src = Path(mesh.__file__).read_text(encoding="utf-8")
ok('"gen_max": _gen_cap["tokens"]' in mesh_src and '_gen_cap = {"tokens": GENERATE_MAX_TOKENS}' in mesh_src,
   "the worker loop sends gen_max (starting at GENERATE_MAX_TOKENS) with each lease request")

# --- luau lookup: NEURO_LUAU, then PATH, then the per-user folder -------------
with tempfile.TemporaryDirectory() as tmp:
    saved = {key: os.environ.get(key) for key in ("NEURO_LUAU", "NEURO_LUAU_BIN", "LOCALAPPDATA", "PATH")}
    try:
        for key in ("NEURO_LUAU", "NEURO_LUAU_BIN"):
            os.environ.pop(key, None)
        os.environ["LOCALAPPDATA"] = str(Path(tmp) / "appdata")
        empty_bin = Path(tmp) / "emptybin"
        empty_bin.mkdir()
        os.environ["PATH"] = str(empty_bin)
        home = mesh.user_luau_path()
        ok(home == Path(tmp) / "appdata" / "Neuro" / "luau" / "luau.exe", f"per-user luau path is %LOCALAPPDATA%\\Neuro\\luau\\luau.exe ({home})")
        ok(mesh.luau_path() is None, "no luau anywhere -> none")
        home.parent.mkdir(parents=True)
        home.write_bytes(b"MZ-not-run")
        ok(mesh.luau_path() == str(home), "luau found in %LOCALAPPDATA%\\Neuro\\luau")
        mesh._kinds_cache["at"] = 0.0
        ok("verify_luau" in mesh.worker_kinds(), "a worker with the per-user luau takes verify_luau")
        on_path = Path(tmp) / "pathbin"
        on_path.mkdir()
        (on_path / "luau.exe").write_bytes(b"MZ-not-run")
        os.environ["PATH"] = str(on_path)
        ok(Path(mesh.luau_path()).parent == on_path, "luau on PATH wins over the per-user folder")
        chosen = Path(tmp) / "chosen.exe"
        chosen.write_bytes(b"MZ-not-run")
        os.environ["NEURO_LUAU"] = str(chosen)
        ok(mesh.luau_path() == str(chosen), "NEURO_LUAU wins over everything")
        ok('Path.home() / ".local" / "share" / "neuro" / "luau" / "luau"' in mesh_src, "Linux/Mac: ~/.local/share/neuro/luau/luau")
    finally:
        for key, value in saved.items():
            if value is None:
                os.environ.pop(key, None)
            else:
                os.environ[key] = value
        mesh._kinds_cache["at"] = 0.0

# --- worker scripts: opt-out luau download, assets[].browser_download_url only --
repo = REPO
for rel in ("start-worker.ps1", "platforms/windows-worker.ps1"):
    raw = (repo / rel).read_bytes()
    text = raw.decode("utf-8")
    ok(raw.count(b"\r\n") == raw.count(b"\n"), f"{rel} keeps CRLF line endings")
    ok("https://api.github.com/repos/luau-lang/luau/releases?per_page=1" in text and "$asset.browser_download_url" in text
       and "luau-windows\\.zip$" in text and "NEURO_NO_LUAU_DOWNLOAD" in text and "Neuro\\luau" in text, f"{rel}: luau from release assets, opt-out")
    ok(text.count("Install-Luau") == 2 and "catch {" in text.split("function Install-Luau", 1)[1], f"{rel}: download step defined, called, failures caught")
    luau_part = text[text.index("function Get-LuauAssetUrl"):text.index("\n}", text.index("function Install-Luau"))]
    ok("& $exe" not in luau_part and "Start-Process" not in luau_part and "Invoke-Expression" not in luau_part, f"{rel}: the downloaded luau is never run by the script")
deck = (repo / "platforms" / "steamdeck.sh").read_bytes()
deck_text = deck.decode("utf-8")
ok(b"\r" not in deck, "steamdeck.sh stays LF")
ok("releases?per_page=1" in deck_text and 'asset.get("browser_download_url")' in deck_text and "luau-ubuntu\\.zip" in deck_text
   and "python3 -m zipfile -e" in deck_text and "chmod +x" in deck_text and ".local/share/neuro/luau" in deck_text, "steamdeck.sh: luau from release assets into ~/.local/share/neuro/luau")
ok("get_luau ||" in deck_text and "NEURO_NO_LUAU_DOWNLOAD" in deck_text, "steamdeck.sh: luau step is opt-out and never stops the script")

# --- seeds: HTML/CSS and TypeScript prompts ask for 4096 tokens ----------------
seed_rows = [json.loads(line) for line in (repo / "seeds" / "coding-v1.jsonl").read_text(encoding="utf-8").splitlines() if line.strip()]
long_kinds = {"verify_html_css", "verify_ts"}
ok(all(r["max_tokens"] == 4096 for r in seed_rows if r["verify"]["kind"] in long_kinds)
   and all(r["max_tokens"] <= 2048 for r in seed_rows if r["verify"]["kind"] not in long_kinds), "seed: 4096 tokens for HTML/CSS and TypeScript only")

# --- page shows truncated next to the pass rate ---------------------------------
page = mesh.DASHBOARD.read_text(encoding="utf-8")
factory = page[page.index("function duration("):page.index("function appRows(")]
ok("row.truncated" in factory and '" truncated"' in factory and "innerHTML" not in factory, "page: truncated count next to the pass rate, textContent only")

# --- full-run loss curve: incremental log parser ---------------------------------
with tempfile.TemporaryDirectory() as tmp:
    log = Path(tmp) / "train.log"
    cache = {}
    head = '2026-11-09T17:57:13 start\r\n{\r\n  "optimizer_steps_per_epoch": 100,\r\n}\r\n'
    log.write_bytes(head.encode())
    st = mesh.update_loss_cache(log, 200, cache)
    ok(st["steps_per_epoch"] == 100 and st["loss"] == [] and st["offset"] == len(head.encode()), "loss cache: reads the head, finds steps per epoch")
    with log.open("ab") as f:
        f.write(b"  1%|  | 5/200 [00:10<06:30,  2.0s/it]\r{'loss': '0.9', 'grad_norm': '1.0', 'epoch': '0.05'}\r\n")
        f.write(b"  5%|  | 10/200 [00:20<06:20,  2.0s/it]\r{'loss': '0.7', 'epoch': '0.1'}\r\n")
        f.write(b"{'eval_loss': '0.8', 'epoch': '0.1'}\r\n{'loss': '0.6', 'ep")  # unfinished last line
    before = st["offset"]
    st = mesh.update_loss_cache(log, 200, cache)
    ok([p[2] for p in st["loss"]] == [0.9, 0.7] and [p[2] for p in st["eval"]] == [0.8], f"loss cache: append adds only new points {st['loss']}")
    ok(st["offset"] > before and st["offset"] < log.stat().st_size, "loss cache: an unfinished line waits for its end")
    with log.open("ab") as f:
        f.write(b"och': '0.15'}\r\n")
    st = mesh.update_loss_cache(log, 200, cache)
    ok([p[2] for p in st["loss"]] == [0.9, 0.7, 0.6] and st["offset"] == log.stat().st_size, "loss cache: the line is parsed once complete")
    again = mesh.update_loss_cache(log, 200, cache)
    ok(len(again["loss"]) == 3, "loss cache: an unchanged file adds nothing")
    series = mesh.loss_series(st, 200)
    ok(series["loss_series"] == [[5, 0.9], [10, 0.7], [15, 0.6]] and series["eval_series"] == [[10, 0.8]], f"loss series: epochs become steps {series}")
    # shrink / rotate: a shorter new file starts over
    log.write_bytes(b"2026-11-09T19:00:00 start\r\n{'loss': '2.5', 'epoch': '0.01'}\r\n")
    st = mesh.update_loss_cache(log, 200, cache)
    ok([p[2] for p in st["loss"]] == [2.5] and st["steps_per_epoch"] is None, f"loss cache: a shrunk file is read from 0 {st['loss']}")
    # same size or larger but different first bytes: rotated, starts over
    log.write_bytes(b"2026-11-09T20:00:00 start\r\n" + b"{'loss': '1.25', 'epoch': '0.02'}\r\n" * 3)
    st = mesh.update_loss_cache(log, 200, cache)
    ok([p[2] for p in st["loss"]] == [1.25, 1.25, 1.25], "loss cache: a rotated file (new first bytes) is read from 0")
    # another path starts over too
    other = Path(tmp) / "other.log"
    other.write_bytes(b"{'loss': '3.0'}\n")
    st = mesh.update_loss_cache(other, 0, cache)
    ok(st["path"] == str(other) and [p[2] for p in st["loss"]] == [3.0], "loss cache: a new path starts over")
    ok(mesh.loss_series(st)["loss_series"] == [[1, 3.0]], "loss series: no epoch or bar -> point index")
    # a new 'start' line in the same file resets the curve
    with other.open("ab") as f:
        f.write(b"2026-11-09T21:00:00 start\n{'loss': '4.0'}\n")
    ok([p[2] for p in mesh.update_loss_cache(other, 0, cache)["loss"]] == [4.0], "loss cache: a new run in the same log starts the curve over")
    # cap: never more than the cap, keeps the first and newest point
    big = Path(tmp) / "big.log"
    big.write_bytes(b"".join(b"%d/1000 [00:01<00:01, 1.0s/it]\r{'loss': '%d'}\n" % (i, i) for i in range(1, 1001)))
    capped = mesh.update_loss_cache(big, 1000, {}, cap=300)
    ok(len(capped["loss"]) <= 300 and capped["loss"][0][2] == 1 and capped["loss"][-1][2] == 1000, f"loss cache: capped at 300 ({len(capped['loss'])}), first and newest kept")
    ok(capped["loss"][-1][1] == 1000 and capped["bar_total"] == 1000, "loss cache: loss points carry the training bar step")
    # an eval bar (smaller total) does not move the step
    small = Path(tmp) / "eval.log"
    small.write_bytes(b"50/1000 [00:01<00:01, 1.0s/it]\r 3/77 [00:01<00:01, 2.0it/s]\r{'eval_loss': '0.5'}\n")
    ok(mesh.update_loss_cache(small, 1000, {})["eval"][0][1] == 50, "loss cache: an eval bar does not move the train step")
    # UTF-16 log (PowerShell redirection)
    u16 = Path(tmp) / "u16.log"
    u16.write_bytes("﻿{'loss': '0.5', 'epoch': '1'}\r\n{'loss': '0.4', 'epoch': '2'}\r\n{'loss'".encode("utf-16-le"))
    st16 = mesh.update_loss_cache(u16, 0, {})
    ok(st16["encoding"] == "utf-16-le" and [p[2] for p in st16["loss"]] == [0.5, 0.4] and st16["offset"] % 2 == 0, "loss cache: UTF-16 logs, offset stays even")
    ok(mesh.loss_series(st16, 0, step=20, epoch=2.0)["loss_series"] == [[10, 0.5], [20, 0.4]], "loss series: step / epoch now converts epochs")
    ok(mesh.update_loss_cache(Path(tmp) / "missing.log", 0, {})["loss"] == [], "loss cache: a missing file gives no points")
    empty_series = mesh.full_loss_series("", 0)
    ok(empty_series["loss_series"] == [] and empty_series["eval_series"] == [] and empty_series["eval_domain_series"] == {}
       and all(empty_series[k] == [] for k in ("grad_norm_series", "lr_series", "step_time_series")), "full_loss_series: no log, empty series")
# downsample
pts = [[i, float(i % 7)] for i in range(5000)]
down = mesh.downsample(pts, 400)
ok(len(down) == 400 and down[-1] == [4999, 1.0] and down[0][0] < 20 and all(down[i][0] <= down[i + 1][0] for i in range(399)), "downsample: 400 points, ordered, newest exact")
ok(mesh.downsample([[1, 2.0], [2, 3.0]], 400) == [[1, 2.0], [2, 3.0]], "downsample: short series unchanged")
ok(all(len(mesh.downsample([[i, 1.0] for i in range(n)], 400)) == min(n, 400) for n in (399, 400, 401, 799, 801, 4321)), "downsample: never over the limit")
mesh._training_cache["at"] = 0.0
tv = mesh.training_view()
ok(isinstance(tv.get("loss_series"), list) and isinstance(tv.get("eval_series"), list) and len(tv["loss_series"]) <= 400, "training_view sends loss_series and eval_series")
ok({"history", "eval_history", "loss", "eval_loss", "step", "max_steps"} <= set(tv), "training_view keeps the old fields")
page = mesh.DASHBOARD.read_text(encoding="utf-8")
gym = page[page.index("function drawTraining("):page.index("function cell(")]
ok("loss_series" in gym and "eval_series" in gym and "innerHTML" not in page, "page: the loss graph uses the full series")
ok('id="gym"' in page and "prefers-reduced-motion" in page and "The training has finished!" in page, "page: brain gym animation, reduced motion, finished message")

# --- premium dashboard: grad norm, learning rate, per-domain eval, step time --------
with tempfile.TemporaryDirectory() as tmp:
    log = Path(tmp) / "metrics.log"
    log.write_bytes(
        b"2026-11-09T17:57:13 start\r\n{\r\n  \"optimizer_steps_per_epoch\": 100,\r\n}\r\n"
        b"  1%|  | 1/200 [00:02<06:30,  2.10s/it]\r  1%|  | 1/200 [00:02<06:30,  2.10s/it]\r  1%|  | 2/200 [00:04<06:30,  2.05s/it]\r"
        b"  5%|  | 10/200 [00:20<06:20,  0.5it/s]\r"
        b"{'loss': '0.9', 'grad_norm': '1.5', 'learning_rate': '2e-05', 'epoch': '0.05'}\r\n"
        b"{'loss': '0.7', 'grad_norm': '0.75', 'learning_rate': '1.8e-05', 'epoch': '0.1'}\r\n"
        b"  3/77 [00:01<00:01, 2.0it/s]\r"
        b"{'eval_loss': '0.8', 'eval_runtime': '3.2', 'epoch': '0.1'}\r\n"
        b"{'eval_all_loss': '0.81', 'epoch': '0.1'}\r\n"
        b"{'eval_cpp_loss': '0.95', 'eval_cpp_runtime': '1.0', 'epoch': '0.1'}\r\n"
        b"{'eval_extra/luau_loss': '1.25', 'eval_python_loss': 0.5, 'epoch': '0.1'}\r\n"
    )
    st = mesh.update_loss_cache(log, 200, {})
    ok([p[2] for p in st["grad"]] == [1.5, 0.75] and [p[2] for p in st["lr"]] == [2e-05, 1.8e-05], f"metrics: grad_norm and learning_rate points {st['grad']} {st['lr']}")
    ok(sorted(st["domains"]) == ["cpp", "extra/luau", "python"] and st["domains"]["cpp"][0][2] == 0.95 and st["domains"]["extra/luau"][0][2] == 1.25,
       f"metrics: per-domain eval losses, eval_all_loss stays the overall curve {sorted(st['domains'])}")
    ok([p[2] for p in st["eval"]] == [0.8, 0.81], "metrics: eval_loss and eval_all_loss feed the overall eval curve")
    ok([r[0] for r in st["speed"]] == [1, 2, 10] and st["speed"][0][1] == 2.1 and st["speed"][2][1] == 2.0 and st["speed"][1][2] == 4,
       f"metrics: one step-time sample per new train step, it/s turned into s/it, elapsed kept {st['speed']}")
    series = mesh.loss_series(st, 200)
    ok(series["grad_norm_series"] == [[5, 1.5], [10, 0.75]] and series["lr_series"] == [[5, 2e-05], [10, 1.8e-05]],
       f"metrics: grad_norm_series and lr_series by step {series['grad_norm_series']} {series['lr_series']}")
    ok(set(series["eval_domain_series"]) == {"cpp", "extra/luau", "python"} and series["eval_domain_series"]["python"] == [[10, 0.5]],
       "metrics: eval_domain_series by step")
    ok(series["step_time_series"] == [[1, 2.1, 2], [2, 2.05, 4], [10, 2.0, 20]], f"metrics: step_time_series rows [step, s/step, elapsed] {series['step_time_series']}")
    with log.open("ab") as f:
        f.write(b"2026-11-09T19:00:00 start\r\n{'loss': '2.0', 'epoch': '0.01'}\r\n")
    st2 = mesh.update_loss_cache(log, 200, {})
    ok(st2["grad"] == [] and st2["lr"] == [] and st2["domains"] == {} and st2["speed"] == [], "metrics: a new run in the same log clears every series")
    many = Path(tmp) / "domains.log"
    many.write_bytes(b"".join(b"{'eval_d%02d_loss': '1.0'}\n" % i for i in range(30)))
    ok(len(mesh.update_loss_cache(many, 0, {})["domains"]) == mesh.EVAL_DOMAINS_CAP == 24, "metrics: at most 24 eval domains")
    long_run = Path(tmp) / "long.log"
    long_run.write_bytes(b"".join(b"%d/3000 [00:%02d<00:01, 1.5s/it]\r{'loss': '1.0', 'grad_norm': '2.0', 'learning_rate': '%se-05'}\n" % (i, i % 60, str(1 + i / 3000).encode())
                                  for i in range(1, 3001)))
    lst = mesh.update_loss_cache(long_run, 3000, {})
    lser = mesh.loss_series(lst, 3000)
    ok(all(len(lser[k]) <= 400 for k in ("loss_series", "grad_norm_series", "lr_series", "step_time_series")) and len(lser["step_time_series"]) == 400,
       "metrics: every series sent is at most 400 points")
    ok(all(p[1] > 1e-6 for p in lser["lr_series"]) and lser["lr_series"][-1] == [3000, 2e-05], "metrics: downsampled learning rates keep their tiny values")
ok(mesh.downsample([[i, 1e-5 * (1 + i)] for i in range(1000)], 100, sig=4)[0][1] > 0, "downsample sig: no rounding to zero")
rows = mesh.downsample_rows([[i, None if i % 2 else 2.0, float(i)] for i in range(1000)], 50)
ok(len(rows) == 50 and rows[-1] == [999, None, 999.0] and rows[0][1] == 2.0 and all(rows[i][0] <= rows[i + 1][0] for i in range(49)), "downsample_rows: None-aware, ordered, newest exact")
ok(mesh.downsample_rows([[1, 2]], 400) == [[1, 2]], "downsample_rows: short list unchanged")
mesh._training_cache["at"] = 0.0
tv2 = mesh.training_view()
ok({"grad_norm_series", "lr_series", "eval_domain_series", "step_time_series"} <= set(tv2) and isinstance(tv2["eval_domain_series"], dict), "training_view sends the new metric series")

# --- GPU telemetry: fixed nvidia-smi query into a 24 h ring buffer -------------------
smi = "0, NVIDIA GeForce RTX 4070, 97, 183.20, 200.00, 71, 11800, 12282\n1, Tiny GPU, [N/A], [N/A], [N/A], 40, 100, 2048\nbad line\n"
cards = mesh.parse_nvidia_telemetry(smi)
ok(len(cards) == 2 and cards[0] == {"index": 0, "name": "NVIDIA GeForce RTX 4070", "util": 97.0, "power_w": 183.2, "power_limit_w": 200.0,
                                     "temp_c": 71.0, "mem_used_mib": 11800.0, "mem_total_mib": 12282.0}, f"telemetry: nvidia-smi line parsed {cards[:1]}")
ok(cards[1]["util"] is None and cards[1]["power_w"] is None and cards[1]["temp_c"] == 40.0, "telemetry: [N/A] fields become None, the line stays")
perf = ('"(PDH-CSV 4.0)","\\\\PC\\GPU Adapter Memory(luid_a_phys_0)\\Dedicated Usage","\\\\PC\\GPU Adapter Memory(luid_b_phys_0)\\Dedicated Usage",'
        '"\\\\PC\\GPU Adapter Memory(luid_a_phys_0)\\Shared Usage","\\\\PC\\GPU Adapter Memory(luid_b_phys_0)\\Shared Usage"\n'
        '"11/11/2026 15:02:47.404","1048576.0","8823767040.0","524288000.0","8192.0"\nExiting, please wait...\n')
ok(mesh.parse_shared_counters(perf) == 0.0, "telemetry: shared MiB of the adapter with the most dedicated memory")
perf2 = perf.replace('"8192.0"', '"2147483648.0"')
ok(mesh.parse_shared_counters(perf2) == 2048.0 and mesh.parse_shared_counters("") is None and mesh.parse_shared_counters("junk\n") is None, "telemetry: spill in MiB; junk gives None")
calls = []


class _Done:
    def __init__(self, out):
        self.stdout = out


def fake_run(argv, **kw):
    calls.append((list(argv), kw.get("timeout")))
    if argv[0] == "nvidia-smi":
        return _Done(smi)
    return _Done(perf2)


gpus_now, err_now = mesh.sample_telemetry(fake_run)
ok(len(gpus_now) == 2 and err_now == "" and calls[0][0] == mesh.TELEMETRY_ARGS and calls[0][1] <= 5, "telemetry: one fixed nvidia-smi command, short timeout")
ok(mesh.TELEMETRY_ARGS == ["nvidia-smi", "--query-gpu=index,name,utilization.gpu,power.draw,power.limit,temperature.gpu,memory.used,memory.total",
                           "--format=csv,noheader,nounits"], "telemetry: the query is a constant")


def missing_run(argv, **kw):
    raise FileNotFoundError(argv[0])


ok(mesh.sample_telemetry(missing_run) == ([], "nvidia-smi was not found on this PC."), "telemetry: no nvidia-smi -> no sample, a clear note")
store = mesh.Telemetry()
ok(store.samples.maxlen == 24 * 3600 // 15 + 8, "telemetry: ring buffer sized for 24 h at 15 s")
calls.clear()
tick_state = {}
for i in range(8):
    mesh.telemetry_tick(store, fake_run, now=1_000_000 + 15 * i, state=tick_state)
ok(sum(1 for argv, _ in calls if argv[0] == "nvidia-smi") == 8 and sum(1 for argv, _ in calls if argv[0] == "typeperf") == (2 if os.name == "nt" else 0),
   "telemetry: nvidia-smi every tick, the shared counter at most once a minute")
view = store.view()
ok(view["available"] and view["count"] == 8 and len(view["gpus"]) == 2 and view["gpus"][0]["mem_total_mib"] == 12282.0 and view["gpus"][0]["power_limit_w"] == 200.0,
   "telemetry: view per GPU with total memory and power limit")
ok(view["gpus"][0]["points"][-1] == [1_000_105, 97.0, 183.2, 71.0, 11800.0, 2048.0 if os.name == "nt" else None] and view["gpus"][1]["points"][-1][5] is None,
   f"telemetry: rows [t, util, power, temp, mem, shared] {view['gpus'][0]['points'][-1]}")
mesh.telemetry_tick(store, missing_run, now=1_000_200, state=tick_state)
ok(store.view()["count"] == 8 and store.view()["error"] == "nvidia-smi was not found on this PC.", "telemetry: a failed sample adds nothing and says why")
big_store = mesh.Telemetry()
start_t = 2_000_000
for i in range(6200):
    big_store.add(start_t + 15 * i, [{"index": 0, "name": "G", "util": float(i % 100), "power_w": 100.0, "power_limit_w": 200.0,
                                      "temp_c": 60.0, "mem_used_mib": 1000.0, "mem_total_mib": 2000.0}])
end_t = start_t + 15 * 6199
ok(len(big_store.samples) <= 24 * 3600 // 15 + 8 and big_store.samples[0][0] >= end_t - 24 * 3600, "telemetry: older than 24 h is dropped")
full = big_store.view()
ok(len(full["gpus"][0]["points"]) <= 400 and full["gpus"][0]["points"][-1][0] == end_t, "telemetry: at most 400 points sent, newest exact")
hour = big_store.view(3600)
ok(hour["window_s"] == 3600 and 238 <= hour["count"] <= 242, f"telemetry: window=3600 keeps the last hour ({hour['count']})")
ok(big_store.view(5)["window_s"] == 300 and big_store.view(10 ** 9)["window_s"] == 24 * 3600, "telemetry: window clamped to 5 min .. 24 h")
ok(mesh.Telemetry().view()["available"] is False and mesh.Telemetry().view()["gpus"] == [], "telemetry: empty buffer -> available false")

# --- /vendor/<name>: allowlisted static files; /api/telemetry ------------------------
import http.client as _hc

mesh.Handler.mode = "serve"
vsrv = ThreadingHTTPServer(("127.0.0.1", 0), mesh.Handler)
threading.Thread(target=vsrv.serve_forever, daemon=True).start()
vport = vsrv.server_address[1]


def raw_get(path, host=None):
    conn = _hc.HTTPConnection("127.0.0.1", vport, timeout=10)
    conn.putrequest("GET", path, skip_host=True, skip_accept_encoding=True)
    conn.putheader("Host", host or f"127.0.0.1:{vport}")
    conn.endheaders()
    resp = conn.getresponse()
    body = resp.read()
    headers = {k.lower(): v for k, v in resp.getheaders()}
    conn.close()
    return resp.status, headers, body


ok(set(mesh.VENDOR_FILES) == {"three.module.min.js", "LICENSE"}, "vendor: fixed allowlist")
for vname, vtype in mesh.VENDOR_FILES.items():
    vstatus, vheaders, vbody = raw_get("/vendor/" + vname)
    ok(vstatus == 200 and vheaders.get("content-type") == vtype and vbody == (mesh.VENDOR_DIR / vname).read_bytes()
       and "max-age=" in vheaders.get("cache-control", "") and vheaders.get("x-content-type-options") == "nosniff", f"vendor: {vname} served with {vtype} and cache headers")
lic = (mesh.VENDOR_DIR / "LICENSE").read_text(encoding="utf-8")
ok("MIT" in lic and "three.js authors" in lic, "vendor: three.js MIT license next to the file")
_three = (mesh.VENDOR_DIR / "three.module.min.js").read_bytes()
ok(b"as REVISION" in _three and b'const t="170"' in _three[:1000], "vendor: three.js r170 pinned")
mesh_src = (Path(mesh.ROOT) / "mesh.py").read_bytes()
for bad in ("/vendor/nope.js", "/vendor/", "/vendor/../mesh.py", "/vendor/..%2fmesh.py", "/vendor/%2e%2e/mesh.py", "/vendor/..\\mesh.py",
            "/vendor/vendor/LICENSE", "/vendor/./LICENSE", "/vendor/LICENSE/", "/vendor/three.module.min.js%00", "/vendor//etc/passwd", "/vendor/LICENSE.bak", "/vendor/C:/Windows/win.ini"):
    bstatus, _, bbody = raw_get(bad)
    ok(bstatus == 404 and mesh_src[:400] not in bbody, f"vendor: {bad} refused")
ok(raw_get("/vendor/LICENSE", host="evil.example.com")[0] == 403, "vendor: same Host check as the page")
tstatus, theaders, tbody = raw_get("/api/telemetry")
tjson = json.loads(tbody)
ok(tstatus == 200 and {"interval_s", "window_s", "available", "gpus", "columns", "error"} <= set(tjson) and theaders.get("cache-control") == "no-store", "GET /api/telemetry answers")
ok(json.loads(raw_get("/api/telemetry?window=abc")[2])["window_s"] == 24 * 3600 and json.loads(raw_get("/api/telemetry?window=3600")[2])["window_s"] == 3600, "/api/telemetry?window= parsed, junk ignored")
vsrv.shutdown()
vsrv.server_close()
mesh.Handler.mode = "work"
wv = ThreadingHTTPServer(("127.0.0.1", 0), mesh.Handler)
threading.Thread(target=wv.serve_forever, daemon=True).start()
vport = wv.server_address[1]
ok(raw_get("/vendor/LICENSE")[0] == 404 and raw_get("/api/telemetry")[0] == 404, "worker port serves no vendor files and no telemetry")
wv.shutdown()
wv.server_close()
mesh.Handler.mode = "serve"

# --- premium page: vendored 3D stage with fallback, metric charts, telemetry --------
page = mesh.DASHBOARD.read_text(encoding="utf-8")
import re as _re
ok('import("/vendor/three.module.min.js")' in page and "unpkg" not in page and "jsdelivr" not in page
   and not _re.search(r"<script[^>]+src=", page), "page: three.js only from the vendored route, no external scripts")
ok('id="gym3d"' in page and 'id="gym"' in page and 'classList.remove("webgl-on")' in page and "webglcontextlost" in page,
   "page: WebGL canvas plus the SVG stage as fallback (import failure, init error, context loss)")
gym3d = page[page.index("function startGym3d("):]
ok("Math.min(2, window.devicePixelRatio" in gym3d and "IntersectionObserver" in gym3d and "document.hidden" in gym3d and "REDUCED.matches" in gym3d,
   "page: 3D stage caps DPR at 2, pauses off-screen and in hidden tabs, still pose for reduced motion")
ok(all(word in gym3d for word in ("IcosahedronGeometry", "MeshPhysicalMaterial", "CapsuleGeometry", "castShadow", "noise3(", "setBend(", "InstancedMesh"))
   and all(f'mode === "{m}"' in gym3d for m in ("lift", "warm", "done", "error")), "page: procedural brain, capsule rig, bending bar, sweat; lift/warm/done/error/rest poses")
ok("1 + Math.floor(pct / 12.5)" in page and "Math.min(8," in page, "page: plates 1 + floor(progress/12.5), at most 8")
for chart_id in ("losses", "eval-losses", "ch-lr", "ch-grad", "ch-speed", "ch-progress", "ch-util", "ch-power", "ch-temp", "ch-mem"):
    ok(f'id="{chart_id}"' in page, f"page: chart #{chart_id}")
ok(page.count('class="explain"') >= 10, "page: every graph has a plain-language note")
ok(all(field in page for field in ("lr_series", "grad_norm_series", "eval_domain_series", "step_time_series", "/api/telemetry?window=")), "page: reads the new series and the telemetry route")
ok("innerHTML" not in page and "insertAdjacentHTML" not in page and "outerHTML" not in page, "page: still textContent / createElement only")
_node_bin = mesh.node_path()
if _node_bin:
    ticks_fn = page[page.index("function niceTicks("):page.index("function timeTicks(")]
    script = ticks_fn + "\nprocess.stdout.write(JSON.stringify([niceTicks(0, 4e-5, 4), niceTicks(-1e-9, 2700, 4), niceTicks(0.6, 0.8, 4)]));"
    out = _sp.run([_node_bin, "-e", script], capture_output=True, text=True, timeout=20, encoding="utf-8")
    tiny, steps_t, loss_t = json.loads(out.stdout or "[[],[],[]]")
    ok(len(tiny) >= 3 and all(0 <= v <= 4e-5 for v in tiny) and steps_t[0] == 0 and 1 / (steps_t[0] or 1) > 0 and len(loss_t) >= 3,
       f"page: chart ticks work for tiny and large ranges {tiny} {steps_t} {loss_t}")

# --- verify_tests / cross_tests: execution-based checks in a sandbox ------------
# Only handcrafted, harmless snippets run here; never generated model output.
import glob as _glob

ok("verify_tests" in mesh.JOB_KINDS and "cross_tests" in mesh.JOB_KINDS and "verify_tests" in mesh.VERIFY_KINDS
   and "cross_tests" not in mesh.VERIFY_KINDS, "verify_tests is a verify kind, cross_tests a job kind")
ok(mesh.KIND_ATTEMPTS == {"verify_tests": 2, "cross_tests": 2} and mesh.TEST_TIMEOUT == 10 and mesh.TEST_MEMORY == 1024 ** 3,
   "test kinds: one lease retry at most, 10 s, 1 GB")
for bad_spec in [{"kind": "verify_tests"}, {"kind": "verify_tests", "lang": "ruby"}, {"kind": "verify_tests", "lang": ["python"]}]:
    try:
        mesh.validate_verify_spec(bad_spec)
        ok(False, f"bad verify_tests spec accepted {bad_spec}")
    except mesh.JobError:
        ok(True, f"verify_tests spec refused {bad_spec}")
ok(mesh.validate_verify_spec({"kind": "verify_tests", "lang": "javascript", "x": 1}) == {"kind": "verify_tests", "lang": "javascript"}, "verify_tests spec keeps lang only")
ANS_PY = "Here:\n```python\ndef double(x):\n    return x * 2\n\nassert double(1) == 2\nassert double(0) == 0\nassert double(-3) == -6\n```\nDone."
ok(mesh.verify_payload({"kind": "verify_tests", "lang": "python"}, ANS_PY) == {"lang": "python", "blocks": ["def double(x):\n    return x * 2\n\nassert double(1) == 2\nassert double(0) == 0\nassert double(-3) == -6\n"]},
   "verify_payload takes the python block for verify_tests")
ok(mesh.verify_payload({"kind": "verify_tests", "lang": "javascript"}, "```js\nconst a = 1;\n```")["blocks"] == ["const a = 1;\n"], "verify_payload maps javascript to js fences")
ok(mesh.validate_job("verify_tests", {"lang": "python", "answer": ANS_PY})["blocks"][0].startswith("def double"), "validate_job verify_tests from an answer")
for bad_row in [{"lang": "python"}, {"lang": "go", "code": "x"}, {"code": "x"}]:
    try:
        mesh.validate_job("verify_tests", bad_row)
        ok(False, f"bad verify_tests row accepted {bad_row}")
    except mesh.JobError:
        ok(True, f"verify_tests row refused {bad_row}")
GOOD_SAMPLES = [{"id": "1" * 16, "code": "x"}, {"id": "2" * 16, "code": "y"}]
for bad_row in [{"lang": "python", "group": "a" * 16, "samples": GOOD_SAMPLES[:1]},
                {"lang": "python", "group": "nothex", "samples": GOOD_SAMPLES},
                {"lang": "python", "group": "a" * 16, "samples": [GOOD_SAMPLES[0], GOOD_SAMPLES[0]]},
                {"lang": "python", "group": "a" * 16, "samples": [{"id": "../x", "code": "y"}, GOOD_SAMPLES[1]]},
                {"lang": "python", "group": "a" * 16, "samples": [{"id": "3" * 16, "code": "x" * 150000}, {"id": "4" * 16, "code": "y" * 60000}]}]:
    try:
        mesh.validate_job("cross_tests", bad_row)
        ok(False, "bad cross_tests row accepted")
    except mesh.JobError as exc:
        ok(True, f"cross_tests row refused ({str(exc)[:40]})")
ok(mesh.job_skill({"verify": {"kind": "verify_tests", "lang": "python"}}) == "code", "verify_tests rows are skill code")

# static checks: enough real asserts, nothing that hides a failure
PY_OK = "def f(x):\n    return x * 2\n\nassert f(1) == 2\nassert f(0) == 0\nassert f(-3) == -6\n"
static_py = {
    "trivial": ("def f(x):\n    return x\nassert True\nassert 1\nassert 2 == 2\nassert f(1) == 1\nassert f(2) == f(2)\n", "trivial"),
    "catch": ("def f(x):\n    return x\ntry:\n    assert f(1) == 1\n    assert f(2) == 2\n    assert f(3) == 3\nexcept AssertionError:\n    pass\n", "catches AssertionError"),
    "swallow": ("def f(x):\n    return x\ntry:\n    assert f(1) == 1\n    assert f(2) == 2\n    assert f(3) == 3\nexcept Exception:\n    print('x')\n", "swallows"),
    "bare except": ("def f(x):\n    return x\ntry:\n    assert f(1) == 1\n    assert f(2) == 2\n    assert f(3) == 3\nexcept:\n    pass\n", "swallows"),
    "suppress": ("import contextlib\ndef f(x):\n    return x\nwith contextlib.suppress(AssertionError):\n    assert f(1) == 1\n    assert f(2) == 2\n    assert f(3) == 3\n", "suppress"),
    "uncalled": ("def f(x):\n    return x\ndef test():\n    assert f(1) == 1\n    assert f(2) == 2\n    assert f(3) == 3\n", "only 0"),
    "rebind": ("AssertionError = Exception\ndef f(x):\n    return x\nassert f(1) == 1\nassert f(2) == 2\nassert f(3) == 3\n", "rebinds"),
    "syntax": ("def f(:\n", "SyntaxError"),
}
for name, (code, why) in static_py.items():
    count, problem = mesh.tests_problems(code, "python")
    ok(why in problem, f"static python: {name} rejected ({problem[:60]})")
ok(mesh.tests_problems(PY_OK, "python") == (3, ""), "static python: three real asserts pass")
ok(mesh.tests_problems("def f(x):\n    return x\ndef test():\n    assert f(1) == 1\n    assert f(2) == 2\n    assert f(3) == 3\nif __name__ == '__main__':\n    test()\n", "python")[1] == "",
   "static python: asserts in a called test function count")
ok(mesh.tests_problems("def f(x):\n    try:\n        return int(x)\n    except ValueError:\n        return None\nassert f('1') == 1\nassert f('x') is None\nassert f('-2') == -2\n", "python")[1] == "",
   "static python: a try/except inside the solution is fine")
ok(mesh.tests_problems("def f(x):\n    return x\ntry:\n    assert f(1) == 1\n    assert f(2) == 2\n    assert f(3) == 3\nexcept Exception as e:\n    print(e)\n    raise\n", "python")[1] == "",
   "static python: a handler that re-raises is fine")
JS_HEAD = "const assert = require('node:assert');\n"
JS_OK = JS_HEAD + "function f(x) { return x * 2; }\nassert.strictEqual(f(1), 2);\nassert.strictEqual(f(0), 0);\nassert.deepStrictEqual([f(2)], [4]);\n"
static_js = {
    "no import": ("function f(x) { return x; }\nassert.strictEqual(f(1), 1);\nassert.strictEqual(f(2), 2);\nassert.strictEqual(f(3), 3);\n", "no node:assert"),
    "console.assert": (JS_HEAD + "function f(x) { return x; }\nconsole.assert(f(1) === 1);\nassert.strictEqual(f(1), 1);\nassert.strictEqual(f(2), 2);\nassert.strictEqual(f(3), 3);\n", "console.assert"),
    "trivial": (JS_HEAD + "function f(x) { return x; }\nassert(true);\nassert.ok(1);\nassert.strictEqual(2, 2);\nassert.strictEqual(f(1), 1);\n", "trivial"),
    "swallow": (JS_HEAD + "function f(x) { return x; }\ntry {\n  assert.strictEqual(f(1), 2);\n  assert.strictEqual(f(2), 2);\n  assert.strictEqual(f(3), 3);\n} catch (e) {\n  console.log(e.message);\n}\n", "swallows"),
    "handler": (JS_HEAD + "process.on('uncaughtException', () => {});\nfunction f(x) { return x; }\nassert.strictEqual(f(1), 1);\nassert.strictEqual(f(2), 2);\nassert.strictEqual(f(3), 3);\n", "global error handler"),
    "in comments": (JS_HEAD + "function f(x) { return x; }\n// assert.strictEqual(f(1), 1);\n/* assert.strictEqual(f(2), 2); */\nconst s = 'assert.strictEqual(f(3), 3)';\nassert.strictEqual(f(4), 4);\n", "only 1"),
}
for name, (code, why) in static_js.items():
    count, problem = mesh.tests_problems(code, "javascript")
    ok(why in problem, f"static javascript: {name} rejected ({problem[:60]})")
ok(mesh.tests_problems(JS_OK, "javascript") == (3, ""), "static javascript: three real node:assert calls pass")
ok(mesh.tests_problems(JS_HEAD + "function f(x) { return x; }\ntry {\n  assert.strictEqual(f(1), 1);\n  assert.strictEqual(f(2), 2);\n  assert.strictEqual(f(3), 3);\n} catch (e) {\n  throw e;\n}\n", "javascript")[1] == "",
   "static javascript: a catch that rethrows is fine")
ok(len(mesh.js_mask("const a = '{'; // }\n/* ( */ const b = `x${1}`;")) == len("const a = '{'; // }\n/* ( */ const b = `x${1}`;")
   and "{" not in mesh.js_mask("'{' // {\n")[:3], "js_mask keeps length and blanks strings and comments")
ok(mesh.extract_asserts(PY_OK + "if __name__ == '__main__':\n    assert f(5) == 10\n", "python") == ["assert f(1) == 2", "assert f(0) == 0", "assert f(-3) == -6", "assert f(5) == 10"],
   "extract_asserts: python top-level and __main__ asserts")
js_extract = mesh.extract_asserts(JS_OK + "function g() {\n  assert.strictEqual(g2(), 1);\n}\nassert.deepStrictEqual(\n  [f(3)],\n  [6]\n);\nassert.rejects(p).catch(() => {});\n", "javascript")
ok(js_extract == ["assert.strictEqual(f(1), 2);", "assert.strictEqual(f(0), 0);", "assert.deepStrictEqual([f(2)], [4]);", "assert.deepStrictEqual(\n  [f(3)],\n  [6]\n);"],
   f"extract_asserts: javascript top-level statements only {js_extract}")

# off switch: Linux/macOS run tests only with NEURO_RUN_TESTS=1
saved_name, saved_flag = mesh.os.name, os.environ.pop("NEURO_RUN_TESTS", None)
try:
    mesh.os.name = "posix"
    posix_off = mesh._sandbox_probe()
finally:
    mesh.os.name = saved_name
    if saved_flag is not None:
        os.environ["NEURO_RUN_TESTS"] = saved_flag
ok(posix_off["ok"] is False and not posix_off["langs"] and "NEURO_RUN_TESTS=1" in posix_off["note"], "POSIX without NEURO_RUN_TESTS=1: no test kinds")
real_status = mesh.sandbox_status
mesh.sandbox_status = lambda refresh=False: {"ok": False, "note": "tests off: test", "langs": {}}
try:
    mesh._kinds_cache["at"] = 0.0
    ok("verify_tests" not in mesh.worker_kinds() and "cross_tests" not in mesh.worker_kinds(), "no sandbox -> verify_tests and cross_tests not advertised")
    unavailable = mesh.job_verify_tests({"lang": "python", "blocks": [PY_OK]})
    ok(unavailable[0] is False and "unavailable" in unavailable[1]["error"], "no sandbox -> the job fails as unavailable, never as rejected")
    ok(mesh.job_cross_tests({"lang": "python", "group": "a" * 16, "samples": GOOD_SAMPLES})[0] is False, "no sandbox -> cross_tests fails as unavailable")
finally:
    mesh.sandbox_status = real_status
    mesh._kinds_cache["at"] = 0.0

# never inside the repository
real_root, real_tempdir = mesh.ROOT, tempfile.tempdir
with tempfile.TemporaryDirectory() as fake_repo:
    try:
        mesh.ROOT = Path(fake_repo)
        tempfile.tempdir = fake_repo
        try:
            mesh.sandbox_workdir()
            ok(False, "a temp folder inside the repository was used")
        except mesh.SandboxError:
            ok(not list(Path(fake_repo).iterdir()), "a temp folder inside the repository is refused and removed")
    finally:
        mesh.ROOT, tempfile.tempdir = real_root, real_tempdir

BOX = mesh.sandbox_status()
print("     sandbox:", BOX["note"])
if os.name == "nt" or os.environ.get("NEURO_RUN_TESTS") == "1":
    ok(BOX["ok"] and "python" in BOX["langs"], "this PC can sandbox Python tests")
    ok("Network access is NOT blocked" in BOX["note"], "the capability text says the network is not blocked")
    mesh._kinds_cache["at"] = 0.0
    ok({"verify_tests", "cross_tests"} <= set(mesh.worker_kinds()), "worker advertises verify_tests and cross_tests")
    before_dirs = set(_glob.glob(os.path.join(tempfile.gettempdir(), "neuro-tests-*")))
    T3 = "\nassert add(1, 2) == 3\nassert add(-1, 1) == 0\nassert add(0, 0) == 0\n"
    runs = {
        "pass": ("def add(a, b):\n    return a + b\n" + T3, 10),
        "fail": ("def add(a, b):\n    return a - b\n" + T3, 10),
        "timeout": ("def add(a, b):\n    while True:\n        pass\n" + T3, 3),
        "memory": ("def add(a, b):\n    block = bytearray(3 * 1024 ** 3)\n    return a + b\n" + T3, 10),
        "child": ("import subprocess, sys\ndef add(a, b):\n    subprocess.run([sys.executable, '-c', '0'], check=True)\n    return a + b\n" + T3, 10),
        "early exit": ("import sys\ndef add(a, b):\n    return a + b\n" + T3 + "sys.exit(0)\nassert add(1, 1) == 3\n", 10),
        "flood": ("import sys\ndef add(a, b):\n    return a + b\n" + T3 + "while True:\n    sys.stdout.write('x' * 65536)\n", 10),
    }
    os.environ["NEURO_CANARY_TOKEN"] = "canary-value"
    os.environ["HTTPS_PROXY"] = "http://127.0.0.1:9"
    try:
        runs["env"] = ("import os\nbad = [k for k in os.environ if 'CANARY' in k or 'PROXY' in k.upper() or 'TOKEN' in k.upper()]\n"
                       "assert not bad, bad\nassert 'neuro-tests-' in os.getcwd(), os.getcwd()\n"
                       "assert 'neuro-cluster' not in os.getcwd().lower()\n"
                       "assert os.environ.get('HOME') == os.getcwd() or os.environ.get('USERPROFILE') == os.getcwd()\n"
                       "def add(a, b):\n    return a + b\n" + T3, 10)
        got = {name: mesh.run_test_program("python", code, BOX, timeout=timeout) for name, (code, timeout) in runs.items()}
    finally:
        os.environ.pop("NEURO_CANARY_TOKEN", None)
        os.environ.pop("HTTPS_PROXY", None)
    ok(got["pass"]["finished"] and got["pass"]["returncode"] == 0, "sandbox: passing tests finish with exit 0")
    ok(not got["fail"]["finished"] and "AssertionError" in got["fail"]["stderr"], "sandbox: a failing assert fails the run")
    ok(not got["timeout"]["finished"] and got["timeout"]["timed_out"] and got["timeout"]["seconds"] < 8, f"sandbox: an endless loop is stopped by the timeout ({got['timeout']['seconds']} s)")
    ok(not got["memory"]["finished"] and "MemoryError" in got["memory"]["stderr"], "sandbox: allocating 3 GB fails under the memory cap")
    ok(not got["child"]["finished"] and got["child"]["returncode"] != 0, "sandbox: starting a child process fails")
    ok(not got["early exit"]["finished"] and got["early exit"]["returncode"] == 0, "sandbox: exit 0 before the end of the tests is not a pass")
    ok(not got["flood"]["finished"] and got["flood"]["flooded"] and got["flood"]["seconds"] < 8, "sandbox: an output flood is stopped early")
    ok(got["env"]["finished"], f"sandbox: no tokens or proxies in the env, cwd is a fresh temp folder outside the repo {mesh._error_line(got['env']['stderr'])}")
    ok(all(run["cleaned"] for run in got.values()), "sandbox: every temp folder is removed")
    ok(set(_glob.glob(os.path.join(tempfile.gettempdir(), "neuro-tests-*"))) == before_dirs, "sandbox: no neuro-tests-* folder left behind")
    verdict = mesh.job_verify_tests({"lang": "python", "blocks": [PY_OK]})
    ok(verdict[0] and verdict[1]["passed"] is True and verdict[1]["asserts"] == 3 and isinstance(verdict[1]["seconds"], float), f"job_verify_tests: pass with runtime {verdict[1]}")
    verdict = mesh.job_verify_tests({"lang": "python", "blocks": [PY_OK.replace("x * 2", "x * 3")]})
    ok(verdict[0] and verdict[1]["passed"] is False and verdict[1]["error"] == "AssertionError" and "main.py" in verdict[1]["stderr"]
       and "harness" not in verdict[1]["stderr"] and len(verdict[1]["stderr"]) <= 300, "job_verify_tests: failure reason and the first 300 chars of stderr")
    verdict = mesh.job_verify_tests({"lang": "python", "blocks": [PY_OK, PY_OK]})
    ok(verdict[0] and verdict[1]["passed"] is False and "one python code block" in verdict[1]["error"], "job_verify_tests: two blocks are rejected")
    verdict = mesh.job_verify_tests({"lang": "python", "blocks": []})
    ok(verdict[0] and verdict[1]["passed"] is False, "job_verify_tests: no block is rejected")
    verdict = mesh.job_verify_tests({"lang": "python", "blocks": ["def f():\n    return 1\nassert True\nassert 1\nassert f() == 1\n"]})
    ok(verdict[1]["passed"] is False and "trivial" in verdict[1]["error"] and "seconds" not in verdict[1], "job_verify_tests: trivial asserts rejected before running")
    if "javascript" in BOX["langs"]:
        JT = "\nassert.strictEqual(add(1, 2), 3);\nassert.strictEqual(add(-1, 1), 0);\nassert.deepStrictEqual([add(0, 0)], [0]);\n"
        js_runs = {
            "pass": JS_HEAD + "function add(a, b) { return a + b; }\n" + JT,
            "fail": JS_HEAD + "function add(a, b) { return a - b; }\n" + JT,
            "child": JS_HEAD + "function add(a, b) { require('node:child_process').execFileSync(process.execPath, ['-e', '0']); return a + b; }\n" + JT,
            "write": JS_HEAD + "function add(a, b) { require('node:fs').writeFileSync('escape.txt', 'x'); return a + b; }\n" + JT,
            "async fail": JS_HEAD + "function add(a, b) { return a + b; }\n" + JT + "setTimeout(() => assert.strictEqual(add(1, 1), 3), 10);\n",
        }
        js_got = {name: mesh.run_test_program("javascript", code, BOX) for name, code in js_runs.items()}
        ok(js_got["pass"]["finished"], "sandbox js: passing node:assert tests finish")
        ok(not js_got["fail"]["finished"] and "ERR_ASSERTION" in js_got["fail"]["stderr"], "sandbox js: a failing assert fails the run")
        ok(not js_got["child"]["finished"], "sandbox js: node cannot start a child process")
        ok(not js_got["write"]["finished"] and not (mesh.ROOT / "escape.txt").exists(), "sandbox js: node cannot write files")
        ok(not js_got["async fail"]["finished"], "sandbox js: an assert failing after the last line still fails the run")
        ok(all(run["cleaned"] for run in js_got.values()), "sandbox js: temp folders removed")
        ok(mesh.job_verify_tests({"lang": "javascript", "blocks": [JS_OK]})[1]["passed"] is True, "job_verify_tests: javascript pass")
    # cross_tests: each sample against the others' asserts
    double_ok = "def double(x):\n    return x * 2\n\nassert double(1) == 2\nassert double(0) == 0\nassert double(-3) == -6\n"
    double_ok2 = "def double(x):\n    return x + x\n\nassert double(2) == 4\nassert double(10) == 20\nassert double(-1) == -2\n"
    double_odd = "def double(x):\n    return 2 if x == 1 else (0 if x == 0 else -6)\n\nassert double(1) == 2\nassert double(0) == 0\nassert double(-3) == -6\n"
    double_hang = "def double(x):\n    if x == 10:\n        while True:\n            pass\n    return x * 2\n\nassert double(1) == 2\nassert double(0) == 0\nassert double(-3) == -6\n"
    cross_ok, cross_out = mesh.job_cross_tests(mesh.validate_job("cross_tests", {"lang": "python", "group": "c" * 16, "samples": [
        {"id": "1" * 16, "code": double_ok}, {"id": "2" * 16, "code": double_ok2}, {"id": "3" * 16, "code": double_odd}, {"id": "4" * 16, "code": double_hang}]}))
    sc = cross_out["scores"]
    ok(cross_ok and sc["1" * 16]["agreement"] == 1.0 and sc["2" * 16]["agreement"] == 1.0, "cross_tests: two correct samples agree fully")
    ok(sc["3" * 16]["agreement"] == round(6 / 9, 4) and sc["3" * 16]["total"] == 9, "cross_tests: a sample that only fits its own tests scores lower")
    ok(sc["4" * 16]["passed"] == 4 and "timed out" in cross_out["detail"], "cross_tests: a run that hangs keeps the asserts it passed before the timeout")
    ok(mesh.check_output("cross_tests", mesh.validate_job("cross_tests", {"lang": "python", "group": "c" * 16, "samples": [
        {"id": "1" * 16, "code": "a"}, {"id": "2" * 16, "code": "b"}, {"id": "3" * 16, "code": "c"}, {"id": "4" * 16, "code": "d"}]}), True, cross_out)[0], "cross_tests output passes check_output")
else:
    print("skip: this computer does not sandbox tests (set NEURO_RUN_TESTS=1 on Linux/macOS)")

cross_payload = {"lang": "python", "group": "d" * 16, "samples": GOOD_SAMPLES}
for bad_out in [{"scores": {"9" * 16: {"agreement": 1.0, "passed": 1, "total": 1}}},
                {"scores": {"1" * 16: {"agreement": 1.5, "passed": 1, "total": 1}}},
                {"scores": {"1" * 16: {"agreement": 0.5, "passed": 3, "total": 2}}},
                {"scores": {}}, {"nope": 1}]:
    ok(mesh.check_output("cross_tests", cross_payload, True, bad_out)[0] is False, f"check_output refuses bad cross_tests output {str(bad_out)[:50]}")


# --- the queue: verify_tests -> one cross_tests per group -> agreement -> export ---
def _finish(q, lease, node, ok_flag, output, at):
    return q.finish({"id": lease["id"], "lease": lease["lease"], "node": node, "ok": ok_flag, "output": output}, at)


def _fake_verify(payload, passed, error=""):
    out = {"passed": passed, "asserts": 3, "seconds": 0.05}
    if error:
        out["error"] = error
    return out


A1 = "```python\ndef inc(x):\n    return x + 1\n\nassert inc(1) == 2\nassert inc(0) == 1\nassert inc(-1) == 0\n```"
A2 = "```python\ndef inc(x):\n    return 1 + x\n\nassert inc(5) == 6\nassert inc(9) == 10\nassert inc(-5) == -4\nassert inc(100) == 101\nassert inc(41) == 42\n```\nThis version is a bit longer to explain."
A3 = "```python\ndef inc(x):\n    return {1: 2, 0: 1, -1: 0}[x]\n\nassert inc(1) == 2\nassert inc(0) == 1\nassert inc(-1) == 0\n```"
A4 = "```python\ndef inc(x):\n    return x + 2\n\nassert inc(1) == 2\nassert inc(0) == 1\nassert inc(-1) == 0\n```"
with tempfile.TemporaryDirectory() as tmp:
    q = mesh.JobQueue(Path(tmp))
    gids = q.add_many([("generate", {"prompt": "Write inc(x) with asserts.", "samples": 4, "max_tokens": 2048,
                                     "verify": {"kind": "verify_tests", "lang": "python"}, "meta": {"lang": "python"}})])
    group = q.jobs[gids[0]]["payload"]["group"]
    ok(q.groups.get(group) == gids, "the queue indexes a verify_tests prompt group")
    t0 = 7000.0
    texts_by = {}
    for index, text in enumerate([A1, A2, A3, A4]):
        lease = q.lease(f"gen{index}", ["generate"], t0 + index)
        texts_by[lease["id"]] = text
        accepted, note = _finish(q, lease, f"gen{index}", True, {"text": text, "finish_reason": "stop"}, t0 + index)
        ok(accepted and "verify_tests queued" in note, f"generate {index} queues verify_tests ({note})")
    vids = kind_ids(q, "verify_tests")
    ok(len(vids) == 4 and not kind_ids(q, "cross_tests"), "four verify_tests jobs, no cross-check yet")
    # an old worker that only knows the parse checks never gets them
    ok(q.lease("oldworker", ["verify_python_syntax"], t0 + 9) is None, "a worker without verify_tests gets none")
    ok(q.lease("jsonly", ["verify_tests"], t0 + 9, test_langs=["javascript"]) is None, "a worker whose sandbox runs only javascript gets no python tests")
    # verdicts: A1, A2, A3 pass their own tests, A4 fails them
    for step in range(4):
        lease = q.lease("tester", ["verify_tests", "cross_tests"], t0 + 10 + step, test_langs=["python", "javascript"])
        ok(lease["kind"] == "verify_tests", f"verify_tests {step} leased to a sandboxed worker")
        parent_text = texts_by[q.jobs[lease["id"]]["parent"]]
        sample_ok = parent_text is not A4
        accepted, note = _finish(q, lease, "tester", True, _fake_verify(lease["payload"], sample_ok, "" if sample_ok else "AssertionError"), t0 + 10 + step)
        if step < 3:
            ok("cross_tests" not in note, f"no cross-check while samples are still open ({step})")
    ok("cross_tests queued for 3 samples" in note, f"the last verdict queues one cross_tests job ({note})")
    cids = kind_ids(q, "cross_tests")
    ok(len(cids) == 1 and group in q.crossed and len(q.jobs[cids[0]]["payload"]["samples"]) == 3, "exactly one cross_tests for the group, over the 3 passing samples")
    cross_job = q.jobs[cids[0]]
    ok(q._maybe_cross(q.jobs[gids[0]]) == "" and len(kind_ids(q, "cross_tests")) == 1, "asking again does not queue a second cross_tests")
    reloaded = mesh.JobQueue(Path(tmp))
    reloaded.sync()
    ok(group in reloaded.crossed and reloaded._maybe_cross(reloaded.jobs[gids[0]]) == "" and len(kind_ids(reloaded, "cross_tests")) == 1,
       "after a restart the group stays cross-checked (no second job)")
    # the cross_tests job: lease, run (real sandbox when available), store agreement
    lease = q.lease("tester", ["cross_tests"], t0 + 30, test_langs=["python"])
    ok(lease and lease["kind"] == "cross_tests" and lease["lease_s"] == 400, "cross_tests leased with its own lease time")
    if BOX["langs"].get("python"):
        run_ok, run_out = mesh.run_job(lease)
    else:
        ids3 = [sample["id"] for sample in lease["payload"]["samples"]]
        run_ok, run_out = True, {"scores": {ids3[0]: {"agreement": 1.0, "passed": 6, "total": 6}, ids3[1]: {"agreement": 1.0, "passed": 6, "total": 6},
                                            ids3[2]: {"agreement": 0.375, "passed": 3, "total": 8}}}
    accepted, note = _finish(q, lease, "tester", run_ok, run_out, t0 + 31)
    ok(accepted and note == "saved", f"cross_tests result saved and queues nothing ({note})")
    by_text = {texts_by[jid]: jid for jid in texts_by}
    ok(q.agreement[by_text[A1]] == 1.0 and q.agreement[by_text[A2]] == 1.0 and q.agreement[by_text[A3]] < 0.5 and by_text[A4] not in q.agreement,
       f"agreement per sample {[q.agreement.get(by_text[a]) for a in (A1, A2, A3, A4)]}")
    reloaded = mesh.JobQueue(Path(tmp))
    reloaded.sync()
    ok(reloaded.agreement == q.agreement, "agreement survives a restart (read from the queue file)")
    s = q.summary()
    agree = s["verifiers"]["verify_tests"]["agreement"]
    ok(s["verifiers"]["verify_tests"]["checked"] == 4 and s["verifiers"]["verify_tests"]["passed"] == 3 and s["verifiers"]["verify_tests"]["pct"] == 75,
       "summary: verify_tests pass rate")
    ok(agree["groups"] == 1 and agree["scored"] == 3 and agree["agree"] == 2 and agree["min"] == 0.5 and 0.7 < agree["avg"] < 0.8, f"summary: cross-check agreement {agree}")
    ok(s["kinds"]["cross_tests"]["done"] == 1, "summary: cross_tests counted in the queue table")
    # export: default keeps the best agreement (ties: shortest), --require-agreement drops < 0.5
    single = q.add("generate", {"prompt": "Unsampled.", "verify": {"kind": "verify_tests", "lang": "python"}})
    lease = q.lease("gen9", ["generate"], t0 + 40)
    _finish(q, lease, "gen9", True, {"text": A1, "finish_reason": "stop"}, t0 + 40)
    lease = q.lease("tester", ["verify_tests"], t0 + 41, test_langs=["python"])
    _finish(q, lease, "tester", True, _fake_verify(lease["payload"], True), t0 + 41)
    ok(not [jid for jid in kind_ids(q, "cross_tests") if jid != cids[0]], "a single (unsampled) prompt gets no cross-check")
    out, count = q.export("tests.jsonl")
    rows = [json.loads(line) for line in out.read_text(encoding="utf-8").splitlines()]
    grouped = [row for row in rows if row.get("group") == group]
    ok(count == 2 and len(grouped) == 1 and grouped[0]["job"] == by_text[A1] and grouped[0]["agreement"] == 1.0 and grouped[0]["verified_by"] == "verify_tests",
       "export: best agreement wins, ties go to the shorter answer")
    out, count = q.export("agreed.jsonl", per_prompt=0, require_agreement=True)
    rows = [json.loads(line) for line in out.read_text(encoding="utf-8").splitlines()]
    ok(count == 2 and {row["job"] for row in rows} == {by_text[A1], by_text[A2]} and all(row["agreement"] >= 0.5 for row in rows),
       "export --require-agreement: only agreement >= 0.5, no rows without a cross-check")
    out, count = q.export_pairs("tests-pairs.jsonl")
    pairs = [json.loads(line) for line in out.read_text(encoding="utf-8").splitlines()]
    ok(count == 1 and pairs[0]["chosen_job"] == by_text[A1] and pairs[0]["rejected_job"] == by_text[A4] and pairs[0]["chosen_agreement"] == 1.0
       and pairs[0]["rejected_error"] == "AssertionError", "pairs: chosen = best agreement, rejected = failed its own tests")
    ok(all(by_text[A3] not in (p["chosen_job"], p["rejected_job"]) for p in pairs), "pairs: a sample below 0.5 agreement is never chosen")
    mesh.JobQueue.__init__.__defaults__ = (Path(tmp),)
    mesh.jobs_main(["export", "--require-agreement", "cli-agreed.jsonl"])
    ok(len((Path(tmp) / "cli-agreed.jsonl").read_text(encoding="utf-8").splitlines()) == 1, "jobs export --require-agreement (one per prompt)")
    mesh.jobs_main(["export", "--per-prompt", "0", "--require-agreement", "cli-agreed-all.jsonl"])
    ok(len((Path(tmp) / "cli-agreed-all.jsonl").read_text(encoding="utf-8").splitlines()) == 2, "jobs export --per-prompt 0 --require-agreement")
    for bad_args in (["export", "--require-agreement"], ["export", "--bogus", "x.jsonl"], ["export", "--per-prompt", "x", "y.jsonl"]):
        try:
            mesh.jobs_main(bad_args)
            ok(False, f"bad export args accepted {bad_args}")
        except SystemExit as exc:
            ok("use:" in str(exc), f"bad export args refused {bad_args}")

# pairs never use cut-off replies or checks that could not run; bounded lease retries
with tempfile.TemporaryDirectory() as tmp:
    q = mesh.JobQueue(Path(tmp))
    gids = q.add_many([("generate", {"prompt": "p", "samples": 3, "max_tokens": 4096, "verify": {"kind": "verify_tests", "lang": "python"}})])
    t0 = 9000.0
    lease = q.lease("g0", ["generate"], t0, gen_max=4096)
    cut_id = lease["id"]
    _finish(q, lease, "g0", True, {"text": "```python\ndef f(", "finish_reason": "length"}, t0)
    ok(not [jid for jid in kind_ids(q, "verify_tests") if q.jobs[jid]["parent"] == cut_id] and cut_id in q.truncated, "a cut-off reply gets no verify_tests")
    for index in (1, 2):
        lease = q.lease(f"g{index}", ["generate"], t0 + index, gen_max=4096)
        _finish(q, lease, f"g{index}", True, {"text": A1 if index == 1 else A4, "finish_reason": "stop"}, t0 + index)
    # the first verify_tests lease expires twice: then it fails for good (no third lease)
    first = q.lease("w1", ["verify_tests"], t0 + 10, test_langs=["python"])
    again = q.lease("w2", ["verify_tests"], t0 + 10 + 121, test_langs=["python"])
    ok(again["id"] == first["id"], "a vanished worker's verify_tests goes back to the queue once")
    later = q.lease("w3", ["verify_tests"], t0 + 10 + 2 * 121 + 1, test_langs=["python"])
    ok(q.jobs[first["id"]]["state"] == "failed" and (later is None or later["id"] != first["id"]), "after the second expiry it fails and is never leased again")
    if later:
        ok(_finish(q, later, "w3", False, {"error": "unavailable: sandbox error"}, t0 + 300)[0], "a check that could not run is saved as failed")
    ok(not kind_ids(q, "cross_tests"), "no cross-check when fewer than two samples passed")
    out, count = q.export_pairs("p.jsonl")
    ok(count == 0, "pairs: no pair from a cut-off reply, an expired check or one that could not run")
    lease_cross = mesh.JobQueue(Path(tmp))
    cid = lease_cross.add("cross_tests", {"lang": "python", "group": "e" * 16, "samples": [{"id": "5" * 16, "code": "x"}, {"id": "6" * 16, "code": "y"}]})
    lease_cross.lease("w1", ["cross_tests"], t0, test_langs=["python"])
    lease_cross.lease("w2", ["cross_tests"], t0 + 401, test_langs=["python"])
    ok(lease_cross.lease("w3", ["cross_tests"], t0 + 803, test_langs=["python"]) is None and lease_cross.jobs[cid]["state"] == "failed",
       "cross_tests: at most one lease retry, then failed")
    ok(mesh.JobQueue(Path(tmp)).lease("w4", ["verify_tests", "cross_tests"], t0 + 2000, test_langs=["python"]) is None,
       "a fresh queue on the same file leases nothing finished again")

# HTTP: /jobs/next passes test_langs; the worker loop sends it
loop_text = Path(mesh.__file__).read_text(encoding="utf-8")
ok('"test_langs": sorted(sandbox_status()["langs"])' in loop_text and "test_langs=body.get(\"test_langs\")" in loop_text, "worker sends test_langs; serve passes it to lease")
ok('"tests": ({"langs"' in loop_text, "worker caps report whether it runs tests")

# seed set: coding-v2-tested
seed2 = Path(mesh.ROOT) / "seeds" / "coding-v2-tested.jsonl"
seed2_rows, seed2_errors = mesh.read_job_file("generate-verify", str(seed2))
raw2 = [json.loads(line) for line in seed2.read_text(encoding="utf-8").splitlines() if line.strip()]
ok(len(seed2_rows) == len(raw2) and not seed2_errors and 380 <= len(raw2) <= 420, f"read_job_file accepts coding-v2-tested ({len(raw2)} rows)")
ok(all(r["verify"]["kind"] == "verify_tests" and r["verify"]["lang"] in ("python", "javascript") and r["samples"] == 4 and r["max_tokens"] == 2048 for r in raw2),
   "seed2: verify_tests, 4 samples, 2048 tokens")
py_share = sum(r["verify"]["lang"] == "python" for r in raw2) / len(raw2)
he_share = sum(r["meta"]["hl"] == "he" for r in raw2) / len(raw2)
ok(0.55 <= py_share <= 0.65 and 0.07 <= he_share <= 0.13, f"seed2: ~60% Python ({py_share:.2f}), ~10% Hebrew ({he_share:.2f})")
ok({r["meta"]["difficulty"] for r in raw2} == {"easy", "medium", "hard"} and len({r["prompt"] for r in raw2}) == len(raw2)
   and len({r["meta"]["id"] for r in raw2}) == len(raw2), "seed2: all difficulties, no duplicate prompts or ids")
ok(all(("```python" in r["prompt"]) == (r["verify"]["lang"] == "python") and ("```javascript" in r["prompt"]) == (r["verify"]["lang"] == "javascript") for r in raw2),
   "seed2: each prompt asks for one block with the right language tag")
ok(all("node:assert" in r["prompt"] for r in raw2 if r["verify"]["lang"] == "javascript") and all("assert" in r["prompt"] for r in raw2),
   "seed2: prompts ask for assert tests (node:assert in JavaScript)")
ok(all(re.search(r"\b5\b", r["prompt"]) for r in raw2), "seed2: prompts ask for at least 5 tests")

# seed set: luau-v2-tested
seed3 = Path(mesh.ROOT) / "seeds" / "luau-v2-tested.jsonl"
seed3_rows, seed3_errors = mesh.read_job_file("generate-verify", str(seed3))
raw3 = [json.loads(line) for line in seed3.read_text(encoding="utf-8").splitlines() if line.strip()]
ok(len(seed3_rows) == len(raw3) and not seed3_errors and 380 <= len(raw3) <= 420, f"read_job_file accepts luau-v2-tested ({len(raw3)} rows)")
ok(all(r["verify"] == {"kind": "verify_luau"} and r["samples"] == 4 and r["max_tokens"] == 2048 and mesh.job_skill(mesh.validate_job("generate", r)) == "luau" for r in raw3),
   "seed3: verify_luau, 4 samples, 2048 tokens, luau skill")
diff3 = {d: sum(r["meta"]["difficulty"] == d for r in raw3) / len(raw3) for d in ("easy", "medium", "hard")}
he3 = sum(r["meta"]["hl"] == "he" for r in raw3) / len(raw3)
ok(0.15 <= diff3["easy"] <= 0.25 and 0.45 <= diff3["medium"] <= 0.55 and 0.25 <= diff3["hard"] <= 0.35 and 0.07 <= he3 <= 0.13,
   f"seed3: ~20/50/30 difficulty {diff3}, ~10% Hebrew ({he3:.2f})")
ok(len({r["prompt"] for r in raw3}) == len(raw3) and len({r["meta"]["id"] for r in raw3}) == len(raw3), "seed3: no duplicate prompts or ids")
ok(all("```lua" in r["prompt"] and "assert(" in r["prompt"] and re.search(r"\b5\b", r["prompt"])
       and all(word in r["prompt"] for word in ("os", "io", "require", "loadstring")) for r in raw3),
   "seed3: one ```lua block, at least 5 asserts, os/io/require/loadstring forbidden")
ok(mesh.verify_payload({"kind": "verify_luau"}, "x\n```lua\nassert(1 == 1)\n```\n")["code"] == "assert(1 == 1)\n", "seed3: verify_luau takes the ```lua block")

# docs and page
readme_text = (Path(mesh.ROOT) / "README.md").read_text(encoding="utf-8")
guide_text = (Path(mesh.ROOT) / "docs" / "GUIDE.md").read_text(encoding="utf-8")
ok("verify_tests" in guide_text and "NEURO_RUN_TESTS" in guide_text and "not blocked" in guide_text and "--require-agreement" in guide_text,
   "guide: verify_tests, NEURO_RUN_TESTS, network not blocked, --require-agreement")
ok("coding-v2-tested" in readme_text and "not blocked" in readme_text, "readme: tested seed set and network note")
ok("luau-v2-tested" in readme_text and "luau-v2-tested" in guide_text, "readme and guide: luau-v2-tested seed set")
page = mesh.DASHBOARD.read_text(encoding="utf-8")
factory = page[page.index("function duration("):page.index("function appRows(")]
ok("function testsAgreement(" in factory and "row.agreement" in factory and "Cross-check agreement" in factory and "innerHTML" not in factory,
   "page: verify_tests agreement next to the pass rate, textContent only")
ok("caps.tests" in factory and "network not blocked" in factory, "page: a worker row says it runs tests and that the network is not blocked")

# --- worker job slots: N generate + M check slots, each with its own lease ------
with tempfile.TemporaryDirectory() as tmp:
    q = mesh.JobQueue(Path(tmp))
    gids = [q.add("generate", {"prompt": f"slot {i}?"}) for i in range(4)]
    vid = q.add("verify_math", {"answer": "\\boxed{2}", "expected": "2"})
    t0 = 5000.0
    a = q.lease("slotbox", ["generate"], t0, slot="g0")
    b = q.lease("slotbox", ["generate"], t0 + 1, slot="g1")
    ok(a["id"] == gids[0] and b["id"] == gids[1], "two generate slots of one computer get two different jobs")
    ok(q.lease("slotbox", ["generate"], t0 + 2, slot="g0")["id"] == gids[0], "the same slot gets its own job back")
    c = q.lease("slotbox", ["verify_math"], t0 + 3, slot="c0")
    ok(c["id"] == vid, "a check slot leases a check job next to the generate slots")
    legacy = q.lease("slotbox", ["generate"], t0 + 4)
    ok(legacy["id"] == gids[2] and q.lease("slotbox", ["generate"], t0 + 5)["id"] == gids[2], "a worker without slots still holds one job at a time")
    ok(q.lease("slotbox", ["generate"], t0 + 6, slot="g9")["id"] == gids[2] and q.lease("slotbox", ["generate"], t0 + 6, slot="x0")["id"] == gids[2]
       and q.lease("slotbox", ["generate"], t0 + 6, slot=["g1"])["id"] == gids[2], "an unknown slot id counts as no slot (at most 16 slots per computer)")
    nodes = q.summary(t0 + 7)["nodes"]
    ok(sorted(nodes["slotbox"]["running"]) == ["generate", "generate", "generate", "verify_math"] and nodes["slotbox"]["now"] == "generate",
       "summary lists every job a computer's slots run; now stays one kind")
    ok(q.finish({"id": a["id"], "lease": a["lease"], "node": "otherbox", "ok": True, "output": {"text": "x"}}, t0 + 8)[0] is False, "a slot's job still only finishes from its own computer")
    ok(q.finish({"id": a["id"], "lease": a["lease"], "node": "slotbox", "ok": True, "output": {"text": "x"}}, t0 + 8)[0], "a slot's result is accepted")
    nxt = q.lease("slotbox", ["generate"], t0 + 9, slot="g0")
    ok(nxt["id"] == gids[3] and q.lease("slotbox", ["generate"], t0 + 9, slot="g1")["id"] == gids[1], "a finished slot leases the next job; the other slot keeps its own")
    q.lease("slotbox", ["generate"], t0 + 10 + 900, slot="g2")  # every generate lease above has expired
    ok(q.by_node.get("slotbox/g1") is None and q.by_node.get("slotbox/g0") is None and q.jobs[gids[3]]["state"] == "queued",
       "an expired slot lease frees the slot and requeues the job")

    # HTTP: /jobs/next passes the slot through.
    mesh._queue = mesh.JobQueue(Path(tmp) / "http")
    for i in range(2):
        mesh._queue.add("verify_math", {"answer": f"\\boxed{{{i}}}", "expected": str(i)})
    mesh.Handler.mode = "serve"
    srv = ThreadingHTTPServer(("127.0.0.1", 0), mesh.Handler)
    threading.Thread(target=srv.serve_forever, daemon=True).start()
    hbase = f"http://127.0.0.1:{srv.server_address[1]}"
    auth = {"Authorization": "Bearer " + FAKE, "Content-Type": "application/json"}

    def jnext(body):
        req = urllib.request.Request(hbase + "/jobs/next", data=json.dumps(body).encode(), headers=auth, method="POST")
        with mesh._DIRECT.open(req, timeout=5) as r:
            return json.loads(r.read())

    j0 = jnext({"node": "httpbox", "kinds": ["verify_math"], "slot": "c0"})
    j1 = jnext({"node": "httpbox", "kinds": ["verify_math"], "slot": "c1"})
    ok(j0["id"] != j1["id"], "/jobs/next gives each slot of one computer its own job")
    srv.shutdown()
    srv.server_close()

# slot counts
saved_env = {key: os.environ.get(key) for key in ("NEURO_GEN_SLOTS", "NEURO_CPU_SLOTS", "NEURO_LOCAL_LLM")}
try:
    for key in ("NEURO_GEN_SLOTS", "NEURO_CPU_SLOTS"):
        os.environ.pop(key, None)
    ok(mesh.gen_slot_count(None) == 1 and mesh.gen_slot_count(2) == 2 and mesh.gen_slot_count(8) == 4, "generate slots: model's count, capped at 4, else 1")
    ok(mesh.cpu_slot_count(2) == 1 and mesh.cpu_slot_count(8) == 2 and mesh.cpu_slot_count(12) == 3 and mesh.cpu_slot_count(64) == 4,
       "check slots: cpu_count // 4, 1 to 4")
    os.environ["NEURO_GEN_SLOTS"], os.environ["NEURO_CPU_SLOTS"] = "3", "0"
    ok(mesh.gen_slot_count(1) == 3 and mesh.cpu_slot_count(64) == 0, "NEURO_GEN_SLOTS / NEURO_CPU_SLOTS override")
    os.environ["NEURO_GEN_SLOTS"], os.environ["NEURO_CPU_SLOTS"] = "99", "two"
    ok(mesh.gen_slot_count(1) == mesh.SLOT_CAP and mesh.cpu_slot_count(8) == 2, "an override is capped at 8; a bad one is ignored")
    os.environ.pop("NEURO_GEN_SLOTS")
    os.environ.pop("NEURO_CPU_SLOTS")

    # model_parallel_slots against a fake llama-server on loopback
    llm_reply = {}

    class FakeLlm(BaseHTTPRequestHandler):
        def log_message(self, *args):
            pass

        def do_GET(self):
            body = llm_reply.get(self.path)
            if body is None:
                self.send_response(404)
                self.end_headers()
                return
            raw = json.dumps(body).encode()
            self.send_response(200)
            self.send_header("Content-Type", "application/json")
            self.send_header("Content-Length", str(len(raw)))
            self.end_headers()
            self.wfile.write(raw)

    llm = ThreadingHTTPServer(("127.0.0.1", 0), FakeLlm)
    threading.Thread(target=llm.serve_forever, daemon=True).start()
    os.environ["NEURO_LOCAL_LLM"] = str(llm.server_address[1])
    llm_reply.update({"/props": {"total_slots": 3, "default_generation_settings": {"n_ctx": 8192}}})
    ok(mesh.model_parallel_slots() == 3, "slots read from GET /props total_slots")
    llm_reply.clear()
    llm_reply["/props"] = {"default_generation_settings": {"total_slots": 2}}
    ok(mesh.model_parallel_slots() == 2, "slots read from /props default_generation_settings")
    llm_reply.clear()
    llm_reply["/slots"] = [{"id": 0}, {"id": 1}]
    ok(mesh.model_parallel_slots() == 2, "slots read from GET /slots when /props says nothing")
    llm_reply.clear()
    llm_reply["/props"] = {"total_slots": True}
    ok(mesh.model_parallel_slots() is None and mesh.gen_slot_count(mesh.model_parallel_slots()) == 1, "no slot count -> one generate slot")
    llm.shutdown()
    llm.server_close()
finally:
    for key, value in saved_env.items():
        if value is None:
            os.environ.pop(key, None)
        else:
            os.environ[key] = value

# machine_busy per lease: only generate pauses for the GPU lock
real = {name: getattr(mesh, name) for name in ("cpu_busy_pct", "ram_mib", "lock_held")}
try:
    mesh.cpu_busy_pct = lambda: 10
    mesh.ram_mib = lambda: {"free_mib": 8000}
    mesh.lock_held = lambda path: True
    if os.name == "nt":
        ok(mesh.machine_busy() == "training holds the GPU" and mesh.machine_busy(gpu=False) == "", "GPU lock pauses generate slots, not check slots")
    mesh.cpu_busy_pct = lambda: 90
    ok(mesh.machine_busy(gpu=False) == "CPU busy", "a busy CPU pauses check slots too")
    saved_on_cpu = os.environ.pop("NEURO_MODEL_ON_CPU", None)
    try:
        mesh.lock_held = lambda path: False
        ok(mesh.machine_busy() == "CPU busy", "a busy CPU pauses generate slots when the model is on a GPU")
        os.environ["NEURO_MODEL_ON_CPU"] = "1"
        ok(mesh.machine_busy() == "" and mesh.machine_busy(gpu=False) == "CPU busy",
           "model on the CPU: its own load does not pause generate slots, check slots still pause")
    finally:
        os.environ.pop("NEURO_MODEL_ON_CPU", None)
        if saved_on_cpu is not None:
            os.environ["NEURO_MODEL_ON_CPU"] = saved_on_cpu
        mesh.lock_held = lambda path: True
    mesh.cpu_busy_pct = lambda: 10
    mesh.ram_mib = lambda: {"free_mib": 500}
    ok(mesh.machine_busy(gpu=False) == "low memory", "low memory pauses check slots too")
finally:
    for name, value in real.items():
        setattr(mesh, name, value)

# slot_once: parallel slots, kinds per slot, thread-safe state, crash isolation
real = {name: getattr(mesh, name) for name in ("slot_plan", "worker_kinds", "machine_busy", "sandbox_status", "slot_once")}
try:
    mesh.slot_plan = lambda refresh=False: {"gen": 2, "cpu": 2}
    mesh.worker_kinds = lambda: ["generate", "verify_math", "dedup_hash"]
    busy_for = {"gpu": "", "cpu": ""}
    mesh.machine_busy = lambda gpu=True: busy_for["gpu" if gpu else "cpu"]
    mesh.sandbox_status = lambda refresh=False: {"langs": {}, "note": ""}
    posted = []
    posted_lock = threading.Lock()
    same_id = {"on": False}

    def fake_post(base, path, body, timeout):
        with posted_lock:
            posted.append((path, json.loads(json.dumps(body))))
        if path == "/jobs/next":
            jid = "shared" if same_id["on"] else "job-" + body["slot"]
            return 200, {"id": jid, "kind": body["kinds"][0], "lease": "L-" + body["slot"], "payload": {}}
        return 200, {"accepted": True}

    release = threading.Event()
    started = threading.Semaphore(0)

    def slow_run(job):
        started.release()
        release.wait(10)
        return True, {"text": "ok"}

    before = mesh.job_state_view()
    results = {}
    threads = [threading.Thread(target=lambda s=s: results.__setitem__(s, mesh.slot_once(s, "http://x", "box", post=fake_post, run=slow_run)))
               for s in ("g0", "g1", "c0")]
    for th in threads:
        th.start()
    for _ in threads:
        started.acquire(timeout=10)
    view = mesh.job_state_view()
    ok(view["slots"]["generate"]["busy"] == 2 and view["slots"]["check"]["busy"] == 1 and view["slots"]["generate"]["total"] >= 2,
       f"three slots run at once: {view['slots']}")
    ok(view["running"] == ["generate", "generate", "verify_math"] and view["current"] == "generate", "running is a list of kinds; current stays one kind")
    release.set()
    for th in threads:
        th.join(10)
    after = mesh.job_state_view()
    ok(all(results[s] == 1.0 for s in ("g0", "g1", "c0")) and after["done"] == before["done"] + 3 and not after["running"], "every slot reports; done counted once per job")
    nexts = {body["slot"]: body for path, body in posted if path == "/jobs/next"}
    ok(nexts["g0"]["kinds"] == ["generate"] and nexts["c0"]["kinds"] == ["verify_math", "dedup_hash"] and nexts["g1"]["node"] == "box",
       "generate slots lease only generate; check slots everything else")
    ok(sorted(body["id"] for path, body in posted if path == "/jobs/result") == ["job-c0", "job-g0", "job-g1"], "each slot posts its own result with its own lease")

    # a crash in one slot's handler fails that job only
    posted.clear()

    def boom(job):
        raise RuntimeError("handler bug")

    before = mesh.job_state_view()
    ok(mesh.slot_once("c1", "http://x", "box", post=fake_post, run=boom) == 1.0, "a crashing handler does not stop its slot")
    res = [body for path, body in posted if path == "/jobs/result"]
    after = mesh.job_state_view()
    ok(len(res) == 1 and res[0]["ok"] is False and "RuntimeError" in res[0]["output"]["error"] and after["failed"] == before["failed"] + 1 and not after["running"],
       "the crashed job is reported failed and its lease handed back")

    # slot_loop: an unexpected error stays inside the slot
    mesh.slot_once = lambda slot, base, node, **kw: (_ for _ in ()).throw(RuntimeError("bad round"))
    try:
        mesh.slot_loop("c0", "http://x", "box", rounds=1)
        ok(True, "slot_loop survives an unexpected error in one round")
    except Exception:
        ok(False, "slot_loop raised")
    mesh.slot_once = real["slot_once"]

    # result posting is retried 3 times, no more
    tries = []

    def flaky_post(base, path, body, timeout):
        if path == "/jobs/result":
            tries.append(1)
            raise OSError("down")
        return fake_post(base, path, body, timeout)

    mesh.slot_once("c0", "http://x", "box", post=flaky_post, run=lambda job: (True, {"text": "x"}), sleep=lambda s: None)
    ok(len(tries) == 3 and not mesh.job_state_view()["running"], "a result that cannot be posted is retried 3 times, then the slot moves on")

    # an older serve hands every slot the same job: only one slot runs it
    same_id["on"] = True
    release.clear()
    runs = []

    def counted(job):
        runs.append(job["id"])
        started.release()
        release.wait(10)
        return True, {"text": "x"}

    first = threading.Thread(target=lambda: mesh.slot_once("g0", "http://x", "box", post=fake_post, run=counted))
    first.start()
    started.acquire(timeout=10)
    ok(mesh.slot_once("g1", "http://x", "box", post=fake_post, run=counted) == 15.0 and runs == ["shared"], "a job another slot already runs is not run twice")
    release.set()
    first.join(10)
    same_id["on"] = False

    # GPU lock: generate slots pause, check slots keep leasing
    posted.clear()
    busy_for["gpu"] = "training holds the GPU"
    ok(mesh.slot_once("g0", "http://x", "box", post=fake_post, run=lambda job: (True, {"text": "x"})) == 15.0
       and not [1 for path, body in posted if body.get("slot") == "g0"], "a generate slot does not lease while training holds the GPU")
    ok(mesh.job_state_view()["paused"] == "training holds the GPU", "the page sees why the generate slots wait")
    ok(mesh.slot_once("c0", "http://x", "box", post=fake_post, run=lambda job: (True, {"result": "x"})) == 1.0
       and [1 for path, body in posted if path == "/jobs/next" and body.get("slot") == "c0"], "a check slot still leases while the GPU lock is held")
    busy_for["gpu"] = ""
    mesh.slot_once("g0", "http://x", "box", post=fake_post, run=lambda job: (True, {"text": "x"}))
    ok(mesh.job_state_view()["paused"] == "", "the pause clears when the GPU lock is gone")

    # slots above the plan stay idle
    posted.clear()
    ok(mesh.slot_once("g3", "http://x", "box", post=fake_post) == 60.0 and mesh.slot_once("c2", "http://x", "box", post=fake_post) == 60.0 and not posted,
       "a slot above the generate/check slot count does not lease")
finally:
    for name, value in real.items():
        setattr(mesh, name, value)
    mesh._kinds_cache["at"] = 0.0
caps_jobs = mesh.job_state_view()
ok({"on", "done", "failed", "paused", "current", "running", "slots"} <= set(caps_jobs) and set(caps_jobs["slots"]) == {"generate", "check"},
   "/hello caps.jobs carries running kinds and busy/total per slot sort")

# page: slots per computer, textContent only
page = mesh.DASHBOARD.read_text(encoding="utf-8")
slot_fn = page[page.index("function slotLine("):page.index("function memoryLine(")]
ok('"generating "' in slot_fn and '"checking "' in slot_fn and '" · "' in slot_fn and "innerHTML" not in slot_fn, "page: slotLine builds 'generating 2/2 · checking 1/3' as text")
ok(page.count("slotLine(") >= 4 and "innerHTML" not in page[page.index("function activity("):page.index("function memoryLine(")]
   and "innerHTML" not in page[page.index("function nowLine("):page.index("function headRow(")], "page: activity, nowLine and the Jobs row use slotLine, textContent only")
node_bin = mesh.node_path()
if node_bin:
    script = slot_fn + "\nconst r = slotLine({on: true, slots: {generate: {busy: 2, total: 2}, check: {busy: 1, total: 3}}, running: ['generate','generate','verify_math']});\n" \
             "const old = slotLine({on: true, current: 'generate'});\nconst bad = slotLine({on: true, slots: {generate: {busy: '<b>', total: 1e9}}});\n" \
             "process.stdout.write(JSON.stringify([r && r.text, old, bad && bad.text]));"
    out = _sp.run([node_bin, "-e", script], capture_output=True, text=True, timeout=20, encoding="utf-8")
    shown = json.loads(out.stdout or "null")
    ok(shown == ["generating 2/2 · checking 1/3", None, "generating 0/99"], f"page: slotLine text {shown}")

# launchers: llama-server with parallel slots, 127.0.0.1 only
for rel in ("platforms/steamdeck.sh", "platforms/mac-worker.sh"):
    raw = (repo / rel).read_bytes()
    text = raw.decode("utf-8")
    line = [l for l in text.splitlines() if "--port 8080" in l][0]
    ok(b"\r" not in raw and '-np "$MODEL_SLOTS" --kv-unified -c "$MODEL_CTX"' in line and "--host 127.0.0.1" in line and "0.0.0.0" not in text
       and 'MODEL_SLOTS="${NEURO_MODEL_SLOTS:-2}"' in text and "MODEL_CTX=$((MODEL_SLOTS * 8192))" in text, f"{rel}: llama-server -np 2 --kv-unified -c 16384 by default, LF, 127.0.0.1")
ps_raw = (repo / "platforms" / "windows-worker.ps1").read_bytes()
ps_text = ps_raw.decode("utf-8")
ok(ps_raw.count(b"\r\n") == ps_raw.count(b"\n") and "-np $ModelSlots --kv-unified -c $ModelCtx" in ps_text and "--host 127.0.0.1" in ps_text
   and "$ModelSlots = 2" in ps_text and "$ModelCtx = $ModelSlots * 8192" in ps_text and "NEURO_MODEL_SLOTS" in ps_text,
   "windows-worker.ps1: llama-server -np 2 --kv-unified -c 16384 by default, CRLF, 127.0.0.1")
for rel in ("README.md", "docs/GUIDE.md"):
    text = (repo / rel).read_text(encoding="utf-8")
    ok("NEURO_GEN_SLOTS" in text and "NEURO_CPU_SLOTS" in text and "slots" in text, f"{rel}: job slots and the two env vars")


# --- fair turns, long start, a model that cannot answer just then ------------
with tempfile.TemporaryDirectory() as tmp:
    q = mesh.JobQueue(Path(tmp))
    old = [q.add("generate", {"prompt": f"luau {i}", "verify": {"kind": "verify_luau"}, "max_tokens": 2048,
                              "meta": {"source": "seeds/luau"}}) for i in range(5)]
    new = q.add("generate", {"prompt": "6*7?", "verify": {"kind": "verify_math", "expected": "42"},
                             "meta": {"source": "seeds/math"}})
    t0 = 5000.0
    first = q.lease("A", ["generate"], t0, gen_max=4096, slot="g0")
    second = q.lease("A", ["generate"], t0, gen_max=4096, slot="g1")
    ok(first["id"] == old[0] and second["id"] == new, "sources take turns: the newer math source is not stuck behind luau")
    ok(first["payload"]["max_tokens"] == 4096 and first["payload"]["long_retry"] is True,
       "a luau task starts with room to finish (4096 tokens, no second long retry)")
    ok(second["payload"].get("max_tokens", 1024) < 4096, "a math task keeps its own max_tokens")
    legacy = q.lease("OLD", ["generate"], t0, gen_max=2048, slot="g0")
    ok(legacy["payload"]["max_tokens"] == 2048, "an old worker (2048 max) gets the task unchanged")
    accepted, note = q.finish({"id": first["id"], "lease": first["lease"], "node": "A", "ok": False,
                               "output": {"error": "HTTP Error 500: failed to find free space in the KV cache"}}, t0 + 1)
    ok(accepted and "back to the queue" in note and q.jobs[first["id"]]["state"] == "queued",
       "HTTP 500 from the model: the job goes back to the queue, not lost")
    again = q.lease("B", ["generate"], t0 + 2, gen_max=4096, slot="g0")
    ok(again["id"] == first["id"], "the returned job is leased again")
    q.finish({"id": again["id"], "lease": again["lease"], "node": "B", "ok": False, "output": {"error": "HTTP Error 500: x"}}, t0 + 3)
    third = q.lease("C", ["generate"], t0 + 4, gen_max=4096, slot="g0")
    q.finish({"id": third["id"], "lease": third["lease"], "node": "C", "ok": False, "output": {"error": "HTTP Error 500: x"}}, t0 + 5)
    ok(third["id"] == first["id"] and q.jobs[first["id"]]["state"] == "failed",
       "after MAX_ATTEMPTS leases a failing job is failed, not queued forever")
    bad = q.lease("D", ["generate"], t0 + 6, gen_max=4096, slot="g0")
    q.finish({"id": bad["id"], "lease": bad["lease"], "node": "D", "ok": False, "output": {"error": "ValueError: bad payload"}}, t0 + 7)
    ok(q.jobs[bad["id"]]["state"] != "queued", "a real error (not the model being busy) still fails at once")
    ok(mesh.job_source({"meta": {"source": "seeds/x"}}) == "seeds/x" and mesh.job_source({"verify": {"kind": "verify_js"}}) == "verify:verify_js",
       "job_source: the seed file, else the verifier")

# The worker slot rests after the model could not answer, and keeps llama-server's reason.
saved = {name: getattr(mesh, name) for name in ("slot_plan", "slot_kinds", "machine_busy", "sandbox_status")}
mesh.slot_plan = lambda: {"gen": 2, "cpu": 1}
mesh.slot_kinds = lambda slot: ["generate"]
mesh.machine_busy = lambda gpu=True: ""
mesh.sandbox_status = lambda: {"langs": []}
try:
    reply = lambda base, path, body, timeout: (200, {"id": "j1", "kind": "generate", "lease": "l", "payload": {}}) if path == "/jobs/next" else (200, {})  # noqa: E731
    rest = mesh.slot_once("g0", "http://x", "node", post=reply,
                          run=lambda job: (False, {"error": "HTTP Error 500: context size exceeded"}), sleep=lambda s: None)
    quick = mesh.slot_once("g0", "http://x", "node", post=reply, run=lambda job: (True, {"text": "ok"}), sleep=lambda s: None)
    asked = []
    def asking(base, path, body, timeout):
        if path == "/jobs/next":
            asked.append(body["gen_max"])
        return reply(base, path, body, timeout)
    cap_before = mesh._gen_cap["tokens"]
    full = lambda job: (False, {"error": "HTTP Error 500: {'code': 500, 'message': 'Context size has been exceeded.'}"})  # noqa: E731
    mesh.slot_once("g0", "http://x", "node", post=asking, run=full, sleep=lambda s: None)
    mesh.slot_once("g0", "http://x", "node", post=asking, run=lambda job: (True, {"text": "ok"}), sleep=lambda s: None)
finally:
    for name, value in saved.items():
        setattr(mesh, name, value)
ok(rest == 90.0 and quick == 1.0, f"a generate slot rests 90 s after the model could not answer ({rest}, {quick})")
ok(cap_before == mesh.GENERATE_MAX_TOKENS and asked == [cap_before, cap_before - 1024],
   f"a full model context lowers the tokens this worker asks for ({asked})")
mesh._gen_cap["tokens"] = mesh.GENERATE_MAX_TOKENS
ok(mesh.looping_answer("I will check it now. " * 60) and not mesh.looping_answer("\n".join(f"assert f({i}) == {i * i}" for i in range(40))),
   "looping_answer: a repeating reply yes, similar test lines no")

# Signals page: every number against a hand count.
NOW = 1_800_000_450.0          # 450 s into a 900 s bucket: the last bucket is half full
W = 21600                      # 6 h window, 900 s buckets
def res(at, node, kind="generate", ok_=True, tokens=0, cut=False, error=""):
    out = {"usage": {"completion_tokens": tokens}} if ok_ and kind == "generate" else {}
    if cut:
        out["finish_reason"] = "length"
    if error:
        out["error"] = error
    return {"id": f"r{at}{node}", "kind": kind, "node": node, "ok": ok_, "output": out, "at": at}
last = int(NOW // 900 * 900)     # start of the last (half-full) bucket
prev = last - 900                # a full bucket
results = [
    res(prev + 10, "mac", tokens=3000), res(prev + 20, "mac", tokens=1500, cut=True),
    res(prev + 30, "deck", tokens=900), res(prev + 40, "deck", ok_=False, error="HTTP Error 500: context size exceeded"),
    res(prev + 50, "deck", ok_=False, error="bad output"), res(prev + 60, "mac", kind="verify_luau"),
    res(last + 10, "mac", tokens=1800), res(last + 20, "mac", kind="verify_math", ok_=False),
    res(NOW - W - 5, "mac", tokens=99999),   # before the window: never counted
]
kind_of = {"g1": "generate", "g2": "generate", "c1": "verify_luau", "c2": "verify_math"}
events = [
    {"op": "add", "id": "g9", "kind": "generate", "payload": {}, "at": prev + 5},
    {"op": "finish", "id": "g1", "ok": True, "at": prev + 10},
    {"op": "finish", "id": "g1", "ok": True, "at": prev + 11},       # a second finish of the same job: ignored
    {"op": "add", "id": "c1", "kind": "verify_luau", "payload": {}, "at": prev + 12},
    {"op": "finish", "id": "c1", "ok": True, "at": last + 5},
    {"op": "finish", "id": "g2", "ok": False, "node": "", "at": last + 6},   # lease expired
]
machines = [
    {"key": "mac", "name": "Mac", "ok": True, "caps": {"local_model": True, "free_ram_mib": 8000, "jobs": {"running": [], "slots": {"generate": {"total": 2}}}}},
    {"key": "pi", "name": "Pi", "ok": False, "caps": {}},
    {"key": "laptop", "name": "Laptop", "ok": True, "caps": {"local_model": False, "free_ram_mib": 300, "jobs": {"running": [], "paused": "low memory", "slots": {"generate": {"total": 0}}}}},
]
sig = mesh.signals_compute(results, events, kind_of, {"generate": 10, "checks": 3}, NOW, W, machines, {"running": True, "state": "running"})
s = sig["series"]
ok(len(s["buckets"]) == 25 and s["buckets"][-1] == last and s["covered_s"][-1] == 450.0 and s["covered_s"][-2] == 900.0,
   f"signals: 25 buckets of 900 s, the last one covers only the 450 s so far ({len(s['buckets'])}, {s['covered_s'][-2:]})")
first_cover = s["covered_s"][0]
ok(abs(sum(s["covered_s"]) - W) < 1.5, f"signals: the buckets cover exactly the window ({sum(s['covered_s'])} of {W} s, first {first_cover})")
ok(s["counts"]["gen_ok"][-2] == 3 and s["counts"]["gen_retry"][-2] == 1 and s["counts"]["gen_failed"][-2] == 1
   and s["counts"]["gen_cut"][-2] == 1 and s["counts"]["gen_tokens"][-2] == 5400,
   "signals: a full bucket counts 3 answers, 1 retry, 1 failure, 1 cut off, 5400 tokens")
ok(s["gen_per_hour"][-2] == 12.0 and s["gen_per_hour"][-1] == 8.0,
   f"signals: answers per hour divide by the seconds covered (3 in 900 s = 12/h, 1 in 450 s = 8/h; {s['gen_per_hour'][-2:]})")
ok(s["tokens_per_s"][-2] == 6.0 and s["tokens_per_s"][-1] == 4.0, f"signals: tokens/s = 5400/900 and 1800/450 ({s['tokens_per_s'][-2:]})")
ok(s["tokens_per_job"][-2] == 1800 and s["failed_pct"][-2] == 40.0 and s["cut_pct"][-2] == 33.3,
   f"signals: 1800 tokens a job, 2 of 5 attempts failed (40%), 1 of 3 cut off ({s['tokens_per_job'][-2]}, {s['failed_pct'][-2]}, {s['cut_pct'][-2]})")
ok(s["checks_per_hour"][-2] == 4.0 and s["checks_per_hour"][-1] == 8.0, f"signals: checks per hour ({s['checks_per_hour'][-2:]})")
ok(all(mesh.TRANSIENT_MODEL_ERROR.search(e) for e in (
       "ConnectionResetError: [WinError 10054] An existing connection was forcibly closed by the remote host",
       "ConnectionAbortedError: [WinError 10053] An established connection was aborted",
       "URLError: <urlopen error [WinError 10061] No connection could be made",
       "ConnectionResetError: [Errno 104] Connection reset by peer"))
   and not mesh.TRANSIENT_MODEL_ERROR.search("The model returned an empty reply format"),
   "a dropped connection to the model (Windows and Linux spellings) is a retry, not a failure")
ok(s["silent"][-3] and s["tokens_per_s"][-3] is None and s["gen_per_hour"][-3] is None
   and sig["series"]["node_tokens_per_s"]["mac"][-3] is None and sig["series"]["node_tokens_per_s"]["deck"][-1] == 0.0,
   "signals: a slice where no computer finished anything has no rate (a gap), a quiet computer in a busy slice has 0")
ok(s["open_generate"][-1] == 10 and s["open_checks"][-1] == 3, "signals: the last point of open jobs is the exact number open now")
# Walking back over the last bucket: c1 and g2 finished in it -> one more of each was open before it.
ok(s["open_checks"][-2] == 4 and s["open_generate"][-2] == 11,
   f"signals: open jobs at the end of the previous bucket add back what finished after it ({s['open_generate'][-2]}, {s['open_checks'][-2]})")
# Before the previous bucket: g9 and c1 were added in it (one fewer open earlier), g1 finished once in it.
ok(s["open_generate"][-3] == 11 and s["open_checks"][-3] == 3,
   f"signals: adds in a bucket mean fewer open before it; a repeated finish counts once ({s['open_generate'][-3]}, {s['open_checks'][-3]})")
mac = next(r for r in sig["nodes"] if r["node"] == "mac")
deck = next(r for r in sig["nodes"] if r["node"] == "deck")
# The hour starts at NOW-3600 but the first attempt is at prev+10: 2260 s with nothing from any
# computer (over the 900 s gap rule), so the hour counts 3600 - 2260 = 1340 s.
counted = 3600 - ((prev + 10) - (NOW - 3600))
ok(sig["now"]["recent_s"] == round(counted) and sig["now"]["recent_left_out_s"] == round(3600 - counted),
   f"signals: a stretch with no attempt from any computer is left out of the hour ({sig['now']['recent_s']} counted)")
ok(mac["gen_done"] == 3 and mac["tokens_per_job"] == 2100 and mac["tokens_per_s"] == round(6300 / counted, 2) and mac["checks"] == 2,
   f"signals: Mac's last hour: 3 answers, 6300 tokens over the counted seconds ({mac})")
ok(deck["gen_done"] == 1 and deck["gen_retry"] == 1 and deck["gen_failed"] == 1 and deck["failed_pct"] == 66.7,
   f"signals: Deck's last hour counts the retry as a failed attempt ({deck})")
ok(sig["now"]["tokens_per_s"] == round(7200 / counted, 2) and sig["now"]["tokens_per_job"] == 1800 and sig["now"]["gen_done"] == 4,
   f"signals: the last hour across machines ({sig['now']})")
pace = sig["now"]["paces"][0]
ok(pace["window_s"] == 3600 and pace["done"] == 4 and pace["counted_s"] == round(counted) and pace["eta_s"] == int(10 * counted / 4),
   f"signals: 10 open answers at 4 per {counted:.0f} counted seconds ({pace})")
busy = [res(NOW - 3600 + i * 300, "pc", tokens=600) for i in range(12)]
busy_sig = mesh.signals_compute(busy, [], {}, {"generate": 6}, NOW, W, [], None)
ok(busy_sig["now"]["recent_s"] == 3600 and busy_sig["now"]["tokens_per_s"] == 2.0 and busy_sig["now"]["paces"][0]["eta_s"] == 1800,
   f"signals: with no gap the hour is the full 3600 s (7200 tokens = 2.0 tok/s, 6 left at 12/h = 30 min) ({busy_sig['now']})")
ok(mesh.active_seconds([100, 200, 1500, 1600], 0, 2000) == (2000 - 1300, 1300) and mesh.active_seconds([], 0, 600) == (600, 0),
   "active_seconds: only gaps over 15 min are left out")
titles = [x["title"] for x in sig["signals"]]
ok(sig["signals"][0]["level"] == "bad" and "Pi does not answer" in titles and "Laptop has no model" in titles
   and "Laptop is low on memory" in titles and "Laptop is pausing new jobs" in titles,
   f"signals: offline, no model, low memory and paused are raised, worst first ({titles})")
ok(not any(x["level"] == "good" for x in sig["signals"]), "signals: no 'everything is working' line next to warnings")
flow_rows = [res(NOW - 100, "mac", tokens=500), res(NOW - 90, "mac", ok_=False, error="bad"),
             dict(res(NOW - 80, "deck", kind="verify_luau"), passed=True), dict(res(NOW - 70, "deck", kind="verify_luau"), passed=False),
             dict(res(NOW - 60, "deck", kind="verify_math", ok_=False), passed=None), res(NOW - 2000, "mac", tokens=9)]
fl = mesh.signals_compute(flow_rows, [], {}, {"generate": 1}, NOW, W, [], None)["flows"]
ok(fl["window_s"] == 900 and {(c["node"], c["kind"], c["ok"], c["failed"]) for c in fl["machine_kind"]}
   == {("mac", "generate", 1, 1), ("deck", "verify_luau", 2, 0), ("deck", "verify_math", 0, 1)},
   f"flows: the last 15 minutes per computer and kind, ok and not ok ({fl['machine_kind']})")
ok({(o["kind"], o["training"], o["rejected"]) for o in fl["kind_outcome"]} == {("verify_luau", 1, 1), ("verify_math", 0, 1)},
   f"flows: only a check that ran and passed goes to training ({fl['kind_outcome']})")
calm = mesh.signals_compute([], [], {}, {"generate": 0, "checks": 0}, NOW, 3600, [], None)
ok(calm["signals"][0]["level"] == "good" and abs(sum(calm["series"]["covered_s"]) - 3600) < 1.5
   and any(x["title"] == "The generation queue is empty" for x in calm["signals"]),
   f"signals: a quiet hour says everything works and that the queue is empty ({[x['title'] for x in calm['signals']]})")
slow = [res(NOW - 3600 * 5 + i * 900 + 10, "oh", tokens=9000) for i in range(16)] + [res(last + 10, "oh", tokens=200)]
slow_sig = mesh.signals_compute(slow, [], {}, {"generate": 5, "checks": 0}, NOW, W, [], None)
ok(any(x["title"] == "oh is writing slowly" for x in slow_sig["signals"]),
   f"signals: a machine far under its usual tokens/s is flagged ({[x['title'] for x in slow_sig['signals']]})")
fails = [res(NOW - 600 + i, "bad", ok_=False, error="bad output") for i in range(3)] + [res(NOW - 500, "bad", tokens=100)]
ok(any(x["title"] == "bad fails often" and x["level"] == "bad" for x in mesh.signals_compute(fails, [], {}, {"generate": 1}, NOW, W, [], None)["signals"]),
   "signals: 3 of 4 failed attempts in the last hour is flagged")
retries = [res(NOW - 600 + i, "slowpc", ok_=False, error="TimeoutError: timed out") for i in range(2)] + [res(NOW - 500 + i, "slowpc", tokens=100) for i in range(2)]
retry_sig = mesh.signals_compute(retries, [], {}, {"generate": 1}, NOW, W, [], None)["signals"]
ok(any(x["title"] == "slowpc often has to retry" and x["level"] == "warn" for x in retry_sig)
   and not any(x["title"] == "slowpc fails often" for x in retry_sig),
   f"signals: time-outs that went back to the queue are a warning, not a failure ({[x['title'] for x in retry_sig]})")
log_path = Path(tempfile.mkdtemp()) / "log.jsonl"
with open(log_path, "w", encoding="utf-8") as handle:
    for i in range(5000):
        handle.write(json.dumps({"at": 1000 + i, "pad": "x" * 400}) + "\n")
tail = mesh.rows_since(log_path, 5900)
ok(len(tail) == 100 and tail[0]["at"] == 5900 and tail[-1]["at"] == 5999,
   f"rows_since: reads back only the rows at or after the time, in order ({len(tail)})")
ok(len(mesh.rows_since(log_path, 0)) == 5000 and mesh.rows_since(log_path.with_name("missing.jsonl"), 0) == [],
   "rows_since: the whole file when asked, nothing for a missing file")


fake.shutdown()
fake.server_close()
print(f"\nALL {passed} CHECKS PASSED")
