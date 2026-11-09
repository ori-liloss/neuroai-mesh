# Neuro guide

Step by step: set up the main PC, add computers (Windows, Mac, Linux, Steam Deck), make verified
coding data with them, and use that data for training. Every command below is meant to be copied.

- [1. What Neuro is](#1-what-neuro-is)
- [2. Set up the main PC](#2-set-up-the-main-pc)
- [3. Add a computer (pairing)](#3-add-a-computer-pairing)
- [4. Per-machine setup](#4-per-machine-setup) — Windows, Mac, Linux, Steam Deck
- [5. A local model so a computer can generate](#5-a-local-model-so-a-computer-can-generate)
- [6. The data factory](#6-the-data-factory)
- [7. Agents](#7-agents)
- [8. Training](#8-training)
- [9. Security model](#9-security-model)
- [10. Troubleshooting](#10-troubleshooting)
- [11. What helps and what does not](#11-what-helps-and-what-does-not)
- [12. Roadmap](#12-roadmap)

## 1. What Neuro is

```
                 your home network (private LAN only)
 ┌────────────────────────────┐        ┌──────────────────────────┐
 │ main PC (NVIDIA GPU)       │        │ worker computers         │
 │  mesh.py serve  :8780      │◄──────►│  mesh.py work    :8781   │
 │  page + job queue + chat   │ pairing│  local model 127.0.0.1   │
 │  trains the model          │  jobs  │  makes + checks data     │
 └────────────────────────────┘        └──────────────────────────┘
```

- **One page** on the main PC shows every computer, what it is doing, and the job queue.
- **One training run** stays on the main PC's GPU. A model is not split across computers over
  Wi-Fi (too slow, and CUDA and Metal cannot share a run).
- **Everything else is shared:** the other computers *make* and *check* training data. A job is data
  for a built-in handler, never a shell command. Only answers that pass a check are kept.
- This repository has no model weights and no private data.

## 2. Set up the main PC

Needs Python 3.10 or newer (no pip packages; developed on 3.14) and `git`.

```bash
git clone https://github.com/ori-liloss/neuroai-mesh.git
cd neuroai-mesh
python mesh.py serve
```

Windows: `python -u mesh.py serve`, or double-click `start-dashboard.cmd`.
Open <http://127.0.0.1:8780>. On first launch `mesh.py serve` creates `cluster.token`
(gitignored). Never commit it or send it to the internet.

Chat uses a llama.cpp server you start yourself on `http://127.0.0.1:8080/v1` (see section 5).

Windows firewall, once, in an Administrator PowerShell (private network only):

```powershell
netsh advfirewall firewall add rule name="Neuro worker" dir=in action=allow protocol=TCP localport=8781 profile=private
```

Never open 8780, 8781 or the llama.cpp RPC port 50052 to the public internet.

## 3. Add a computer (pairing)

No file copying. It works like pairing a Bluetooth device:

1. Start the worker on the new computer (section 4). If it has no token, or the main PC rejects
   its token, it prints a code: `Pairing code: 482 913`.
2. On the **main PC**, open its own page (<http://127.0.0.1:8780>). A **New computer wants to
   join** card shows the name, address and the same code.
3. Press **Approve** only if the code matches the one on that computer's screen. Press **Deny**
   otherwise.
4. The worker saves `cluster.token` (an old one is kept as `cluster.token.old`) and starts taking
   jobs without a restart. An unapproved code expires after 15 minutes; the worker then asks again.

Approval only works from the page on the main PC itself. The token is handed out once, only to
the computer that asked, and only with the secret that computer generated.

**How the worker finds the main PC.** In this order: `NEURO_MESH_URL` if set, then the address
remembered in `mesh_url.txt`, then a search of the worker's own /24 network for a Neuro page on
port 8780 (it is saved in `mesh_url.txt` once found). A worker that already holds a token only
accepts a main PC that proves it holds the same token. If nothing is found it tries
`127.0.0.1:8780` (this computer). Delete `mesh_url.txt` to search again. If the search cannot see
the main PC (another subnet, client isolation on Wi-Fi), set the address yourself:

```bash
# Mac / Linux / Steam Deck
export NEURO_MESH_URL=http://192.168.1.20:8780      # use your main PC's LAN IP
```
```powershell
# Windows
$env:NEURO_MESH_URL = "http://192.168.1.20:8780"
```

Find the main PC's address with `ipconfig` (Windows) or `hostname -I` (Linux).

Fallback if pairing cannot be used: copy `cluster.token` from the main PC into the worker's folder
by hand (USB stick), then restart the worker.

## 4. Per-machine setup

Every worker runs the same file (`mesh.py work`, port 8781). The start scripts do **not** update
the code by themselves: run `git pull` first when there is a new version.

### Windows laptop or desktop

```bat
git clone https://github.com/ori-liloss/neuroai-mesh.git
cd neuroai-mesh
python -u mesh.py work
```

If `python` is missing, use `py -3 -u mesh.py work`. Leave the window open. Once, as Administrator:

```bat
netsh advfirewall firewall add rule name="Neuro worker" dir=in action=allow protocol=TCP localport=8781 profile=private
```

### Windows laptop with an NVIDIA GPU (one command)

For a laptop that should also generate (for example an RTX 3050 Ti with 4 GB). Needs Git and
Python 3.10+ (`winget install Git.Git` and `winget install Python.Python.3.14`, then a new CMD
window). Plug it in, then in CMD:

```bat
git clone https://github.com/ori-liloss/neuroai-mesh.git %USERPROFILE%\neuroai-mesh
%USERPROFILE%\neuroai-mesh\platforms\windows-worker.cmd
```

It downloads llama.cpp (CUDA 12 build; Vulkan if `nvidia-smi` is missing) and a 2.9 GB model once
into `%LOCALAPPDATA%\Neuro`, starts the model hidden on `127.0.0.1:8080` (with part of it on the
GPU if 4 GB is not enough), adds the firewall rule when run as Administrator (otherwise it prints
the command), starts the worker and keeps the laptop awake. A pairing code appears after about
30 seconds; approve it on the main PC (section 3). Ctrl+C stops the worker and the model it
started. Next time, and for updates:

```bat
cd /d %USERPROFILE%\neuroai-mesh && git pull && platforms\windows-worker.cmd
```

If the main PC is not found: run `set NEURO_MESH_URL=http://<main PC>:8780` first, in the same window.

### Mac

```bash
git clone https://github.com/ori-liloss/neuroai-mesh.git
cd neuroai-mesh
bash platforms/start-mac.sh
```

With a local model too (recommended; one window, keeps the Mac awake): `brew install llama.cpp` once, then

```bash
cd ~/neuroai-mesh && git pull && sh platforms/mac-worker.sh
```

macOS has no `python`; the script uses `python3` (`brew install python` if missing). Do not open
127.0.0.1 on the Mac — that is the Mac itself. Do not run `mesh.py serve` there.

### Linux

```bash
git clone https://github.com/ori-liloss/neuroai-mesh.git
cd neuroai-mesh
sh platforms/start-linux.sh
```

### Steam Deck (a real Linux computer with a 16 GB GPU)

A Steam Deck is a good extra worker: it can run a small model and double the speed of data
generation. It is too weak for training.

**One command (recommended).** Hold the power button, choose **Switch to Desktop**, plug it in,
open **Konsole** (start menu) and paste (Ctrl+Shift+V):

```bash
git clone https://github.com/ori-liloss/neuroai-mesh.git ~/neuroai-mesh && sh ~/neuroai-mesh/platforms/steamdeck.sh
```

It downloads llama.cpp (Vulkan) and a 2.9 GB model once, starts the model on `127.0.0.1:8080`,
starts the worker, and keeps the Deck awake. A pairing code appears after about 30 seconds;
approve it on the main PC (section 3). Ctrl+C stops everything. Next time, and for updates:

```bash
cd ~/neuroai-mesh && git pull && sh platforms/steamdeck.sh
```

If the main PC is not found: `NEURO_MESH_URL=http://<main PC>:8780 sh platforms/steamdeck.sh`.

**Step by step (what the script does):**

1. Hold the power button, choose **Switch to Desktop**. Plug it in. In *System Settings → Power
   Management* set suspend and screen saving to *Never*. Open **Konsole** (start menu).
2. Download the project:

```bash
git clone https://github.com/ori-liloss/neuroai-mesh.git ~/neuroai-mesh
```

3. Download llama.cpp built for the Deck's GPU (Vulkan). Use the releases *list*, not
   `releases/latest` — for llama.cpp `latest` points to a release without binaries:

```bash
mkdir -p ~/llama && cd ~/llama && URL=$(curl -s "https://api.github.com/repos/ggml-org/llama.cpp/releases?per_page=1" | grep -oE '"browser_download_url": *"[^"]*ubuntu-vulkan-x64\.tar\.gz"' | head -1 | grep -oE 'https://[^"]*') && echo "$URL" && curl -L -o pkg.tar.gz "$URL" && tar xzf pkg.tar.gz && rm pkg.tar.gz && echo READY
```

   It must print a URL ending in `ubuntu-vulkan-x64.tar.gz` and then `READY`.

4. Download a small model (Qwen3.5-4B, about 3 GB) and start it. First tab:

```bash
mkdir -p ~/models && curl -L -o ~/models/Qwen3.5-4B-UD-Q4_K_XL.gguf https://huggingface.co/unsloth/Qwen3.5-4B-GGUF/resolve/main/Qwen3.5-4B-UD-Q4_K_XL.gguf
```
```bash
cd "$(dirname "$(find ~/llama -name llama-server -type f | head -1)")" && chmod +x ./* && systemd-inhibit --what=sleep:idle ./llama-server -m ~/models/Qwen3.5-4B-UD-Q4_K_XL.gguf --host 127.0.0.1 --port 8080 -c 8192 --jinja -ngl 99
```

   Wait for `server is listening`. Leave this tab open. (If `systemd-inhibit` errors, run the same
   command without `systemd-inhibit --what=sleep:idle `.)

5. New tab (Ctrl+Shift+T):

```bash
cd ~/neuroai-mesh && systemd-inhibit --what=sleep:idle sh platforms/start-linux.sh
```

6. A pairing code appears after about 30 seconds. Approve it on the main PC (section 3).

After a restart you only repeat steps 4 (second command) and 5; no new pairing is needed.
SteamOS is Arch-based: if the Ubuntu llama.cpp build fails to start with a missing-library error,
send the error text in an issue.

## 5. A local model so a computer can generate

A worker offers the `generate` job only when it sees a model server on `127.0.0.1:8080`
(checked once a minute at `/v1/models`). Keep the model server on **127.0.0.1**, never `0.0.0.0`.

| Computer | Model server |
| --- | --- |
| NVIDIA laptop (6–8 GB) | llama.cpp `llama-server` with a 2B–4B Q4 GGUF, `-ngl 99` |
| Apple Silicon Mac | `brew install llama.cpp`, then the command below |
| Steam Deck / AMD / Intel | Vulkan build (section 4) |
| Old laptop, no GPU | no model: it runs the checks (`verify_*`, `dedup_hash`) |

Mac, with automatic download (needs about 3 GB):

```bash
llama-server -hf unsloth/Qwen3.5-4B-GGUF:UD-Q4_K_XL --host 127.0.0.1 --port 8080 -c 8192 --jinja
```

Run it in its own terminal window, separate from the worker.
Another port: set `NEURO_LOCAL_LLM=8090` for the worker.

## 6. The data factory

The main PC keeps a queue in `jobs/` (gitignored). Idle workers pull jobs into job slots.

**Job slots.** Each worker runs several jobs at once: generate slots (as many as its local
`llama-server` serves in parallel, read from `/props` or `/slots`, at most 4; 1 if it does not say)
and check slots for `verify_*`, `cross_tests` and `dedup_hash` (one per 4 CPU cores, 1 to 4).
Every slot leases its own job; a busy CPU, low memory or the GPU lock pauses new leases, and the
GPU lock pauses only generate. Override with `NEURO_GEN_SLOTS` and `NEURO_CPU_SLOTS` (0 to 8).
The `platforms/` scripts start the model with `-np 2 --kv-unified -c 16384` (2 slots of 8k
context; `NEURO_MODEL_SLOTS` changes the count). The Jobs page shows e.g. `generating 2/2 · checking 1/3`.

| Job kind | Checks | Notes |
| --- | --- | --- |
| `generate` | asks the worker's own model | thinking off, up to 4096 tokens (a cut-off reply is retried once with 4096), never leaves the worker |
| `verify_python_syntax` | `compile()` | code is never run |
| `verify_js` | `node --check` | parse only; needs Node on the worker |
| `verify_ts` | Node's built-in TypeScript stripper + `node --check` | Node 22.13+ / 23.2+; enums are "inconclusive" |
| `verify_html_css` | tag nesting, unclosed script/style, CSS braces | stdlib only |
| `verify_sql` | SQLite in memory, whitelisted statements, `EXPLAIN` only | ATTACH/PRAGMA blocked |
| `verify_luau` | runs `luau` with limits | needs `luau` on PATH or `NEURO_LUAU` |
| `verify_math` | final answer vs expected | |
| `dedup_hash` | normalized hash + MinHash | |
| `verify_tests` | **runs** the answer's own asserts (Python via `python -I`, JavaScript via `node`) | sandboxed, opt-in; see below |
| `cross_tests` | **runs** each passing sample of a prompt against the other samples' asserts | queued once per prompt group |

The parse checks never run generated code. `verify_luau`, `verify_tests` and `cross_tests` do, with
limits, and no job is ever a shell command: a job is data for a built-in handler.

### Tests that really run (`verify_tests`, `cross_tests`)

"Verified" from a parse check only means the code parses. `verify_tests` means it works: the answer
holds the solution and at least 5 assert tests in **one** fenced block, the worker runs it, and it
passes only when

- the code has at least 3 real asserts that run (`assert True`, `assert 1`, `assert x == x`,
  `assert.strictEqual(2, 2)` do not count; Python asserts inside a test function count only when the
  file calls that function),
- nothing hides a failure (catching `AssertionError`, a `try`/`except Exception` or `catch` around
  the asserts that does not re-raise, `contextlib.suppress`, `process.on('uncaughtException')`,
  `console.assert`), and
- the run exits with code 0 within 10 s **and** reaches the end of the file (exiting early with
  code 0 is a failure).

The result keeps the runtime and the failure reason (the error line and the first 300 characters
of stderr). JavaScript tests use `const assert = require('node:assert')`.

When all samples of one prompt group are back and at least two passed their own tests, the main PC
queues one `cross_tests` job for that group (CodeT-lite): every passing sample runs once more with
the **other** samples' plain top-level asserts appended, and gets an agreement score, the share of
those asserts it passes. A sample that only satisfies its own (possibly wrong) tests scores low.

**Safety, and what is not covered.** Running model-written code is a real risk, so a worker only
takes these jobs when it can sandbox them:

| Where | When | Limits |
| --- | --- | --- |
| Windows | always (needs the Job Object API) | Job Object: 1 GB memory, one process (no child processes), 10 s CPU time, killed when closed |
| Linux / macOS | only with `NEURO_RUN_TESTS=1`, never as root | rlimits in `preexec_fn`: CPU 10 s, address space 1 GB, file size 16 MB, no core files, NPROC 0 (for node, which cannot start its threads under NPROC 0 on Linux, node's permission model blocks child processes instead) |

Always: a fresh temp folder as the working directory (never inside the repository), a scrubbed
environment (no tokens, proxies, home folder or interpreter options), stdin closed, a 10 s wall
timeout, output capped (a run that prints more than 4 MB is stopped), the folder deleted afterwards.
Node also runs with `--permission`, so it can read only its temp folder and cannot write files or
start processes. **Network access is not blocked**: a test could still open a connection. Do not
run these jobs on a computer whose network you do not trust with that. A worker's capability text
on the page (Jobs tab) says whether it runs tests.

Each test job has no verdict retry and at most one lease retry (a worker that vanished);
`cross_tests` is queued once per group and never queues anything itself.

### Use the ready coding set

`seeds/coding-v1.jsonl` has 321 original prompts (JavaScript, TypeScript, HTML/CSS, SQLite,
Python, Luau; about 10% Hebrew), each with `"samples": 4`.

```bash
python mesh.py jobs add generate-verify seeds/coding-v1.jsonl     # queues 1284 jobs
python mesh.py jobs status                                        # progress
python mesh.py jobs export coding-v1-verified.jsonl               # one verified answer per prompt
python mesh.py jobs export --pairs coding-v1-pairs.jsonl          # chosen vs rejected, for DPO/KTO
```

`--pairs` refuses paths inside a training-data folder. Answers whose check could not run are never
used as "rejected".

### The tested coding set

`seeds/coding-v2-tested.jsonl` has 400 original function-level tasks (Python 60%, JavaScript 40%;
easy, medium and hard; about 10% in Hebrew), each asking for the implementation plus at least 5
assert tests in one block, checked with `verify_tests`:

```bash
python mesh.py jobs add generate-verify seeds/coding-v2-tested.jsonl    # 1600 jobs
python mesh.py jobs export coding-v2-tested.jsonl                       # passed their own tests
python mesh.py jobs export --require-agreement coding-v2-agreed.jsonl   # and agreement >= 0.5
python mesh.py jobs export --pairs coding-v2-pairs.jsonl                # chosen vs failed own tests
```

`export` keeps one answer per prompt: with a cross-check, the highest agreement (ties: the
shortest). `--require-agreement` keeps only answers whose agreement is at least 0.5 (prompts where
fewer than two samples passed have no cross-check and are left out). In `--pairs`, chosen is the best
passing answer (never one below 0.5 agreement) and rejected is one that ran and failed its own
tests; cut-off replies and checks that could not run are never used.

### Your own prompts

A row of `prompts.jsonl`:

```json
{"prompt": "Write a JavaScript function that reverses words in a string. Answer in one ```javascript block.",
 "verify": {"kind": "verify_js"}, "samples": 4, "max_tokens": 1024}
```

Math: `{"prompt": "What is 6*7?", "verify": {"kind": "verify_math", "expected": "42"}}`.
Tested code: `{"prompt": "... plus at least 5 assert tests at the bottom of one python block", "verify": {"kind": "verify_tests", "lang": "python"}, "samples": 4}`
(`lang` is `python` or `javascript`).
Ask for the answer in **one fenced code block with the language tag**; the extractor looks there first.
Then `python mesh.py jobs add generate-verify prompts.jsonl`. If `jobs add` says
"Restart serve first", restart `mesh.py serve` — it is running older code.

A worker takes no job while its CPU is above 75%, it has under 1 GB free memory, or the GPU lock
is held. Every job has a time limit (its lease), so a worker that disappears does not block the queue.

## 7. Agents

`agents.py` is one file for every computer (`python agents.py agent` runs a worker,
`python agents.py drill` runs the self-test, `neuro` opens Neuro Code in CMD).

- **Task router:** each task is classified once as `quick`, `research`, `review` or `code`
  (English and Hebrew keywords). The kind sets thinking, token budget, turns and which tools the
  agent can use. `/deep` at the start forces thinking.
- **Code tasks** must run a compile or test after writing files before they may report "done".
- **Cheap on tokens:** a stable prompt prefix (the model server caches it), a larger context with
  the oldest tool results trimmed first, one retry instead of four on server errors.
- **Safety:** secret files cannot be read or written, dangerous `RUN` commands are blocked,
  writes stay in `workspace/`, a read that is cut off mid-file is refused.
- Finished tasks are logged under `agent-logs/` (gitignored). Four tasks run at once on one GPU;
  a hundred copies of a model do not fit on a 12 GB card.

## 8. Training

This repository has no trainer of its own. It gives you the data, a small experiment script, and a
live view of the run you start with your own trainer (for example a Hugging Face `Trainer` script).

**Show your run on the Training page.** Copy `training.example.json` to `training.json` next to
`mesh.py` (gitignored) and set:

```json
{"log": "D:/models/logs/training.log", "run_dir": "D:/models/runs/my-run", "name": "my-run"}
```

- `log`: the file your trainer's output is written to (redirect stdout and stderr into it). The
  page reads only the first 16 KB and the last 96 KB, so a log of any size is fine. It understands
  tqdm progress bars (`120/14688 [10:05<20:31:00, 5.07s/it]`), `Trainer` dicts with `loss` and
  `eval_loss`, a `Traceback`, the final `train_runtime` dict, and timestamped lines such as
  `2026-01-01T10:00:00 Training. Run folder: D:/models/runs/my-run`.
- `run_dir` (optional): the output folder. The newest `checkpoint-N/trainer_state.json` there fills
  in the loss history and the total steps when the log tail lacks them.
- Optional keys: `gpu_lock` (a file your trainer holds while it owns the GPU; default
  `../train_local/gpu.lock`), `data_root` (default `../neuro-ai-data`), `process_match` (a piece of
  the trainer's command line, used to tell a crashed run from a quiet one).

Environment variables win over the file: `NEURO_TRAINING_LOG`, `NEURO_TRAINING_RUN_DIR`,
`NEURO_GPU_LOCK`, `NEURO_AI_DATA`. Restart `mesh.py serve` after changing them.

States: **Loading** (only model-loading bars so far), **Training** / **Evaluating**, **Finished**,
**Error** (a traceback after the last progress), **Stopped** (the log stopped changing and no
trainer is running), **Stopped · lock held** (no trainer, but the GPU lock file is still there).
The page never starts or stops the trainer.

**Data format** (`jobs export` output): one JSON object per line,
`{"messages": [{"role": "user", "content": "..."}, {"role": "assistant", "content": "..."}], ...}`.
Hold out a slice for evaluation and keep your benchmark prompts out of the training data.

**Quick experiments on a side GPU:** `side_train.py` trains a LoRA on a small local model and
reports eval loss, so you can test a learning rate or a data mix before spending days on the big run:

```bash
python side_train.py --base /path/to/small-model --data coding-v1-verified.jsonl --out runs/lr1e-4 --learning-rate 1e-4 --max-steps 300
```

It refuses to start while a main training lock is held.

**Settings that worked for one home setup** (a starting point, not a rule):

| | RTX 4070 12 GB | 128 GB unified memory (DGX Spark) |
| --- | --- | --- |
| Base | Qwen3.5-9B | Qwen3.8-27B |
| Method | QLoRA 4-bit | LoRA bf16 |
| Memory | about 9–10 GB | about 66 GB |
| Learning rate | 1e-4, cosine, 3% warmup | same |
| LoRA | r=16, alpha=32, all linear layers, not the vision tower | same |
| Packing | **off** for Qwen3.5/3.8 (hybrid linear attention would mix conversations) | off |
| Loss | assistant tokens only | same |

Tips that save days: a learning rate of 5e-6 is far too low for LoRA (use 1e-4 to
2e-4); save every few hundred steps, not once per epoch; check that no benchmark question is in
the training data; a LoRA adapter only works on the exact base model it was trained on.

**Order of work that pays off most:** clean verified data first → one training run → measure on a
fixed benchmark (pass@1 and tokens per task) → a second round on verified answers with DPO/KTO.

## Settings reference

| Variable | Where | Meaning |
| --- | --- | --- |
| `NEURO_MESH_URL` | worker | The main PC's page, e.g. `http://192.168.1.20:8780`. Normally found by itself. |
| `NEURO_JOBS=0` | worker | Take no data jobs. |
| `NEURO_LOCAL_LLM` | worker | Port of the local model server (default 8080). |
| `NEURO_GEN_SLOTS`, `NEURO_CPU_SLOTS` | worker | Parallel generate / check job slots (default: the model server's slot count up to 4; cores/4, 1 to 4). |
| `NEURO_MODEL_SLOTS` | `platforms/` scripts | Parallel slots the started `llama-server` gets (default 2, 8k context each). |
| `NEURO_LUAU` | worker | Path to the `luau` binary for `verify_luau`. |
| `NEURO_RUN_TESTS=1` | Linux/macOS worker | Allow `verify_tests` / `cross_tests` (runs model-written code under rlimits; network not blocked). Windows uses a Job Object and needs no switch. |
| `NEURO_RPC_BIN` | any | Path to `ggml-rpc-server` for memory sharing. |
| `NEURO_TRAINING_LOG`, `NEURO_TRAINING_RUN_DIR` | main PC | Training log and run folder (or `training.json`). |
| `NEURO_GPU_LOCK`, `NEURO_AI_DATA` | main PC | GPU lock file and data folder (defaults: `../train_local/gpu.lock`, `../neuro-ai-data`). |

## 9. Security model

- Dashboard and workers answer only loopback and private LAN addresses. VPN adapters, VirtualBox
  host-only networks and link-local addresses get 403. A web page open in a browser on another
  site cannot press Stop (Origin and Host are checked).
- Every route that changes something on a worker needs `Authorization: Bearer <cluster.token>`.
- Before the main PC sends the token to any computer, that computer must prove it already holds
  the same token (HMAC over a fresh nonce and its own address). A stranger on the LAN never receives it.
- Pairing: only the page on the main PC itself can approve; the token is handed out once.
- The token and all worker traffic cross your LAN as plain HTTP. Anyone who can watch the LAN
  can see them. Keep the ports off the internet; use a VPN you control to reach it remotely.
- A worker never shells out for a job; job kinds are a fixed list.
- `verify_tests` and `cross_tests` run model-written code in a sandbox (Job Object on Windows,
  rlimits with `NEURO_RUN_TESTS=1` elsewhere, temp folder, scrubbed env, 10 s). The network is not
  blocked for that code.

## 10. Troubleshooting

| Symptom | Cause and fix |
| --- | --- |
| Worker stays "Jobs paused: token" | It holds a token the main PC did not issue. Restart the worker; it pairs by itself. Approve the card on the main PC. |
| No pairing code appears | Wrong `NEURO_MESH_URL`, or the main PC firewall blocks 8780 on the private network. Open `http://<main PC>:8780/` from the worker's browser. |
| Computer is red on the page | Worker not running, port 8781 blocked (see firewall command), or different networks. |
| `generate` never lights up | No model on `127.0.0.1:8080`, or the model server and the worker share one terminal window. Use two. |
| `jobs add` says "Restart serve first" | `mesh.py serve` is older than the file on disk. Restart it. |
| llama.cpp download URL is empty | You used `releases/latest`. Use the releases list command in section 4. |
| Steam Deck falls asleep | Turn off suspend in Power Management, keep `systemd-inhibit` in the commands. |
| `Access is denied` on a data folder (Windows) | The folder only allows SYSTEM. Run the step as a scheduled task as SYSTEM instead of changing permissions. |
| Page says "Stopped · lock held" | `gpu.lock` exists but no trainer is running. Delete it only if its pid is not a live trainer. |
| Training page says "No training log found" | Create `training.json` (section 8) or set `NEURO_TRAINING_LOG`, then restart `mesh.py serve`. |
| Training state "Unknown" | The log or its folder cannot be read by the user that runs `mesh.py serve` (for example a trainer started as SYSTEM). Write the log somewhere that user can read. |

## 11. What helps and what does not

| Hardware | Verdict |
| --- | --- |
| Another PC or Mac with 16 GB | Good worker: generates (small model) and checks. |
| Steam Deck | Good worker; about doubles generation speed. Not for training. |
| Old laptop without a GPU | Checks only; still useful. |
| Game consoles (PlayStation, Xbox) | Cannot run Linux or CUDA. No use. |
| Nintendo Switch Lite | Needs a soldered mod chip; 4 GB memory. Not worth it. |
| Rented GPU (A100/H100 by the hour) | Best way to train a 27B before you own the hardware. |
| DGX Spark (128 GB) | Trains Qwen3.8-27B in bf16 and serves many agents with vLLM. |

Splitting one training run across slow home links is not worth it; spend the other computers on
making and checking data instead.

## 12. Roadmap

- Publish the coding benchmark (61 tasks with hidden tests) and the trainer.
- Real unit-test execution for generated code in an isolated sandbox.
- A token and TLS option for traffic over untrusted networks.
- vLLM serving and many-agent mode on large-memory machines.

Issues and pull requests are welcome.
