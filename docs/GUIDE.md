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

**Tell the worker where the main PC is.** Set `NEURO_MESH_URL` to the main PC's LAN address
before starting the worker. The built-in default is the author's home address, so on your network
you must set it:

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
set NEURO_MESH_URL=http://192.168.1.20:8780
python -u mesh.py work
```

If `python` is missing, use `py -3 -u mesh.py work`. Leave the window open. Once, as Administrator:

```bat
netsh advfirewall firewall add rule name="Neuro worker" dir=in action=allow protocol=TCP localport=8781 profile=private
```

### Mac

```bash
git clone https://github.com/ori-liloss/neuroai-mesh.git
cd neuroai-mesh
export NEURO_MESH_URL=http://192.168.1.20:8780
bash platforms/start-mac.sh
```

macOS has no `python`; the script uses `python3` (`brew install python` if missing). Do not open
127.0.0.1 on the Mac — that is the Mac itself. Do not run `mesh.py serve` there.

### Linux

```bash
git clone https://github.com/ori-liloss/neuroai-mesh.git
cd neuroai-mesh
export NEURO_MESH_URL=http://192.168.1.20:8780
sh platforms/start-linux.sh
```

### Steam Deck (a real Linux computer with a 16 GB GPU)

A Steam Deck is a good extra worker: it can run a small model and double the speed of data
generation. It is too weak for training.

**One command (recommended).** Hold the power button, choose **Switch to Desktop**, plug it in,
open **Konsole** (start menu) and paste (Ctrl+Shift+V):

```bash
git clone https://github.com/ori-liloss/neuroai-mesh.git ~/neuro-cluster && sh ~/neuro-cluster/platforms/steamdeck.sh
```

It downloads llama.cpp (Vulkan) and a 2.9 GB model once, starts the model on `127.0.0.1:8080`,
starts the worker, and keeps the Deck awake. A pairing code appears after about 30 seconds;
approve it on the main PC (section 3). Ctrl+C stops everything. Next time, and for updates:

```bash
cd ~/neuro-cluster && git pull && sh platforms/steamdeck.sh
```

Another main PC address: `NEURO_MESH_URL=http://192.168.1.20:8780 sh platforms/steamdeck.sh`.

**Step by step (what the script does):**

1. Hold the power button, choose **Switch to Desktop**. Plug it in. In *System Settings → Power
   Management* set suspend and screen saving to *Never*. Open **Konsole** (start menu).
2. Download the project:

```bash
git clone https://github.com/ori-liloss/neuroai-mesh.git ~/neuro-cluster
```

3. Download llama.cpp built for the Deck's GPU (Vulkan). Use the releases *list*, not
   `releases/latest` — for llama.cpp `latest` points to a release without binaries:

```bash
mkdir -p ~/llama && cd ~/llama && URL=$(curl -s "https://api.github.com/repos/ggml-org/llama.cpp/releases?per_page=1" | grep -oE 'https://[^"]*ubuntu-vulkan-x64\.tar\.gz' | head -1) && echo "$URL" && curl -L -o pkg.tar.gz "$URL" && tar xzf pkg.tar.gz && rm pkg.tar.gz && echo READY
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
cd ~/neuro-cluster && export NEURO_MESH_URL=http://192.168.1.20:8780 && systemd-inhibit --what=sleep:idle sh platforms/start-linux.sh
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

The main PC keeps a queue in `jobs/` (gitignored). Idle workers pull one job at a time.

| Job kind | Checks | Notes |
| --- | --- | --- |
| `generate` | asks the worker's own model | thinking off, up to 2048 tokens, never leaves the worker |
| `verify_python_syntax` | `compile()` | code is never run |
| `verify_js` | `node --check` | parse only; needs Node on the worker |
| `verify_ts` | Node's built-in TypeScript stripper + `node --check` | Node 22.13+ / 23.2+; enums are "inconclusive" |
| `verify_html_css` | tag nesting, unclosed script/style, CSS braces | stdlib only |
| `verify_sql` | SQLite in memory, whitelisted statements, `EXPLAIN` only | ATTACH/PRAGMA blocked |
| `verify_luau` | runs `luau` with limits | needs `luau` on PATH or `NEURO_LUAU` |
| `verify_math` | final answer vs expected | |
| `dedup_hash` | normalized hash + MinHash | |

Nothing here runs generated code except `verify_luau` (limited), and no job is ever a shell command.

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

### Your own prompts

A row of `prompts.jsonl`:

```json
{"prompt": "Write a JavaScript function that reverses words in a string. Answer in one ```javascript block.",
 "verify": {"kind": "verify_js"}, "samples": 4, "max_tokens": 1024}
```

Math: `{"prompt": "What is 6*7?", "verify": {"kind": "verify_math", "expected": "42"}}`.
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

The full trainer and the coding benchmark live in the author's own tree and are not in this
repository yet. What this repository gives you is the data and a small experiment script.

**Data format** (`jobs export` output): one JSON object per line,
`{"messages": [{"role": "user", "content": "..."}, {"role": "assistant", "content": "..."}], ...}`.
Hold out a slice for evaluation and keep your benchmark prompts out of the training data.

**Quick experiments on a side GPU:** `side_train.py` trains a LoRA on a small local model and
reports eval loss, so you can test a learning rate or a data mix before spending days on the big run:

```bash
python side_train.py --base /path/to/small-model --data coding-v1-verified.jsonl --out runs/lr1e-4 --learning-rate 1e-4 --max-steps 300
```

It refuses to start while a main training lock is held.

**What the author uses for the real run** (a good starting point for yours):

| | RTX 4070 12 GB | 128 GB unified memory (DGX Spark) |
| --- | --- | --- |
| Base | Qwen3.5-9B | Qwen3.8-27B |
| Method | QLoRA 4-bit | LoRA bf16 |
| Memory | about 9–10 GB | about 66 GB |
| Learning rate | 1e-4, cosine, 3% warmup | same |
| LoRA | r=16, alpha=32, all linear layers, not the vision tower | same |
| Packing | **off** for Qwen3.5/3.8 (hybrid linear attention would mix conversations) | off |
| Loss | assistant tokens only | same |

Tips that cost the author days: a learning rate of 5e-6 is far too low for LoRA (use 1e-4 to
2e-4); save every few hundred steps, not once per epoch; check that no benchmark question is in
the training data; a LoRA adapter only works on the exact base model it was trained on.

**Order of work that pays off most:** clean verified data first → one training run → measure on a
fixed benchmark (pass@1 and tokens per task) → a second round on verified answers with DPO/KTO.

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

- Auto-discovery of the main PC, so `NEURO_MESH_URL` is not needed.
- Publish the coding benchmark (61 tasks with hidden tests) and the trainer.
- Real unit-test execution for generated code in an isolated sandbox.
- A token and TLS option for traffic over untrusted networks.
- vLLM serving and many-agent mode on large-memory machines.

Issues and pull requests are welcome.
