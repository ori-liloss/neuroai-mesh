# Neuro

A local console for one Windows GPU machine plus Mac and Linux workers. The dashboard stays on the Windows machine. Chat goes to a llama.cpp server you start yourself. This repository does not contain model weights.

People use this the way they use other local AI screens: one page, the machines in front of them, one command to join, and a chat box.

**New here? Read the step-by-step [guide](docs/GUIDE.md)**: setup, pairing a computer, Steam Deck, the data factory, training and troubleshooting.

## What you need

- Python 3 on every machine. No pip packages.
- A llama-server already running on the Windows machine at `http://127.0.0.1:8080/v1`, pointed at your own GGUF.
- The machines on the same private network.

## 1. Start the host

On the Windows machine, in this folder:

```powershell
python -u mesh.py serve
```

Or double-click `start-dashboard.cmd`. That starts the page without a PowerShell script.

Open [http://127.0.0.1:8780](http://127.0.0.1:8780).

The first launch of `mesh.py serve` creates `cluster.token` next to `mesh.py`. That file is the shared password for workers; they get it by pairing (see below). It is gitignored. Do not commit it and do not send it to the public internet.

You should see:

- The Windows bay is **Live**.
- Temperature, VRAM, and RAM are numbers from this PC, not placeholders.
- Chat accepts a message. The GPU is busy only while the reply is generating. An idle model still occupies memory.

## 2. Add a Mac

On the Mac, once:

```bash
git clone https://github.com/ori-liloss/neuroai-mesh.git
cd neuroai-mesh
bash platforms/start-mac.sh
```

The next time, `git pull` when there is a new version, then `bash platforms/start-mac.sh`. The script does not update the code by itself. The Windows page finds the machine. On the Mac, do not open 127.0.0.1. That address is the Mac itself. Open the Windows address shown on the Share page, which looks like http://192.168.7.14:8780. There is no file to copy and no address to paste into the worker.

`mesh.py serve` belongs on Windows. macOS has no `python` command. The script uses `python3`. If Python is missing: `brew install python`, then run the script again.

If an older worker is still running in that terminal, press Ctrl+C, then run the script.

## 3. Add a Linux machine

```bash
git clone https://github.com/ori-liloss/neuroai-mesh.git
cd neuroai-mesh
bash platforms/start-linux.sh
```

The Windows page finds the machine. Nothing to paste.

## 4. Add a Windows laptop

On that laptop, clone this repository into any folder that belongs to the person using it. Then open that folder and run:

```bat
python -u mesh.py work
```

Leave that window open. If `python` is missing, run `py -3 -u mesh.py work` in the same folder.

Once, in a second window opened as Administrator:

```bat
netsh advfirewall firewall add rule name="Neuro worker" dir=in action=allow protocol=TCP localport=8781 profile=private
```

Laptop with an NVIDIA GPU (it also runs a small model and generates): one command does everything, `git clone https://github.com/ori-liloss/neuroai-mesh.git %USERPROFILE%\neuro-cluster` then `%USERPROFILE%\neuro-cluster\platforms\windows-worker.cmd` (details in [docs/GUIDE.md](docs/GUIDE.md) section 4).

The Windows page adds the laptop by the computer name Windows already uses. Nothing is typed into the page. The laptop stays off the page until this command is running. A second laptop uses the same command in its own copy of the folder. Its own computer name is added beside the first. There is no empty seat waiting for a computer that is not here. The weight update stays on the NVIDIA PC.

## 5. If the bay stays red

On the Windows machine, in an Administrator PowerShell, allow the worker port on the private network only:

```powershell
netsh advfirewall firewall add rule name="Neuro worker" dir=in action=allow protocol=TCP localport=8781 profile=private
```

Do not open 8781, or the llama.cpp RPC port 50052, to the public internet. RPC has no login.

The dashboard and the worker answer only loopback and private home-LAN addresses. Requests from a VPN adapter (26.x), the VirtualBox host-only net (192.168.56.x) or link-local addresses get 403. The LAN sweep never scans those ranges. A POST from another web site open in a browser is refused, so a page elsewhere cannot press Stop.

## What a joined computer does

It reports its memory and checks the agent code on its own processor. The weight update stays on the NVIDIA PC. If `ggml-rpc-server` is installed on the Mac, the script listens on the Mac's LAN address at port 50052 and offers that RAM. The model already loaded on the Windows GPU does not move. The Mac joins the next time that server starts. One training run is not split across the computers. The page does not run shell commands on the Mac.

`bash platforms/build-mac-rpc.sh` builds that listener from llama.cpp commit `c0ab6e5c0`, the same commit as the Windows build in this project. The two builds have to match.

## Side work for training: the data factory

One training run stays on the NVIDIA PC. A 7B model cannot be trained across the Mac and the laptops over Wi-Fi, and CUDA and Metal cannot share one run. The other computers instead make and check training data. The main PC keeps a small job queue in `jobs/` (gitignored, never inside the training data folder). Each worker pulls one job at a time when it is idle, runs it with a built-in handler, and sends the result back. A job is data for one of these handlers, never a command:

| Kind | What it does | Runs on |
| --- | --- | --- |
| `generate` | Asks the worker's own local model at `http://127.0.0.1:8080/v1` (or the port in `NEURO_LOCAL_LLM`). Thinking off, at most 2048 tokens. Never sent to another host. | A computer with a local llama-server |
| `verify_luau` | Runs the code with `luau` (on PATH or `NEURO_LUAU`), 10 s limit, 1 MB output limit. | A computer with luau |
| `verify_python_syntax` | `compile()` only. The code is never run. | Every worker |
| `verify_math` | Compares the final or `\boxed{}` answer with the expected one. | Every worker |
| `dedup_hash` | Normalized sha256 plus a 32-value MinHash for each text. | Every worker |

On the main PC:

```powershell
python mesh.py jobs add generate-verify prompts.jsonl
python mesh.py jobs status
python mesh.py jobs export verified.jsonl
```

A `prompts.jsonl` row looks like `{"prompt": "What is 6*7?", "verify": {"kind": "verify_math", "expected": "42"}, "max_tokens": 512}`. When a generated reply comes back, the matching verify job is queued by itself. `export` writes only replies that passed into `jobs/verified.jsonl`, as chat rows for the next training run. Other kinds: `python mesh.py jobs add verify_math rows.jsonl` with rows like `{"answer": "...", "expected": "42"}`, `dedup_hash` with `{"texts": ["...", "..."]}`.

A ready coding set is in `seeds/coding-v1.jsonl`: 321 prompts (JavaScript, TypeScript, HTML/CSS, SQLite, Python, Luau; about 10% in Hebrew), each with its code verifier and `"samples": 4`.
Queue it on the main PC with `python mesh.py jobs add generate-verify seeds/coding-v1.jsonl` (1284 generate jobs); workers with a local model pick them up once they hold `cluster.token`.
Check progress with `python mesh.py jobs status`, then `python mesh.py jobs export coding-v1-verified.jsonl` keeps one verified answer per prompt.

The **Jobs** page shows queued, leased, done and failed counts per kind and per computer, and the Machines page shows which kinds each computer can run.

### Each worker, once

1. If the main PC is not `192.168.7.14`, set `NEURO_MESH_URL=http://<main PC>:8780` first. `NEURO_JOBS=0` turns jobs off.
2. `git pull`, then start `mesh.py work` (Mac: `bash platforms/start-mac.sh`). A worker with no `cluster.token`, or with one the main PC rejects, starts pairing by itself and prints `Pairing code: 482 913`.
3. On the main PC, open its own Neuro page (http://127.0.0.1:8780). A **New computer wants to join** card shows the name, address and code. Press **Approve** only if the code matches the one on that computer's screen. The worker saves the token as `cluster.token` (an old one is kept as `cluster.token.old`) and its jobs resume without a restart. An unapproved code expires after 15 minutes; the worker then asks again with a new code.

Fallback: copy `cluster.token` from the main PC into the worker's folder by hand (USB stick or a private copy), then restart the worker.

How the token is used: the job endpoints need the main PC's token, and the worker's control routes (Stop, RPC start/stop, agent check) need it too. `mesh.py serve` adds it for the page, only after that computer answers a fresh challenge (`/hello/proof`) proving it already holds the same token. A worker never makes up its own token. Approval works only from the page on the main PC itself, and the token is handed out once, to the computer that asked, and only with its pairing secret. Like all Bearer traffic in this project it travels over plain HTTP on the home LAN, so anyone who can watch that LAN can see it; keep these ports off the internet.

A worker takes no job while its CPU is above 75 %, it has under 1 GB free memory, or the GPU lock is held.

### A local model so `generate` lights up

- **RTX 3050 laptop (LAPTOP-M906OEFU):** llama.cpp `llama-server` with a small GGUF, e.g. a Qwen3.5 2B or 4B Q4: `llama-server -m model.gguf --host 127.0.0.1 --port 8080 -ngl 99 -c 8192`.
- **Mac mini:** the same llama.cpp built with Metal (`cmake -B build && cmake --build build -j`), `./build/bin/llama-server -m model.gguf --host 127.0.0.1 --port 8080 -ngl 99`.
- **Old laptop (OhProggraming):** no model. It runs `verify_*` and `dedup_hash` only.

Keep the model server on `127.0.0.1`. The worker checks `/v1/models` once a minute and only then offers `generate`.

## The page

Open http://127.0.0.1:8780.

- **Machines** shows every joined computer, what it is doing, and a Stop button on each heavy program.
- **Share** is the split view. It says which computer is training or serving chat, and which ones only joined. Asking them to listen does not move a model that is already loaded.
- **Models** lists GGUF files on this PC. Adding one does not load it while the GPU is full.
- **Try** is the chat with the loaded model.
- **Training** shows the live step, the recent losses, and the plan name. The button does not start a second run while one is already on the GPU.

A Linux computer is drawn as a penguin. A Mac mini stays a small Mac. A laptop stays a laptop. A Windows desktop stays a desktop.

## Change the training plan

The page can show the plans that are already on disk. It does not rewrite them. To train a different plan, stop the run that is using the GPU, then start the script with your own folder:

```powershell
powershell -File scripts\run-neuro-ai-training.ps1 -RunId my-plan -Data D:\path\to\your-mix -Epochs 3 -LearningRate 5e-6
```

`your-mix` needs `manifest.json`, `train.jsonl`, and `validation.jsonl`. The script checks those files and refuses to start if they do not match the saved run. One training run stays on the NVIDIA PC. It is not split across a Mac.

Neuro Code can write files under `workspace/`. A request to make a folder and a game is handled there. Every heavy program has a Stop process button on this PC and on a joined computer.

A Mac still joins with `bash platforms/start-mac.sh` even if its GPU name is missing. The bay stays live. The chip name appears after that script is started again.

## Agents, in CMD only

```powershell
neuro
```

`neuro` opens Neuro Code from the project folder, or from any folder after the command is on PATH. It uses one local model. The worker can list, search, read, write, run a command in this repo, and read its own saved session. Every finished task is appended on this computer under `agent-logs/`, which git does not upload. Closing the window leaves the workers running.

A hundred copies of the model do not fit on one 12 GB GPU. The queue can hold many tasks. Four of them generate at once.

## Files

| File | Role |
| --- | --- |
| `mesh.py` | Dashboard on port 8780, worker on port 8781 |
| `dashboard.html` | The console |
| `platforms/start-mac.sh` | Mac worker |
| `platforms/start-linux.sh` | Linux worker |
| `start-dashboard.cmd` | Windows dashboard, no PowerShell script |
| `start-worker.ps1` | Windows worker, for a second Windows PC |


I will keep updating this Project Hope it will help you !
