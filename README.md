# Neuro

A local console for one Windows GPU machine plus Mac and Linux workers. The dashboard stays on the Windows machine. Chat goes to a llama.cpp server you start yourself. This repository does not contain model weights.

People use this the way they use other local AI screens: one page, the machines in front of them, one command to join, and a chat box.

## What you need

- Python 3 on every machine. No pip packages.
- A llama-server already running on the Windows machine at `http://127.0.0.1:8080/v1`, pointed at your own GGUF.
- The machines on the same private network.

## 1. Start the host

On the Windows machine, in this folder:

```powershell
python -u mesh.py serve
```

Or double-click `start-dashboard.ps1`.

Open [http://127.0.0.1:8780](http://127.0.0.1:8780).

The first launch creates `cluster.token` next to `mesh.py`. That file is the shared password for workers. It is gitignored. Do not commit it and do not send it to the public internet.

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

The next time, only `bash platforms/start-mac.sh`. The script updates itself and starts the worker. The Windows page finds the machine. On the Mac, do not open 127.0.0.1. That address is the Mac itself. Open the Windows address shown on the Share page, which looks like http://192.168.7.14:8780. There is no file to copy and no address to paste into the worker.

`mesh.py serve` belongs on Windows. macOS has no `python` command. The script uses `python3`. If Python is missing: `brew install python`, then run the script again.

If an older worker is still running in that terminal, press Ctrl+C, then run the script.

## 3. Add a Linux machine

```bash
git clone https://github.com/ori-liloss/neuroai-mesh.git
cd neuroai-mesh
bash platforms/start-linux.sh
```

The Windows page at http://127.0.0.1:8780 finds the machine. Nothing to paste.

## 4. If the bay stays red

On the Windows machine, in an Administrator PowerShell, allow the worker port on the private network only:

```powershell
netsh advfirewall firewall add rule name="Neuro worker" dir=in action=allow protocol=TCP localport=8781 profile=private
```

Do not open 8781, or the llama.cpp RPC port 50052, to the public internet. RPC has no login.

## What a joined computer does

It reports its memory. If `ggml-rpc-server` is installed on the Mac, the script listens on the Mac's LAN address at port 50052 and offers that RAM. The model already loaded on the Windows GPU does not move. The Mac joins the next time that server starts. Training stays on the CUDA machine. One training run is not split across the two computers. The page does not run shell commands on the Mac.

`bash platforms/build-mac-rpc.sh` builds that listener from llama.cpp commit `c0ab6e5c0`, the same commit as the Windows build in this project. The two builds have to match.

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
| `start-dashboard.ps1` | Windows dashboard |
| `start-worker.ps1` | Windows worker, for a second Windows PC |


I will keep updating this Project Hope it will help you !
