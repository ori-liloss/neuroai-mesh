# NeuroAI Mesh 1.0

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

Copy this whole folder onto the Mac, including `cluster.token` beside `mesh.py`.

On the Mac, inside the folder:

```bash
bash platforms/start-mac.sh
```

The script prints an address like `http://192.168.1.20:8781`. On the Windows page, open the Mac bay and paste that address. Fill in the Mac's name, CPU, RAM, and GPU, then choose **Save bay**.

The bay turns **Live** only when the Mac answers. **Waiting** means you have not saved an address. **No reply** means the script is not running, or Windows is blocking TCP 8781.

## 3. Add a Linux machine

Same folder and the same `cluster.token`.

```bash
bash platforms/start-linux.sh
```

Paste the printed `http://IP:8781` into the Linux bay.

## 4. If the bay stays red

On the Windows machine, in an Administrator PowerShell, allow the worker port on the private network only:

```powershell
netsh advfirewall firewall add rule name="NeuroAI Mesh worker" dir=in action=allow protocol=TCP localport=8781 profile=private
```

Do not open 8781, or the llama.cpp RPC port 50052, to the public internet. RPC has no login.

## What a joined computer does

It reports its memory and can later hold part of one inference run. Training stays on the CUDA machine. The page does not run shell commands on the Mac or the Linux machine, and it does not train one adapter across Metal and CUDA.

Weight split starts only after the chat server stops, because one GPU cannot serve chat and lend its memory at the same time.

## Files

| File | Role |
| --- | --- |
| `mesh.py` | Dashboard on port 8780, worker on port 8781 |
| `dashboard.html` | The console |
| `platforms/start-mac.sh` | Mac worker |
| `platforms/start-linux.sh` | Linux worker |
| `start-dashboard.ps1` | Windows dashboard |
| `start-worker.ps1` | Windows worker, for a second Windows PC |
