# NeuroAI Mesh 1.0

English console for a local NeuroAI cluster. One Windows machine with an NVIDIA GPU holds the dashboard and the chat. Mac and Linux machines join as workers on the local network.

Weights are not in this repository. NeuroAI 1.0 is the name of this build. The chat process on the host is a local llama.cpp server pointed at your own GGUF.

## What a second computer does

It reports memory and can later hold part of one inference run through llama.cpp RPC. Training stays on the CUDA machine. This project does not run shell commands on the other computer, and it does not train one adapter across Metal and CUDA.

## Host

Requires Python 3. No extra packages.

```powershell
python -u mesh.py serve
```

Open http://127.0.0.1:8780

The first launch writes `cluster.token` next to `mesh.py`. Keep that file private. It is gitignored.

Chat is proxied to a llama-server already listening on http://127.0.0.1:8080/v1 . Start that server yourself and point it at your GGUF. The page calls the model NeuroAI.

## Mac

Copy this folder, including `cluster.token`, onto the Mac. From the folder:

```bash
bash platforms/start-mac.sh
```

The script prints `http://THIS-MAC-IP:8781`. Paste that address into the Mac bay, with the machine spec.

## Linux

```bash
bash platforms/start-linux.sh
```

Paste the printed `http://THIS-LINUX-IP:8781` into the Linux bay.

## Firewall

The dashboard binds to 127.0.0.1. The worker listens on TCP 8781. If the bay stays red while the worker is running, allow 8781 on the private network only. Do not publish 8781 or the llama.cpp RPC port 50052 to the internet. RPC has no authentication.

## Layout

- `mesh.py` — dashboard and worker
- `dashboard.html` — NeuroAI Mesh console
- `platforms/start-mac.sh`, `platforms/start-linux.sh` — worker launchers
- `start-dashboard.ps1`, `start-worker.ps1` — Windows launchers
