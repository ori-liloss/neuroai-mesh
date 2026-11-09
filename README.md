# Neuro

A local dashboard for a small home AI cluster: one main PC with an NVIDIA GPU that trains and serves a model, plus any Windows, Mac, Linux or Steam Deck computers on the same private network. The other computers do not join the training run. They make and check training data instead: generate answers with their own small local model, check code and math, hash text for dedup.

Plain Python 3.10+, no pip packages, no model weights in this repository. Everything stays on your LAN.

**Step-by-step setup, pairing, the data factory, training and troubleshooting: [docs/GUIDE.md](docs/GUIDE.md).**

## Quick start

On the main PC:

```bash
git clone https://github.com/ori-liloss/neuroai-mesh.git
cd neuroai-mesh
python mesh.py serve            # Windows: python -u mesh.py serve, or start-dashboard.cmd
```

Open <http://127.0.0.1:8780>. The first start creates `cluster.token` (gitignored; never share it).

On each other computer:

```bash
git clone https://github.com/ori-liloss/neuroai-mesh.git
cd neuroai-mesh
python3 mesh.py work            # or: bash platforms/start-mac.sh, sh platforms/start-linux.sh,
                                # sh platforms/steamdeck.sh, platforms\windows-worker.cmd
```

The worker finds the main PC on the local network by itself and remembers it in `mesh_url.txt`. If it cannot, set `NEURO_MESH_URL=http://<main PC>:8780`. A new worker shows a pairing code; approve the same code on the main PC's page.

Chat and generate jobs use a llama.cpp `llama-server` you run yourself on `127.0.0.1:8080`. The Steam Deck, Mac and Windows-laptop scripts in `platforms/` can download and start one for you.

## The page

| Tab | Shows |
| --- | --- |
| Machines | Every computer, what it is doing (training, generating, checking, idle, paused and why), memory, and its heavy programs grouped by name. Stop appears only on ordinary apps, never on the trainer, the worker or system/desktop processes. |
| Jobs | Queue speed, time left, per-computer rates, pass rate per check, truncated replies. |
| Training | Step, speed, ETA, loss and eval-loss sparklines, run folder and state (loading, running, finished, stopped, error) read from your trainer's log. Point it at the log with `training.json` (copy `training.example.json`) or `NEURO_TRAINING_LOG`. |
| Models | GGUF files on the main PC. |
| Try | Chat with the model on the main PC. |
| Share | Agent checks on joined computers, optional llama.cpp RPC memory sharing. |

## Making training data

```bash
python mesh.py jobs add generate-verify seeds/coding-v1.jsonl   # 321 coding prompts x 4 samples
python mesh.py jobs status
python mesh.py jobs export coding-v1-verified.jsonl            # only answers that passed a check
```

A job is data for a built-in handler (generate, verify_js/ts/python/sql/html_css/luau/math, dedup_hash), never a shell command.

## Security, in short

Both ports answer only loopback and private LAN addresses; Origin and Host are checked; every worker control route needs the cluster token, which the main PC only sends to a computer that proves it already holds it. Traffic is plain HTTP on your LAN: keep 8780, 8781 and 50052 off the internet. Details in the guide, section 9.

## Files

| File | Role |
| --- | --- |
| `mesh.py` | `serve` (page and job queue, port 8780), `work` (worker, port 8781), `jobs` (queue CLI) |
| `dashboard.html` | The page, one file, no external scripts |
| `agents.py` | Local coding agents (`neuro` in CMD) |
| `side_train.py` | Small LoRA experiments on a side GPU |
| `platforms/` | One-command workers for Mac, Linux, Steam Deck and Windows laptops |
| `seeds/coding-v1.jsonl` | A ready set of coding prompts with verifiers |

MIT license.
