# seg.bio MVP

An agentic, natural‑language interface for end‑to‑end biomedical EM segmentation. Researchers describe tasks (“train Lucchi++ for 50 epochs and run inference on slices 120–180”), and the system orchestrates training, inference, SLURM jobs, and QC loops with minimal manual steps. This repo contains:
- FastAPI supervisor + multi‑agent chat backend (Ollama + LangChain/LangGraph)
- PyTC worker service for training/inference jobs
- React frontend with the chat assistant docked alongside visualization/training/inference flows

## Quick Start (host)

Requirements: Python 3.10+, Node 18+, git, pyenv (optional but recommended), Ollama running with your target models.

```bash
# 1) Backend deps (from repo root)
eval "$(pyenv init -)" && pyenv shell pytc-venv   # or activate your env
pip install -r server_api/requirements.txt

# 2) Frontend deps
cd client && npm install && cd ..

# 3) Run backend (keeps chat + API at :4242)
PYTHONPATH=$(pwd) uvicorn server_api.main:app --host 0.0.0.0 --port 4242 --log-level info

# 4) Run PyTC worker (separate shell, for training/inference endpoints)
PYTHONPATH=$(pwd) uvicorn server_pytc.main:app --host 0.0.0.0 --port 4243 --log-level info

# 5) Run frontend dev server
cd client && npm start
```

Navigate to http://localhost:3000 (or your Nginx domain). The chat panel uses `/api/chat/query` routed to port 4242.

### Production build + deploy (static frontend)

```bash
cd client
npm run build
sudo rsync -a --delete build/ /var/www/view.seg.bio/   # adjust target as needed
```

### Minimal run commands (runnable scripts)
- `PYTHONPATH=$(pwd) uvicorn server_api.main:app --host 0.0.0.0 --port 4242`
- `PYTHONPATH=$(pwd) uvicorn server_pytc.main:app --host 0.0.0.0 --port 4243`
- `cd client && npm start` (dev) or `npm run build` + rsync to your web root

## Features
- **Natural-language agent** coordinating training/inference/QC with SLURM job submission.
- **Checkpoint and config discovery** for PyTorch Connectomics workflows.
- **LLM-powered fallback** if an agent tool fails, so chat stays responsive.
- **Frontend assistant** docked to the right with quick prompts and session persistence.

## Repo structure
- `server_api/` — FastAPI supervisor + chat agents (LangChain/LangGraph/Ollama)
- `server_pytc/` — Worker API for training/inference
- `client/` — React frontend
- `pytorch_connectomics/` — submodule/vendorized connectomics code

## Creating & pushing the GitHub repo
If starting fresh:
```bash
git init
git add .
git commit -m "Initial seg.bio MVP"
git remote add origin git@github.com:<your-username>/seg.bio-mvp.git
git push -u origin main
```
If `origin` exists, use `git remote set-url origin git@github.com:<your-username>/seg.bio-mvp.git` then `git push`.

## Notes
- Ensure Ollama is running and accessible at the configured `OLLAMA_BASE_URL`.
- Keep the backend process alive (tmux/systemd or Docker with `--restart unless-stopped`) so Nginx proxying `/api/` succeeds.

## Contributions

Hanson Pan — Ingestion UI, Neuroglancer integration, and end-to-end pipeline wiring across the FastAPI supervisor, PyTorch Connectomics worker, and SLURM submission layer. Implemented major backend–frontend glue code and overall system orchestration.

Tian-Hao Zhao — Agent supervisor design and implementation (LangChain/LangGraph), training/inference sub-agents, SLURM execution tooling, and monitoring flows within server_api/ and server_pytc/.

Ethan Shen — Error-detection and proofreading workflow, dataset triage logic, and improvements to stability of request/response handling.

Team — Joint architectural design, documentation, experiment design, writing, and editing across the MVP.
