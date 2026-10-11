# Installation & vLLM Setup

[← Back to README](../../README.md)

## Installation

### Prerequisites
- **Python**: 3.10 or higher.
- **OS**: Linux (recommended for robust subprocess isolation and timeout enforcement) or macOS.

### 1. Install from source (editable mode)

```bash
git clone https://github.com/taindp98/myirace.git
cd myirace/EvoKit
pip install -e .
```

### 2. Or install dependencies via `requirements.txt`

```bash
pip install -r requirements.txt
```

### 3. Optional extras (dashboard & dev tools)

```bash
pip install -e ".[dashboard,dev]"
```

### 4. Verify

```bash
evokit --help
```

---

## LLM Backends

EvoKit's `LLMGateway` talks to any OpenAI-compatible chat-completions endpoint. You have two options, selected by the `gateway` block of a config:

1. **Self-hosted vLLM** (`provider: "vllm"`) — run the model on your own GPU. Best for high-throughput, zero per-token cost, and private/offline use.
2. **Cloud API provider** (`provider: "openai"`, OpenRouter, etc.) — point at a hosted endpoint with an API key. Best when you have no GPU.

A third `provider: "mock"` returns deterministic canned responses for zero-cost pipeline tests (`--dry-run`).

Keys and endpoints are read from environment variables via `${VAR}` interpolation in the YAML (the loader expands `${VAR}` and `${VAR:-default}`), so you never commit secrets.

### `.env` file

Copy the template and fill in your values:

```bash
cp .env.example .env     # .env is gitignored — never commit real keys
```

It holds `VLLM_BASE_URL`, `LLM_MODEL`, and cloud keys (`OPENAI_API_KEY`, `OPENROUTER_API_KEY`). The `scripts/run_*_3seeds.sh` helpers auto-load `.env`. For other entrypoints, export it into your shell first:

```bash
set -a; source .env; set +a
```

---

## Option A — Self-hosted vLLM

High-throughput local inference via **[vLLM](https://github.com/vllm-project/vllm)** (PagedAttention + continuous batching lets concurrent mutation requests share the GPU).

### Step 1: Install vLLM on the GPU host (CUDA 12.1+)

```bash
pip install vllm
# or: docker pull vllm/vllm-openai:latest
```

### Step 2: Launch the OpenAI-compatible API server

```bash
python -m vllm.entrypoints.openai.api_server \
    --model mistralai/Devstral-Small-2-24B-Instruct-2512 \
    --host 0.0.0.0 \
    --port 8000 \
    --tensor-parallel-size 1 \
    --max-model-len 8192 \
    --gpu-memory-utilization 0.90 \
    --trust-remote-code
```

> **Key flags:** `--host 0.0.0.0` (allow worker nodes to connect) · `--tensor-parallel-size N` (GPUs to shard across) · `--max-model-len` (context window) · `--gpu-memory-utilization` (VRAM fraction).

Docker alternative:

```bash
docker run --gpus all \
    -v ~/.cache/huggingface:/root/.cache/huggingface \
    -p 8000:8000 --ipc=host \
    vllm/vllm-openai:latest \
    --model mistralai/Devstral-Small-2-24B-Instruct-2512 \
    --max-model-len 8192
```

### Step 3: Verify the endpoint

```bash
curl http://localhost:8000/v1/models

curl http://localhost:8000/v1/chat/completions \
  -H "Content-Type: application/json" \
  -d '{"model": "mistralai/Devstral-Small-2-24B-Instruct-2512",
       "messages": [{"role": "user", "content": "Write a python function returning 42."}],
       "max_tokens": 50}'
```

### Step 4: Point EvoKit at vLLM

In a config YAML:

```yaml
gateway:
  provider: "vllm"
  base_url: "http://localhost:8000/v1"
  model: "mistralai/Devstral-Small-2-24B-Instruct-2512"
  max_parallel_requests: 8
  temperature: 0.8
  max_tokens: 4096
  timeout: 120.0
```

Or override on the CLI:

```bash
evokit run -c evokit/configs/eoh_obp.yaml \
  --vllm-url http://localhost:8000/v1 \
  --model mistralai/Devstral-Small-2-24B-Instruct-2512
```

> **Remote cluster / SSH:** forward the port to your machine:
> ```bash
> ssh -L 8000:localhost:8000 user@gpu-cluster-node
> ```

---

## Option B — Cloud API providers (OpenAI, OpenRouter, …)

Any provider exposing an OpenAI-compatible `/v1/chat/completions` endpoint works: set `base_url`, `model`, and an `api_key`. The gateway sends `Authorization: Bearer <api_key>`.

### Step 1: Export your API key

```bash
export OPENAI_API_KEY="sk-..."          # OpenAI
# or
export OPENROUTER_API_KEY="sk-or-..."   # OpenRouter
```

Put it in a `.env` / your shell profile rather than committing it. EvoKit does not auto-read a specific variable name — you wire the chosen variable into the config with `${...}` (next step), so any name works.

### Step 2: Point the gateway at the provider

**OpenAI:**

```yaml
gateway:
  provider: "openai"
  base_url: "https://api.openai.com/v1"
  model: "gpt-4o-mini"
  api_key: "${OPENAI_API_KEY}"     # expanded from the environment at load time
  temperature: 0.8
  max_tokens: 4096
  max_parallel_requests: 8
  timeout: 120.0
```

**OpenRouter** (gateway to many models — Anthropic, Google, Qwen, etc.):

```yaml
gateway:
  provider: "openai"               # OpenRouter speaks the OpenAI protocol
  base_url: "https://openrouter.ai/api/v1"
  model: "qwen/qwen3-coder"        # any model slug OpenRouter serves
  api_key: "${OPENROUTER_API_KEY}"
  max_parallel_requests: 8
```

Any other OpenAI-compatible service (Together, Groq, Fireworks, a self-hosted gateway, …) follows the same shape — change `base_url`, `model`, and the key variable.

### Step 3: Verify

```bash
evokit run -c evokit/configs/eoh_obp.yaml --dry-run    # pipeline check, no tokens spent
evokit run -c evokit/configs/eoh_obp.yaml -g 1         # 1 generation against the live provider
```

> **Cost & rate limits:** cloud providers bill per token and throttle concurrency. Start with `max_parallel_requests: 1–2` and a small `-g`/`candidates_per_generation`, then scale up. `--dry-run` always costs nothing.
