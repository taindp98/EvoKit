# EvoKit Tutorials

Step-by-step guides for **using and building on** EvoKit. Start here if you want to run
experiments, connect your own LLM, visualize results, or add a new method.

| Guide | What it covers |
|---|---|
| [setup.md](setup.md) | Install EvoKit; run an LLM backend — self-hosted vLLM **or** cloud API (OpenAI / OpenRouter, API keys, `.env`). |
| [methods.md](methods.md) | Run the built-in method × benchmark presets and the faithful reproductions (`examples/`). |
| [cli.md](cli.md) | `evokit run` / `evaluate` / `dashboard`, flags, and the two-tier parallelism model. |
| [telemetry.md](telemetry.md) | Inspect results: Optuna Dashboard, TensorBoard (switchable x-axis), and `.py` code artifacts. |
| [extending.md](extending.md) | Add your own method: implement a population/mutation plugin, register it, wire a config. |
| [plugin_guide.md](plugin_guide.md) | Deep plugin-development reference with code templates and expansion recipes. |
| [spec.md](spec.md) | Architecture specification — the design rationale and component contracts. |

Suggested reading order for a first-time user: **setup → methods → telemetry**, then
**extending → plugin_guide** when you want to contribute a new paradigm.

Development records (milestones, changelog, bug fixes) live in [../dev/](../dev/).
