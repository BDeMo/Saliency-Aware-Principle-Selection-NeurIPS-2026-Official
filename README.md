# Saliency-Aware Principle Selection (SAP)

Official implementation of **Saliency-Aware Principle Selection (SAP)**, accepted at **NeurIPS 2026**.

SAP is a training-free inference-time scaling method for vision-language reasoning. Instead of extending one increasingly text-dominated chain, SAP evolves a population of short reasoning principles, generates independent routes from the original image and prompt, and uses visual grounding as advisory evidence for selecting the next generation.

## Highlights

- **Training-free:** no parameter updates or task-specific finetuning.
- **Route-parallel:** scales over independent reasoning routes rather than a longer serial chain.
- **Visually grounded:** a frozen grounding stack evaluates whether cited objects are supported by the image.
- **Tool-advisory:** grounding affects selection but never writes the final answer.
- **OpenAI-compatible:** the clean wrapper exposes SAP through a chat-completions endpoint for evaluation tools.

## Repository layout

```text
.
├── launcher.py              # Config-driven service launcher
├── cfg/
│   ├── launcher.yaml        # Model, grounding, SAP, and wrapper services
│   └── benchmark/           # Evaluation configurations
├── run/
│   ├── sap.py               # Core SAP evolution and fitness logic
│   ├── template.py          # Principle, route, and aggregation prompts
│   ├── sam_amg.py           # Frozen mask-generation service
│   ├── proxy_qwen_sam.py    # Object labeling and grounding proxy
│   ├── sap_clean.py         # OpenAI-compatible SAP wrapper
│   └── run_eval.py          # EvalScope/VLMEvalKit entry point
├── scripts/                 # Reproducible launch commands
└── website/                 # Project page and camera-ready PDF
```

## Installation

Python 3.10+ and CUDA-capable GPUs are recommended.

```bash
git clone https://github.com/BDeMo/Saliency-Aware-Principle-Selection-NeurIPS-2026-Official.git
cd Saliency-Aware-Principle-Selection-NeurIPS-2026-Official
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

Place the model checkpoints referenced by `cfg/launcher.yaml` under `checkpoints/`, or update each `model_path` and `MODEL_ID` for your environment.

## Run SAP

Start the complete stack:

```bash
python launcher.py --cfg cfg/launcher.yaml --profile full_sap_stack
```

This launches the VLM backend, frozen grounding services, SAP core, and the OpenAI-compatible wrapper. The default endpoints are:

| Service | Endpoint |
|---|---|
| VLM backend | `http://127.0.0.1:8001` |
| SAM mask generator | `http://127.0.0.1:8010` |
| Grounding proxy | `http://127.0.0.1:8020` |
| SAP core | `http://127.0.0.1:8030` |
| SAP clean wrapper | `http://127.0.0.1:8040/v1/chat/completions` |

Use `--dry` to inspect all resolved commands without starting services:

```bash
python launcher.py --cfg cfg/launcher.yaml --profile full_sap_stack --dry
```

## Evaluation

The included configs use EvalScope with the VLMEvalKit backend:

```bash
python run/run_eval.py cfg/benchmark/SAP-Qwen-VL-8B-Instruct-ALL.yaml
```

Edit the `data` list in the benchmark YAML to select datasets. Evaluation outputs are written to `outputs/` and are intentionally excluded from version control.

## Key SAP controls

The main settings live under `services.proxy_sap.env` in `cfg/launcher.yaml`:

- `MU_PLUS_LAMBDA`, `MU`, `LAMBDA`: population, elite, and newborn counts.
- `EVO_GENS`: number of evolutionary generations.
- `TAU`: number of independent routes per principle.
- `CONS_W`, `DIV_W`, `UNC_W`, `EVID_W`: route-level fitness weights.
- `MAX_MASKS`: maximum grounded objects retained per image.

## Citation

Please cite the NeurIPS 2026 paper if this code is useful in your research. Citation metadata is provided in [`CITATION.cff`](CITATION.cff).

## Acknowledgements

This work was supported in part by the National Science Foundation under Grants IIS-2144209, IIS-2223769, CMMI-2411248, and IIBR-2601942; the Office of Naval Research under Grant N000142412636; grant support from Meta; and the NVIDIA Academic Grant Program using DGX Spark.

