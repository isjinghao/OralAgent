<div align="center">

<img src="assets/logo_OralAgent.png" alt="OralAgent logo" width="200"/>

# OralAgent

**A multimodal reasoning agent for dental image analysis** — tool use, RAG over oral corpora, and a Gradio UI or OpenAI-compatible HTTP API.

[![111](https://img.shields.io/badge/HuggingFace-Models-ffbf00?style=flat-square&logo=huggingface&logoColor=white)](https://huggingface.co/OralGPT/OralAgent-ToolHub)
[![Model weights](https://img.shields.io/badge/HuggingFace-Models-ffbf00?style=flat-square&logo=huggingface&logoColor=white)](YOUR_HF_MODEL_URL)
[![Benchmark / data](https://img.shields.io/badge/HuggingFace-Benchmark-ffbf00?style=flat-square&logo=huggingface&logoColor=white)](YOUR_HF_BENCHMARK_OR_DATASET_URL)
[![License](https://img.shields.io/badge/License-See%20LICENSE-blue?style=flat-square)](LICENSE)

[English](#oralagent) · [Custom links](#custom-links-table) · [快速配置（中文）](#quick-config-zh)

</div>

---

## OralAgent

OralAgent orchestrates **vision–language reasoning** with **modality-aware routing** and a library of **oral/dental expert tools** (panoramic X-ray, periapical, cephalometric, intraoral imaging, cytopathology, histopathology, etc.), optional **OralGPT-Omni** integration, and **RAG** backed by an oral corpus. It is built with **LangGraph / LangChain** and ships with:

- **Gradio chat UI** (`main.py`) — upload images or DICOM, multi-turn chat.
- **FastAPI service** (`launch_OralAgent.py`) — OpenAI-style `POST /v1/chat/completions` for integration.
- **Multi-GPU workers** — `gunicorn` + `uvicorn` via `run_launch_OralAgent_multi_workers.sh` and `gunicorn_conf.py`.

> **Logo:** place your image at `assets/logo_OralAgent.png` (or change the path in the banner above).

---

<a id="custom-links-table"></a>

## Links & badges (fill these for sharing)

Replace the placeholder URLs in the badge block at the top of this file, and update the table below so visitors can find models, benchmarks, and demos in one place.

| Resource | URL (replace `YOUR_*`) |
|----------|-------------------------|
| GitHub repository | `YOUR_GITHUB_REPO_URL` |
| Project / demo page | `YOUR_PROJECT_OR_DEMO_URL` |
| MMOral (NeurIPS 2025) paper | `YOUR_MMORAL_PAPER_URL` |
| OralGPT-Omni (CVPR 2026) paper | `YOUR_ORALGPT_OMNI_PAPER_URL` |
| OralGPT-Plus (CVPR 2026) paper | `YOUR_ORALGPT_PLUS_PAPER_URL` |
| Expert model weights (e.g. Hugging Face) | `YOUR_HF_MODEL_URL` |
| Benchmark / dataset (e.g. Hugging Face) | `YOUR_HF_BENCHMARK_OR_DATASET_URL` |
| Oral corpus / RAG documents (if public) | `YOUR_ORAL_CORPUS_URL` |

---

## Requirements

- **Python** ≥ 3.10 (see `pyproject.toml`)
- **NVIDIA GPU + CUDA** strongly recommended (vision tools and embeddings are GPU-oriented)
- An **OpenAI-compatible API** (OpenAI, Azure, DashScope, vLLM, Ollama, etc.) for the chat model

---

## Installation

```bash
git clone YOUR_GITHUB_REPO_URL
cd OralAgent   # use the folder name of this package inside your clone (e.g. OralGPT-Agent/OralAgent)

python -m venv .venv
source .venv/bin/activate   # Windows: .venv\Scripts\activate

pip install -e .
```

**Note:** `pyproject.toml` pins a specific `transformers` git revision and includes `faiss-gpu`. If `faiss-gpu` fails on your platform, install a CPU variant or adjust dependencies locally to match your environment.

---

## Configuration

### 1. Environment variables

Create a `.env` in the project root (do not commit secrets). Typical variables:

```bash
OPENAI_API_KEY=your_key
# Optional: custom base URL for OpenAI-compatible providers
# OPENAI_BASE_URL=https://api.openai.com/v1
```

For local OpenAI-compatible servers (e.g. Ollama):

```bash
export OPENAI_BASE_URL=http://127.0.0.1:11434/v1
export OPENAI_API_KEY=ollama
```

### 2. Model weights directory (`model_dir`)

Expert checkpoints and configs (DINO / MaskDINO / BioMedCLIP modality ID, RAG embeddings, etc.) are resolved from a single **`model_dir`** on disk.

In **`main.py`** and **`launch_OralAgent.py`**, set paths such as:

- `model_dir` / `expert_model_dir` → directory containing OralGPT expert weights and `categories_*.json` / `config_*` files as expected by `oralagent/tools/get_tools.py`.
- **`main.py`** also references modality identification weights, e.g. `OralGPT_Modality_Identification_BioMedCLIP_8modalities.pth` under the same tree.

Download or sync files from **`YOUR_HF_MODEL_URL`** (or your release page) into that folder.

### 3. Tools and RAG

- **Which tools load:** edit `DEFAULT_SELECTED_TOOL_NAMES` in `oralagent/tools/get_tools.py` (comment or uncomment tool names). The default in-repo may be minimal (e.g. RAG-only); enable panoramic / periapical / other tools when weights are available.
- **RAG:** in `main.py`, adjust `RAGConfig` (`persist_dir`, `use_OralCorpus`, `corpus_language`, `local_docs_dir`, embedding/rerank model IDs) to match your machines and **`YOUR_ORAL_CORPUS_URL`** if you host documents locally.

---

## Running

### Gradio UI (recommended for first run)

```bash
python main.py
```

Default launch uses `demo.launch(...)` with host/port as set in `main.py` (e.g. `0.0.0.0:8552`). Change `model`, `model_dir`, and `device` in `main.py` first.

### FastAPI (single process, dev)

```bash
python launch_OralAgent.py
```

Default bind is `0.0.0.0:8124` with reload; adjust `expert_model_dir`, `ROOT`, `model_name`, and tool list inside `launch_OralAgent.py` / `get_tools` as needed.

### Multi-GPU / production-style workers

```bash
pip install gunicorn "uvicorn[standard]"
# Optional tuning, see script comments:
# ORAL_AGENT_MAX_WORKERS_PER_GPU=2 ./run_launch_OralAgent_multi_workers.sh
./run_launch_OralAgent_multi_workers.sh
```

See `gunicorn_conf.py` and `gpu_utils.py` for worker count and `CUDA_VISIBLE_DEVICES` behavior.

---

## Benchmarks and experiments

Scripts under `experiments/` support evaluation and analysis (e.g. MMOral-style benchmarks). Point datasets to **`YOUR_HF_BENCHMARK_OR_DATASET_URL`** or local paths as in each script’s arguments. **`quickstart.py`** can run lightweight API-based evaluation when the benchmark is configured similarly to the Hugging Face dataset layout expected by the script.

---

## Repository layout (high level)

| Path | Role |
|------|------|
| `oralagent/` | Agent, tools, prompts, RAG assets |
| `oralagent/tools/get_tools.py` | Tool registry and default tool list |
| `main.py` | Gradio app entry |
| `launch_OralAgent.py` | FastAPI app and `/v1/chat/completions` |
| `interface.py` | Gradio chat UI |
| `experiments/` | Benchmarks and analysis |
| `benchmark/`, `data/` | Data layouts (as used by your workflows) |

---

## Citation

If you use this repository or the associated work, please cite the relevant papers:

```bibtex
@article{hao2025mmoral,
  title={Towards Better Dental AI: A Multimodal Benchmark and Instruction Dataset for Panoramic X-ray Analysis},
  author={Hao, Jing and Fan, Yuxuan and Sun, Yanpeng and Guo, Kaixin and Lin, Lizhuo and Yang, Jinrong and Ai, Qi Yong H and Wong, Lun M and Tang, Hao and Hung, Kuo Feng},
  journal={NeurIPS 2025},
  year={2025}
}
@article{hao2025oralgpt-omni,
  title={OralGPT-Omni: A Versatile Dental Multimodal Large Language Model},
  author={Hao, Jing and Liang, Yuci and Lin, Lizhuo and Fan, Yuxuan and Zhou, Wenkai and Guo, Kaixin and Ye, Zanting and Sun, Yanpeng and Zhang, Xinyu and Yang, Yanqi and others},
  journal={CVPR 2026},
  year={2025}
}
@article{fan2026oralgpt-plus,
  title={OralGPT-Plus: Learning to Use Visual Tools via Reinforcement Learning for Panoramic X-ray Analysis},
  author={Fan, Yuxuan and Hao, Jing and Chen, Hong and Bao, Jiahao and Shao, Yihua and Liang, Yuci and Hung, Kuo Feng and Tang, Hao},
  journal={CVPR 2026},
  year={2026}
}
@article{hao2025oraldataset,
  title={Characteristics, licensing, and ethical considerations of openly accessible oral-maxillofacial imaging datasets: a systematic review},
  author={Hao, Jing and Nalley, Andrew and Yeung, Andy Wai Kan and Tanaka, Ray and Ai, Qi Yong H and Lam, Walter Yu Hang and Shan, Zhiyi and Leung, Yiu Yan and AlHadidi, Abeer and Bornstein, Michael M and others},
  journal={npj Digital Medicine},
  volume={8},
  number={1},
  pages={412},
  year={2025},
  publisher={Nature Publishing Group UK London}
}
```

---

<a id="quick-config-zh"></a>

## 快速配置摘要

1. 将 Logo 放到 **`assets/logo_OralAgent.png`**，并替换 README 顶部所有 **`YOUR_*`** 链接。  
2. 配置 **`.env`**（`OPENAI_API_KEY` 等）。  
3. 在 **`main.py`** / **`launch_OralAgent.py`** 中把 **`model_dir`**（及 `ROOT`）改成你存放 OralGPT 专家权重的目录。  
4. 在 **`oralagent/tools/get_tools.py`** 里按需启用工具；显存不足时少开工具或使用量化（若工具支持）。  
5. **`python main.py`** 体验 Gradio；需要 HTTP 对接时运行 **`launch_OralAgent.py`** 或 **`run_launch_OralAgent_multi_workers.sh`**。

---

## License

See [`LICENSE`](LICENSE).

---

<div align="center">

**OralAgent** — dental multimodal reasoning with tools and RAG.

</div>
