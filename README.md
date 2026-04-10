<div align="center">

<img src="assets/logo_OralAgent.png" alt="OralAgent logo" width="200"/>

# OralAgent

[**OralAgent: Integrating Reasoning, Tools, and Knowledge for Interactive Dental Image Analysis**](YOUR_ORALAGENT_PAPER_URL)

**A multimodal reasoning agent for dental image analysis** — tool use, RAG over oral corpora, and a Gradio UI or OpenAI-compatible HTTP API.

[![ToolHub](https://img.shields.io/badge/HuggingFace-Tool_Hub-f59e0b?style=flat-square&logo=huggingface&logoColor=white)](https://huggingface.co/OralGPT/OralAgent-ToolHub)
[![OralCorpus](https://img.shields.io/badge/HuggingFace-OralCorpus-059669?style=flat-square&logo=huggingface&logoColor=white)](https://huggingface.co/datasets/OralGPT/OralCorpus)
[![OralQA-ZH](https://img.shields.io/badge/HuggingFace-OralQA_ZH-6366f1?style=flat-square&logo=huggingface&logoColor=white)](https://huggingface.co/datasets/OralGPT/OralQA-ZH)
[![License](https://img.shields.io/badge/License-See%20LICENSE-1d4ed8?style=flat-square)](LICENSE)

[English](#oralagent) · [Custom links](#custom-links-table) · [快速配置（中文）](#quick-config-zh)

</div>

---

## OralAgent

OralAgent orchestrates **vision–language reasoning** with **modality-aware routing** and a library of **oral/dental expert tools** (panoramic X-ray, periapical, cephalometric, intraoral imaging, cytopathology, histopathology, etc.), optional **OralGPT-Omni** integration, and **RAG** backed by an OralCorpus (368 widely-used classical dental textbooks). It is built with **LangGraph / LangChain** and ships with:

- **Gradio chat UI** (`main.py`) — upload images or DICOM, multi-turn chat.
- **FastAPI service** (`launch_OralAgent.py`) — OpenAI-style `POST /v1/chat/completions` for integration.
- **Local-brain FastAPI service** (`launch_OralAgent_local.py`) — same API, but uses a locally served LLM (Lingshu, HuatuoGPT, HealthGPT, etc.) as the agent's reasoning backend.
- **Multi-GPU workers** — `gunicorn` + `uvicorn` via `run_launch_OralAgent_multi_workers.sh` and `gunicorn_conf.py`.

---

<a id="custom-links-table"></a>

## Links & badges

| Resource | URL (replace `YOUR_*`) |
|----------|-------------------------|
| OralAgent (this system) paper | `YOUR_ORALAGENT_PAPER_URL` |
| MMOral (NeurIPS 2025) paper | `https://arxiv.org/pdf/2509.09254` |
| OralGPT-Omni (CVPR 2026) paper | `https://arxiv.org/abs/2511.22055` |
| OralGPT-Plus (CVPR 2026) paper | `https://arxiv.org/abs/2603.06366` |
| Dental ToolHub (e.g. Hugging Face) | `https://huggingface.co/OralGPT/OralAgent-ToolHub` |
| OralQA-ZH Benchmark | `https://huggingface.co/datasets/OralGPT/OralQA-ZH` |
| Oral corpus / RAG documents | `https://huggingface.co/datasets/OralGPT/OralCorpus` |

---

## Requirements

- **Python** ≥ 3.10 (see `pyproject.toml`)
- **NVIDIA GPU + CUDA** strongly recommended (vision tools and embeddings are GPU-oriented)
- An **OpenAI-compatible API** (OpenAI, Azure, DashScope, vLLM, Ollama, etc.) for the chat model

---

## Installation

```bash
git clone https://github.com/isjinghao/OralAgent.git
cd OralAgent   # use the folder name of this package inside your clone (e.g. OralGPT-Agent/OralAgent)
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

Download or sync files from **`https://huggingface.co/OralGPT/OralAgent-ToolHub`** into that folder.

### 3. Tools and RAG

- **Which tools load:** edit `DEFAULT_SELECTED_TOOL_NAMES` in `oralagent/tools/get_tools.py` (comment or uncomment tool names). 
- **RAG:** in `main.py`, adjust `RAGConfig` (`persist_dir`, `use_OralCorpus`, `corpus_language`, `local_docs_dir`, embedding/rerank model IDs) to match your machines and **`https://huggingface.co/datasets/OralGPT/OralCorpus`** if you host documents locally.

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

### FastAPI with a local model as the brain

```bash
python launch_OralAgent_local.py
```

Uses a locally served LLM (e.g. Lingshu, HuatuoGPT, HealthGPT via `launch_server.py` or vLLM) as the agent's reasoning backend. Switch `backend = "custom"` in `startup_event` and set `base_url` (e.g. `http://localhost:8125/v1`) to point at the local model; `api_key` is not required.

### Multi-GPU / production-style workers

```bash
pip install gunicorn "uvicorn[standard]"
# Optional tuning, see script comments:
# ORAL_AGENT_MAX_WORKERS_PER_GPU=2 ./run_launch_OralAgent_multi_workers.sh
./run_launch_OralAgent_multi_workers.sh
```

See `gunicorn_conf.py` and `gpu_utils.py` for worker count and `CUDA_VISIBLE_DEVICES` behavior.

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

1. 配置 **`.env`**（`OPENAI_API_KEY` 等）。  
2. 在 **`main.py`** / **`launch_OralAgent.py`** 中把 **`model_dir`**（及 `ROOT`）改成你存放 OralGPT 专家权重的目录。  
3. 在 **`oralagent/tools/get_tools.py`** 里按需启用工具；显存不足时少开工具或使用量化（若工具支持）。  
4. **`python main.py`** 体验 Gradio；需要 HTTP 对接时运行 **`launch_OralAgent.py`** 或 **`run_launch_OralAgent_multi_workers.sh`**。

---

## License

See [`LICENSE`](LICENSE).

---

<div align="center">

**OralAgent** — dental multimodal reasoning with tools and RAG.

</div>
