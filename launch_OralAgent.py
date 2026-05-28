import warnings
import json
import base64
from pathlib import Path
from datetime import datetime
from dotenv import load_dotenv
from langchain_openai import ChatOpenAI
from langchain_core.messages import AIMessage, ToolMessage
from oralagent.agent import Agent
from oralagent.utils import (
    load_prompts_from_file,
    parse_bool_env,
    apply_enriched_query_ablation,
)
from langgraph.checkpoint.memory import MemorySaver
from oralagent.tools import *
from langchain_core.runnables import Runnable
from fastapi import FastAPI, HTTPException, Request
from fastapi.responses import FileResponse
from pydantic import BaseModel
from typing import List, Optional, Dict, Any
import os
import re
import time
import uuid

warnings.filterwarnings("ignore")
PROJECT_ROOT = Path(__file__).resolve().parent
_ = load_dotenv()

# 请求日志目录：在此目录下以「本次 OralAgent 启动时间」为子文件夹落盘；设为空字符串可关闭
DEFAULT_REQUEST_LOG_DIR = os.getenv("ORALAGENT_REQUEST_LOG_DIR", "logs/requests")

OralAgent = FastAPI()
UPLOAD_DIR = PROJECT_ROOT / "temp" / "uploads"
PUBLIC_FILE_ROOTS = [
    (PROJECT_ROOT / "temp").resolve(),
    Path("temp").resolve(),
]
IMAGE_SUFFIXES = {".png", ".jpg", ".jpeg", ".webp", ".bmp"}
OUTPUT_IMAGE_KEYS = {
    "visualization_path",
    "segmentation_image_path",
    "output_path",
    "processed_image_path",
    "generated_image_path",
    "image_path",
}

# 声明全局变量（不再使用全局 thread，每次请求用独立 thread_id 隔离状态）
agent = None
# 本次进程启动时间，用作请求日志子文件夹名（格式 20260313_143022），在 startup_event 中设置
REQUEST_LOG_SESSION_DIR: Optional[str] = None

# 定义请求体模型
class ChatCompletionRequest(BaseModel):
    messages: List[Dict[str, Any]]  # 消息列表，符合 vllm 的输入格式


def _get_usage_from_message(msg) -> Optional[Dict[str, Any]]:
    """从 AIMessage 提取 usage_metadata，转为可 JSON 序列化的 dict。"""
    if not isinstance(msg, AIMessage):
        return None
    meta = getattr(msg, "usage_metadata", None)
    if not meta:
        return None
    return {
        "input_tokens": meta.get("input_tokens"),
        "output_tokens": meta.get("output_tokens"),
        "total_tokens": meta.get("total_tokens"),
    }


def _serialize_message_for_log(msg) -> Dict[str, Any]:
    """将单条 message 转为可 JSON 序列化的 dict，用于请求日志。"""
    if isinstance(msg, AIMessage):
        out = {
            "type": "ai",
            "content": getattr(msg, "content", None) or "",
            "tool_calls": [
                {"id": tc.get("id"), "name": tc.get("name"), "args": tc.get("args")}
                for tc in (getattr(msg, "tool_calls", None) or [])
            ],
        }
        usage = _get_usage_from_message(msg)
        if usage:
            out["usage"] = usage
        return out
    if isinstance(msg, ToolMessage):
        return {
            "type": "tool",
            "tool_call_id": getattr(msg, "tool_call_id", None),
            "name": getattr(msg, "name", None),
            "args": getattr(msg, "args", None),
            "content": (getattr(msg, "content", None) or "")[:2000],  # 截断过长结果便于查看
        }
    # 其他类型（HumanMessage 等）只记类型和 content 摘要
    content = getattr(msg, "content", None)
    if isinstance(content, list):
        content = f"<list len={len(content)}>"
    elif isinstance(content, str) and len(content) > 500:
        content = content[:500] + "..."
    return {"type": type(msg).__name__, "content": content}


def _get_first_image_name_from_messages(messages: List[Dict[str, Any]]) -> Optional[str]:
    """从请求的 messages（dict 列表）中解析第一个图像路径/URL，返回可用于文件名的 basename。"""
    for msg in messages:
        content = msg.get("content")
        if not content:
            continue
        if isinstance(content, list):
            for item in content:
                if isinstance(item, dict) and item.get("type") == "image_url":
                    url_obj = item.get("image_url") or {}
                    path = url_obj.get("image_path")
                    if path:
                        return os.path.basename(path)
                    url = url_obj.get("url") or ""
                    if url:
                        return os.path.basename(url.split("?")[0].rstrip("/") or "image")
        if isinstance(content, str) and "image_path:" in content:
            part = content.split("image_path:")[1].strip().split()[0]
            return os.path.basename(part)
    return None


def _get_first_image_name_from_stream_events(stream_events: List[Dict[str, Any]]) -> Optional[str]:
    """从 stream_events 中第一个带 image_path 的 tool_call 解析图像 basename。"""
    for step in stream_events:
        for m in step.get("messages") or []:
            for tc in (m.get("tool_calls") or []):
                args = (tc.get("args") or {})
                path = args.get("image_path")
                if path:
                    return os.path.basename(path)
    return None


def _sanitize_log_basename(name: str, max_len: int = 180) -> str:
    """将图像名等转为安全文件名：去掉非法字符并截断长度。"""
    safe = re.sub(r'[/\\:*?"<>|]', "_", name)
    return safe[:max_len] if len(safe) > max_len else safe


def _image_suffix_from_mime(mime: str) -> str:
    mime = (mime or "").lower()
    if "png" in mime:
        return ".png"
    if "webp" in mime:
        return ".webp"
    if "bmp" in mime:
        return ".bmp"
    return ".jpg"


def _safe_upload_filename(name: str, suffix: str) -> str:
    filename = _sanitize_log_basename(os.path.basename(name or ""), max_len=120)
    if filename and Path(filename).suffix.lower() in IMAGE_SUFFIXES:
        return filename
    if filename:
        return f"{filename}{suffix}"
    return f"{uuid.uuid4().hex}{suffix}"


def _save_data_url_image(data_url: str, requested_name: Optional[str] = None) -> Optional[str]:
    """Save an OpenAI-style data:image/...;base64 payload and return server-local path."""
    if not isinstance(data_url, str) or not data_url.startswith("data:image/"):
        return None

    match = re.match(r"^data:(?P<mime>image/[^;]+);base64,(?P<data>.+)$", data_url, re.S)
    if not match:
        return None

    UPLOAD_DIR.mkdir(parents=True, exist_ok=True)
    suffix = _image_suffix_from_mime(match.group("mime"))
    filename = _safe_upload_filename(requested_name or "", suffix)
    target = UPLOAD_DIR / filename
    if target.exists():
        target = UPLOAD_DIR / f"{target.stem}_{uuid.uuid4().hex[:8]}{target.suffix}"

    try:
        target.write_bytes(base64.b64decode(match.group("data")))
    except Exception as exc:
        raise ValueError(f"Invalid base64 image payload: {exc}") from exc
    return str(target.resolve())


def _materialize_base64_images(messages: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    """Convert incoming base64 image_url blocks into real files before Agent preprocessing."""
    for msg in messages:
        content = msg.get("content")
        if not isinstance(content, list):
            continue

        replacements: Dict[str, str] = {}
        for item in content:
            if not isinstance(item, dict) or item.get("type") != "image_url":
                continue
            image_url = item.get("image_url") or {}
            if not isinstance(image_url, dict):
                continue

            old_path = image_url.get("image_path") or ""
            saved_path = _save_data_url_image(image_url.get("url"), old_path)
            if not saved_path:
                continue

            if old_path:
                replacements[str(old_path)] = saved_path
            image_url["image_path"] = saved_path

        if not replacements:
            continue

        for item in content:
            if not isinstance(item, dict) or item.get("type") != "text":
                continue
            text = item.get("text")
            if not isinstance(text, str):
                continue
            for old_path, new_path in replacements.items():
                text = text.replace(f"image_path: {old_path}", f"image_path: {new_path}")
            item["text"] = text

    return messages


def _path_is_relative_to(path: Path, root: Path) -> bool:
    try:
        path.resolve().relative_to(root.resolve())
        return True
    except ValueError:
        return False


def _resolve_image_file_path(raw_path: str) -> Optional[Path]:
    if not raw_path:
        return None
    raw_path = raw_path.strip().strip("'\"")
    if raw_path.startswith("data:") or raw_path.startswith("http://") or raw_path.startswith("https://"):
        return None

    path = Path(raw_path)
    candidates = [path]
    if not path.is_absolute():
        candidates.extend([PROJECT_ROOT / path, Path.cwd() / path])

    for candidate in candidates:
        resolved = candidate.resolve()
        if resolved.is_file() and resolved.suffix.lower() in IMAGE_SUFFIXES:
            return resolved
    return None


def _extract_input_image_paths(messages: List[Dict[str, Any]]) -> set[str]:
    paths: set[str] = set()
    for msg in messages:
        content = msg.get("content")
        if isinstance(content, str) and "image_path:" in content:
            paths.add(content.split("image_path:", 1)[1].strip().split()[0])
        if not isinstance(content, list):
            continue
        for item in content:
            if not isinstance(item, dict):
                continue
            if item.get("type") == "text" and isinstance(item.get("text"), str) and "image_path:" in item["text"]:
                paths.add(item["text"].split("image_path:", 1)[1].strip().split()[0])
            if item.get("type") == "image_url":
                image_url = item.get("image_url") or {}
                if isinstance(image_url, dict) and image_url.get("image_path"):
                    paths.add(str(image_url["image_path"]))
    return paths


def _extract_output_image_paths_from_text(text: str) -> List[str]:
    if not text:
        return []

    suffix_pattern = r"(?:png|jpg|jpeg|webp|bmp)"
    keys = "|".join(re.escape(key) for key in OUTPUT_IMAGE_KEYS)
    patterns = [
        rf"""['"](?:{keys})['"]\s*:\s*['"]([^'"]+\.{suffix_pattern})['"]""",
        rf"""(?:{keys}|image path|image\(s\)|processed output image\(s\)|returned processed image path\(s\)|annotated visualization image path)\s*:\s*`?([^`\s,;]+\.{suffix_pattern})`?""",
        rf"""!\[[^\]]*\]\(([^)]+\.{suffix_pattern})\)""",
        rf"""`([^`\n\r]+\.{suffix_pattern})`""",
    ]
    paths: List[str] = []
    for pattern in patterns:
        for match in re.findall(pattern, text, flags=re.I):
            if isinstance(match, tuple):
                match = next((part for part in match if part), "")
            paths.append(str(match).strip().strip("`'\"").rstrip(".,;)]"))

    for match in re.findall(
        rf"""(?<![A-Za-z0-9_./-])((?:temp|outputs?|results?)/[^\s`'"<>]+\.{suffix_pattern})""",
        text,
        flags=re.I,
    ):
        paths.append(str(match).strip().rstrip(".,;)]"))

    deduped: List[str] = []
    seen: set[str] = set()
    for path in paths:
        if path and path not in seen:
            seen.add(path)
            deduped.append(path)
    return deduped


def _request_base_url(request: Request) -> str:
    proto = request.headers.get("x-forwarded-proto") or request.url.scheme
    host = request.headers.get("x-forwarded-host") or request.headers.get("host")
    if host:
        return f"{proto}://{host}".rstrip("/")
    return str(request.base_url).rstrip("/")


def _public_url_for_file(path: Path, request: Request) -> Optional[str]:
    resolved = path.resolve()
    for root in PUBLIC_FILE_ROOTS:
        if not _path_is_relative_to(resolved, root):
            continue
        try:
            rel = resolved.relative_to(PROJECT_ROOT.resolve()).as_posix()
        except ValueError:
            rel = resolved.relative_to(root.resolve()).as_posix()
        return f"{_request_base_url(request)}/files/{rel}"
    return None


def _build_output_images(paths: List[str], input_paths: set[str], request: Request) -> List[Dict[str, str]]:
    images: List[Dict[str, str]] = []
    seen: set[str] = set()
    resolved_inputs = {
        str(path.resolve())
        for raw in input_paths
        for path in [_resolve_image_file_path(raw)]
        if path
    }

    for raw_path in paths:
        path = _resolve_image_file_path(raw_path)
        if not path:
            continue
        resolved = str(path.resolve())
        if resolved in seen or resolved in resolved_inputs:
            continue
        url = _public_url_for_file(path, request)
        if not url:
            continue
        seen.add(resolved)
        images.append({"url": url, "name": path.name})
    return images


def get_agent(
    tools,
    prompt_file,
    model_name,
    temperature,
    model_dir,
    device="cuda",
):
    # Load prompts
    prompts = load_prompts_from_file(prompt_file)
    system_prompt = prompts["MEDICAL_ASSISTANT"]
    intent_recognition_prompt = prompts["INTENT_RECOGNITION_ASSISTANT"]
    enriched_query_template = prompts.get("ENRICHED_QUERY_TEMPLATE")
    modality_section_template = prompts.get("MODALITY_SECTION_TEMPLATE")

    # Optional ablation switches (default keeps original template unchanged)
    include_intent_recognition = parse_bool_env(
        "ORALAGENT_INCLUDE_INTENT_RECOGNITION_IN_PROMPT", default=True
    )
    include_modality_section = parse_bool_env(
        "ORALAGENT_INCLUDE_MODALITY_SECTION_IN_PROMPT", default=True
    )
    enriched_query_template = apply_enriched_query_ablation(
        enriched_query_template,
        include_intent_recognition=include_intent_recognition,
        include_modality_section=include_modality_section,
    )

    # Initialize the agent
    checkpointer = MemorySaver()
    # Bedrock Claude endpoints may reject requests when temperature and top_p
    # are both provided. Keep only temperature by default.
    # model = ChatOpenAI(model=model_name, temperature=temperature, top_p=0.95)
    model = ChatOpenAI(model=model_name, temperature=temperature)

    intent_classifier_model = BioMedCLIPClassifier(
        checkpoint_path=f"{model_dir}/OralGPT_Modality_Identification_BioMedCLIP_8modalities.pth",
        coco_names_path=f"{model_dir}/categories_Modality_Identification_BioMedCLIP_8modalities.json",
        num_classes=8,
        device=device,
    )

    agent = Agent(
        model,
        intent_classifier_model=intent_classifier_model,
        tools=tools,
        log_tools=True,
        log_dir="logs",
        system_prompt=system_prompt,
        intent_recognition_prompt=intent_recognition_prompt,
        enriched_query_template=enriched_query_template,
        modality_section_template=modality_section_template,
        checkpointer=checkpointer,
    )
    return agent

def run_OralAgent(
    agent,
    messages,
    thread_id: Optional[str] = None,
    request_id: Optional[str] = None,
    request_log_dir: Optional[str] = None,
    request_log_session_dir: Optional[str] = None,
):
    """Run agent for one request. Each request uses an isolated thread so state does not leak.
    Pass thread_id only when you want to continue a specific conversation.
    When request_log_dir and request_log_session_dir are set, saves one JSON per request (path: .../session_dir/YYYYMMDD_HHMMSS_request_id.json)."""
    thread = {"configurable": {"thread_id": thread_id or str(uuid.uuid4())}}
    thread_id_str = thread["configurable"]["thread_id"]
    request_id = request_id or str(uuid.uuid4())
    start_time = time.time()
    start_iso = datetime.now().isoformat()

    final_response = None
    stream_events: List[Dict[str, Any]] = []
    input_image_paths = _extract_input_image_paths(messages)
    output_image_paths: List[str] = []
    # 汇总本请求内所有 LLM 调用的 token 消耗
    total_input_tokens = 0
    total_output_tokens = 0
    total_tokens = 0
    token_usage_by_node: List[Dict[str, Any]] = []  # 每步的 token 消耗，便于按节点统计

    for event in agent.workflow.stream({"messages": messages}, thread):
        for node_name, v in event.items():
            final_response = v
            # 只记录本步新增的 messages（state 更新中的 messages）
            if isinstance(v, dict) and "messages" in v:
                step_messages = [_serialize_message_for_log(m) for m in v["messages"]]
                step_log = {"node": node_name, "messages": step_messages}
                for m in v["messages"]:
                    if isinstance(m, ToolMessage):
                        output_image_paths.extend(
                            _extract_output_image_paths_from_text(getattr(m, "content", "") or "")
                        )
                # 本步 token 消耗（来自 AIMessage.usage_metadata）
                step_input = 0
                step_output = 0
                for m in v["messages"]:
                    u = _get_usage_from_message(m)
                    if u:
                        step_input += (u.get("input_tokens") or 0)
                        step_output += (u.get("output_tokens") or 0)
                if step_input or step_output:
                    step_log["token_usage"] = {
                        "input_tokens": step_input,
                        "output_tokens": step_output,
                        "total_tokens": (step_input + step_output),
                    }
                    total_input_tokens += step_input
                    total_output_tokens += step_output
                    total_tokens += step_input + step_output
                    token_usage_by_node.append({"node": node_name, "token_usage": step_log["token_usage"]})
                stream_events.append(step_log)

    final_response = final_response["messages"][-1].content.strip()
    output_image_paths.extend(_extract_output_image_paths_from_text(final_response))
    agent_state = agent.workflow.get_state(thread)
    end_time = time.time()
    duration_sec = round(end_time - start_time, 3)

    # 按请求落盘一份完整日志，便于统计工具调用过程（目录=启动时间，文件名=请求时间_图像名或request_id）
    if request_log_dir and request_log_session_dir:
        log_path = Path(request_log_dir) / request_log_session_dir
        log_path.mkdir(parents=True, exist_ok=True)
        request_time_str = datetime.now().strftime("%Y%m%d_%H%M%S")  # 如 20260313_143052
        # 优先用本请求中出现的图像名作为日志文件名，无图像时用 request_id
        image_name = _get_first_image_name_from_stream_events(stream_events)
        if not image_name and messages:
            # messages 可能是 LangChain 对象或 dict；若是 dict 列表则直接解析
            dict_messages = [
                m if isinstance(m, dict) else {"content": getattr(m, "content", None)}
                for m in messages
            ]
            image_name = _get_first_image_name_from_messages(dict_messages)
        # 只取文件名并去掉后缀（如 2191.jpg -> 2191）
        if image_name:
            image_name = Path(image_name).stem
        log_basename = _sanitize_log_basename(image_name) if image_name else request_id
        log_file = log_path / f"{request_time_str}_{log_basename}.json"
        request_log = {
            "request_id": request_id,
            "thread_id": thread_id_str,
            "start_time": start_iso,
            "end_time": datetime.now().isoformat(),
            "duration_sec": duration_sec,
            "input_messages_count": len(messages),
            "token_usage": {
                "prompt_tokens": total_input_tokens,
                "completion_tokens": total_output_tokens,
                "total_tokens": total_tokens,
            },
            "token_usage_by_node": token_usage_by_node,
            "stream_events": stream_events,
            "final_response_preview": (final_response[:500] + "...") if len(final_response) > 500 else final_response,
        }
        try:
            with open(log_file, "w", encoding="utf-8") as f:
                json.dump(request_log, f, ensure_ascii=False, indent=2)
        except Exception as e:
            import traceback
            print(f"[RequestLog] Failed to write {log_file}: {e}\n{traceback.format_exc()}")

    return final_response, str(agent_state), output_image_paths, input_image_paths

@OralAgent.get("/test")
def test_endpoint():
    return {"status": "OralAgent is working"}


@OralAgent.get("/files/{file_path:path}")
def public_file_endpoint(file_path: str):
    requested = Path(file_path)
    candidates = [(PROJECT_ROOT / requested).resolve()]
    candidates.extend((root / requested).resolve() for root in PUBLIC_FILE_ROOTS)

    for candidate in candidates:
        allowed = any(_path_is_relative_to(candidate, root) for root in PUBLIC_FILE_ROOTS)
        if allowed and candidate.is_file() and candidate.suffix.lower() in IMAGE_SUFFIXES:
            return FileResponse(candidate)

    raise HTTPException(status_code=404, detail="File not found")


@OralAgent.on_event("startup")
async def startup_event():
    global agent, REQUEST_LOG_SESSION_DIR
    # 多 worker 时由 gunicorn on_starting 设置 ORALAGENT_REQUEST_LOG_SESSION_DIR，保证共用一个文件夹
    REQUEST_LOG_SESSION_DIR = os.getenv("ORALAGENT_REQUEST_LOG_SESSION_DIR") or datetime.now().strftime("%Y%m%d_%H%M%S")
    expert_model_dir = "/data/OralGPT/OralGPT-expert-model-repository"
    temp_dir = "temp"
    # 单进程时用默认 "cuda"；多 worker 时由 gunicorn_conf 的 post_fork 设置 CUDA_VISIBLE_DEVICES，本进程内 cuda:0 即对应分配的那张卡
    device = "cuda"
    model_name = "qwen3.5-27b"
    temperature = 0.2

    ROOT = "/home/jinghao/projects/OralGPT-Agent/OralAgent"
    PROMPT_FILE = f"{ROOT}/oralagent/docs/system_prompts.txt"

    tools = get_tools(
        model_dir=expert_model_dir,
        temp_dir=temp_dir,
        device=device
    )
    agent = get_agent(
        tools,
        prompt_file=PROMPT_FILE,
        model_name=model_name,
        temperature=temperature,
        model_dir=expert_model_dir,
        device=device,
    )
    print("OralAgent successfully initialized and ready to use.")

    # for route in OralAgent.routes:
    #     print(route.path, route.name)


# 添加 API 路由
@OralAgent.post("/v1/chat/completions")
def run_agent_endpoint(chat_request: ChatCompletionRequest, request: Request):
    try:
        request_id = str(uuid.uuid4())
        messages = _materialize_base64_images(chat_request.messages)
        # 每次请求使用新的 thread_id，多次请求之间状态互不影响；若配置了日志目录则写入请求级日志便于统计工具调用
        response, state, output_image_paths, input_image_paths = run_OralAgent(
            agent=agent,
            messages=messages,
            request_id=request_id,
            request_log_dir=DEFAULT_REQUEST_LOG_DIR or None,
            request_log_session_dir=REQUEST_LOG_SESSION_DIR,
        )
        output_images = _build_output_images(output_image_paths, input_image_paths, request)
        return {
            "id": f"chatcmpl-{request_id.replace('-', '')[:24]}",
            "object": "chat.completion",
            "created": int(time.time()),  # 当前时间戳
            "response": response,
            "output_images": output_images,
            "choices": [
                {
                    "index": 0,
                    "message": {
                        "role": "assistant",
                        "content": response,  # 使用生成的响应内容
                        "images": output_images,
                        "annotations": [],
                        "refusal": None,
                    }
                }
            ],

            # "state": state,
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

if __name__ == "__main__":
    import uvicorn
    # 单进程开发/调试；多卡多 worker 请用: ./run_multi_gpu.sh 或 gunicorn -k uvicorn.workers.UvicornWorker -c gunicorn_conf.py launch_OralAgent:OralAgent
    uvicorn.run("launch_OralAgent:OralAgent", host="0.0.0.0", port=8124, reload=True)
