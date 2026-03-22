import warnings
import json
from pathlib import Path
from datetime import datetime
from dotenv import load_dotenv
from langchain_openai import ChatOpenAI
from langchain_core.messages import AIMessage, ToolMessage
from medrax.agent import Agent
from medrax.utils import (
    load_prompts_from_file,
    parse_bool_env,
    apply_enriched_query_ablation,
)
from langgraph.checkpoint.memory import MemorySaver
from medrax.tools import *
from langchain_core.runnables import Runnable
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from typing import List, Optional, Dict, Any
import os
import re
import time
import uuid

warnings.filterwarnings("ignore")
_ = load_dotenv()

# 请求日志目录：在此目录下以「本次 OralAgent 启动时间」为子文件夹落盘；设为空字符串可关闭
DEFAULT_REQUEST_LOG_DIR = os.getenv("ORALAGENT_REQUEST_LOG_DIR", "logs/requests")

OralAgent = FastAPI()

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
    model = ChatOpenAI(model=model_name, temperature=temperature, top_p=0.95)

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

    return final_response, str(agent_state)

@OralAgent.get("/test")
def test_endpoint():
    return {"status": "OralAgent is working"}

@OralAgent.on_event("startup")
async def startup_event():
    global agent, REQUEST_LOG_SESSION_DIR
    # 多 worker 时由 gunicorn on_starting 设置 ORALAGENT_REQUEST_LOG_SESSION_DIR，保证共用一个文件夹
    REQUEST_LOG_SESSION_DIR = os.getenv("ORALAGENT_REQUEST_LOG_SESSION_DIR") or datetime.now().strftime("%Y%m%d_%H%M%S")
    expert_model_dir = "/data/OralGPT/OralGPT-expert-model-repository"
    temp_dir = "temp"
    # 单进程时用默认 "cuda"；多 worker 时由 gunicorn_conf 的 post_fork 设置 CUDA_VISIBLE_DEVICES，本进程内 cuda:0 即对应分配的那张卡
    device = "cuda"
    model_name = "gpt-5-nano"
    temperature = 0.2

    ROOT = "/home/jinghao/projects/OralGPT-Agent/OralAgent"
    PROMPT_FILE = f"{ROOT}/medrax/docs/system_prompts.txt"

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
def run_agent_endpoint(request: ChatCompletionRequest):
    try:
        request_id = str(uuid.uuid4())
        # 每次请求使用新的 thread_id，多次请求之间状态互不影响；若配置了日志目录则写入请求级日志便于统计工具调用
        response, state = run_OralAgent(
            agent=agent,
            messages=request.messages,
            request_id=request_id,
            request_log_dir=DEFAULT_REQUEST_LOG_DIR or None,
            request_log_session_dir=REQUEST_LOG_SESSION_DIR,
        )
        return {
            "id": f"chatcmpl-{request_id.replace('-', '')[:24]}",
            "object": "chat.completion",
            "created": int(time.time()),  # 当前时间戳
            "response": response,
            "choices": [
                {
                    "index": 0,
                    "message": {
                        "role": "assistant",
                        "content": response,  # 使用生成的响应内容
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