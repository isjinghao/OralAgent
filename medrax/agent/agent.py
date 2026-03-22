import json
import operator
from pathlib import Path
from dotenv import load_dotenv
from datetime import datetime
from typing import List, Dict, Any, TypedDict, Annotated, Optional, Union

from langgraph.graph import StateGraph, END
from langchain_core.messages import AnyMessage, SystemMessage, ToolMessage, AIMessage

from langchain_core.language_models import BaseLanguageModel
from langchain_core.tools import BaseTool
from transformers import AutoModelForCausalLM, AutoTokenizer

import logging
logging.basicConfig(level=logging.INFO)

_ = load_dotenv()


class ToolCallLog(TypedDict):
    """
    A TypedDict representing a log entry for a tool call.

    Attributes:
        timestamp (str): The timestamp of when the tool call was made.
        tool_call_id (str): The unique identifier for the tool call.
        name (str): The name of the tool that was called.
        args (Any): The arguments passed to the tool.
        content (str): The content or result of the tool call.
    """

    timestamp: str
    tool_call_id: str
    name: str
    args: Any
    content: str


class AgentState(TypedDict):
    """
    A TypedDict representing the state of an agent.

    Attributes:
        messages (Annotated[List[AnyMessage], operator.add]): A list of messages
            representing the conversation history. The operator.add annotation
            indicates that new messages should be appended to this list.
    """

    messages: Annotated[List[AnyMessage], operator.add]


class Agent:
    """
    A class representing an agent that processes requests and executes tools based on
    language model responses.

    Attributes:
        model (BaseLanguageModel): The language model used for processing.
        tools (Dict[str, BaseTool]): A dictionary of available tools.
        checkpointer (Any): Manages and persists the agent's state.
        system_prompt (str): The system instructions for the agent.
        workflow (StateGraph): The compiled workflow for the agent's processing.
        log_tools (bool): Whether to log tool calls.
        log_path (Path): Path to save tool call logs.
    """

    # Default templates when not loaded from file (keys: user_text_query, user_intent, modality_section)
    _DEFAULT_ENRICHED_QUERY_TEMPLATE = (
        "The user's query: {user_text_query}\n"
        "Intent recognition: {user_intent}\n"
        "{modality_section}"
    )
    _DEFAULT_MODALITY_SECTION_TEMPLATE = (
        "Image modality: {modality}\n"
        "Please select the most appropriate tools based on the image modality above."
    )
    # 多图（如 intraoral video）时注入的指令，与 ENRICHED_QUERY_TEMPLATE 中规则一致，因多图不会走 enriched query 替换
    MULTI_IMAGE_NO_TOOLS_INSTRUCTION = (
        "When handling intraoral‑video queries, all tools except <OralGPT‑Omni> are prohibited. Responses must rely on the intraoral video and the outputs of <OralGPT‑Omni>, and should be thoroughly refined using your own visual reasoning."
    )

    def __init__(
        self,
        model: BaseLanguageModel,
        tools: List[BaseTool],
        intent_recognition_model: str = "Qwen/Qwen3-0.6B", # add by Bryce
        intent_classifier_model: Any = None,
        checkpointer: Any = None,
        system_prompt: str = "",
        intent_recognition_prompt: str = "",
        enriched_query_template: Optional[str] = None,
        modality_section_template: Optional[str] = None,
        log_tools: bool = True,
        log_dir: Optional[str] = "logs",
    ):
        """
        Initialize the Agent.

        Args:
            model (BaseLanguageModel): The language model to use.
            tools (List[BaseTool]): A list of available tools.
            checkpointer (Any, optional): State persistence manager. Defaults to None.
            system_prompt (str, optional): System instructions. Defaults to "".
            enriched_query_template (str, optional): Template for enriched user query with placeholders
                {user_text_query}, {user_intent}, {modality_section}. Defaults to built-in template.
            modality_section_template (str, optional): Template for image modality section with
                {modality}, {confidence}. Used only when user provides an image. Defaults to built-in.
            log_tools (bool, optional): Whether to log tool calls. Defaults to True.
            log_dir (str, optional): Directory to save logs. Defaults to 'logs'.
        """
        self.system_prompt = system_prompt
        self.intent_recognition_prompt = intent_recognition_prompt
        self.enriched_query_template = (
            enriched_query_template or self._DEFAULT_ENRICHED_QUERY_TEMPLATE
        )
        self.modality_section_template = (
            modality_section_template or self._DEFAULT_MODALITY_SECTION_TEMPLATE
        )
        self.log_tools = log_tools

        if self.log_tools:
            self.log_path = Path(log_dir or "logs")
            self.log_path.mkdir(exist_ok=True)

        # Define the agent workflow
        workflow = StateGraph(AgentState)
        workflow.add_node("process", self.process_request)
        workflow.add_node("execute", self.execute_tools)
        workflow.add_conditional_edges(
            "process", self.has_tool_calls, {True: "execute", False: END}
        )
        workflow.add_edge("execute", "process")
        workflow.set_entry_point("process")
        self.workflow = workflow.compile(checkpointer=checkpointer)
        self.tools = {t.name: t for t in tools}
        self.model = model.bind_tools(tools)

        # Add by Bryce
        self.intent_recognition_tokenizer = AutoTokenizer.from_pretrained(intent_recognition_model)
        self.intent_recognition_model = AutoModelForCausalLM.from_pretrained(
            intent_recognition_model,
            torch_dtype="auto",
            device_map="auto"
        )
        self.modality_classifier = intent_classifier_model
        
    def process_request(self, state: AgentState) -> Dict[str, List[AnyMessage]]:
        """
        Process the request using the language model.

        Args:
            state (AgentState): The current state of the agent.

        Returns:
            Dict[str, List[AnyMessage]]: A dictionary containing the model's response.
        """

        state = self.preprocess_request(state)  # Preprocess the request to recognize intent

        messages = state["messages"]
        if self.system_prompt:
            messages = [SystemMessage(content=self.system_prompt)] + messages
        response = self.model.invoke(messages)
        logging.info(f"Process node output: {response}")

        return {"messages": [response]}

    def has_tool_calls(self, state: AgentState) -> bool:
        """
        Check if the response contains any tool calls.

        Args:
            state (AgentState): The current state of the agent.

        Returns:
            bool: True if tool calls exist, False otherwise.
        """
        response = state["messages"][-1]
        return len(response.tool_calls) > 0

    def execute_tools(self, state: AgentState) -> Dict[str, List[ToolMessage]]:
        """
        Execute tool calls from the model's response.

        Args:
            state (AgentState): The current state of the agent.

        Returns:
            Dict[str, List[ToolMessage]]: A dictionary containing tool execution results.
        """
        tool_calls = state["messages"][-1].tool_calls
        results = []

        for call in tool_calls:
            print(f"Executing tool: {call}")
            if call["name"] not in self.tools:
                print("\n....invalid tool....")
                result = "invalid tool, please retry"
            else:
                result = self.tools[call["name"]].invoke(call["args"])

            results.append(
                ToolMessage(
                    tool_call_id=call["id"],
                    name=call["name"],
                    args=call["args"],
                    content=str(result),
                )
            )

        self._save_tool_calls(results)
        print("Returning to model processing!")

        return {"messages": results}

    def _save_tool_calls(self, tool_calls: List[ToolMessage]) -> None:
        """
        Save tool calls to a JSON file with timestamp-based naming.

        Args:
            tool_calls (List[ToolMessage]): List of tool calls to save.
        """
        if not self.log_tools:
            return

        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        filename = self.log_path / f"tool_calls_{timestamp}.json"

        logs: List[ToolCallLog] = []
        for call in tool_calls:
            log_entry = {
                "tool_call_id": call.tool_call_id,
                "name": call.name,
                "args": call.args,
                "content": call.content,
                "timestamp": datetime.now().isoformat(),
            }
            logs.append(log_entry)

        with open(filename, "w") as f:
            json.dump(logs, f, indent=4)

    def preprocess_request(self, state: AgentState) -> AgentState:
        """
        Preprocess the latest user input before the language model runs.

        We determine the "current turn" (all consecutive user messages at the end of
        history until an AI/Tool message) and count how many images appear in that
        turn. For a single image (or no image), we run intent recognition and
        modality detection, build an enriched query (user query + intent + modality +
        image path), and replace the user's text content with this enriched query so
        the model and tools get a single, structured prompt. For multiple images in
        the same turn, we skip intent and modality and return state unchanged.

        Args:
            state (AgentState): The current state of the agent.

        Returns:
            AgentState: The same state, with user message content possibly replaced by the enriched query.
        """
        # print(f"state: {state}")
        user_input = state["messages"][-1]

        if isinstance(user_input, ToolMessage) or isinstance(user_input, AIMessage):
            return state

        # Current turn: all user messages from the end of history until an AI/Tool message (state may contain multiple turns).
        current_turn_messages = self._get_current_turn_user_messages(state["messages"])
        # Collect all image paths in the current turn (API 可传入多图，如 content 中多个 image_url；Gradio 仅单图).
        image_paths: List[Optional[str]] = []
        for msg in current_turn_messages:
            image_paths.extend(self.extract_image_paths_from_message(msg))
        # 直接统计最后一则用户消息 content 中 image_url 块数量，避免因路径解析差异漏判多图
        n_image_url_blocks = self._count_image_url_blocks_in_message(state["messages"][-1])
        # 无论图片数量多少，都提取并写入 benchmark index，保证模型一定能看到
        benchmark_index = self.extract_benchmark_index_from_messages(current_turn_messages)
        # 多图（仅 API 会传多图）：跳过意图识别与模态检测，直接进入模型。
        is_multi_image = len(image_paths) > 1 or n_image_url_blocks > 1
        logging.info(f"[preprocess] image_paths len={len(image_paths)}, n_image_url_blocks={n_image_url_blocks}, is_multi_image={is_multi_image}")
        if is_multi_image:
            last_msg = state["messages"][-1]
            last_content = last_msg.get("content") if isinstance(last_msg, dict) else getattr(last_msg, "content", None)
            if isinstance(last_content, list):
                # 多图时不走 enriched query，需显式注入「禁止工具」指令，与 prompt 中 intraoral video 规则一致
                instruction_injected = False
                for item in last_content:
                    if item.get("type") != "text":
                        continue
                    raw = (item.get("text") or "").strip()
                    if raw.startswith("image_path:"):
                        continue
                    if "All tools are prohibited" not in (item.get("text") or ""):
                        item["text"] = self.MULTI_IMAGE_NO_TOOLS_INSTRUCTION + (item.get("text") or "")
                    instruction_injected = True
                    break
                if not instruction_injected:
                    # 仅有图片无用户文字时，在末尾加一条仅含指令的 text 块
                    last_content.append({"type": "text", "text": self.MULTI_IMAGE_NO_TOOLS_INSTRUCTION.strip()})
                if benchmark_index is not None:
                    for item in last_content:
                        if item.get("type") == "text" and not (item.get("text") or "").strip().startswith("image_path:"):
                            if "Benchmark case index:" not in (item.get("text") or ""):
                                item["text"] = (item.get("text") or "") + f"\nBenchmark case index: {benchmark_index}"
                            break
            elif isinstance(last_content, str) and benchmark_index is not None and "Benchmark case index:" not in last_content:
                if isinstance(last_msg, dict):
                    state["messages"][-1]["content"] = last_content + f"\nBenchmark case index: {benchmark_index}"
                else:
                    setattr(state["messages"][-1], "content", last_content + f"\nBenchmark case index: {benchmark_index}")
            return state

        user_text_query = self.extract_text_content(user_input)

        # 单图或无图：取当前用户输入中的唯一图片路径（若有）
        image_path = image_paths[0] if image_paths else None
        if image_path is None:
            newest_image_content = self.extract_image_content(state["messages"])
            if newest_image_content and isinstance(newest_image_content.get("content"), list):
                for item in newest_image_content["content"]:
                    if isinstance(item, dict) and item.get("type") == "image_url":
                        url_obj = item.get("image_url")
                        if isinstance(url_obj, dict):
                            if "image_path" in url_obj:
                                path = url_obj.get("image_path")
                            elif "url" in url_obj:
                                path = url_obj.get("url")
                            if path and isinstance(path, str):
                                image_path = path.strip()
                                break
            if image_path is None and newest_image_content and isinstance(newest_image_content.get("content"), str):
                content = newest_image_content["content"]
                if "image_path:" in content:
                    image_path = content.split("image_path:")[1].strip()

        # 网页上传时 image_url 为 base64，无法作为文件路径；从同轮对话中查找显式的 image_path: 路径
        if image_path and (image_path.startswith("data:") or (not Path(image_path).is_file())):
            for msg in reversed(state["messages"]):
                content = msg.get("content") if isinstance(msg, dict) else getattr(msg, "content", None)
                if isinstance(content, str) and "image_path:" in content:
                    image_path = content.split("image_path:")[1].strip()
                    break

        modality_section = ""
        if image_path and self.modality_classifier:
            try:
                modality_result = self.modality_classifier.predict(image_path)
                modality_section = self.modality_section_template.format(
                    modality=modality_result["modality"],
                    # confidence=f"{modality_result['confidence']:.2f}",
                )
                # print(f"modality_section: {modality_section}")
            except Exception as e:
                print(f"Modality classification error: {e}")
        if image_path:
            modality_section += f"\nImage file path (you MUST use this exact path for the image_path argument in tool calls): {image_path}"

        if benchmark_index is not None:
            modality_section += f"\nBenchmark case index: {benchmark_index}"

        user_intent = self.recognize_intent(user_text_query)

        processed_user_query = self.enriched_query_template.format(
            user_text_query=user_text_query or "",
            user_intent=user_intent,
            modality_section=modality_section,
        )
        # 当 ORALAGENT_INCLUDE_MODALITY_SECTION_IN_PROMPT=0 时，apply_enriched_query_ablation 会移除含 {modality_section} 的整行，
        # 导致 image_path 与 benchmark_index 未被插入。此处补回这两项必要信息，避免模型拿不到图像路径和 case index。
        if image_path and "Image file path" not in processed_user_query:
            processed_user_query += f"\nImage file path (you MUST use this exact path for the image_path argument in tool calls): {image_path}"
        if benchmark_index is not None and "Benchmark case index" not in processed_user_query:
            processed_user_query += f"\nBenchmark case index: {benchmark_index}"
        print(f"enriched query:\n{processed_user_query}")

        # 把用于意图的 text 块替换为 enriched query；若无（仅图片无文字）则替换第一个 text 块，保证模型收到 modality/intent
        last_content = state["messages"][-1].get("content") if isinstance(state["messages"][-1], dict) else getattr(state["messages"][-1], "content", None)
        if isinstance(last_content, list):
            updated = False
            for item in last_content:
                if item.get("type") != "text":
                    continue
                raw = (item.get("text") or "").strip()
                if raw.startswith("image_path:"):
                    continue
                item["text"] = processed_user_query
                updated = True
                break
            if not updated:
                for item in last_content:
                    if item.get("type") == "text":
                        item["text"] = processed_user_query
                        break

        return state


    def extract_text_content(self, input_dict):
        """
        Extract the text content for type 'text'，用于意图识别等。若存在多段 text（如 image_path 行 + 用户问题），
        优先返回用户问题（非 image_path: 开头的段落）。

        Args:
            input_dict (dict): The input dictionary containing 'content' list.

        Returns:
            str: 用作 user query 的文本；无则 None。
        """
        text_content = None
        fallback_text = None
        for item in input_dict.get("content", []):
            if item.get("type") != "text":
                continue
            raw = item.get("text") or ""
            if not raw.strip():
                continue
            if raw.strip().startswith("image_path:"):
                fallback_text = fallback_text or raw
                continue
            text_content = raw
            break
        return text_content if text_content is not None else fallback_text

    def _count_image_url_blocks_in_message(self, message: Union[Dict, Any]) -> int:
        """统计单条消息 content 中 type 为 image_url 的块数量，用于可靠判断 API 多图请求。"""
        content = message.get("content") if isinstance(message, dict) else getattr(message, "content", None)
        if not isinstance(content, list):
            return 0
        n = 0
        for item in content:
            t = item.get("type") if isinstance(item, dict) else getattr(item, "type", None)
            if t == "image_url":
                n += 1
        return n

    def _get_current_turn_user_messages(self, messages: List[AnyMessage]) -> List[AnyMessage]:
        """
        从多轮 state 中取出「当前轮」的用户消息：从末尾向前，直到遇到 AIMessage 或 ToolMessage 为止。

        Args:
            messages: state["messages"]

        Returns:
            当前轮内的所有用户消息（顺序与 state 中一致）。
        """
        if not messages:
            return []
        end = len(messages)
        for i in range(len(messages) - 1, -1, -1):
            msg = messages[i]
            if isinstance(msg, (AIMessage, ToolMessage)):
                return list(messages[i + 1 : end])
        return list(messages[0 : end])

    def extract_image_content(self, messages):
        """
        Reverse iterate through the list to find the first ToolMessage class and check for image_path.

        Args:
            messages (list): The list of messages to process.

        Returns:
            dict or None: The dictionary containing image_path if found, otherwise None.
        """
        tool_message_index = None

        # Find the first ToolMessage instance in reverse order
        for i in range(len(messages) - 1, -1, -1):
            if isinstance(messages[i], AIMessage):
                tool_message_index = i
                break
        
        # If no ToolMessage is found, set index to the start of the list
        if tool_message_index is None:
            tool_message_index = 0

        # Iterate from the last element to the ToolMessage instance (or start of the list)
        for i in range(len(messages) - 1, tool_message_index - 1, -1):
            message = messages[i]
            if isinstance(message, dict) and "content" in message:
                content = message["content"]
                if isinstance(content, str) and "image_path:" in content:
                    return message  # Return the message containing 'image_path'
                # User's first message: content is list with type "image_url" (e.g. HumanMessage with image)
                if isinstance(content, list):
                    for item in content:
                        if isinstance(item, dict) and item.get("type") == "image_url":
                            return message
            # LangChain message may be object with .content (e.g. HumanMessage)
            if hasattr(message, "content"):
                content = getattr(message, "content", None)
                if isinstance(content, list):
                    for item in content:
                        if isinstance(item, dict) and item.get("type") == "image_url":
                            return message if isinstance(message, dict) else {"content": content}

        return None

    def extract_image_paths_from_message(self, message: Union[Dict, Any]) -> List[Optional[str]]:
        """
        从单条用户消息中解析出所有图片路径（或 url），返回路径列表。
        用于判断当前输入是否包含多图，以及获取本轮用户输入的全部图片。

        Args:
            message: 单条消息，可为 dict（含 "content"）或具 .content 属性的对象。

        Returns:
            当前消息中所有 image_url 对应的路径列表；无图片时返回 []。
        """
        paths: List[Optional[str]] = []
        content = message.get("content") if isinstance(message, dict) else getattr(message, "content", None)
        if content is None:
            return paths
        if isinstance(content, str):
            # 支持一条消息内多行 "image_path: xxx"，全部解析出来，才能正确判断多图并跳过模态识别
            if "image_path:" in content:
                parts = content.split("image_path:")
                for part in parts[1:]:
                    path = part.strip().split("\n")[0].strip() if part.strip() else None
                    if path:
                        paths.append(path)
            return paths
        if isinstance(content, list):
            # 先只从 image_url 块收集路径（用于数量判断）
            for item in content:
                item_type = item.get("type") if isinstance(item, dict) else getattr(item, "type", None)
                url_obj = item.get("image_url") if isinstance(item, dict) else getattr(item, "image_url", None)
                if item_type == "image_url" and url_obj is not None:
                    if isinstance(url_obj, dict):
                        p = url_obj.get("image_path") or url_obj.get("url")
                    else:
                        p = getattr(url_obj, "image_path", None) or getattr(url_obj, "url", None)
                    if p and isinstance(p, str):
                        paths.append(p.strip())
                    else:
                        paths.append(None)
            # 若本条消息没有任何 image_url 块，则从 text 块解析 image_path:（兼容服务端 API 仅用 text 传图的情况）
            # 若有 image_url 块则不再从 text 解析，避免网页端「1 张图 = image_url + text(image_path:)」被重复计为 2 张
            if not paths:
                for item in content:
                    item_type = item.get("type") if isinstance(item, dict) else getattr(item, "type", None)
                    if item_type == "text":
                        raw = (item.get("text") or "") if isinstance(item, dict) else (getattr(item, "text", None) or "")
                        raw = (raw or "").strip()
                        if "image_path:" in raw:
                            parts = raw.split("image_path:")
                            for part in parts[1:]:
                                path = part.strip().split("\n")[0].strip() if part.strip() else None
                                if path:
                                    paths.append(path)
        return paths

    def extract_benchmark_index_from_messages(self, messages: List[Any]) -> Optional[Union[int, str]]:
        """
        从当前轮用户消息中解析 benchmark case 的 index（若请求中包含）。
        兼容：请求中不包含 index 时返回 None，不报错。

        Returns:
            index 值（int 或 str），未找到则 None。
        """
        for msg in (messages or []):
            content = msg.get("content") if isinstance(msg, dict) else getattr(msg, "content", None)
            if content is None:
                continue
            if isinstance(content, list):
                for item in content:
                    if isinstance(item, dict) and item.get("type") == "image_url":
                        url_obj = item.get("image_url")
                        if isinstance(url_obj, dict) and "index" in url_obj:
                            return url_obj["index"]
                    if isinstance(item, dict) and item.get("type") == "text":
                        raw = (item.get("text") or "").strip()
                        if raw.startswith("index:") or raw.startswith("benchmark_index:"):
                            part = raw.split(":", 1)[1].strip().split()[0]
                            try:
                                return int(part)
                            except ValueError:
                                return part
            if isinstance(content, str):
                if "index:" in content:
                    part = content.split("index:")[1].strip().split()[0]
                    try:
                        return int(part)
                    except ValueError:
                        return part
                if "benchmark_index:" in content:
                    part = content.split("benchmark_index:")[1].strip().split()[0]
                    try:
                        return int(part)
                    except ValueError:
                        return part
        return None

    def recognize_intent(self, query: str) -> str:
        """
        Recognize the intent of a user query using the Qwen-3 model.

        Args:
            query (str): The user input query.

        Returns:
            str: The recognized intent.
        """
        # Tokenize the input query
        text = self.intent_recognition_tokenizer.apply_chat_template(
            [
                {"role": "system", "content": self.intent_recognition_prompt},  # System prompt
                {"role": "user", "content": query}  # User query
            ],
            tokenize=False,
            add_generation_prompt=True,
            enable_thinking=True
        )
        model_inputs = self.intent_recognition_tokenizer([text], return_tensors="pt").to(self.intent_recognition_model.device)

        # Conduct text completion
        generated_ids = self.intent_recognition_model.generate(
            **model_inputs,
            max_new_tokens=32768
        )
        output_ids = generated_ids[0][len(model_inputs.input_ids[0]):].tolist()

        # Parse the thinking content
        try:
            # Find the index of the </think> token (151668)
            index = len(output_ids) - output_ids[::-1].index(151668)
        except ValueError:
            index = 0

        # Decode the intent content
        intent_content = self.intent_recognition_tokenizer.decode(output_ids[index:], skip_special_tokens=True).strip("\n")

        return intent_content