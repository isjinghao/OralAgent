import os
import json
from typing import Dict, List


def load_prompts_from_file(file_path: str) -> Dict[str, str]:
    """
    Load multiple prompts from a file.

    Args:
    file_path (str): Path to the file containing prompts.

    Returns:
    Dict[str, str]: A dictionary of prompt names and their content.

    Raises:
    FileNotFoundError: If the specified file is not found.
    """
    if not os.path.exists(file_path):
        raise FileNotFoundError(f"Prompts file not found: {file_path}")

    prompts = {}
    current_prompt = None
    current_content = []

    with open(file_path, "r") as file:
        for line in file:
            line = line.strip()
            if line.startswith("[") and line.endswith("]"):
                if current_prompt:
                    prompts[current_prompt] = "\n".join(current_content).strip()
                current_prompt = line[1:-1]
                current_content = []
            elif line:
                current_content.append(line)

    if current_prompt:
        prompts[current_prompt] = "\n".join(current_content).strip()

    return prompts


def load_tool_prompts(tools: List[str], tools_json_path: str) -> str:
    """
    Load prompts for specified tools from the tools.json file.

    Args:
    tools (List[str]): List of tool names to load prompts for.
    tools_json_path (str): Path to the tools.json file.

    Returns:
    str: A string containing prompts for the specified tools.

    Raises:
    FileNotFoundError: If the tools.json file is not found.
    """
    if not os.path.exists(tools_json_path):
        raise FileNotFoundError(f"Tools JSON file not found: {tools_json_path}")

    with open(tools_json_path, "r") as file:
        tools_data = json.load(file)

    tool_prompts = []
    for tool in tools:
        if tool in tools_data:
            tool_info = tools_data[tool]
            tool_prompt = f"Tool: {tool}\n"
            tool_prompt += f"Description: {tool_info['description']}\n"
            tool_prompt += f"Usage: {tool_info['prompt']}\n"
            tool_prompt += f"Input type: {tool_info['input_type']}\n"
            tool_prompt += f"Return type: {tool_info['return_type']}\n\n"
            tool_prompts.append(tool_prompt)

    return "\n".join(tool_prompts)


def load_system_prompt(
    system_prompts_file: str,
    system_prompt_type: str,
    tools: List[str],
    tools_json_path: str,
) -> str:
    """
    Load the system prompt by combining the system prompt and tool information.

    Args:
    system_prompts_file (str): Path to the file containing system prompts.
    system_prompt_type (str): The type of system prompt to use.
    tools (List[str]): List of tool names to include in the prompt.
    tools_json_path (str): Path to the tools.json file.

    Returns:
    str: The system prompt combining system prompt and tool information.
    """
    prompts = load_prompts_from_file(system_prompts_file)
    system_prompt = prompts.get(system_prompt_type, "GENERAL_ASSISTANT")
    tool_prompts = load_tool_prompts(tools, tools_json_path)

    return f"{system_prompt}\n\nTools:\n{tool_prompts}".strip()


def parse_bool_env(name: str, default: bool) -> bool:
    """
    Parse a boolean-like environment variable.

    Accepted truthy values: 1/true/t/yes/y/on (case-insensitive).
    Everything else is treated as false.
    """
    val = os.getenv(name)
    if val is None:
        return default
    return val.strip().lower() in ("1", "true", "t", "yes", "y", "on")


def apply_enriched_query_ablation(
    enriched_query_template: str,
    *,
    include_intent_recognition: bool,
    include_modality_section: bool,
) -> str:
    """
    Ablate the enriched query prompt by conditionally removing specific lines.

    - Remove the line starting with `Intent recognition:` when `include_intent_recognition` is False.
    - Remove any line containing `{modality_section}` when `include_modality_section` is False.
    """
    if not enriched_query_template:
        return enriched_query_template

    kept_lines: List[str] = []
    for line in enriched_query_template.splitlines():
        stripped = line.strip()
        if not include_intent_recognition and stripped.startswith("Intent recognition:"):
            continue
        if not include_modality_section and "{modality_section}" in stripped:
            continue
        kept_lines.append(line)

    return "\n".join(kept_lines).strip()
