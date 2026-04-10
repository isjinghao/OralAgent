"""
OralAgent Application Main Module
This module serves as the entry point for the OralAgent medical imaging AI assistant.
It provides functionality to initialize an AI agent with various medical imaging tools
and launch a web interface for interacting with the system.
The system uses OpenAI's language models for reasoning and can be configured
with different model weights, tools, and parameters.
"""
import base64
import gradio_client.utils as _gcu

_orig_get_type = _gcu.get_type
def _patched_get_type(schema):
    if isinstance(schema, bool):
        return "Any"
    return _orig_get_type(schema)
_gcu.get_type = _patched_get_type

_orig_json = _gcu._json_schema_to_python_type
def _patched_json(schema, defs=None):
    if isinstance(schema, bool):
        return "Any"
    return _orig_json(schema, defs)
_gcu._json_schema_to_python_type = _patched_json

import os
import warnings
from typing import *
from dotenv import load_dotenv
from transformers import logging

from langgraph.checkpoint.memory import MemorySaver
from langchain_openai import ChatOpenAI

from interface import create_demo
from oralagent.agent import *
from oralagent.tools import *
from oralagent.tools.get_tools import get_all_tools_factories, DEFAULT_SELECTED_TOOL_NAMES
from oralagent.utils import *

warnings.filterwarnings("ignore")
logging.set_verbosity_error()
_ = load_dotenv()


def initialize_agent(
    prompt_file,
    tools_to_use=None,
    model_dir="/model-weights",
    temp_dir="temp",
    device="cuda",
    model="chatgpt-4o-latest",
    temperature=0.7,
    top_p=0.95,
    rag_config: Optional[RAGConfig] = None,
    openai_kwargs={}
):
    """Initialize the OralAgent agent with specified tools and configuration.

    Args:
        prompt_file (str): Path to file containing system prompts
        tools_to_use (List[str], optional): List of tool names to initialize. If None, all tools are initialized.
        model_dir (str, optional): Directory containing model weights. Defaults to "/model-weights".
        temp_dir (str, optional): Directory for temporary files. Defaults to "temp".
        device (str, optional): Device to run models on. Defaults to "cuda".
        model (str, optional): Model to use. Defaults to "chatgpt-4o-latest".
        temperature (float, optional): Temperature for the model. Defaults to 0.7.
        top_p (float, optional): Top P for the model. Defaults to 0.95.
        rag_config (RAGConfig, optional): Configuration for the RAG tool. Defaults to None.
        openai_kwargs (dict, optional): Additional keyword arguments for OpenAI API, such as API key and base URL.

    Returns:
        Tuple[Agent, Dict[str, BaseTool]]: Initialized agent and dictionary of tool instances
    """

    # Load system prompts from file
    prompts = load_prompts_from_file(prompt_file)
    system_prompt = prompts["MEDICAL_ASSISTANT"]
    prompt_for_intent_recognition = prompts["INTENT_RECOGNITION_ASSISTANT"]
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

    # Get tool factories from single source of truth (oralagent.tools.get_tools)
    all_tools = get_all_tools_factories(model_dir, temp_dir, device, rag_config)
    tools_to_use = tools_to_use or list(all_tools.keys())
    tools_dict: Dict[str, BaseTool] = {}
    for tool_name in tools_to_use:
        if tool_name in all_tools:
            tools_dict[tool_name] = all_tools[tool_name]()

    # Set up checkpointing for conversation state
    checkpointer = MemorySaver()
    # Initialize the language model
    model = ChatOpenAI(model=model, temperature=temperature, **openai_kwargs)

    intent_classifier_model = BioMedCLIPClassifier(
        checkpoint_path=f"{model_dir}/OralGPT_Modality_Identification_BioMedCLIP_8modalities.pth",
        coco_names_path=f"{model_dir}/categories_Modality_Identification_BioMedCLIP_8modalities.json", 
        num_classes=8
        )

    # Create the agent with the specified model, tools, and configuration
    agent = Agent(
        model,
        intent_classifier_model=intent_classifier_model,
        tools=list(tools_dict.values()),
        log_tools=True,
        log_dir="logs",
        system_prompt=system_prompt,
        intent_recognition_prompt=prompt_for_intent_recognition,
        enriched_query_template=enriched_query_template,
        modality_section_template=modality_section_template,
        checkpointer=checkpointer,
    )

    print("Agent initialized")
    return agent, tools_dict


if __name__ == "__main__":
    """
    This is the main entry point for the OralAgent application.
    It initializes the agent with the selected tools and creates the demo.
    """
    print("Starting server...")

    # Use default tool list from oralagent.tools.get_tools (customize DEFAULT_SELECTED_TOOL_NAMES there)
    selected_tools = DEFAULT_SELECTED_TOOL_NAMES

    ################## for RAG ##################
    # Configure the Retrieval Augmented Generation (RAG) system
    # This allows the agent to access and use medical knowledge documents
    rag_config = RAGConfig(
        model="command-a-03-2025",  # Invalid in current version; TODO: support Qwen3 model
        embedding_model="Qwen/Qwen3-Embedding-8B",  # "0.6B, 4B, 8B"
        rerank_model="Qwen/Qwen3-Reranker-8B",  # "0.6B, 4B, 8B"
        temperature=0.7,
        retriever_k=7,
        persist_dir="oralagent/rag/",  # Base path for vector DB; subdir added when use_OralCorpus=True
        use_OralCorpus=True,  # Set to True to load Oral corpus when creating vectorstore
        corpus_language="chinese",  # "english" -> vectorDB_OralCorpus_English, "chinese" -> vectorDB_OralCorpus_Chinese
        local_docs_dir="",  # Path to Oral corpus (EN or CN) for RAG; also used for custom docs
        chunk_size=1000,  # Only valid for private documents
        chunk_overlap=100,  # Only valid for private documents
    )

    # Prepare OpenAI API configuration from environment variables
    openai_kwargs: Dict[str, str] = {}
    # if api_key := os.getenv("OPENAI_API_KEY"):
    #     openai_kwargs["api_key"] = api_key

    # if base_url := os.getenv("OPENAI_BASE_URL"):
    #     openai_kwargs["base_url"] = base_url

    # Initialize the agent with all configured components
    agent, tools_dict = initialize_agent(
        "oralagent/docs/system_prompts.txt",
        tools_to_use=selected_tools,
        model_dir="/data/OralGPT/OralGPT-expert-model-repository",  # Change this to the path of the model weights
        temp_dir="temp",  # Change this to the path of the temporary directory
        device="cuda",  # Change this to the device you want to use
        model="gpt-5.4",  # Change this to the model you want to use, e.g. gpt-4o-mini
        temperature=0.7,
        # top_p=0.95,
        rag_config=rag_config,
        openai_kwargs=openai_kwargs
    )

    # Create and launch the web interface
    demo = create_demo(agent, tools_dict)

    demo.launch(server_name="0.0.0.0", server_port=8552, share=True)
