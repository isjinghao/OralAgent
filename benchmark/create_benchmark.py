#!/usr/bin/env python3
"""
Medical X-ray Question Generation Benchmark aka ChestAgentBench

This script generates clinical questions from X-ray case data of Eurorad dataset using GPT-4o.
It structures questions across different analytical categories and saves them as JSON.
"""

import os
import re
import json
from typing import *
from pprint import pprint

import openai
import numpy as np
from scipy import stats
import plotly.graph_objects as go
from tqdm import tqdm

from benchmark.utils import load_eurorad_dataset
from benchmark.llm import get_llm_response

# Constants
DATA_DIR = "set your data directory here, e.g. /home/OralAgent/data"
DATASET_PATH = os.path.join(DATA_DIR, "eurorad_metadata.json")

SYSTEM_PROMPT = """
You are an expert medical benchmark creation assistant.
Your goal is to generate questions that evaluate a multimodal medical AI agent's ability to interpret and reason about chest X-rays.
""".strip()

CATEGORIES_META = {
    "detection": "Identify and locate specific findings in the chest X-ray.",
    "classification": "Determine whether specific findings are present or absent in the chest X-ray.",
    "enumeration": "Count the number of target findings in the chest X-ray.",
    "localization": "Locate a given finding in the chest X-ray.",
    "comparison": "Compare the size or position of a specific finding in the chest X-ray.",
    "relationship": "Determine the relationship between two or more findings in the chest X-ray.",
    "diagnosis": "Make a diagnosis or determine a treatment plan by interpreting the chest X-ray.",
    "characterization": "Describe specific attributes (shape, density, margins, etc.) of findings.",
    "reasoning": "Explain the medical rationale and thought process behind findings and conclusions.",
}
CATEGORIES = list(CATEGORIES_META.keys())

CATEGORY_COMBINATIONS = [
    ["detection", "localization", "characterization", "reasoning"],  # Detailed Finding Analysis
    ["detection", "classification", "relationship", "reasoning"],  # Pattern Recognition & Relations
    ["localization", "comparison", "relationship", "reasoning"],  # Spatial Understanding
    ["classification", "comparison", "diagnosis", "reasoning"],  # Clinical Decision Making
    ["classification", "characterization", "diagnosis", "reasoning"],  # Diagnostic Characterization
]

DEFAULT_SECTIONS = [
    "history",
    "image_finding",
    "discussion",
    "differential_diagnosis",
    "diagnosis",
    "figures",
]


class Question:
    """A class to generate clinical questions from case data.

    This class handles creating structured clinical questions by combining case data with
    specified categories and difficulty levels.

    Attributes:
        type (str): The type of question (e.g. multiple choice)
        difficulty (str): Difficulty level of the question
        case_data (Dict[str, Any]): Dictionary containing the clinical case data
        case_content (str): Formatted case data from selected sections
        case_id (str): Unique identifier for the case
        categories (List[str]): List of analytical categories this question tests
        sections (List[str]): Case sections to include in question
        raw_content (Optional[str]): Raw LLM response to the question prompt
        content (Optional[Dict[str, str]]): Extracted content from the raw LLM response
    """

    def __init__(
        self,
        type: str,
        difficulty: str,
        case_data: Dict[str, Any],
        categories: List[str],
        sections: List[str] = [
            "history",
            "image_finding",
            "discussion",
            "differential_diagnosis",
            "diagnosis",
            "figures",
        ],
        system_prompt: str = "You are an expert medical benchmark creation assistant.",
    ) -> None:
        self.type = type
        self.difficulty = difficulty
        self.case_data = case_data
        self.case_id = case_data["case_id"]
        self.categories = categories
        self.sections = sections
        self.system_prompt = system_prompt
        self.case_content = self.select_case_sections()
        self.raw_content: Optional[str] = None
        self.content: Optional[Dict[str, str]] = None

    def create_question_prompt(self) -> str:
        """Creates a formatted prompt for generating a clinical question.

        Returns:
            str: A structured prompt containing the question parameters and clinical data
        """
        category_descriptions = "\n".join(
            f"{category}: {desc}"
            for category, desc in CATEGORIES_META.items()
            if category in self.categories
        )

        return f"""
        You must follow these guidelines:
        1. Questions must be answerable using only context and chest X-rays.
        - Questions must explicitly mention the referenced figures
        - Questions can only reference the chest X-ray figures

        2. Questions must have unambiguous, verifiable answers, and should:
        - Challenge the agent's analytical capabilities
        - Require multi-step reasoning
        - Test ability to make precise observations
        - Evaluate capability to derive insights and findings from the chest X-ray

        3. The agent has access to tools like classification, report generation, segmentation, grounding, visual question answering, etc. Your question should be complex to require the use of such tools.


        Create a {self.difficulty} {self.type} clinical question that integrates the following:

        {category_descriptions}

        based on the following clinical case:

        {self.case_content}

        Do not use any infomration derived from the CT and MRI images. Do not provide any information and findings about the chest X-rays.
        Your question should require the agent to derive insights and findings from the chest X-ray by itself.
        Your answer should be verifiable directly in the context of the case.
        You can only use the image findings that come from the chest X-ray figures.

        Your response must follow this exact format:
        THOUGHTS: [Think about different reasoning steps and tools the agent should use to answer the question]
        QUESTION: [complete question with relevant context. Incorrect choices should be very close to the correct answer.]
        FIGURES: [list of required figures, e.g. ["Figure 1", "Figure 2a"]]
        EXPLANATION: [short explanation of why your answer is verifiable in the case]
        ANSWER: [correct answer e.g. "A"]
        """.strip().replace(
            "        ", ""
        )  # remove tabs

    def select_case_sections(self) -> str:
        """Extract and format selected sections from case data into paragraphs.

        Returns:
            str: Formatted string with case sections and content
        """
        section_mapping = {
            "history": ("history", "No history provided."),
            "image_finding": ("image_finding", "No findings provided."),
            "discussion": ("discussion", "No discussion provided."),
            "differential_diagnosis": (
                "differential_diagnosis",
                "No differential diagnosis provided.",
            ),
            "diagnosis": ("diagnosis", "No diagnosis provided."),
            "figures": ("figures", "No figures provided."),
        }

        formatted = []
        for section in self.sections:
            if section in section_mapping:
                key, default = section_mapping[section]
                content = self.case_data.get(key, default)

                if key == "figures":
                    figures_text = []
                    for figure in content:
                        for subfig in figure["subfigures"]:
                            figures_text.append(f"{subfig['number']}: {subfig['caption']}")
                    content = "\n".join(figures_text)

                formatted.append(f"{section}:\n{content}")

        return "\n\n".join(formatted)

    def create_question(
        self,
        client: openai.OpenAI,
        temperature: float = 0.7,
        top_p: float = 0.95,
        max_tokens: int = 500,
        model: str = "gpt-4o",
    ) -> str:
        """Create a clinical question using LLM.

        Args:
            client (openai.OpenAI): OpenAI client instance
            temperature (float): Controls randomness in responses. Defaults to 0.7.
            top_p (float): Controls diversity via nucleus sampling. Defaults to 0.95.
            max_tokens (int): Max tokens in model response. Defaults to 500.
            model (str): OpenAI model to use. Defaults to "gpt-4o".

        Returns:
            str: LLM response containing formatted question components
        """
        self.raw_content = get_llm_response(
            client=client,
            prompt=self.create_question_prompt(),
            system_prompt=self.system_prompt,
            temperature=temperature,
            top_p=top_p,
            max_tokens=max_tokens,
            model=model,
        )
        self.content = self.extract_content()

        return self.raw_content

    def extract_content(self) -> Dict[str, str]:
        """Extract sections from raw LLM response using regex patterns.

        Returns:
            Dict[str, str]: Extracted sections including thoughts, question, figures, explanation, and answer
        """
        keywords = ["THOUGHTS", "QUESTION", "FIGURES", "EXPLANATION", "ANSWER"]

        content = {}
        for kw in keywords:
            pattern = rf"{kw}:\s*(.*?)(?=\n[A-Z]+:|$)"
            match = re.search(pattern, self.raw_content, re.DOTALL)
            content[kw.lower()] = match.group(1).strip() if match else None

        return content

    def save(self, output_path: str) -> Dict[str, Any]:
        """Save question content and metadata as a JSON file.

        Args:
            output_path (str): Directory path where the JSON file will be saved

        Returns:
            Dict[str, Any]: Question data including content (thoughts, question, figures, options,
                explanation, answer) and metadata (type, difficulty, categories, etc.)
        """
        question_metadata = self.content.copy()

        # Add metadata
        question_metadata["metadata"] = {
            "case_id": self.case_id,
            "type": self.type,
            "difficulty": self.difficulty,
            "categories": self.categories,
            "sections": self.sections,
        }

        # Create a directory for the case
        case_dir = os.path.join(output_path, str(self.case_id))
        os.makedirs(case_dir, exist_ok=True)

        # Save the question metadata to a JSON file
        output_file = os.path.join(case_dir, f"{self.case_id}_{self.__hash__()}.json")
        with open(output_file, "w") as f:
            json.dump(question_metadata, f, indent=2)

        return question_metadata


def generate_questions(
    dataset: Dict[str, Any],
    client: openai.OpenAI,
    output_dir: str,
    skip_first: int = 100,
    temperature: float = 0.7,
    top_p: float = 0.95,
    max_tokens: int = 1200,
    model: str = "gpt-4o",
) -> None:
    """Generate questions for each case and category combination.

    Args:
        dataset: Dictionary of case data
        client: OpenAI client instance
        output_dir: Directory to save generated questions
        skip_first: Number of initial cases to skip
        temperature: LLM temperature parameter
        top_p: LLM top_p parameter
        max_tokens: Maximum tokens for LLM response
        model: LLM model name
    """
    target_cases = sorted(list(dataset.keys()), key=int)[-len(dataset) : -skip_first]

    for case_id in tqdm(target_cases, desc="Processing cases"):
        case_data = dataset[case_id]

        for category in tqdm(CATEGORY_COMBINATIONS, desc=f"Categories for case {case_id}"):
            question = Question(
                type="multiple choice (A/B/C/D/E/F)",
                difficulty="complex",
                case_data=case_data,
                categories=category,
                sections=DEFAULT_SECTIONS,
                system_prompt=SYSTEM_PROMPT,
            )

            response = question.create_question(
                client=client,
                temperature=temperature,
                top_p=top_p,
                max_tokens=max_tokens,
                model=model,
            )
            question.save(output_dir)


def main():
    """Main execution function."""
    client = openai.OpenAI()

    # Load and verify dataset
    dataset = load_eurorad_dataset(
        DATASET_PATH,
        section="Chest Imaging",
        as_dict=True,
        filter_by_caption=[
            "xray",
            "x-ray",
            "x ray",
            "ray",
            "xr",
            "radiograph",
        ],
    )
    print(f"\n---\nFound {len(dataset)} cases with X-ray mentions\n---\n")

    # Optional: Print sample case for verification
    case_data = dataset["16798"]
    pprint(case_data, sort_dicts=False)

    # Generate questions
    generate_questions(dataset=dataset, client=client, output_dir="benchmark/questions")


if __name__ == "__main__":
    main()
