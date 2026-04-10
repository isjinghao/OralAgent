"""
OralGPT-Omni: Pseudo tool that returns pre-computed results from an Excel table.

OralGPT-Omni is a dental versatile multimodal large language model across multiple
oral modalities. This tool does not run the model; it looks up offline-computed
results by data index from an Excel file for efficient agent execution.
"""

from typing import Any, Dict, Optional, Tuple, Type

import pandas as pd
from pathlib import Path
from pydantic import BaseModel, Field
from langchain_core.callbacks import AsyncCallbackManagerForToolRun, CallbackManagerForToolRun
from langchain_core.tools import BaseTool


# ---------------------------------------------------------------------------
# Input / Output schemas (readable names for agent and downstream code)
# ---------------------------------------------------------------------------

class OralGPTOmniLookupInput(BaseModel):
    """Input schema for the OralGPT-Omni lookup tool."""

    data_index: int = Field(
        ...,
        description="Index of the data sample to look up (must match the index column in the pre-computed Excel)."
    )


# ---------------------------------------------------------------------------
# Tool implementation: lookup by index from Excel, return pre-computed result
# ---------------------------------------------------------------------------

# Default Excel column names (user can override via constructor if their Excel uses different names)
DEFAULT_INDEX_COLUMN_NAME = "index"
DEFAULT_MODEL_OUTPUT_COLUMN_NAME = "prediction"


class OralGPTOmniTool(BaseTool):
    """
    Pseudo tool for OralGPT-Omni: returns pre-computed model outputs from an Excel table by data index.

    OralGPT-Omni is a dental versatile multimodal large language model across multiple dental imaging modalities.
    Results are pre-computed offline and stored in an Excel file; this tool only performs lookup
    for efficient agent runs without re-executing the model.
    """

    name: str = "OralGPT-Omni"
    description: str = (
        "Looks up pre-computed OralGPT-Omni model output by data index from an Excel table. "
        "Input the data_index that corresponds to the row in the pre-computed results Excel."
    )

    args_schema: Type[BaseModel] = OralGPTOmniLookupInput

    # Path to the Excel file containing index and model output columns
    excel_path: str = ""
    # Column name in Excel for the data index (row identifier)
    index_column_name: str = DEFAULT_INDEX_COLUMN_NAME
    # Column name in Excel for the model output to return
    model_output_column_name: str = DEFAULT_MODEL_OUTPUT_COLUMN_NAME
    # In-memory cache of the DataFrame to avoid repeated file reads
    _results_dataframe: Optional[pd.DataFrame] = None

    def __init__(
        self,
        excel_path: str,
        index_column_name: str = DEFAULT_INDEX_COLUMN_NAME,
        model_output_column_name: str = DEFAULT_MODEL_OUTPUT_COLUMN_NAME,
        load_excel_on_init: bool = True,
    ):
        """
        Initialize the OralGPT-Omni lookup tool.

        Args:
            excel_path: Path to the Excel file with columns for data index and model output.
            index_column_name: Name of the column that stores the data index (default: "index").
            model_output_column_name: Name of the column that stores the model output (default: "model_output").
            load_excel_on_init: If True, load the Excel into memory at init for faster lookups.
        """
        super().__init__()
        self.excel_path = excel_path
        self.index_column_name = index_column_name
        self.model_output_column_name = model_output_column_name
        if load_excel_on_init and excel_path and Path(excel_path).exists():
            self._results_dataframe = self._load_excel(excel_path)

    def _load_excel(self, path: str) -> pd.DataFrame:
        """Load the pre-computed results Excel into a DataFrame."""
        path_obj = Path(path)
        if not path_obj.exists():
            raise FileNotFoundError(f"OralGPT-Omni results Excel not found: {path}")
        if path_obj.suffix.lower() in (".xlsx", ".xls"):
            df = pd.read_excel(path)
        else:
            # Support .csv as well for flexibility
            df = pd.read_csv(path)
        return df

    def _get_dataframe(self) -> pd.DataFrame:
        """Return the results DataFrame, loading from file if not yet cached."""
        if self._results_dataframe is not None:
            return self._results_dataframe
        self._results_dataframe = self._load_excel(self.excel_path)
        return self._results_dataframe

    def _run(
        self,
        data_index: int,
        run_manager: Optional[CallbackManagerForToolRun] = None,
    ) -> Tuple[Dict[str, Any], Dict[str, Any]]:
        """
        Look up the pre-computed model output for the given data index.

        Returns:
            Tuple of (output_dict, metadata_dict) consistent with other OralAgent tools.
        """
        try:
            df = self._get_dataframe()

            if self.index_column_name not in df.columns:
                error_msg = (
                    f"Excel missing index column '{self.index_column_name}'. "
                    f"Available columns: {list(df.columns)}"
                )
                return (
                    {"error": error_msg},
                    {
                        "data_index": data_index,
                        "excel_path": self.excel_path,
                        "analysis_status": "failed",
                        "error_reason": "missing_index_column",
                    },
                )
            if self.model_output_column_name not in df.columns:
                error_msg = (
                    f"Excel missing model output column '{self.model_output_column_name}'. "
                    f"Available columns: {list(df.columns)}"
                )
                return (
                    {"error": error_msg},
                    {
                        "data_index": data_index,
                        "excel_path": self.excel_path,
                        "analysis_status": "failed",
                        "error_reason": "missing_output_column",
                    },
                )

            # Match row by index (allow int or string comparison for flexibility)
            mask = df[self.index_column_name].astype(str) == str(data_index)
            if not mask.any():
                return (
                    {
                        "error": f"No row found for data_index={data_index} in Excel.",
                        "data_index": data_index,
                    },
                    {
                        "data_index": data_index,
                        "excel_path": self.excel_path,
                        "analysis_status": "failed",
                        "error_reason": "index_not_found",
                    },
                )

            row = df.loc[mask].iloc[0]
            model_output_value = row[self.model_output_column_name]
            # Coerce to native Python type for JSON-friendly output
            if pd.isna(model_output_value):
                model_output_value = ""
            elif hasattr(model_output_value, "item"):
                model_output_value = model_output_value.item()
            else:
                model_output_value = str(model_output_value) if not isinstance(model_output_value, (str, int, float, bool)) else model_output_value

            output = {
                "model_output": model_output_value,
                "data_index": data_index,
                "source": "oralgpt_omni_precomputed",
            }
            metadata = {
                "data_index": data_index,
                "excel_path": self.excel_path,
                "analysis_status": "completed",
                "source_tool": "oralgpt_omni",
            }
            return output, metadata

        except Exception as e:
            return (
                {"error": str(e), "data_index": data_index},
                {
                    "data_index": data_index,
                    "excel_path": self.excel_path,
                    "analysis_status": "failed",
                    "error_reason": "exception",
                    "error_detail": str(e),
                },
            )

    async def _arun(
        self,
        data_index: int,
        run_manager: Optional[AsyncCallbackManagerForToolRun] = None,
    ) -> Tuple[Dict[str, Any], Dict[str, Any]]:
        """Async version: same as _run (lookup is I/O bound and fast)."""
        return self._run(data_index, run_manager)
