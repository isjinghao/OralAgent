import argparse
import re
from typing import Dict, Optional, Tuple

import pandas as pd


OPTION_HEAD_PATTERN = re.compile(
    r'^[\s"]*([A-G])[\.、．:：\)]\s*(.*)$', flags=re.IGNORECASE
)


def parse_question_and_options(q_with_options: str) -> Tuple[str, Dict[str, str]]:
    """
    Parse a full string that contains both the question and the options
    into (question, options_dict), where options_dict maps 'A'~'G' to text.

    适配类似下面的格式（单元格内部可能是多行）:
        女，29岁......该患者应诊断为
        A. 根尖周囊肿
        B. 根尖周脓肿
        C. 急性根尖周炎
        D. 根尖周肉芽肿
        E. 根尖周致密性骨炎
        （题目可有 F、G 等更多选项）
    """
    if not isinstance(q_with_options, str):
        q_with_options = "" if pd.isna(q_with_options) else str(q_with_options)

    text = q_with_options.strip()
    if not text:
        return "", {}

    # 统一换行符
    lines = re.split(r"\r?\n", text)

    question_lines = []
    options: Dict[str, str] = {}
    current_opt: Optional[str] = None

    for raw_line in lines:
        line = raw_line.rstrip()
        m = OPTION_HEAD_PATTERN.match(line)
        if m:
            # 新的选项头
            opt_label = m.group(1).upper()
            opt_text = m.group(2).strip()
            options[opt_label] = opt_text
            current_opt = opt_label
        else:
            # 不是新的选项头，如果已经在某个选项里，则视为该选项的续行
            if current_opt is not None:
                if line.strip():
                    options[current_opt] = (options[current_opt] + " " + line.strip()).strip()
            else:
                question_lines.append(line)

    question = "\n".join(l for l in question_lines if l.strip()).strip()
    return question, options


def convert_excel_to_tsv(
    input_path: str,
    output_path: str,
    id_col: str,
    category_col: Optional[str],
    qa_col: str,
    answer_col: str,
    split: str,
    sheet_name: Optional[str] = None,
) -> None:
    """
    Convert an Excel file of multiple-choice questions into a TSV file
    compatible with VLMEvalKit's ImageMCQ-style datasets (text-only).

    Output TSV columns (MedQ-Bench MCQ 风格, 纯文本版):
      - index: integer, unique per row
      - question: question text only
      - A~G: option texts (某些题目选项数更少, 则对应列为空)
      - answer: correct answer label(s), e.g. 'A' or 'AC'
      - category
      - split
    """
    read_kwargs = {}
    if sheet_name is not None:
        read_kwargs["sheet_name"] = sheet_name

    df = pd.read_excel(input_path, **read_kwargs)

    if id_col not in df.columns:
        raise ValueError(f"ID column '{id_col}' not found in Excel columns: {list(df.columns)}")
    if qa_col not in df.columns:
        raise ValueError(f"QA/Question column '{qa_col}' not found in Excel columns: {list(df.columns)}")
    if answer_col not in df.columns:
        raise ValueError(f"Answer column '{answer_col}' not found in Excel columns: {list(df.columns)}")
    if category_col is not None and category_col not in df.columns:
        raise ValueError(
            f"Category column '{category_col}' not found in Excel columns: {list(df.columns)}"
        )

    records = []
    for idx, row in df.iterrows():
        full_q = row[qa_col]
        question, options_dict = parse_question_and_options(full_q)

        answer_val = row[answer_col]
        if pd.isna(answer_val):
            answer_str = ""
        else:
            answer_str = str(answer_val).strip()

        if category_col is not None:
            category_val = row[category_col]
            category = "" if pd.isna(category_val) else str(category_val).strip()
        else:
            category = ""

        record = {
            "index": int(row[id_col]) if not pd.isna(row[id_col]) else int(idx),
            "question": question,
        }

        # 选项列 A~G，若不存在则置为空字符串
        for opt_label in ["A", "B", "C", "D", "E", "F", "G"]:
            record[opt_label] = options_dict.get(opt_label, "")

        record.update(
            {
                "answer": answer_str,
                "category": category,
                "split": split,
            }
        )
        records.append(record)

    out_df = pd.DataFrame.from_records(records)
    out_df.to_csv(output_path, sep="\t", index=False, encoding="utf-8")


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Convert an Excel file of oral multiple-choice questions "
            "into a VLMEvalKit-compatible TSV file (ImageMCQ-style)."
        )
    )
    parser.add_argument("--input", "-i", required=True, help="Path to input Excel file.")
    parser.add_argument("--output", "-o", required=True, help="Path to output TSV file.")
    parser.add_argument(
        "--sheet-name",
        help="Optional sheet name in the Excel file (default: first sheet).",
    )
    parser.add_argument(
        "--id-col",
        default="Number",
        help="Column name for unique ID / 数据id (default: 'id').",
    )
    parser.add_argument(
        "--category-col",
        default="Classification",
        help="Column name for category / 分类 (default: None, meaning empty category).",
    )
    parser.add_argument(
        "--qa-col",
        default="Questions",
        help=(
            "Column name that contains the full question string, "
            "including options / 问题+选项整体字符串 (default: 'question')."
        ),
    )
    parser.add_argument(
        "--answer-col",
        default="Answer",
        help="Column name for the ground-truth answer label(s) / 答案 (default: 'answer').",
    )
    parser.add_argument(
        "--split",
        default="test",
        help="Split name to write into TSV (e.g. 'val', 'test', 'train'). Default: 'val'.",
    )
    return parser


def main() -> None:
    parser = build_arg_parser()
    args = parser.parse_args()

    convert_excel_to_tsv(
        input_path=args.input,
        output_path=args.output,
        id_col=args.id_col,
        category_col=args.category_col,
        qa_col=args.qa_col,
        answer_col=args.answer_col,
        split=args.split,
        sheet_name=args.sheet_name,
    )


if __name__ == "__main__":
    main()

