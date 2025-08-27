import json
import logging
import re
from pathlib import Path
from typing import Callable

import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

from deprecated import deprecated

logger = logging.getLogger("data_preprocessing")

MODEL_PREPROCESSING: dict[str, Callable] = {}
FILE_PREPARATION: dict[str, Callable] = {}  # Framework {ollama, gemini} -> function


def register(name: str, container: dict[str, Callable]):
    def decorator(fn: Callable):
        container[name] = fn
        return fn

    return decorator


@register("gemma3", MODEL_PREPROCESSING)
def _gemma3_preprocessing(predictions_df):
    # For Gemma we need to be more careful becuase the format is different, it encapsulated the json output in the with the tokens:
    # ```
    # ```json\n
    # <actual_answer>
    # \n```
    # ```

    json_pattern = r"^\s*(?:```json\s)?({[^}]+})(?:\s```)?"
    json_mask = predictions_df["answer"].str.match(json_pattern, flags=re.DOTALL)
    predictions_df["json_mask"] = json_mask
    matches_json_template = json_mask.sum()

    print(f"Total answers: {len(predictions_df)}")
    print(f"Answers following JSON template: {matches_json_template}")
    print(
        f"Percentage following JSON template: {(matches_json_template / len(predictions_df)) * 100:.2f}%"
    )

    predictions_df = predictions_df.loc[json_mask].copy()
    predictions_df["answer"] = predictions_df["answer"].apply(
        lambda x: re.search(json_pattern, x).group(1)
    )

    # ---------------- Removing enoding errors
    # Replace new line (lead to EOF Errors) with whitespace
    predictions_df["answer"] = (
        predictions_df["answer"]
        .str.replace("\n+", " ", regex=True)
        .str.replace(
            "[\u2018-\u201b]", "'", regex=True
        )  # Replace left and right quotation mark with simple quotation mark
        .str.replace("[\u201c\u201d]", '"', regex=True)
    )

    # ------------------ Removing inner double quotes --------------------
    # It may happen that the text may contain inner double quotes before the
    # attribute end. This will cause the parser to termiate early and spout
    # errors for the remaining text. With this snippet we replace those inner
    # double quotes with single quotes.
    #
    # we first match the text of the reason paramter inside the double quotes
    # then we escape/replace all the double quotes inside the text
    inside_doublequotes = r"(?<=\"answer\": \")(.*)(?=\"(?:,|}))"

    predictions_df["answer"] = predictions_df.apply(
        func=lambda row: re.sub(
            inside_doublequotes,
            lambda matchobj: matchobj.group(0).replace('"', ""),
            row["answer"],
        ),
        axis=1,
    )

    return predictions_df


@register("qwen2.5vl", MODEL_PREPROCESSING)
def _qwen25_preprocessing(predictions_df):
    json_pattern = r"^(?:```json\s)?({[^}]+})(?:\s*```)?"
    json_mask = predictions_df["answer"].str.match(json_pattern)
    predictions_df["json_mask"] = json_mask
    matches_json_template = json_mask.sum()

    print(f"Total answers: {len(predictions_df)}")
    print(f"Answers following JSON template: {matches_json_template}")
    print(
        f"Percentage following JSON template: {(matches_json_template / len(predictions_df)) * 100:.2f}%"
    )

    predictions_df = predictions_df.loc[json_mask].copy()
    predictions_df["answer"] = predictions_df["answer"].apply(
        lambda x: re.search(json_pattern, x).group(1)
    )

    return predictions_df


@register("ollama", FILE_PREPARATION)
def _ollama_file_prepartion(input_filepath):
    input_filepath = Path(input_filepath)

    predictions = []
    with open(input_filepath, mode="r", encoding="utf-8", errors="strict") as f:
        predictions = [json.loads(line) for line in f.readlines()]

    # transforming the id key from `qid` to `id` for consistency and `response` to `answer`
    predictions_df = pd.DataFrame(predictions, dtype="string").rename(
        columns={"qid": "id", "response": "answer"}
    )

    predictions_df["chat_history"] = predictions_df["chat_history"].apply(
        lambda x: eval(x)
    )

    return predictions_df


@register("gemini", FILE_PREPARATION)
def _gemini_file_preparation(input_filepath):
    input_filepath = Path(input_filepath)

    predictions = []
    with open(input_filepath, mode="r", encoding="utf-8", errors="strict") as f:
        predictions = [json.loads(line) for line in f.readlines()]

    new_compatible_pred = []
    for pred in predictions:
        new_pred = {}
        new_pred["id"] = pred["key"]
        new_pred["chat_history"] = []

        for content in pred["request"]["contents"]:
            text = []
            for part in content["parts"]:
                if "text" in part.keys():
                    text.append("\n" + part["text"])

            text = "".join(text)
            new_pred["chat_history"].append({"role": content["role"], "content": text})

        new_compatible_pred.append(new_pred)

    return pd.DataFrame(new_compatible_pred)


def ans_extract(input_filepath, model, format="ollama"):
    file_preparator = FILE_PREPARATION[format]
    predictions_df = file_preparator(input_filepath)

    predictions_df = predictions_df.drop_duplicates(subset=["id"])
    predictions_df.set_index("id", inplace=True)

    original_df = predictions_df[["chat_history"]].copy()
    # the final answer is contained in the last message
    # responded by the assistant
    predictions_df["answer"] = predictions_df["chat_history"].apply(
        lambda x: x[-1]["content"]
    )

    preprocess = MODEL_PREPROCESSING[model]
    predictions_df = preprocess(predictions_df)

    # -------------- Extracting answers

    def safe_eval(x):
        try:
            evaluated = eval(x)
            return evaluated
        except (SyntaxError, NameError, KeyError, AttributeError):
            # Handle the exception and return a default value
            logger.warning(f"Error executing eval on: {x}.")
            return None

    predictions_df["answer"] = predictions_df["answer"].apply(
        lambda x: result["answer"].strip() if (result := safe_eval(x)) else ""
    )

    ans_regex_pattern = r"^(?:[A-Z]\.)\s+((?:\w+(?:\s|\/)?){,10}\.?)"
    contains_answer = predictions_df["answer"].str.contains(
        ans_regex_pattern, regex=True
    )

    n_valid_answers = contains_answer.sum()
    print(
        f"Answer with a valid alternative: {n_valid_answers}\n"
        f"{n_valid_answers / predictions_df.shape[0]:.2%} of the total"
    )

    print(
        f"\nInvalid answers: {contains_answer.shape[0] - n_valid_answers }"
    )

    ans_df = (
        predictions_df[contains_answer]["answer"]
        .apply(lambda x: re.findall(ans_regex_pattern, x)[-1])
        .apply(lambda x: x + "." if not x.endswith(".") else x)
        .to_frame(name="answer")
    )

    ans_df["answer"] = ans_df["answer"].str.strip()

    original_df.loc[:, "contains_answer"] = False
    original_df.loc[contains_answer[contains_answer].index, "contains_answer"] = (
        contains_answer
    )

    original_df.loc[:, "answer"] = ""
    original_df.loc[contains_answer[contains_answer].index, "answer"] = ans_df["answer"]

    return original_df


@deprecated('Use ans_extract(model="gemma3"')
def gemma3_ans_extract(input_filepath):
    input_filepath = Path(input_filepath)

    predictions = []
    with open(input_filepath, mode="r", encoding="utf-8", errors="strict") as f:
        predictions = [json.loads(line) for line in f.readlines()]

    # transforming the id key from `qid` to `id` for consistency and `response` to `answer`
    predictions_df = pd.DataFrame(predictions, dtype="string").rename(
        columns={"qid": "id", "response": "answer"}
    )

    predictions_df = predictions_df.drop_duplicates(subset=["id"])
    predictions_df.set_index("id", inplace=True)

    predictions_df["chat_history"] = predictions_df["chat_history"].apply(
        lambda x: eval(x)
    )

    original_df = predictions_df[["chat_history"]].copy()
    # the final answer is contained in the last message
    # responded by the assistant
    predictions_df["answer"] = predictions_df["chat_history"].apply(
        lambda x: x[-1]["content"]
    )

    predictions_df = _gemma3_preprocessing(predictions_df)

    # -------------- Extracting answers

    predictions_df["answer"] = predictions_df["answer"].apply(
        lambda x: eval(x)["answer"].strip()
    )

    ans_regex_pattern = r"^(?:[A-Z]\.)\s+((?:\w+(?:\s|\/)?){,10}\.?)"
    contains_answer = predictions_df["answer"].str.contains(
        ans_regex_pattern, regex=True
    )

    print(
        f"Answer following the template: {contains_answer.value_counts()[True]}\n"
        f"{contains_answer.value_counts()[True] / predictions_df.shape[0]:.2%} of the total"
    )

    print(
        f"\nOnly {contains_answer.shape[0] - contains_answer.value_counts()[True]} samples do not contain the answer in the response with the specified format"
    )

    ans_df = (
        predictions_df[contains_answer]["answer"]
        .apply(lambda x: re.findall(ans_regex_pattern, x)[-1])
        .apply(lambda x: x + "." if not x.endswith(".") else x)
        .to_frame(name="answer")
    )

    ans_df["answer"] = ans_df["answer"].str.strip()

    original_df.loc[:, "contains_answer"] = False
    original_df.loc[contains_answer[contains_answer].index, "contains_answer"] = (
        contains_answer
    )

    original_df.loc[:, "answer"] = ""
    original_df.loc[contains_answer[contains_answer].index, "answer"] = ans_df["answer"]

    return original_df


def accuracy(eval_df, on_what="text"):
    hits_text = (
        eval_df[f"pred_{on_what}"].str.lower() == eval_df[on_what].str.lower()
    ).sum()

    return hits_text / eval_df.shape[0]


def print_acc(eval_df, acc_fn):
    print(f"{'Question type':<15}{'Total':^15}{'Accuracy':^10}\n")

    total = eval_df.index.str.startswith("Interaction").sum()
    acc = acc_fn(eval_df[eval_df.index.str.startswith("Interaction")])
    print(f"{'Interaction':<15}{total:^15}{acc:^10.2%}")

    total = eval_df.index.str.startswith("Sequence").sum()
    acc = acc_fn(eval_df[eval_df.index.str.startswith("Sequence")])
    print(f"{'Sequence':<15}{total:^15}{acc:^10.2%}")

    total = eval_df.index.str.startswith("Prediction").sum()
    acc = acc_fn(eval_df[eval_df.index.str.startswith("Prediction")])
    print(f"{'Prediction':<15}{total:^15}{acc:^10.2%}")

    total = eval_df.index.str.startswith("Feasibility").sum()
    acc = acc_fn(eval_df[eval_df.index.str.startswith("Feasibility")])
    print(f"{'Feasibility':<15}{total:^15}{acc:^10.2%}")

    print()
    total = eval_df.shape[0]
    acc = acc_fn(eval_df)
    print(f"{'Average':<15}{total:^15}{acc:^10.2%}")


def print_ans_perc(eval_df, gt_df):
    print(f"{'Question type':<15}{'Total':^15}{'Answered':^10}\n")

    total = gt_df.index.str.startswith("Interaction").sum()
    acc = (
        len(
            gt_df.index.intersection(
                eval_df[eval_df.index.str.startswith("Interaction")].index
            )
        )
        / total
    )
    print(f"{'Interaction':<15}{total:^15}{acc:^10.2%}")

    total = gt_df.index.str.startswith("Sequence").sum()
    acc = (
        len(
            gt_df.index.intersection(
                eval_df[eval_df.index.str.startswith("Sequence")].index
            )
        )
        / total
    )
    print(f"{'Sequence':<15}{total:^15}{acc:^10.2%}")

    total = gt_df.index.str.startswith("Prediction").sum()
    acc = (
        len(
            gt_df.index.intersection(
                eval_df[eval_df.index.str.startswith("Prediction")].index
            )
        )
        / total
    )
    print(f"{'Prediction':<15}{total:^15}{acc:^10.2%}")

    total = gt_df.index.str.startswith("Feasibility").sum()
    acc = (
        len(
            gt_df.index.intersection(
                eval_df[eval_df.index.str.startswith("Feasibility")].index
            )
        )
        / total
    )
    print(f"{'Feasibility':<15}{total:^15}{acc:^10.2%}")

    total = gt_df.shape[0]
    acc = eval_df.shape[0] / total
    print(f"{'Overall':<15}{total:^15}{acc:^10.2%}")


def plot_acc(eval_df, acc_fn):
    """
    Creates a minimal and concise visualization of model accuracy.

    Args:
        eval_df (pd.DataFrame): The DataFrame with evaluation results.
        acc_fn (function): A function that takes a DataFrame and returns the
                           accuracy score as a float.
    """
    # 1. --- Data Preparation ---
    question_types = ["Interaction", "Sequence", "Prediction", "Feasibility"]
    summary_data = []
    for q_type in question_types:
        filtered_df = eval_df[eval_df.index.str.startswith(q_type)]
        accuracy = acc_fn(filtered_df) * 100 if not filtered_df.empty else 0
        summary_data.append({"Question type": q_type, "Accuracy": accuracy})

    summary_df = pd.DataFrame(summary_data)
    average_accuracy = acc_fn(eval_df) * 100 if not eval_df.empty else 0

    # 2. --- Visualization ---
    plt.style.use('seaborn-v0_8-whitegrid') # Use a clean, minimal style
    fig, ax = plt.subplots(figsize=(10, 6))

    # Use a gradient color palette
    barplot = sns.barplot(
        x='Question type',
        y='Accuracy',
        data=summary_df,
        color='Blue', # Use a dark blue gradient
        ax=ax
    )

    # Add average accuracy line
    ax.axhline(y=average_accuracy, color='#ff0000', linestyle='--', linewidth=1)
    ax.text(
        3.5, average_accuracy, # x, y positions
        f'Avg: {average_accuracy:.2f}%',
        va='center', ha='right', fontsize=11, color='#ff0000',
        bbox=dict(color="#ffffff", edgecolor='#ff0000', pad=0.2)
    )

    # Annotate bars
    for p in ax.patches:
        ax.annotate(f"{p.get_height():.2f}%",
                    (p.get_x() + p.get_width() / 2., p.get_height()),
                    ha='center', va='bottom', fontsize=10,
                    textcoords='offset points', xytext=(0, 5))

    # 3. --- Polishing ---
    ax.text(-0.5, 118, 'Model Accuracy by Question Type', fontsize=20, fontweight='bold', ha='left')
    ax.text(-0.5, 110, f'Performance across four categories based on {len(eval_df)} answered samples',
            fontsize=14, ha='left', style='italic', color='#666666')

    ax.set_xlabel('') # The category names are self-explanatory
    ax.set_ylabel('Accuracy', fontsize=12, labelpad=15)
    ax.set_ylim(0, 105)
    ax.set_yticks(range(0, 101, 20))
    ax.set_yticklabels([f'{y}%' for y in range(0, 101, 20)])
    ax.tick_params(axis='x', length=0) # Remove x-axis ticks

    # Remove unnecessary spines and ticks
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.tick_params(axis='x', length=0)

    plt.tight_layout()
    plt.show()


def plot_ans_perc(eval_df, gt_df):
    """
    Creates an elegant, publication-quality visualization of answered percentage,
    inspired by the aesthetics of Anthropic and distill.pub.

    Args:
        eval_df (pd.DataFrame): DataFrame with answered samples.
        gt_df (pd.DataFrame): Ground truth DataFrame with all samples.
    """
    # 1. --- Data Preparation ---
    question_types = ["Interaction", "Sequence", "Prediction", "Feasibility"]
    summary_data = []
    for q_type in question_types:
        total = gt_df.index.str.startswith(q_type).sum()
        answered_count = len(eval_df[eval_df.index.str.startswith(q_type)])
        percentage = (answered_count / total) * 100 if total > 0 else 0
        summary_data.append({"Question type": q_type, "Answered": percentage})

    overall_percentage = (len(eval_df) / len(gt_df)) * 100 if not gt_df.empty else 0
    summary_df = pd.DataFrame(summary_data)
    
    # 2. --- Style and Color Definition ---
    # A different but equally professional palette
    plt.style.use('default')
    
    # 3. --- Visualization ---
    fig, ax = plt.subplots(figsize=(12, 7))

    barplot = sns.barplot(
        x='Question type', y='Answered', color="Blue", data=summary_df, ax=ax
    )
    
    ax.axhline(y=overall_percentage, color="#ff0000", linestyle='--', linewidth=1.5)
    
    ax.text(
        3.5, overall_percentage,
        f'Avg: {overall_percentage:.2f}%  ',
        va='center', ha='right', fontsize=12, color='#ff0000',
        bbox=dict(color="#ffffff", edgecolor='#ff0000', pad=0.2)
    )

    for p in ax.patches:
        ax.annotate(
            f"{p.get_height():.2f}%", (p.get_x() + p.get_width() / 2., p.get_height()),
            ha='center', va='center', xytext=(0, 10), textcoords='offset points',
            fontsize=11, fontweight='medium'
        )

    # 4. --- Polishing ---
    ax.text(-0.5, 118, 'Percentage of Questions Answered by Type', fontsize=20, fontweight='bold', ha='left')
    ax.text(-0.5, 110, f'Comparing {len(eval_df)} answered samples against {len(gt_df)} total in the ground truth',
            fontsize=14, ha='left', style='italic', color='#666666')

    ax.set_xlabel('')
    ax.set_ylabel('Answered', fontsize=12, labelpad=15)
    ax.set_ylim(0, 105)
    ax.set_yticks(range(0, 101, 20))
    ax.set_yticklabels([f'{y}%' for y in range(0, 101, 20)])
    ax.tick_params(axis='x', length=0)
    
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.grid(axis='y', linestyle='--', linewidth=0.5, alpha=0.7)
    
    plt.tight_layout(pad=2)
    plt.show()
