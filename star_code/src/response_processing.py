import json
import logging
import re
from pathlib import Path
from typing import Callable

import matplotlib.pyplot as plt
import pandas as pd
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

    print(f"\nInvalid answers: {contains_answer.shape[0] - n_valid_answers}")

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

    original_df.loc[:, "answer"] = predictions_df["answer"]
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


def print_acc(eval_df, acc_fn=None):
    if acc_fn is None:
        acc_fn = accuracy
    print(f"{'Question type':<15}{'Total':^15}{'Accuracy':^10}\n")

    results = {}

    total = eval_df.index.str.startswith("Interaction").sum()
    acc = acc_fn(eval_df[eval_df.index.str.startswith("Interaction")])
    results["Interaction"] = acc
    print(f"{'Interaction':<15}{total:^15}{acc:^10.2%}")

    total = eval_df.index.str.startswith("Sequence").sum()
    acc = acc_fn(eval_df[eval_df.index.str.startswith("Sequence")])
    results["Sequence"] = acc
    print(f"{'Sequence':<15}{total:^15}{acc:^10.2%}")

    total = eval_df.index.str.startswith("Prediction").sum()
    acc = acc_fn(eval_df[eval_df.index.str.startswith("Prediction")])
    results["Prediction"] = acc
    print(f"{'Prediction':<15}{total:^15}{acc:^10.2%}")

    total = eval_df.index.str.startswith("Feasibility").sum()
    acc = acc_fn(eval_df[eval_df.index.str.startswith("Feasibility")])
    results["Feasibility"] = acc
    print(f"{'Feasibility':<15}{total:^15}{acc:^10.2%}")

    print()

    total = eval_df.shape[0]
    acc = acc_fn(eval_df)
    results["Average"] = acc
    print(f"{'Average':<15}{total:^15}{acc:^10.2%}")

    return results


def print_ans_perc(eval_df, gt_df):
    print(f"{'Question type':<15}{'Total':^15}{'Answered':^10}\n")

    total = gt_df.index.str.startswith("Interaction").sum()
    results = {}

    acc = (
        len(
            gt_df.index.intersection(
                eval_df[eval_df.index.str.startswith("Interaction")].index
            )
        )
        / total
    )
    results["Interaction"] = acc
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
    results["Sequence"] = acc
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
    results["Prediction"] = acc
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
    results["Feasibility"] = acc
    print(f"{'Feasibility':<15}{total:^15}{acc:^10.2%}")

    total = gt_df.shape[0]
    acc = eval_df.shape[0] / total
    results["Average"] = acc
    print(f"{'Overall':<15}{total:^15}{acc:^10.2%}")

    return results


def plot_acc(eval_df, acc_fn, labels=None):
    """
    Creates a minimal and concise visualization of model accuracy.

    This function is polymorphic: it can accept a single DataFrame or a list of
    DataFrames for `eval_df`. When a list is provided, it plots a grouped
    bar chart for comparison, with sample counts in the legend.

    Args:
        eval_df (pd.DataFrame or list[pd.DataFrame]): The DataFrame(s) with
                                                     evaluation results.
        acc_fn (function): A function that takes a DataFrame and returns the
                           accuracy score as a float.
        labels (list[str], optional): A list of names for the models/dataframes,
                                      corresponding to the DataFrames in eval_df.
                                      Defaults to None.
    """
    # 1. --- Data Preparation ---
    is_list = isinstance(eval_df, list)
    if not is_list:
        eval_df = [eval_df]
        if labels is None:
            labels = ["Model"]

    if labels is None:
        labels = [f"Model {i + 1}" for i in range(len(eval_df))]

    question_types = ["Interaction", "Sequence", "Prediction", "Feasibility"]
    summary_data = []
    average_accuracies = {}
    samples_per_label = {}

    for i, (df, label) in enumerate(zip(eval_df, labels)):
        samples_per_label[label] = len(df)
        for q_type in question_types:
            filtered_df = df[df.index.str.startswith(q_type)]
            accuracy = acc_fn(filtered_df) * 100 if not filtered_df.empty else 0
            summary_data.append(
                {"Question type": q_type, "Accuracy": accuracy, "Label": label}
            )
        average_accuracies[label] = acc_fn(df) * 100 if not df.empty else 0

    summary_df = pd.DataFrame(summary_data)

    # 2. --- Visualization ---
    plt.style.use("seaborn-v0_8-whitegrid")
    fig, ax = plt.subplots(figsize=(16, 9))

    # Use distinct plotting logic for single vs. multiple dataframes
    if is_list:
        # For multiple DFs, use a palette and hue for grouping
        palette = sns.color_palette(n_colors=len(eval_df))
        sns.barplot(
            x="Question type",
            y="Accuracy",
            hue="Label",
            data=summary_df,
            palette=palette,
            ax=ax,
        )
        # Add average accuracy lines matching the hue colors
        for i, (label, avg_acc) in enumerate(average_accuracies.items()):
            ax.axhline(
                y=avg_acc, color=palette[i % len(palette)], linestyle="--", linewidth=1
            )
            ax.text(
                3.6,
                avg_acc,
                f"Avg ({label}): {avg_acc:.2f}%",
                va="center",
                ha="right",
                fontsize=10,
                color=palette[i % len(palette)],
                bbox=dict(
                    boxstyle="round,pad=0.3",
                    fc="white",
                    ec=palette[i % len(palette)],
                    lw=1,
                ),
            )
    else:
        # For a single DF, use the 'color' argument to avoid warnings
        sns.barplot(
            x="Question type", y="Accuracy", data=summary_df, color="Blue", ax=ax
        )
        # Add a single, distinct average accuracy line
        label, avg_acc = list(average_accuracies.items())[0]
        ax.axhline(y=avg_acc, color="#ff0000", linestyle="--", linewidth=1)
        ax.text(
            3.5,
            avg_acc,
            f"Avg: {avg_acc:.2f}%",
            va="center",
            ha="right",
            fontsize=11,
            color="#ff0000",
            bbox=dict(color="#ffffff", edgecolor="#ff0000", pad=0.2),
        )

    # Annotate bars
    for p in ax.patches:
        ax.annotate(
            f"{p.get_height():.1f}%",
            (p.get_x() + p.get_width() / 2.0, p.get_height()),
            ha="center",
            va="bottom",
            fontsize=9,
            textcoords="offset points",
            xytext=(0, 5),
        )

    # 3. --- Polishing ---
    title = "Model Comparison" if is_list else "Model Accuracy"
    ax.text(
        -0.5,
        118,
        f"{title} by Question Type",
        fontsize=20,
        fontweight="bold",
        ha="left",
    )

    if not is_list:
        sample_count = samples_per_label[labels[0]]
        ax.text(
            -0.5,
            110,
            f"Performance across four categories based on {sample_count} answered samples",
            fontsize=14,
            ha="left",
            style="italic",
            color="#666666",
        )

    ax.set_xlabel("")
    ax.set_ylabel("Accuracy", fontsize=12, labelpad=15)
    ax.set_ylim(0, 105)
    ax.set_yticks(range(0, 101, 20))
    ax.set_yticklabels([f"{y}%" for y in range(0, 101, 20)])
    ax.tick_params(axis="x", length=0)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)

    # If multiple models, modify the legend to include sample counts
    if is_list:
        legend = ax.get_legend()
        if legend:
            legend.set_title("Label")
            for i, label in enumerate(labels):
                sample_count = samples_per_label[label]
                legend.texts[i].set_text(f"{label} (n_questions={sample_count})")

    plt.tight_layout()
    plt.show()


def plot_ans_perc(eval_df, gt_df, labels=None):
    """
    Creates an elegant, publication-quality visualization of answered percentage.

    This function is polymorphic: it can accept a single DataFrame or a list of
    DataFrames for `eval_df`. When a list is provided, it plots a grouped
    bar chart for comparison, with sample counts in the legend.

    Args:
        eval_df (pd.DataFrame or list[pd.DataFrame]): The DataFrame(s) with
                                                     evaluation results.
        gt_df (pd.DataFrame): The ground truth DataFrame with all samples.
        labels (list[str], optional): A list of names for the models/dataframes,
                                      corresponding to the DataFrames in eval_df.
                                      Defaults to None.
    """
    # 1. --- Data Preparation ---
    is_list = isinstance(eval_df, list)
    if not is_list:
        eval_df = [eval_df]
        if labels is None:
            labels = ["Model"]

    if labels is None:
        labels = [f"Model {i + 1}" for i in range(len(eval_df))]

    question_types = ["Interaction", "Sequence", "Prediction", "Feasibility"]
    summary_data = []
    average_percentages = {}
    samples_per_label = {}

    for i, (df, label) in enumerate(zip(eval_df, labels)):
        samples_per_label[label] = len(df)
        for q_type in question_types:
            # More robust filtering - handle case sensitivity and exact matching
            gt_mask = gt_df.index.str.startswith(q_type)
            total = gt_mask.sum()
            
            # Ensure df has an index to work with
            if hasattr(df, 'index'):
                eval_mask = df.index.str.startswith(q_type)
                answered_count = eval_mask.sum()
            else:
                # Fallback if df doesn't have proper index
                answered_count = 0
                
            percentage = (answered_count / total) * 100 if total > 0 else 0
            
            # Debug print to help identify the issue
            print(f"Debug - {label}, {q_type}: total={total}, answered={answered_count}, percentage={percentage:.1f}%")
            
            summary_data.append(
                {"Question type": q_type, "Answered": percentage, "Label": label}
            )
        average_percentages[label] = (
            (len(df) / len(gt_df)) * 100 if not gt_df.empty else 0
        )

    summary_df = pd.DataFrame(summary_data)

    # 2. --- Visualization Setup ---
    plt.style.use("seaborn-v0_8-whitegrid")
    # Standard figure width since legend won't be on the side
    fig, ax = plt.subplots(figsize=(16, 9))

    if is_list:
        palette = sns.color_palette(n_colors=len(eval_df))
        sns.barplot(
            x="Question type",
            y="Answered",
            hue="Label",
            data=summary_df,
            palette=palette,
            ax=ax,
        )
    else:
        sns.barplot(
            x="Question type", y="Answered", data=summary_df, color="Green", ax=ax
        )

    # 3. --- Add percentage annotations on bars ---
    for p in ax.patches:
        ax.annotate(
            f"{p.get_height():.1f}%",
            (p.get_x() + p.get_width() / 2.0, p.get_height()),
            ha="center",
            va="bottom",
            fontsize=9,
            textcoords="offset points",
            xytext=(0, 5),
        )

    # 4. --- Create separated average section with grey line ---
    # Add vertical grey line to separate main chart from average section (thinner)
    ax.axvline(x=3.7, color='gray', linestyle='-', linewidth=1, alpha=0.5)

    # Plot average bars in the separated section
    for i, (label, avg_perc) in enumerate(average_percentages.items()):
        if is_list:
            # For multiple models, create small bars showing averages
            bar_width = 0.6 / len(average_percentages) if len(average_percentages) > 1 else 0.3
            x_position = 4.1 + (i - (len(average_percentages) - 1) / 2) * bar_width
            
            # Create average bar with same color as corresponding model
            color = palette[i % len(palette)]
            ax.bar(x_position, avg_perc, width=bar_width, color=color, alpha=0.8)
            
            # Add percentage label on top of average bar
            ax.annotate(
                f"{avg_perc:.1f}%",
                (x_position, avg_perc),
                ha="center",
                va="bottom",
                fontsize=9,
                textcoords="offset points",
                xytext=(0, 5),
            )
        else:
            # For single model, create one average bar
            ax.bar(4.1, avg_perc, width=0.3, color="Green", alpha=0.8)
            ax.annotate(
                f"{avg_perc:.1f}%",
                (4.1, avg_perc),
                ha="center",
                va="bottom",
                fontsize=9,
                textcoords="offset points",
                xytext=(0, 5),
            )

    # 5. --- Title and Subtitle ---
    title = "Percentage of Answered Questions"

    if not is_list:
        ax.text(
            x=0.5,
            y=118,
            s=f"{title} by Type",
            fontsize=16,
            fontweight="bold",
            ha="left",
        )
        sample_count = samples_per_label[labels[0]]
        ax.text(
            -0.5,
            110,
            f"Comparing {sample_count} answered samples against {len(gt_df)} total in the ground truth",
            fontsize=14,
            ha="left",
            style="italic",
            color="#666666",
        )
    else:
        ax.set_title(title, fontsize=16, fontweight='bold', pad=20)

    # 6. --- Axis formatting ---
    ax.set_xlabel('Question Type', fontsize=12)
    ax.set_ylabel("Answered", fontsize=12, labelpad=15)
    ax.set_ylim(0, 105)
    ax.set_yticks(range(0, 101, 20))
    ax.set_yticklabels([f"{y}%" for y in range(0, 101, 20)])
    
    # Extend x-axis to accommodate the average section
    ax.set_xlim(-0.5, 4.5)
    
    # Update x-tick labels to include average section
    original_labels = question_types + ["Average"]
    ax.set_xticks(list(range(len(question_types))) + [4.1])
    ax.set_xticklabels(original_labels)
    
    ax.tick_params(axis="x", length=0)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)

    # 7. --- Legend positioning (top right quadrant with white background) ---
    if is_list:
        legend = ax.get_legend()
        if legend:
            # Get the original legend handles (which contain the color information)
            original_handles = legend.legend_handles
            
            # Remove the old legend first
            legend.remove()
            
            # Create new legend labels with sample counts, preserving original handles
            legend_labels = []
            for i, label in enumerate(labels):
                sample_count = samples_per_label[label]
                legend_labels.append(f"{label} (n_questions={sample_count})")
            
            # Position legend in the top right quadrant with white background
            legend = ax.legend(original_handles, legend_labels, title="Models", 
                             loc='lower right', 
                             frameon=True, fancybox=True, 
                             facecolor='white', edgecolor='gray',
                             framealpha=1.0)

    
    return fig
