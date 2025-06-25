def compact_print_qa(idx, gt_dataset_df, predictors, predictor_labels=None):

    if predictor_labels:
        assert len(predictors) == len(predictor_labels)
    else:
        predictor_labels = [f"Prediction {i}" for i in range(1, len(predictors) + 1)]

    question = gt_dataset_df.loc[idx]["question"]
    gt_answer = gt_dataset_df.loc[idx]["answer"]

    print(f"\n┌─ Sample: {str(idx)} " + "─" * (80 - len(str(idx))))
    print("│")
    print("│ Question:")
    print(f"│    {question}")
    print("│ Alternatives:")
    print(
        "\n".join(
            [
                f"│    {c['choice_id']}. {c['choice']}"
                for c in gt_dataset_df.loc[idx]["choices"]
            ]
        )
    )
    print("│")
    print("│ Ground Truth:")
    print(f"│    {gt_answer}")
    print("│")

    for pred, label in zip(predictors, predictor_labels):
        reasoning = pred.loc[idx]["chat_history"][1]["content"]
        answer = pred.loc[idx]["answer"]

        status = "[CORRECT]" if answer.lower() == gt_answer.lower() else "[WRONG]"
        print("|")
        print(f"│ Model Predictions - {label}:")
        print(f"│    Prediction:  {answer} {status}")
        print("│    Reasoning:")
        print("\n".join([f"│        {line}" for line in reasoning.split("\n")]))
    print("│")
    print("└" + "─" * 85)
