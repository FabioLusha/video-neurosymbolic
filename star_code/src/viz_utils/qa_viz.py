import requests
import webbrowser
import os

def compute_answer_freq(answers):
    """
        answers: pandas.Series
    """
    freq = answers.copy().value_counts().reset_index()
    freq.columns = ["word", "freq"]

    return freq


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

def upload_and_visualize_video(videopath, server_url="http://localhost:10882"):
    """
    Uploads a video to the Django server and opens the browser to visualize it.
    Args:
        videopath (str): Path to the video file to upload.
        server_url (str): Base URL of the Django server.
    """
    upload_url = f"{server_url}/upload/"
    video_title = os.path.basename(videopath)
    with open(videopath, 'rb') as f:
        files = {'file': (video_title, f, 'video/mp4')}
        data = {'title': video_title}
        response = requests.post(upload_url, files=files, data=data)
        if response.status_code == 200 or response.status_code == 302:
            print(f"Video '{video_title}' uploaded successfully.")
            webbrowser.open(server_url)
        else:
            print(f"Failed to upload video. Status code: {response.status_code}")
            print(response.text)
