import requests
import webbrowser
import os
from pathlib import Path

import base64
import io
from PIL import Image
import matplotlib.pyplot as plt



from src import video_tools

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


def vis_video_frames(data, raw_video_dir, save_video_dir, fps=1):

    start = round(data['start'], 2) # start time
    end = round(data['end'], 2) # end time
    video_id = data['video_id']

    in_path = raw_video_dir / f"{video_id}.mp4"
    out_path = save_video_dir / f"{data['question_id']}.mp4"

    print('\tVideo Seg: ', str(start) + 's', '-', str(end) + 's')
    frames = video_tools.generate_video_frames(in_path, fps, start, end)
    if not frames:
        print("No frames to display.")
        return

    num_frames = len(frames)
    fig, axes = plt.subplots(1, num_frames, figsize=(num_frames * 3, 3))
    for frame_data, ax in zip(frames, axes):
        img_data = base64.b64decode(frame_data["encoding"])
        img = Image.open(io.BytesIO(img_data))
        ax.imshow(img)
        ax.axis("off")
        ax.set_title(f"Frame {frame_data['frame_id']}")

    plt.tight_layout()
    plt.show()
