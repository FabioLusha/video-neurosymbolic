import json
import os
from torch.utils.data import Dataset


class PromptDataset(Dataset):
    def __init__(
        self,
        qa_file_path,
        prompt_formatter,
        stsg_file_path=None,
        ids=None,
        limit=None,
    ):
        if not os.path.exists(qa_file_path):
            raise OSError(f"No such file or directory: '{qa_file_path}'")
        self.qa_file_path = qa_file_path
        self.stsg_file_path = stsg_file_path
        self.prompt_formatter = prompt_formatter

        # Load QA data
        self.qa = self._load_qa_file()

        # Get question ID key (auto-detect between 'qid' and 'question_id')
        if len(self.qa) > 0:
            self.q_id_key = "qid" if "qid" in self.qa[0] else "question_id"
        # Set in _load_stsg_data
        self.stsg_id_key = None
        # Filter by IDs if provided
        if ids:
            self.qa = [q for q in self.qa if q[self.q_id_key] in ids]

        # Apply limit if provided
        if limit:
            self.qa = self.qa[:limit]

        # Load STSG data (video_id -> stsg mapping)
        self.stsgs = {}
        if self.stsg_file_path:
            self._load_stsg_data()

        self.preprocess()
        self.print_stats()

    def print_stats(self):
        """Print statistics about the dataset."""
        print("\nDataset Statistics:")
        print("=" * 40)

        # QA stats
        print(f"QA File: {os.path.basename(self.qa_file_path)}")
        print(f"Number of QA samples: {len(self.qa)}")

        if len(self.qa) > 0:
            # Print example keys in QA data
            sample_keys = list(self.qa[0].keys())
            print(f"QA sample keys: {', '.join(sample_keys)}")

        # STSG stats
        if self.stsg_file_path:
            print(f"\nSTSG File: {os.path.basename(self.stsg_file_path)}")
            print(f"Number of unique video IDs with STSG: {len(self.stsgs)}")

            if len(self.stsgs) > 0:
                # Print example of first video_id and STSG keys if available
                first_vid = next(iter(self.stsgs))
                if isinstance(self.stsgs[first_vid], dict):
                    stsg_keys = list(self.stsgs[first_vid].keys())
                    print(f"STSG keys: {', '.join(stsg_keys)}")

        print("=" * 40 + "\n")

    def load_jsons(self, filepath):
        """Load a JSON or JSONL file."""
        ext = os.path.splitext(filepath)[1].lower()
        with open(filepath, "r") as f:
            if ext == ".json":
                return json.load(f)
            elif ext in (".jsonl", ".ndjson"):
                return [json.loads(line) for line in f]
            else:
                raise IOError(f"{self.qa_file_path} must be either JSON or JSONL")

    def _load_qa_file(self):
        """Load QA data from JSON or JSONL file."""
        ext = os.path.splitext(self.qa_file_path)[1].lower()
        with open(self.qa_file_path, "r") as f:
            if ext == ".json":
                return json.load(f)
            elif ext in (".jsonl", ".ndjson"):
                return [json.loads(line) for line in f]
            else:
                raise IOError(f"{self.qa_file_path} must be either JSON or JSONL")

    def _load_stsg_data(self):
        """Load all STSG data into memory (video_id -> stsg dict)."""
        if not os.path.exists(self.stsg_file_path):
            raise OSError(f"STSG file not found: {self.stsg_file_path}")

        data = self.load_jsons(self.stsg_file_path)
        for item in data:
            try:
                if "question_id" in item.keys():
                    self.stsg_id_key = "question_id"
                elif "video_id" in item.keys():
                    self.stsg_id_key = "video_id"
                else:
                    raise ValueError(
                        "Expected 'question_id' or 'video_id' as id in the STSG file"
                    )

                id = item.get(self.stsg_id_key)
                stsg = item.get("stsg")
                # equivalent:
                # if video_id is not None and stsg is not None
                if id and stsg:
                    self.stsgs[id] = stsg
            except json.JSONDecodeError:
                continue

    def preprocess(self):
        pass

    def __len__(self):
        return len(self.qa)

    def __getitem__(self, idx):
        if idx >= len(self):
            raise IndexError

        sample = self.qa[idx]
        question_id = sample.get(self.q_id_key)
        video_id = sample.get("video_id")

        # Add STSG to sample if available
        if video_id and video_id in self.stsgs:
            sample["stsg"] = self.stsgs[video_id]

        sample["qid"] = sample["question_id"]
        sample["prompt"] = self.prompt_formatter.format(sample)

        return sample


class STARDataset(PromptDataset):

    def _load_stsg_data(self):
        """Load all STSG data into memory (video_id -> stsg dict)."""
        if not os.path.exists(self.stsg_file_path):
            raise OSError(f"STSG file not found: {self.stsg_file_path}")

        data = self.load_jsons(self.stsg_file_path)
        for item in data:
            try:
                if "question_id" in item.keys():
                    self.stsg_id_key = "question_id"

                    id = item.get("question_id")
                    stsg = item.get("stsg")

                    self.stsgs[id] = stsg
                elif "video_id" in item.keys():
                    self.stsg_id_key = "video_id"

                    video_id = item.get("video_id")
                    stsg = item.get("stsg")
                    if video_id is not None and stsg is not None:
                        # if the key is not present in the dict is initialized with the empty
                        # list and then append the new value.
                        # If the key exists setdefault returns the value (the list), to which we
                        # append the new element
                        self.stsgs.setdefault(video_id, []).append(
                            {
                                "stsg": stsg,
                                "start": item.get("start", None),
                                "end": item.get("end", None),
                            }
                        )
                else:
                    raise ValueError(
                        "Expected 'question_id' or 'video_id' as id in the STSG file"
                    )

            except json.JSONDecodeError:
                continue

    def preprocess(self):
        """If a stsg file is specified, than filter out the qa without an stsg"""
        filtered_qas = []
        if self.stsg_file_path:
            for sample in self.qa:
                if self.stsg_id_key == "question_id":
                    if isinstance(sample["choices"][0], dict):
                        sample["stsg"] = self.stsgs[sample.get("question_id")]
                        choices = {
                            str(choice["choice_id"]): choice["choice"]
                            for choice in sample["choices"]
                        }
                        sample["choices"] = choices
                elif self.stsg_id_key == "video_id":
                    video_id = sample.get("video_id")
                    start = sample.get("start")
                    end = sample.get("end")

                    if video_id in self.stsgs:
                        # Look inside all the sub-clip of the video for the one referenced
                        # by the question (i.e. matching video_id, start, and end)
                        for situation in self.stsgs[video_id]:
                            if situation["start"] == start and situation["end"] == end:
                                sample["stsg"] = situation["stsg"]

                                # check if the choices attribute is a string or the dict
                                # used in STAR
                                if isinstance(sample["choices"][0], dict):
                                    sample["choices"] = {
                                        str(choice["choice_id"]): choice["choice"]
                                        for choice in sample["choices"]
                                    }

                                filtered_qas.append(sample)
                    else:
                        # TODO: warn qa is not associated to a stsg
                        continue
        # self.qa updated with filtered_qas if filtered is not null
        # i.e. 'question_id' => filtered_qas == [] --> (A => B) == (not A) or B
        assert not (self.stsg_id_key == "question_id") or (filtered_qas == [])
        self.qa = filtered_qas or self.qa
        return

    def __getitem__(self, idx):
        if idx >= len(self):
            raise IndexError

        sample = self.qa[idx]

        sample["qid"] = sample[self.q_id_key]  # question_id
        sample["prompt"] = self.prompt_formatter.format(sample)
        return sample


class CVRRDataset(PromptDataset):

    def preprocess(self):
        for item in self.qa:
            item["question_id"] = item.pop("unique_id")
            item["question"] = item.pop("Q")
            item["video_id"] = item.pop("video_path").split(".")[0]
            item["answer"] = item.pop("A")


class JudgeDataset(PromptDataset):

    def __init__(self, dataset, predictions_filepath, prompt_formatter):
        self._wrapped = dataset

        data = self.load_jsons(predictions_filepath)
        predictions = {}
        for pred in data:
            key = pred.get("qid", None)
            if key is None:
                key = pred.get("question_id")

            predictions[key] = pred

        for sample in self.qa:
            sample["gt_answer"] = sample["answer"]
            sample["response"] = predictions[sample["question_id"]]

        self._wrapped.prompt_formatter = prompt_formatter

    def __getattr__(self, name):
        return getattr(self._wrapped, name)

    def __setattr__(self, name, value):
        if name == "_wrapped":
            object.__setattr__(self, name, value)
        else:
            setattr(self._wrapped, name, value) 