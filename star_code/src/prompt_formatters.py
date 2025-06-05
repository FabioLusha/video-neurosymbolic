import re
from typing import Dict


class PromptFormatter:

    def __init__(self, prompt_format, fields=None):
        self.prompt_format = prompt_format
        self.fields = fields

    def init_fields(self, sample) -> Dict[str, str]:
        if self.fields:
            args = dict()
            args = {field: sample[field] for field in self.fields}
            return args
        raise NotImplementedError("You need to implement how to extract the fields from the sample.")

    def validate_fields(self, fields):
        pattern = r'\{([a-zA-Z_][a-zA-Z0-9_]*)\}'
        required_fields = set(re.findall(pattern, self.prompt_format))

        if not required_fields:
            return

        missing_fields = required_fields - set(fields.keys())
        if missing_fields:
            missing_list = sorted(missing_fields)
            raise ValueError(
                f"Missing required format fields: {', '.join(missing_list)}"
            )

    def format(self, sample):
        fields = self.init_fields(sample)
        self.validate_fields(fields)

        return self.prompt_format.format(**fields)

class OpenEndedPrompt(PromptFormatter):

    def init_fields(self, sample):
        args = dict()

        args["question"] = sample["question"]
        args["stsg"] = sample.get("stsg", None)

        return args


class MCQPrompt(PromptFormatter):

    def init_fields(self, sample):
        args = dict()

        args["question"] = sample["question"]

        args["c1"] = sample["choices"]["0"]
        args["c2"] = sample["choices"]["1"]
        args["c3"] = sample["choices"]["2"]
        args["c4"] = sample["choices"]["3"]

        args["stsg"] = str(sample["stsg"])

        return args


class MCQPromptWoutSTSG(PromptFormatter):

    def init_fields(self, sample):
        args = dict()

        args["question"] = sample["question"]

        choices = [f"{key}. {val}" for key, val in sample["choices"].items()]
        args["c1"] = sample["choices"]["0"]
        args["c2"] = sample["choices"]["1"]
        args["c3"] = sample["choices"]["2"]
        args["c4"] = sample["choices"]["3"]

        return args


class LlmAsJudgePrompt(PromptFormatter):

    def init_fields(self, sample):
        args = dict()

        qid = sample["question_id"]

        args["question"] = sample["question"]
        args["gt_answer"] = sample["answer"]
        args["prediction"] = sample["response"]

        return args


class LlmAsJudgePromptForMCQ(LlmAsJudgePrompt):
    def init_fields(self, sample):
        args = dict()

        qid = sample["question_id"]

        args["question"] = sample["question"]

        if sample.get("choices", None):
            args["c1"] = sample["choices"]["0"]
            args["c2"] = sample["choices"]["1"]
            args["c3"] = sample["choices"]["2"]
            args["c4"] = sample["choices"]["3"]

        args["gt_answer"] = sample["answer"]
        args["prediction"] = sample["response"]

        return args
