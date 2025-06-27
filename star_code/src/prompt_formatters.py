import re
from typing import Dict


class PromptFormatter:

    # TODO: Partially string substitution is not allowed for str.format
    # Migrate to string.Template
    def __init__(self, prompt_format, fields=None, required_fields=None):
        self.prompt_format = prompt_format
        self.fields = fields
        self.field_values = {}
        self._required_fields = required_fields or self._init_required_fields()

    def init_fields(self, sample):
        if self.fields:
            self.field_values = {field: sample[field] for field in self.fields}
        return

    def _init_required_fields(self):
        pattern = r'\{([a-zA-Z_][a-zA-Z0-9_]*)\}'
        required_fields = set(re.findall(pattern, self.prompt_format))

        return required_fields

    def validate_fields(self, fields):
        if not self._required_fields:
            return

        missing_fields = set(self._required_fields) - set(fields.keys())
        if missing_fields:
            missing_list = sorted(missing_fields)
            raise ValueError(
                f"Missing required format fields: {', '.join(missing_list)}"
            )

    def format(self, sample):
        self.init_fields(sample)
        self.validate_fields(self.field_values)

        return self.prompt_format.format(**self.field_values)

class OpenEndedPrompt(PromptFormatter):

    def init_fields(self, sample):
        args = dict()

        args["question"] = sample["question"]
        args["stsg"] = sample.get("stsg", None)

        self.field_values = args

class MCQPrompt(PromptFormatter):

    def init_fields(self, sample):
        args = dict()

        args["question"] = sample["question"]

        args["c1"] = sample["choices"]["0"]
        args["c2"] = sample["choices"]["1"]
        args["c3"] = sample["choices"]["2"]
        args["c4"] = sample["choices"]["3"]

        args["stsg"] = str(sample["stsg"])

        self.field_values = args


class MCQPromptWoutSTSG(PromptFormatter):

    def init_fields(self, sample):
        args = dict()

        args["question"] = sample["question"]

        args["c1"] = sample["choices"]["0"]
        args["c2"] = sample["choices"]["1"]
        args["c3"] = sample["choices"]["2"]
        args["c4"] = sample["choices"]["3"]

        self.field_values = args


class LlmAsJudgePrompt(PromptFormatter):

    def init_fields(self, sample):
        args = dict()

        args["question"] = sample["question"]
        args["gt_answer"] = sample["answer"]
        args["prediction"] = sample["response"]

        self.field_values = args


class LlmAsJudgePromptForMCQ(LlmAsJudgePrompt):
    def init_fields(self, sample):
        args = dict()

        args["question"] = sample["question"]

        if sample.get("choices", None):
            args["c1"] = sample["choices"]["0"]
            args["c2"] = sample["choices"]["1"]
            args["c3"] = sample["choices"]["2"]
            args["c4"] = sample["choices"]["3"]

        args["gt_answer"] = sample["answer"]
        args["prediction"] = sample["response"]

        self.field_values = args


class ImgPromptDecorator:

    def __init__(self, wrapped_formatter, img_field="images", tag="[img]"):
        self.wrapped_formatter = wrapped_formatter
        self.img_field = img_field
        self.tag = tag

    def format(self, sample):
        n_imgs = len(sample[self.img_field])
        images_text = "".join([f"\n\nImage {i}:\n{self.tag}" for i in range(n_imgs)])

        self.wrapped_formatter.field_values[self.img_field] = images_text
        
        return self.wrapped_formatter.format(sample)
        
    def __getattr__(self, name):
        return getattr(self.wrapped_formatter, name)
