class PromptFormatter:

    def __init__(self, prompt_format):
        self.prompt_format = prompt_format

    def format(self, sample):
        pass


class OpenEndedPrompt(PromptFormatter):

    def format(self, sample):
        args = dict()

        args["question"] = sample["question"]
        args["stsg"] = sample.get("stsg", None)

        return self.prompt_format.format(**args)


class MCQPrompt(PromptFormatter):

    def format(self, sample):
        args = dict()

        args["question"] = sample["question"]

        args["c1"] = sample["choices"]["0"]
        args["c2"] = sample["choices"]["1"]
        args["c3"] = sample["choices"]["2"]
        args["c4"] = sample["choices"]["3"]

        args["stsg"] = str(sample["stsg"])

        return self.prompt_format.format(**args)


class MCQPromptWoutSTSG(PromptFormatter):

    def format(self, sample):
        args = dict()

        args["question"] = sample["question"]

        choices = [f"{key}. {val}" for key, val in sample["choices"].items()]
        args["c1"] = sample["choices"]["0"]
        args["c2"] = sample["choices"]["1"]
        args["c3"] = sample["choices"]["2"]
        args["c4"] = sample["choices"]["3"]

        return self.prompt_format.format(**args)


class LlmAsJudgePrompt(PromptFormatter):

    def format(self, sample):
        args = dict()

        qid = sample["question_id"]

        args["question"] = sample["question"]
        args["gt_answer"] = sample["answer"]
        args["prediction"] = sample["response"]

        return self.prompt_format.format(**args)


class LlmAsJudgePromptForMCQ(LlmAsJudgePrompt):
    def format(self, sample):
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

        return self.prompt_format.format(**args)
