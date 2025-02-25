import json


class PromptFormatter:

    def __init__(self, prompt_format):
        self.prompt_format = prompt_format

    def format(self):
        pass


class OpenEndedPrompt(PromptFormatter):

    def format(self, sample):
        args = dict()

        args['question'] = sample['question']
        args['stsg'] = sample['stsg']

        return self.prompt_format.format(**args)


class MCQPrompt(PromptFormatter):

    def format(self, sample):
        args = dict()

        args['question'] = sample['question']

        choices = [f"{key}. {val}" for key, val in sample['choices'].items()]
        args['c1'] = choices[0]
        args['c2'] = choices[1]
        args['c3'] = choices[2]
        args['c4'] = choices[3]

        args['stsg'] = str(sample['stsg'])

        return self.prompt_format.format(**args)


class LlmAsJudgePrompt(PromptFormatter):

    def __init__(self, prompt_format, predictions_path):
        super(prompt_format)

        with open(predictions_path, 'r') as in_f:
            self.predictions = {i['question_id']: i for i in json.load(in_f)}

    def format(self, sample):
        args = dict()

        qid = sample['question_id']

        args['question'] = sample['question']
        args['gt_answer'] = sample['choices'][str(sample['answer'])]

        args['prediction'] = self.predictions[qid]['pred']

        return self.prompt_format.format(**args)
