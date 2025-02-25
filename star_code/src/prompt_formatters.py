import json


class PromptFormatter:

    def __init__(self, prompt_format):
        self.prompt_format = prompt_format

    def format(self):
        pass


class OpeEndedPrompt(PromptFormatter):

    def format(self, sample):
        args = dict()

        args['question'] = sample['question']
        args['stsg'] = sample['stsg']


class MCQPrompt(PromptFormatter):

    def format(self, sample):
        args = dict()

        args['question'] = sample['question']

        choices = [f"{key}. {val}" for key, val in sample['choices'].items()]
        args['c1'] = choices['c1']
        args['c2'] = choices['c2']
        args['c3'] = choices['c3']
        args['c4'] = choices['c4']

        args['stsg'] = str(sample['stsg'])

        self.prompt_format.format(**args)
        return self.prompt_format


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

        self.prompt.format(**args)
        return self.prompt
