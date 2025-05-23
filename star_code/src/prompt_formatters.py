import json
import string


class PromptFormatter:

    def __init__(self, prompt_format):
        self.prompt_format = prompt_format

    def format(self):
        pass


class OpenEndedPrompt(PromptFormatter):

    def format(self, sample):
        args = dict()

        args['question'] = sample['question']
        args['stsg'] = sample.get('stsg', None)

        return self.prompt_format.format(**args)


class MCQPrompt(PromptFormatter):

    def format(self, sample):
        args = dict()

        args['question'] = sample['question']

        args['c1'] = sample['choices']['0']
        args['c2'] = sample['choices']['1']
        args['c3'] = sample['choices']['2']
        args['c4'] = sample['choices']['3']

        args['stsg'] = str(sample['stsg'])

        return self.prompt_format.format(**args)


class MCQPromptWoutSTSG(PromptFormatter):

    def format(self, sample):
        args = dict()

        args['question'] = sample['question']

        choices = [f"{key}. {val}" for key, val in sample['choices'].items()]
        args['c1'] = sample['choices']['0']
        args['c2'] = sample['choices']['1']
        args['c3'] = sample['choices']['2']
        args['c4'] = sample['choices']['3']

        return self.prompt_format.format(**args)


class LlmAsJudgePrompt(PromptFormatter):

    def __init__(self, prompt_format, predictions_path):

        self.prompt_format = prompt_format
        self.predictions = {}
        if not predictions_path.endswith('.jsonl'):
            raise ValueError("File type not supported. Expected a .jsonl file.")

        with open(predictions_path, 'r') as in_f:
            # expect a JSONL file
            for line in in_f.readlines():
                data = json.loads(line)
                key = data.get('qid', None)
                if key is None:
                    key = data.get('question_id')
                    
            self.predictions[key] = data

    def format_user(self, sample):
        args = dict()

        qid = sample['question_id']

        args['question'] = sample['question']
        args['gt_answer'] = sample['answer']

        args['prediction'] = self.predictions[qid]['response']

        return self.prompt_format.format(**args)

    def raw_format(self, sample):
        prompt = [
            '<|begin_of_text|>',
            '<|start_header_id|>system<|end_header_id|>',
            self.format_system(sample),
            '<|eot_id|>',
            '<start_header_id|>user<|end_header_id|>',
            self.format_user(sample),
            '<|eot_id|>',
            '<start_header_id|>assistant<|end_header_id|>'
        ]

        return ''.join(prompt)

    def format(self, sample):
        return self.format_user(sample)

class LlmAsJudgePromptForMCQ(LlmAsJudgePrompt):
    def format_user(self, sample):
        args = dict()

        qid = sample['question_id']

        args['question'] = sample['question']
        
        if sample.get('choices', None):
            args['c1'] = sample['choices']['0']
            args['c2'] = sample['choices']['1']
            args['c3'] = sample['choices']['2']
            args['c4'] = sample['choices']['3']

        args['gt_answer'] = sample['choices'][str(sample['answer'])]
        args['prediction'] = self.predictions[qid]['response']

        return self.prompt_format.format(**args)
