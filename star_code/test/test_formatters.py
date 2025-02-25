import unittest
import random
import json
import sys

sys.path.append('../src')


# noqa: E402 - disables the warning for this line
from ollama_manager import STARPromptGenerator  # noqa: E402
import prompt_formatters as pf  # noqa: E402

SEED = 11270525022025
random.seed(SEED)


class PromptFormatterTest(unittest.TestCase):
    def setUp(self):
        input_filename = '../data/datasets/STAR_QA_and_stsg_val.json'

        with open(input_filename, 'r') as in_file:
            q_stsg_data = json.load(in_file)
            idx = random.randint(0, len(q_stsg_data))
            self.sample = q_stsg_data[idx]

    def test_simple_prompt(self):
        prompt_format = "QUESTION: {question}\n"\
                        "SPATIO-TEMPORAL SCENE-GRAPH: {stsg}"

        question = self.sample['question']
        stsg = self.sample['stsg']

        gt_prompt = prompt_format.format(
            question=question,
            stsg=stsg
        )

        test_prompt = pf.OpenEndedPrompt(prompt_format).format(self.sample)

        self.assertEqual(gt_prompt, test_prompt)

    def test_mcq_prompt(self):
        prompt_format = "Q: {question}\n"\
                      "{c1}\n{c2}\n{c3}\n{c4}\n"\
                      "STSG: {stsg}\n"\
                      "A:"

        question = self.sample['question']
        stsg = self.sample['stsg']
        choices = [f"{key}. {val}" for key, val in self.sample['choices'].items()]
        c1, c2, c3, c4 = choices

        gt_prompt = prompt_format.format(
            question=question,
            stsg=stsg,
            c1=c1,
            c2=c2,
            c3=c3,
            c4=c4
        )

        test_prompt = pf.MCQPrompt(prompt_format).format(self.sample)

        self.assertEqual(gt_prompt, test_prompt)


if __name__ == "__main__":
    unittest.main()
