import json
import random
import sys
import tempfile
import unittest

sys.path.append("../src")


import prompt_formatters as pf  # noqa: E402

SEED = 11270525022025
random.seed(SEED)


class PromptFormatterTest(unittest.TestCase):
    def setUp(self):
        input_filename = "../data/datasets/STAR_QA_and_stsg_val.json"

        with open(input_filename, "r") as in_file:
            q_stsg_data = json.load(in_file)
            idx = random.randint(0, len(q_stsg_data))
            self.sample = q_stsg_data[idx]

    def test_simple_prompt(self):
        prompt_format = "QUESTION: {question}\n" "SPATIO-TEMPORAL SCENE-GRAPH: {stsg}"

        question = self.sample["question"]
        stsg = self.sample["stsg"]

        gt_prompt = prompt_format.format(question=question, stsg=stsg)

        test_prompt = pf.OpenEndedPrompt(prompt_format).format(self.sample)

        self.assertEqual(gt_prompt, test_prompt)

    def test_mcq_prompt(self):
        prompt_format = (
            "Q: {question}\n" "{c1}\n{c2}\n{c3}\n{c4}\n" "STSG: {stsg}\n" "A:"
        )

        question = self.sample["question"]
        stsg = self.sample["stsg"]
        choices = [val for _, val in self.sample["choices"].items()]
        c1, c2, c3, c4 = choices

        gt_prompt = prompt_format.format(
            question=question, stsg=stsg, c1=c1, c2=c2, c3=c3, c4=c4
        )

        test_prompt = pf.MCQPrompt(prompt_format).format(self.sample)

        self.assertEqual(gt_prompt, test_prompt)

    def test_llmjudge(self):
        prompt_format = """\
        Please evaluate the following question-answer pair:
        Question: {question}

        Ground truth correct Answer: 
        [START ANSWER]
        {gt_answer}
        [END ANSWER]

        Predicted Answer:
        [START PREDICTION]
        {prediction}
        [END Prediction]

        Provide your evaluation as a correct/incorrect prediction along with the score where the score is an
        integer value between 0 (fully wrong) and 5 (fully correct). The middle score provides the percentage of
        correctness.
        Please generate the response in the form of a Python dictionary string with keys 'pred', 'score' and
        'reason', where value of 'pred' is a string of 'correct' or 'incorrect', value of 'score' is in INTEGER, not STRING
        and value of 'reason' should provide the reason behind the decision.
        Only provide the Python dictionary string.
        For example, your response should look like this: {{'pred': 'correct', 'score': 4, 'reason': reason}}.\
        """

        question = self.sample["question"]
        pred = "This is the answer prediction."
        answer = self.sample["choices"][str(self.sample["answer"])]

        gt_prompt = prompt_format.format(
            question=question, gt_answer=answer, prediction=pred
        )
        self.sample["answer"] = self.sample["choices"][str(self.sample["answer"])]
        self.sample["response"] = pred
        formatter = pf.LlmAsJudgePrompt(prompt_format)
        test_prompt = formatter.format(self.sample)

        self.assertEqual(gt_prompt, test_prompt)


if __name__ == "__main__":
    unittest.main()
