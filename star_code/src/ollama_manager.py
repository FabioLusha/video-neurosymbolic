import requests
import json
import os
from datetime import datetime


class OllamaRequestManager:

    def __init__(self, model, base_url, **options):
        self.base_url = base_url
        self.model = model
        self.options = options

    def make_request(self, prompt):
        try:
            response = requests.post(
                f'{self.base_url}/api/generate',
                json={
                    'model': self.model,
                    'prompt': prompt,
                    # Set to False to get full response at once
                    'stream': False,
                    'options': self.options
                }
            )

            # Raise an exception for HTTP errors
            response.raise_for_status()

            response_data = response.json()
            return response_data.get('response', '')

        except requests.RequestException as e:
            print(f"Error making request to Ollama server: {e}")
            return None

    def batch_requests(self, prompts, output_dir):

        # Create directory if it doesn't exist
        os.makedirs(output_dir, exist_ok=True)

        # Saving the response as JSON in jsonl format,
        # where each json object is saved in one line
        start_timestamp = datetime.now().strftime("%Y%m%d_%H:%M:%S")
        output_file = os.path.join(
            output_dir, f'responses_{self.model}_{start_timestamp}.jsonl')

        error_file = os.path.join(
            output_dir, f'errors_{self.model}_{start_timestamp}.txt')

        print(f"Responses will be saved to: {output_file}")
        print(f"Errors will be logged to: {error_file}")
        print("============== Starting Response Generation ==============")

        # Using line buffering
        with open(output_file, 'w', buffering=1, encoding='utf-8') as f:

            consecutive_errors = 0
            error = False
            for i, sample in enumerate(prompts, 1):
                id = sample['qid']
                prompt = sample['prompt']

                try:
                    print(f"Generating response for prompt {i}")

                    response = self.make_request(prompt)
                    if response:
                        # Create response object
                        response_obj = {
                            'qid': id,
                            'response': response,
                        }

                        f.write(json.dumps(response_obj,
                                ensure_ascii=True) + '\n')
                        f.flush()

                        error = False
                        consecutive_errors = 0

                except Exception as e:
                    if error:
                        consecutive_errors += 1

                    error = True
                
                    with open(error_file, 'w', buffering=1, encoding='utf-8') as error_f:
                        error_msg = {
                            'qid': id,
                            'prompt': prompt,
                            'error': str(e),
                            'timestamp': datetime.now().isoformat()
                        }

                        error_f.write(
                            json.dumps(error_msg, indent=2, ensure_ascii=False))
                        error_f.flush()

                        print(f"Error at iteration {i}\n"
                              f"Prompt id:{id}\n"
                              f"Look at the log file for specifics on the error"
                              )

                finally:
                    if consecutive_errors > 5:
                        raise Exception(
                            "T_here have been {5} consecutive errors!\n"
                            "The process has stopped!")


class STARPromptGenerator:
    def __init__(self, input_filename, system_prompt_filename=None):
        self.input_filename = input_filename
        self.system_prompt = ''

        if system_prompt_filename:
            self.system_prompt = self._load_system_prompt(
                system_prompt_filename)

    def _load_system_prompt(self, filename):
        try:
            with open(filename) as in_file:
                return in_file.read().strip()

        except IOError as e:
            raise IOError(f"Error reading system prompt file: {e}")

    def generate(self, limit=None):
        try:
            with open(self.input_filename, 'r') as in_file:
                q_stsg_data = json.load(in_file)

                for i, sample in enumerate(q_stsg_data, 1):
                    if limit is not None and i > limit:
                        break

                    prompt = (
                        f"{self.system_prompt}\n\n"
                        f"QUESTION: {sample['question']}\n"
                        f"SPATIO-TEMPORAL SCENE-GRAPH: {str(sample['stsg'])}"
                    )

                    yield {'qid': sample['question_id'], 'prompt': prompt}

        except IOError as e:
            raise IOError(f"Error reading question and stsg file: {e}")

    def generate_and_save_prompts(self, output_dir, limit=None):
        # Create output directory if it doesn't exist
        os.makedirs(output_dir, exist_ok=True)

        # Generate filename with timestamp
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        output_file = os.path.join(
            output_dir, f'prompts_{timestamp}.jsonl')

        try:
            # Open file with line buffering
            with open(output_file, 'w', buffering=1, encoding='utf-8') as f:
                for prompt_data in self.generate(limit):
                    # Write each prompt as a JSON line
                    f.write(json.dumps(prompt_data, ensure_ascii=False) + '\n')
                    f.flush()

                print(f"Prompts saved to {output_file}")

                return

        except IOError as e:
            raise IOError(f"Error saving prompts: {e}")
