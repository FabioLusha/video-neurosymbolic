import traceback
import os
from datetime import datetime
import logging

import requests
import json


class OllamaRequestManager:

    def __init__(self, base_url, ollama_params, verbosity=None):
        self.base_url = base_url
        self.model = ollama_params['model']

        # Setting some of the params to pass to ollama
        self.ollama_params = ollama_params

        # TODO: implement logging functionality
        self.verbosity = verbosity or int(os.getenv('DEBUG_LEVEL', 0))
        logger = self._setup_logger()

    def _setup_logger(self):
        # TODO:
        # map verbosity to logging levels?

        class ScaffoldLogger():
            def log(self, message):
                print(message)
                logger = logging.getLogger(self.__class__.__name__)

            def info(self, msg):
                self.log(msg)

            def debug(self, msg):
                self.log(msg)

            def warning(self, msg):
                self.log(msg)

        return ScaffoldLogger()

    def make_request(self, prompt):
        try:
            # adding the prompt param to the other ollama_params
            self.ollama_params['prompt'] = prompt

            server_response = requests.post(
                f'{self.base_url}/api/generate',
                json=self.ollama_params,
                timeout=300,
                stream=True
            )

            # Raise an exception for HTTP errors
            server_response.raise_for_status()

            llm_generated_txt = []
            for chunk in server_response.iter_lines():
                # Filter out keep-alive chunks
                if chunk: 
                    data = json.loads(chunk)

                    token = data.get('response', '')
                    llm_generated_txt.append(token)
                    print(''.join(llm_generated_txt), '\r')

        except requests.RequestException as e:
            response_sofar = ''.join(llm_generated_txt)
            e.response = response_sofar
            raise

    return '.'.join(llm_generated_txt)

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
        with open(output_file, 'w', buffering=1, encoding='utf-8') as res_f:

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

                        res_f.write(json.dumps(response_obj) + '\n')
                        res_f.flush()

                        error = False
                        consecutive_errors = 0

                except requests.RequestException as e:
                    if error:
                        consecutive_errors += 1

                    error = True
                    with open(error_file, 'a', buffering=1, encoding='utf-8') as error_f:
                        response = getattr(e, 'response')

                        error_msg = {
                            'qid': id,
                            'prompt': prompt,
                            'response': response,
                            'error': traceback.format_exc(),
                            'timestamp': datetime.now().isoformat()
                        }

                        error_f.write(json.dumps(error_msg, indent=2, ensure_ascii=False) + '\n')
                        error_f.flush()

                        response_file_msg = {
                            'qid':id,
                            'response': "CAREFUL! THE FOLLOWING RESPONSE GENERATED AN ERROR.\n" + response
                        }

                        res_f.write(json.dumps(response_file_err_msg) + '\n')
                        res_f.flush()

                        print(f"Error at iteration {i}\n"
                              f"Prompt id:{id}\n"
                              f"Look at the log file for specifics on the error"
                             )

                finally:
                    if consecutive_errors > 5:
                        raise Exception("There have been {5} consecutive errors!\n"
                                        "The process has stopped!")


class STARPromptGenerator:

    def __init__(self, input_filename):
        if not os.path.exists(input_filename):
            raise OSError(f"No such file or directory: '{input_filename}'")

        self.input_filename = input_filename

    def generate(self, prompt_format, start=0, limit=None, mcq=False):
        """
        Args:
            prompt_format (str): a string wich needs to have the two idenifier {question} and {stsg}

            start (int): from which sample to start generation

            limit (int): how many prompt to generate

            mcq (boolean): specifies if we need to use the MCQ prompt


        Returns:
            prompt
        """
        try:
            with open(self.input_filename, 'r') as in_file:
                q_stsg_data = json.load(in_file)

                for i, sample in enumerate(q_stsg_data, 1):
                    if i < start:
                        continue
                    if limit and i > (limit + start):
                        break

                    if mcq:
                        choices = [f"{key}. {val}" for key, val in sample['choices'].items()]
                        c1, c2, c3, c4 = choices
                        prompt = prompt_format.format(
                            question=sample['question'],
                            c1      =c1,
                            c2      =c2,
                            c3      =c3,
                            c4      =c4,
                            stsg    =str(sample['stsg'])
                        )
                    else:
                        prompt = prompt_format.format(
                            question=sample['question'],
                            stsg    =str(sample['stsg'])
                        )

                    yield {'qid': sample['question_id'], 'prompt': prompt}

        except IOError as e:
            raise IOError(f"Error reading question and stsg file: {e}") from e

    def generate_and_save_prompts(self, output_file, prompt_format, mcq=False, limit=None):
        
        # Create output directory if it doesn't exist
        os.makedirs(os.path.dirname(output_file), exist_ok=True)

        try:
            # Open file with line buffering
            with open(output_file, 'w', buffering=1, encoding='utf-8') as f:
                for prompt_data in self.generate(prompt_format, limit, mcq):
                    
                    f.write(json.dumps(prompt_data, ensure_ascii=False) + '\n')
                    f.flush()

                print(f"Prompts saved to {output_file}")

                return True

        except IOError as e:
            raise IOError(f"Error saving prompts") from e
