from ollama_manager import STARPromptGenerator, OllamaRequestManager
import os

def main():
    

    # system_prompt = _load_system_prompt('data/system_prompt.txt')
    # prompt_format = "QUESTION: {question}\n"\
    #                 "SPATIO-TEMPORAL SCENE-GRAPH: {stsg}"

    # mcq_system_prompt = _load_system_prompt('data/MCQ_system_prompt_v2.txt')
    # mcq_pformat = "Q: {question}\n"\
    #               "{c1}\n{c2}\n{c3}\n{c4}\n"\
    #               "STSG: {stsg}\n"\
    #               "A:"
    
    mcq_system_prompt = _load_system_prompt('data/MCQ_system_prompt_v3.txt')
    mcq_pformat = "<Question>\n"\
                  "{question}\n"\
                  "Alternatives:\n"\
                  "{c1}\n{c2}\n{c3}\n{c4}\n"\
                  "<\Question>\n"\
                  "<STSG>\n{stsg}\n<\STSG>"
    
    # Initialize Ollama manager
    OLLAMA_URL = os.getenv('OLLAMA_URL', 'http://localhost:11434')
    ollama = OllamaRequestManager(
        base_url=OLLAMA_URL, 
        # model='llama3.1:8b',
        # model='llama3.2',
        # model='phi3:3.8b',
        model='deepseek-r1:7b',
        system=mcq_system_prompt,
        options={
            'num_ctx': 20480,     # increasing the context window
            'temperature': 0.1,   # less createive and more focuesed generation (default: 0.8)
            'num_predict': 10240   # let's check if fixing a number of max output token fixes the bug
        }
        ) 
    
    # Initialize the prompt generator
    prompt_generator = STARPromptGenerator(
        # input_filename='data/datasets/STAR_question_and_stsg.json',    # Generative
        input_filename='data/datasets/STAR_QA_and_stsg_val.json',    # MCQ
    )
    
    # start from where the server crashed (repeat the last generation to test start parm
    # actually works)
    prompts = list(prompt_generator.generate(mcq_pformat, mcq=True))
    # generate responses
    ollama.batch_requests(
        prompts=prompts,
        output_dir='outputs'
    )

def _load_system_prompt(filename):
    try:
        with open(filename) as in_file:
            return in_file.read().strip()

    except IOError as e:
        raise IOError(f"Error reading system prompt file: {e}")

        
if __name__ == "__main__":
    main()
