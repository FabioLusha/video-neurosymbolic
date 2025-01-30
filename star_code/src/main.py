from ollama_manager import STARPromptGenerator, OllamaRequestManager
import os

def main():
    

    system_prompt = _load_system_prompt('data/system_prompt.txt')
    prompt_format = "QUESTION: {question}\n"\
                    "SPATIO-TEMPORAL SCENE-GRAPH: {stsg}"

    #prompt_format = system_prompt + '\n' + prompt_format

    # Initialize Ollama manager
    OLLAMA_URL = os.getenv('OLLAMA_URL', 'http://localhost:11434')
    ollama = OllamaRequestManager(
        base_url=OLLAMA_URL, 
        model='llama3.2',
        num_ctx=8192,     # increasing the context window
        temperature=0.1,   # less createive and more focuesed generation (default: 0.8)
        num_predict=128,  # limits the number of tokens the LLM can generate as response -> useful to not fill context window
        system=system_prompt
        ) 
    
    # Initialize the prompt generator
    prompt_generator = STARPromptGenerator(
        input_filename='data/datasets/STAR_question_and_stsg.json',
    )
    
    prompts = list(prompt_generator.generate(prompt_format))
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
