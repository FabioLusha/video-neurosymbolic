from ollama_manager import STARPromptGenerator, OllamRequestManager
import os

def main():
    # Initialize the prompt generator
    prompt_generator = STARPromptGenerator(
        input_filename='data/datasets/STAR_question_and_stsg.json',
        system_prompt_filename='data/system_prompt.txt'
    )

    
    # Initialize Ollama manager
    OLLAMA_URL = os.getenv('OLLAMA_URL', 'http://localhost:11434')
    ollama = OllamRequestManager(
        base_url=OLLAMA_URL, 
        model='llama3.2')

    # Generate 50 prompts and collect them in a list
    prompts = list(prompt_generator.generate())
    prompt_generator.generate_and_save_prompts(output_dir='outputs')
    # Save prompts and generate responses
    ollama.batch_requests(
        prompts=prompts,
        output_dir='outputs'
    )


if __name__ == "__main__":
    main()
