import json
import sys
sys.path.append('../src')


# noqa: E402 - disables the warning for this line
from ollama_manager import STARPromptGenerator, OllamaRequestManager  # noqa: E402

# Test data - creating a small JSON file with sample data
test_data = [
    {
        "question_id": f"q{i}",
        "question": f"Test question {i}",
        "stsg": f"Test graph {i}"
    } for i in range(10)
]

# Save test data
with open('test_data.json', 'w') as f:
    json.dump(test_data, f)

# Create a simple system prompt
with open('test_system_prompt.txt', 'w') as f:
    f.write("You are a helpful assistant.")

# Test script


def main():
    # Initialize the generator
    generator = STARPromptGenerator(
        input_filename='test_data.json'
    )

    prompt_format = "QUESTION: {question}\n"\
                    "STSG: {stsg}"
    
    with open('test_system_prompt.txt', 'r') as f:
          prompt_format = '\n'.join([f.read(), prompt_format])
          
    # Initialize the Ollama manager
    manager = OllamaRequestManager(
        model='llama2',
        base_url='http://localhost:11434')

    # Create output directory
    output_dir = "test_output"

    # Get first 10 prompts
    prompts = []
    for i, prompt in enumerate(generator.generate(prompt_format)):
        if i >= 10:
            break
        prompts.append(prompt)

    # Process the prompts
    manager.batch_requests(prompts, output_dir)


if __name__ == "__main__":
    main()
