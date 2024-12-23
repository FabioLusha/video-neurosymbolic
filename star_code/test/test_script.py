import json
import sys
sys.path.append('../')


# noqa: E402 - disables the warning for this line
from ollama_manager import STARPromptGenerator, OllamRequestManager  # noqa: E402

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
        input_filename='test_data.json',
        system_prompt_filename='test_system_prompt.txt'
    )

    # Initialize the Ollama manager
    manager = OllamRequestManager(model="llama2")

    # Create output directory
    output_dir = "test_output"

    # Get first 10 prompts
    prompts = []
    for i, prompt in enumerate(generator.generate()):
        if i >= 10:
            break
        prompts.append(prompt)

    # Process the prompts
    manager.batch_requests(prompts, output_dir)


if __name__ == "__main__":
    main()
