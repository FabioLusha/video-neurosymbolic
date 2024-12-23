import argparse
import sys
import json
import os
from datetime import datetime
from ollama_manager import STARPromptGenerator


def main():
    # Create the main argument parser
    parser = argparse.ArgumentParser(
        description='STAR Prompt Generator and Ollama Request Manager',
        epilog='Process STAR prompts using a local Ollama server'
    )

    # Add mutually exclusive group for main actions
    action_group = parser.add_mutually_exclusive_group(required=True)

    # Generate Prompts Subcommand
    action_group.add_argument(
        '-g', '--generate-prompts',
        action='store_true',
        help='Generate prompts from input file'
    )

    # Request Responses Subcommand
    action_group.add_argument(
        '-r', '--request-responses',
        action='store_true',
        help='Send requests to Ollama and get responses'
    )

    # Common Arguments
    parser.add_argument(
        '-i', '--input-file',
        required=True,
        help='Input JSON file containing questions and scene graphs'
    )

    parser.add_argument(
        '-s', '--system-prompt',
        help='Optional system prompt file',
        default=None
    )

    # Prompt generation arguments
    parser.add_argument(
        '-n', '--num-prompts',
        type=int,
        default=100,
        help='Number of prompts to generate (default: 100, -1 for all)'
    )

    # Output arguments
    parser.add_argument(
        '-o', '--output-dir',
        default='generated_prompts',
        help='Directory to save generated prompts (default: generated_prompts)'
    )

    # Parse arguments
    args = parser.parse_args()

    try:
        # Prompt Generation Flow
        if args.generate_prompts:
            # Create output directory if it doesn't exist
            os.makedirs(args.output_dir, exist_ok=True)

            # Initialize prompt generator
            prompt_generator = STARPromptGenerator(
                input_filename=args.input_file,
                system_prompt_filename=args.system_prompt
            )

            # Generate prompts
            generated_prompts = prompt_generator.generate_prompts(
                N=args.num_prompts)

            # Reformat prompts to desired output format
            output_prompts = []
            for prompt_dict in generated_prompts:
                for question_id, prompt in prompt_dict.items():
                    output_prompts.append({
                        'id': question_id,
                        'prompt': prompt
                    })

            # Generate unique filename
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            output_filename = os.path.join(
                args.output_dir,
                f'generated_prompts_{timestamp}.json'
            )

            # Save prompts to file
            try:
                with open(output_filename, 'w', encoding='utf-8') as f:
                    json.dump(output_prompts, f, ensure_ascii=False, indent=4)

                print(f"Generated {len(output_prompts)} prompts")
                print(f"Prompts saved to {output_filename}")

            except IOError as e:
                print(f"Error saving prompts: {e}")
                sys.exit(1)

        # Response Request Flow (placeholder - you can implement this similarly)
        elif args.request_responses:
            print("Response request flow not implemented in this version")
            sys.exit(1)

    except Exception as e:
        print(f"An error occurred: {e}")
        sys.exit(1)


if __name__ == "__main__":
    main()
