#!/usr/bin/env python3
"""
Run inference with the fine-tuned GPT OSS 20B model.
Generate human-like responses to given prompts.
"""

import argparse
import yaml
from pathlib import Path
from unsloth import FastLanguageModel
import torch


def load_config(config_path: str) -> dict:
    """Load configuration from YAML file."""
    with open(config_path, 'r') as f:
        return yaml.safe_load(f)


def format_prompt(instruction: str, prompt_template: str) -> str:
    """Format instruction with the prompt template."""
    # Remove response section for inference
    template_parts = prompt_template.split("### Response:")
    return template_parts[0].format(instruction=instruction) + "### Response:\n"


def generate_response(model, tokenizer, prompt: str, config: dict) -> str:
    """Generate a response for the given prompt."""
    # Tokenize input
    inputs = tokenizer(prompt, return_tensors="pt").to(model.device)
    
    # Generate response
    with torch.no_grad():
        outputs = model.generate(
            **inputs,
            max_new_tokens=config['generation']['max_new_tokens'],
            temperature=config['generation']['temperature'],
            top_p=config['generation']['top_p'],
            top_k=config['generation']['top_k'],
            repetition_penalty=config['generation']['repetition_penalty'],
            do_sample=True,
        )
    
    # Decode and return response
    response = tokenizer.decode(outputs[0], skip_special_tokens=True)
    
    # Extract only the generated part (after the prompt)
    if "### Response:" in response:
        response = response.split("### Response:")[-1].strip()
    
    return response


def interactive_mode(model, tokenizer, config: dict):
    """Run in interactive mode for continuous prompting."""
    print("\n" + "=" * 80)
    print("Interactive Mode - Enter your prompts (type 'quit' or 'exit' to stop)")
    print("=" * 80 + "\n")
    
    while True:
        try:
            instruction = input("\nYour instruction: ").strip()
            
            if instruction.lower() in ['quit', 'exit', 'q']:
                print("\nExiting interactive mode...")
                break
            
            if not instruction:
                continue
            
            # Format prompt and generate response
            prompt = format_prompt(instruction, config['dataset']['prompt_template'])
            print("\nGenerating response...")
            response = generate_response(model, tokenizer, prompt, config)
            
            print("\n" + "-" * 80)
            print("Response:")
            print("-" * 80)
            print(response)
            print("-" * 80)
            
        except KeyboardInterrupt:
            print("\n\nExiting interactive mode...")
            break
        except Exception as e:
            print(f"\nError: {e}")


def main(args):
    # Load configuration
    config = load_config(args.config)
    
    print("=" * 80)
    print("GPT OSS 20B Inference with Unsloth")
    print("=" * 80)
    
    # Load fine-tuned model
    model_path = args.model_path
    print(f"\nLoading fine-tuned model from: {model_path}")
    
    model, tokenizer = FastLanguageModel.from_pretrained(
        model_name=model_path,
        max_seq_length=config['model']['max_seq_length'],
        dtype=config['model']['dtype'],
        load_in_4bit=args.load_in_4bit,
    )
    
    # Set model to inference mode
    FastLanguageModel.for_inference(model)
    
    print("Model loaded successfully!\n")
    
    # Run inference
    if args.interactive:
        interactive_mode(model, tokenizer, config)
    else:
        if not args.instruction:
            print("Error: --instruction is required in non-interactive mode")
            return
        
        # Single prompt inference
        prompt = format_prompt(args.instruction, config['dataset']['prompt_template'])
        print("Generating response...\n")
        response = generate_response(model, tokenizer, prompt, config)
        
        print("=" * 80)
        print("Instruction:")
        print(args.instruction)
        print("\nResponse:")
        print("=" * 80)
        print(response)
        print("=" * 80)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run inference with fine-tuned GPT OSS 20B")
    parser.add_argument(
        "--model-path",
        type=str,
        default="./models/fine_tuned_model",
        help="Path to fine-tuned model directory"
    )
    parser.add_argument(
        "--config",
        type=str,
        default="./config/training_config.yaml",
        help="Path to configuration file"
    )
    parser.add_argument(
        "--instruction",
        type=str,
        default=None,
        help="Instruction/prompt for single inference"
    )
    parser.add_argument(
        "--interactive",
        action="store_true",
        help="Run in interactive mode for continuous prompting"
    )
    parser.add_argument(
        "--load-in-4bit",
        action="store_true",
        help="Load model in 4-bit quantization for lower memory usage"
    )
    
    args = parser.parse_args()
    main(args)
