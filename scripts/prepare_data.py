#!/usr/bin/env python3
"""
Prepare training data for fine-tuning GPT OSS 20B.
This script converts raw data into the format expected by the training pipeline.
"""

import json
import argparse
from pathlib import Path
from typing import List, Dict
import pandas as pd


def load_raw_data(input_path: str) -> List[Dict]:
    """Load raw data from various formats (JSON, JSONL, CSV)."""
    input_path = Path(input_path)
    
    if input_path.suffix == '.jsonl':
        with open(input_path, 'r') as f:
            return [json.loads(line) for line in f]
    elif input_path.suffix == '.json':
        with open(input_path, 'r') as f:
            data = json.load(f)
            # Handle both list and dict formats
            if isinstance(data, dict) and 'data' in data:
                return data['data']
            return data
    elif input_path.suffix == '.csv':
        df = pd.read_csv(input_path)
        return df.to_dict('records')
    else:
        raise ValueError(f"Unsupported file format: {input_path.suffix}")


def format_for_training(raw_data: List[Dict], instruction_key: str = "instruction", 
                       response_key: str = "response") -> List[Dict]:
    """Format data for training with instruction and response fields."""
    formatted_data = []
    
    for item in raw_data:
        # Extract instruction and response with flexible key names
        instruction = item.get(instruction_key) or item.get("prompt") or item.get("input")
        response = item.get(response_key) or item.get("output") or item.get("completion")
        
        if instruction and response:
            formatted_data.append({
                "instruction": instruction.strip(),
                "response": response.strip()
            })
        else:
            print(f"Warning: Skipping item without valid instruction/response: {item}")
    
    return formatted_data


def split_data(data: List[Dict], train_ratio: float = 0.9) -> tuple:
    """Split data into training and evaluation sets."""
    split_idx = int(len(data) * train_ratio)
    return data[:split_idx], data[split_idx:]


def save_jsonl(data: List[Dict], output_path: str):
    """Save data in JSONL format."""
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    
    with open(output_path, 'w') as f:
        for item in data:
            f.write(json.dumps(item) + '\n')
    
    print(f"Saved {len(data)} examples to {output_path}")


def main(args):
    print("=" * 80)
    print("Preparing training data for GPT OSS 20B fine-tuning")
    print("=" * 80)
    
    # Load raw data
    print(f"\nLoading raw data from: {args.input}")
    raw_data = load_raw_data(args.input)
    print(f"Loaded {len(raw_data)} examples")
    
    # Format data
    print("\nFormatting data...")
    formatted_data = format_for_training(
        raw_data,
        instruction_key=args.instruction_key,
        response_key=args.response_key
    )
    print(f"Formatted {len(formatted_data)} valid examples")
    
    # Split data
    if args.eval_ratio > 0:
        print(f"\nSplitting data (train: {1-args.eval_ratio:.1%}, eval: {args.eval_ratio:.1%})")
        train_data, eval_data = split_data(formatted_data, 1 - args.eval_ratio)
        print(f"Train examples: {len(train_data)}")
        print(f"Eval examples: {len(eval_data)}")
    else:
        train_data = formatted_data
        eval_data = []
        print(f"\nUsing all {len(train_data)} examples for training (no evaluation split)")
    
    # Save processed data
    output_dir = Path(args.output_dir)
    save_jsonl(train_data, output_dir / "train.jsonl")
    
    if eval_data:
        save_jsonl(eval_data, output_dir / "eval.jsonl")
    
    # Display sample
    if train_data:
        print("\n" + "=" * 80)
        print("Sample training example:")
        print("=" * 80)
        print(json.dumps(train_data[0], indent=2))
    
    print("\n" + "=" * 80)
    print("Data preparation completed successfully!")
    print("=" * 80)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Prepare data for GPT OSS 20B fine-tuning")
    parser.add_argument(
        "--input",
        type=str,
        required=True,
        help="Path to raw data file (JSON, JSONL, or CSV)"
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default="./data/processed",
        help="Output directory for processed data"
    )
    parser.add_argument(
        "--instruction-key",
        type=str,
        default="instruction",
        help="Key name for instruction field in raw data"
    )
    parser.add_argument(
        "--response-key",
        type=str,
        default="response",
        help="Key name for response field in raw data"
    )
    parser.add_argument(
        "--eval-ratio",
        type=float,
        default=0.1,
        help="Ratio of data to use for evaluation (0-1)"
    )
    
    args = parser.parse_args()
    main(args)
