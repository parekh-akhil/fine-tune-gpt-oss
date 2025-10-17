#!/usr/bin/env python3
"""
Fine-tune GPT OSS 20B model using Unsloth for human-like responses.
This script uses efficient 4-bit quantization and LoRA for memory-efficient training.
"""

import os
import yaml
import argparse
from pathlib import Path
from datasets import load_dataset
from unsloth import FastLanguageModel
from trl import SFTTrainer
from transformers import TrainingArguments
import torch


def load_config(config_path: str) -> dict:
    """Load configuration from YAML file."""
    with open(config_path, 'r') as f:
        return yaml.safe_load(f)


def format_prompts(examples, prompt_template):
    """Format examples according to the prompt template."""
    texts = []
    for instruction, response in zip(examples['instruction'], examples['response']):
        text = prompt_template.format(instruction=instruction, response=response)
        texts.append(text)
    return {"text": texts}


def main(args):
    # Load configuration
    config = load_config(args.config)
    
    print("=" * 80)
    print("Fine-tuning GPT OSS 20B with Unsloth")
    print("=" * 80)
    
    # Initialize model with Unsloth
    print(f"\nLoading model: {config['model']['name']}")
    model, tokenizer = FastLanguageModel.from_pretrained(
        model_name=config['model']['name'],
        max_seq_length=config['model']['max_seq_length'],
        dtype=config['model']['dtype'],
        load_in_4bit=config['model']['load_in_4bit'],
    )
    
    # Add LoRA adapters
    print("\nAdding LoRA adapters for efficient fine-tuning...")
    model = FastLanguageModel.get_peft_model(
        model,
        r=config['lora']['r'],
        target_modules=config['lora']['target_modules'],
        lora_alpha=config['lora']['lora_alpha'],
        lora_dropout=config['lora']['lora_dropout'],
        bias=config['lora']['bias'],
        use_gradient_checkpointing="unsloth",
        random_state=42,
    )
    
    # Load dataset
    print(f"\nLoading training data from: {config['dataset']['train_file']}")
    dataset = load_dataset('json', data_files={
        'train': config['dataset']['train_file'],
        'eval': config['dataset']['eval_file'] if os.path.exists(config['dataset']['eval_file']) else None
    })
    
    # Limit samples if specified
    if config['dataset']['max_samples']:
        dataset['train'] = dataset['train'].select(range(min(config['dataset']['max_samples'], len(dataset['train']))))
    
    # Format dataset with prompt template
    print("\nFormatting dataset with prompt template...")
    dataset = dataset.map(
        lambda x: format_prompts(x, config['dataset']['prompt_template']),
        batched=True,
    )
    
    # Setup training arguments
    training_args = TrainingArguments(
        output_dir=config['training']['output_dir'],
        num_train_epochs=config['training']['num_train_epochs'],
        per_device_train_batch_size=config['training']['per_device_train_batch_size'],
        gradient_accumulation_steps=config['training']['gradient_accumulation_steps'],
        warmup_steps=config['training']['warmup_steps'],
        learning_rate=config['training']['learning_rate'],
        fp16=config['training']['fp16'],
        bf16=config['training']['bf16'],
        logging_steps=config['training']['logging_steps'],
        save_steps=config['training']['save_steps'],
        save_total_limit=config['training']['save_total_limit'],
        optim=config['training']['optim'],
        weight_decay=config['training']['weight_decay'],
        max_grad_norm=config['training']['max_grad_norm'],
        lr_scheduler_type=config['training']['lr_scheduler_type'],
        report_to="wandb" if config['wandb']['enabled'] else "none",
    )
    
    # Initialize trainer
    print("\nInitializing trainer...")
    trainer = SFTTrainer(
        model=model,
        tokenizer=tokenizer,
        train_dataset=dataset['train'],
        eval_dataset=dataset.get('eval'),
        dataset_text_field="text",
        max_seq_length=config['model']['max_seq_length'],
        args=training_args,
    )
    
    # Start training
    print("\n" + "=" * 80)
    print("Starting training...")
    print("=" * 80 + "\n")
    trainer.train()
    
    # Save the fine-tuned model
    output_path = Path(args.output) if args.output else Path("./models/fine_tuned_model")
    output_path.mkdir(parents=True, exist_ok=True)
    
    print(f"\nSaving fine-tuned model to: {output_path}")
    model.save_pretrained(str(output_path))
    tokenizer.save_pretrained(str(output_path))
    
    # Save model in 16-bit precision for inference (optional)
    if args.save_16bit:
        print("\nSaving model in 16-bit precision for inference...")
        model.save_pretrained_merged(
            str(output_path / "16bit"),
            tokenizer,
            save_method="merged_16bit",
        )
    
    print("\n" + "=" * 80)
    print("Training completed successfully!")
    print("=" * 80)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Fine-tune GPT OSS 20B with Unsloth")
    parser.add_argument(
        "--config",
        type=str,
        default="./config/training_config.yaml",
        help="Path to training configuration file"
    )
    parser.add_argument(
        "--output",
        type=str,
        default=None,
        help="Output directory for fine-tuned model (default: ./models/fine_tuned_model)"
    )
    parser.add_argument(
        "--save-16bit",
        action="store_true",
        help="Save model in 16-bit precision for faster inference"
    )
    
    args = parser.parse_args()
    main(args)
